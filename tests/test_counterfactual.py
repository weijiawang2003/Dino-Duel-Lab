import copy,json
import numpy as np
import pytest
import torch
from dino_duel.counterfactual import branch_rollouts,PublicPolicy,PLANS
from dino_duel.env import DuelEnv,EnvConfig
from dino_duel.models import ActorCritic

class Simple:
    def __init__(self): self.seen=[]
    def act(self, obs, masks, uniforms):
        self.seen.append(obs.copy())
        a=np.zeros((len(obs),2),np.int64)
        # Stay grounded, allowing simple attacks to reveal branch differences.
        return a


def root():
    e=DuelEnv(EnvConfig(max_seconds=None,natural_obstacles=False,
        initial_coins=60,initial_speed=8,acceleration=0,attack_every_n_steps=6))
    e.reset(100)
    return {'state':e.get_state(),'seat':0,'source_seed':100}


def test_branch_order_invariant_and_no_snapshot_mutation():
    r=root(); before=json.dumps(r,sort_keys=True)
    plans=PLANS[[0,3,8]]
    a=branch_rollouts(r,Simple(),Simple(),2,30,plans=plans,horizons=(30,60))
    b=branch_rollouts(r,Simple(),Simple(),2,30,plans=plans[::-1],horizons=(30,60))
    np.testing.assert_array_equal(a['labels'],b['labels'][::-1])
    np.testing.assert_array_equal(a['target'],b['target'][::-1])
    assert json.dumps(r,sort_keys=True)==before
    assert a['labels'][0,0,-1]==0  # Ongoing, NOT a draw
    assert a['labels'][1,0,-1]==1  # attack eliminated opponent


def test_opponent_reacts_to_branch_specific_states():
    opp=Simple()
    a=branch_rollouts(root(),Simple(),opp,1,50,plans=PLANS[[0,3]],horizons=(30,60))
    assert any(len(o)>1 and not np.array_equal(o[0],o[1]) for o in opp.seen)
    assert np.all(a['elapsed'][1,0] <= 4.00001)


def test_no_reset_after_terminal():
    a=branch_rollouts(root(),Simple(),Simple(),1,50,plans=PLANS[[3]],horizons=(60,120))
    assert a['labels'][0,0,0] == a['labels'][0,0,1] == 1
    np.testing.assert_array_equal(a['target'][0,0,0],a['target'][0,0,1])
    assert a['elapsed'][0,0,0]==a['elapsed'][0,0,1]


def test_nonlinear_fusion_actually_interacts():
    torch.manual_seed(123)
    linear=ActorCritic(hidden=32,variant='jepa',prediction_horizons=(2,4))
    nonlinear=ActorCritic(hidden=32,variant='jepa',prediction_horizons=(2,4),predictor_fusion='nonlinear')
    state1=torch.randn(1,32);state2=torch.randn(1,32);a=torch.randn(1,32);b=torch.randn(1,32)
    def difference(m,s):return m.prediction_head(torch.cat((s,a),-1))-m.prediction_head(torch.cat((s,b),-1))
    assert torch.allclose(difference(linear,state1),difference(linear,state2),atol=1e-6)
    assert not torch.allclose(difference(nonlinear,state1),difference(nonlinear,state2),atol=1e-5)
    m=ActorCritic(**nonlinear.model_config)
    m.load_state_dict(nonlinear.state_dict())


def test_nonlinear_sequence_loss_gradients():
    m=ActorCritic(hidden=32,variant='jepa',prediction_horizons=(2,4),predictor_fusion='nonlinear')
    x=torch.randn(8,53);a=torch.zeros(8,4,4,dtype=torch.long);y=torch.randn(8,2,53)
    loss,_=m.sequence_auxiliary_loss(x,a,y,torch.ones(8,2,dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert m.prediction_head[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.target_encoder.parameters())


def test_boundary_required():
    r=root();r['state']['ticks']=4
    with pytest.raises(ValueError,match='boundary'):
        branch_rollouts(r,Simple(),Simple(),1,50,horizons=(30,))


def test_macro_public_input_shape_and_nonadditivity():
    from dino_duel.counterfactual_models import MacroWorldModel
    torch.manual_seed(72)
    linear=MacroWorldModel('latent_additive',16)
    nonlinear=MacroWorldModel('latent_nonlinear',16)
    x=torch.randn(2,58);plans=torch.tensor(PLANS[[0,3]])
    def effects(model):
        yy=model(x.repeat_interleave(2,0),plans.repeat(2,1)).reshape(2,2,3,16)
        return yy[:,1]-yy[:,0]
    e=effects(linear);assert torch.allclose(e[0],e[1],atol=1e-6)
    e=effects(nonlinear);assert not torch.allclose(e[0],e[1],atol=1e-5)


def test_selector_signature_has_no_hidden_future_input():
    import inspect
    from dino_duel.counterfactual_models import MacroSelector
    assert list(inspect.signature(MacroSelector.select).parameters)==['self','observations','masks']


def test_planner_roundtrip_and_legal_actions(tmp_path):
    from pathlib import Path
    from dino_duel.counterfactual_models import MacroWorldModel,OutcomeDecoder,MacroSelector
    from dino_duel.macro_planner import PlanningPolicy
    torch.manual_seed(22)
    m=MacroWorldModel('latent_nonlinear',16);d=OutcomeDecoder(16)
    p=tmp_path/'model.pt';torch.save({'model_config':m.config,'model_state':m.state_dict(),
        'decoder_state':d.state_dict(),'x_mean':torch.zeros(58),'x_std':torch.ones(58)},p)
    selector=MacroSelector([p]);x=np.zeros((2,53),np.float32);mask=np.ones((2,5),bool)
    prob=selector.probabilities(x,mask)
    assert prob.shape==(2,12,4);np.testing.assert_allclose(prob.sum(-1),1,atol=1e-6)
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'elimination/runs/baseline_seed11/policy.pt'
    if not checkpoint.exists():pytest.skip('Optional shipped policy not present')
    policy=PlanningPolicy(checkpoint,[p])
    env=DuelEnv(EnvConfig(max_seconds=None,natural_obstacles=False,initial_coins=60,attack_every_n_steps=6))
    for t in range(20):
        ob=env.observe();ma=env.attack_masks();a=policy.act(ob[:1],ma[:1])
        assert ma[0,a[0,1]]
        env.step(np.concatenate((a,np.zeros((1,2),dtype=int))))
    assert policy.selection_calls>0


def test_nonlinear_predictor_runs_in_real_ppo_trainer_and_resumes(tmp_path):
    from dino_duel.learning import train
    config={'env':{'max_seconds':0.4,'natural_obstacles':False},
            'model':{'hidden':16,'prediction_horizons':[2,4],'predictor_fusion':'nonlinear'},
            'training':{'updates':1,'num_envs':1,'rollout_steps':8,'epochs':1,
                        'minibatch_size':8,'checkpoint_interval':1,'snapshot_interval':1,
                        'opponent_pool_size':0,'device':'cpu'}}
    train(config,tmp_path/'jepa','jepa',seed=123)
    c=torch.load(tmp_path/'jepa/latest.pt',weights_only=True)
    assert c['model_config']['predictor_fusion']=='nonlinear'
    config['training']['updates']=2
    train(config,tmp_path/'jepa','jepa',seed=123,resume=True)
    c=torch.load(tmp_path/'jepa/latest.pt',weights_only=True)
    assert c['update']==2


def test_planner_selecting_ppo_reproduces_controller_exactly():
    from pathlib import Path
    from dino_duel.macro_planner import PlanningPolicy
    path=Path(__file__).resolve().parents[1]/'elimination/runs/baseline_seed11/policy.pt'
    if not path.exists():pytest.skip('Optional shipped policy not present')
    a=PlanningPolicy(path);b=PlanningPolicy(path)
    class AlwaysPPO:
        def select(self,obs,masks):return np.zeros(len(obs),dtype=int)
    b.selector=AlwaysPPO();a.reset(55);b.reset(55)
    env=DuelEnv(EnvConfig(max_seconds=None,natural_obstacles=False,initial_coins=60,attack_every_n_steps=6))
    for t in range(40):
        obs=env.observe();masks=env.attack_masks()
        aa=a.act(obs[:1],masks[:1]);bb=b.act(obs[:1],masks[:1])
        np.testing.assert_array_equal(aa,bb)
        _,_,term,_,_=env.step(np.concatenate((aa,np.zeros((1,2),dtype=int))))
        if term:break


def test_committed_plan_executes_wait_then_reactive_ppo_tail():
    from pathlib import Path
    from dino_duel.macro_planner import PlanningPolicy
    path=Path(__file__).resolve().parents[1]/'elimination/runs/baseline_seed11/policy.pt'
    if not path.exists():pytest.skip('Optional shipped policy not present')
    p=PlanningPolicy(path,attack_every=1,commit_opportunities=20,fixed_plan=1)
    class AlwaysAttack:
        def act(self,obs,masks,uniforms):
            return np.tile(np.array([[0,1]]),(len(obs),1))
    p.controller=AlwaysAttack()
    obs=np.zeros((1,53),np.float32); masks=np.ones((1,5),bool)
    actions=[int(p.act(obs,masks)[0,1]) for _ in range(41)]
    assert actions[:8]==[0]*8
    assert actions[8:20]==[1]*12
    assert actions[20:28]==[0]*8
    assert actions[28:40]==[1]*12
    assert actions[40]==0
    assert p.selection_calls==0
    assert p.plan_counts[1]==3
