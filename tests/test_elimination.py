import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from dino_duel.env import DuelEnv, EnvConfig, OBS_DIM
from dino_duel.evaluation import play_episode, evaluate
from dino_duel.policies import TeacherPolicy, StochasticAttackPolicy
from dino_duel.learning import build_sequence_targets, train, load_policy, _load_checkpoint
from dino_duel.models import ActorCritic


def test_no_clock_is_not_episode_end():
    env=DuelEnv({'max_seconds':None,'natural_obstacles':False,'acceleration':0})
    for _ in range(1800):
        obs,r,term,trunc,info=env.step([[0,0],[0,0]])
    assert env.time>119 and not term and not trunc and info['reason']=='running'
    assert np.all(obs[:,52]==0) and not env._done


def test_external_budget_is_censored_not_draw():
    config={'max_seconds':None,'natural_obstacles':False,'acceleration':0}
    l=TeacherPolicy('none');r=TeacherPolicy('none')
    row,_=play_episode(l,r,config,4,safety_seconds=.2)
    assert row['censored'] and row['left_score'] is None
    assert not row['terminated'] and not row['truncated']
    assert row['winner_policy']=='censored'
    result=evaluate(l,r,config,[4,5],safety_seconds=.2)
    assert result['censored']==4 and result['draws']==0
    assert result['left_mean_score'] is None
    assert result['left_score_identification_bounds']==[0.,1.]


def test_collision_remains_true_terminal():
    config={'max_seconds':None,'natural_obstacles':False,'acceleration':0}
    env=DuelEnv(config);env._spawn(0,1,'player_1',.8)
    _,r,t,u,info=env.step([[0,0],[0,0]])
    assert t and not u and info['winner']==1
    assert info['collision_sources'][0]=='player_1'


def test_multistep_targets_do_not_cross_reset():
    T,N,S,D=6,1,2,3
    obs=np.zeros((T,N,S,D),np.float32)
    nxt=np.broadcast_to(np.arange(T)[:,None,None,None],(T,N,S,D)).copy().astype(np.float32)
    acts=np.zeros((T,N,S,4),dtype=np.int64)
    term=np.zeros((T,N),bool);term[2]=True
    batch={'obs':obs,'next_obs':nxt,'joint_actions':acts,'terminated':term,'truncated':np.zeros_like(term)}
    seq,tgt,valid=build_sequence_targets(batch,(1,2,3))
    assert valid[0,0,0,2]  # endpoint terminal permitted
    assert not valid[1,0,0,2]  # crosses terminal
    assert valid[2,0,0,0] and not valid[2,0,0,1]
    assert not valid[-1,0,0,1]
    assert tgt[0,0,0,2,0]==2


def test_multistep_gradient_and_motor_share_encoder():
    model=ActorCritic(hidden=16,variant='jepa',train_motor=True,residual_motor=True,prediction_horizons=(2,4))
    obs=torch.randn(8,OBS_DIM);nxt=torch.randn(8,2,OBS_DIM)
    action=torch.zeros(8,4,4,dtype=torch.long);valid=torch.ones(8,2,dtype=torch.bool)
    loss,m=model.auxiliary_loss(obs,action,nxt,valid);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert 'prediction_h4' in m
    model.zero_grad()
    with torch.no_grad(): model.motor_residual_head.weight.fill_(.1)
    model(obs)['motor_logits'].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.runner.parameters())


def test_multistep_end_to_end_resume_and_numpy(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config={'env':{'max_seconds':None,'natural_obstacles':False,'initial_coins':20},
            'model':{'hidden':16,'residual_motor':True,'prediction_horizons':[2,4],'motor_logit_scale':2},
            'training':{'updates':2,'num_envs':2,'rollout_steps':16,'epochs':1,'minibatch_size':16,
                        'train_motor':True,'bootstrap_time_limits':True,'motor_anchor_coef':.02,
                        'motor_entropy_coef':.001,'checkpoint_interval':1,'snapshot_interval':1}}
    runner=root/'assets/runner_bc.npz'
    train(config,tmp_path/'full','jepa',1,runner_path=runner)
    c=copy.deepcopy(config);c['training']['updates']=1
    train(c,tmp_path/'split','jepa',1,runner_path=runner)
    train(config,tmp_path/'split','jepa',1,resume=True)
    a=_load_checkpoint(tmp_path/'full/latest.pt');b=_load_checkpoint(tmp_path/'split/latest.pt')
    for k,v in a['model_state'].items(): assert torch.equal(v,b['model_state'][k]),k
    pol=load_policy(tmp_path/'full/policy.pt');np_pol=StochasticAttackPolicy(pol);np_pol.sampling_mode='greedy'
    env=DuelEnv(config['env']);obs=env.observe();mask=env.attack_masks();ro=env.runner_observations()
    assert np.array_equal(pol.act(obs,mask,ro),np_pol.act(obs,mask,ro))


def test_configurable_obstacles_respect_geometry():
    e=DuelEnv({'wide_cactus_width':2.2,'wide_cactus_height':1.2,'natural_obstacles':False})
    e._spawn(0,2,'player_1',22.)
    assert e.tracks[0][0]['width']==2.2 and e.tracks[0][0]['height']==1.2
    restored=DuelEnv(e.config);restored.set_state(e.get_state())
    assert restored.tracks==e.tracks


def test_default_geometry_remains_legacy():
    e=DuelEnv({'natural_obstacles':False})
    e._spawn(0,2,'player_1',22.)
    assert e.tracks[0][0]['width']==1.4 and e.tracks[0][0]['height']==.9
    state=e.get_state()
    del state['config']['wide_cactus_width'];del state['config']['wide_cactus_height']
    e.set_state(state)


def test_viability_mask_does_not_change_environment():
    e=DuelEnv({'natural_obstacles':False,'max_seconds':None})
    e._spawn(0,1,'player_1',3.)
    before=copy.deepcopy(e.get_state())
    mask=e.viable_motor_mask(0)
    assert mask.shape==(3,) and mask.any()
    assert e.get_state()==before


def test_censored_replay_no_false_winner(tmp_path):
    from dino_duel.replay import make_replay
    r=make_replay(TeacherPolicy('none'),TeacherPolicy('none'),
                  {'max_seconds':None,'natural_obstacles':False},1,tmp_path/'r.html',safety_seconds=.2)
    assert r['episode']['censored'] and r['episode']['left_score'] is None
    text=(tmp_path/'r.html').read_text()
    assert "Unresolved · observation budget" in text


def test_noop_resume_counts_residual_and_frozen_base_correctly(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config={'env':{'max_seconds':None,'natural_obstacles':False},
            'model':{'hidden':16,'residual_motor':True,'prediction_horizons':[2]},
            'training':{'updates':1,'num_envs':1,'rollout_steps':4,'epochs':1,'minibatch_size':4,'train_motor':True}}
    a=train(config,tmp_path/'run','jepa',7,runner_path=root/'assets/runner_adversarial.npz')
    b=train(config,tmp_path/'run','jepa',7,resume=True)
    for key in ('parameter_count','trainable_parameter_count','acting_parameter_count'):
        assert a[key]==b[key],(key,a[key],b[key])


def test_unseen_scripts_are_legal_and_attack_intervention_is_pure():
    from dino_duel.adversaries import ScriptedAdversary,NoAttackPolicy
    root=Path(__file__).resolve().parents[1]
    env=DuelEnv({'natural_obstacles':False,'initial_coins':40,'max_seconds':None})
    p=ScriptedAdversary('alternating',root/'assets/runner_adversarial.npz')
    q=NoAttackPolicy(ScriptedAdversary('wide',root/'assets/runner_adversarial.npz'))
    for _ in range(50):
        obs=env.observe();mask=env.attack_masks();ro=env.runner_observations()
        left=p.act(obs[:1],mask[:1],ro[:1]);right=q.act(obs[1:],mask[1:],ro[1:])
        assert mask[0,left[0,1]] and right[0,1]==0
        _,_,done,_,_=env.step(np.concatenate([left,right]))
        if done:break
