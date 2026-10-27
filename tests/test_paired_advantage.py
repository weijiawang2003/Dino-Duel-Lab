import inspect
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from dino_duel.paired_advantage import PairedAdvantageModel,AdvantageSelector,WAIT,VARIANTS
from dino_duel.paired_control import GuardedSelector

@pytest.mark.parametrize('variant',VARIANTS)
def test_wait_is_exact_zero(variant):
    m=PairedAdvantageModel(variant,auxiliary_dim=16)
    q,z=m(torch.randn(3,58))
    assert q.shape==(3,12)
    assert torch.equal(q[:,WAIT],torch.zeros(3))
    if z is not None:
        assert z.shape==(3,12,3,16)
        assert torch.equal(z[:,WAIT],torch.zeros(3,3,16))

def test_matching_shared_initialization():
    weights=[]
    for v in VARIANTS:
        torch.manual_seed(202)
        m=PairedAdvantageModel(v)
        weights.append({k:p for k,p in m.state_dict().items() if not k.startswith('auxiliary.')})
    for name in weights[0]:
        assert all(torch.equal(weights[0][name],w[name]) for w in weights[1:])

def test_effects_depend_on_current_state():
    torch.manual_seed(762);m=PairedAdvantageModel()
    q,_=m(torch.randn(2,58))
    assert not torch.allclose(q[0],q[1],atol=1e-6)

def test_public_inputs_only():
    assert list(inspect.signature(AdvantageSelector.select).parameters)==['self','observations','masks']

def test_reload_identical(tmp_path):
    torch.manual_seed(22);m=PairedAdvantageModel('latent_aux',auxiliary_dim=16).eval()
    p=tmp_path/'m.pt';torch.save(dict(model_config=m.config,model_state=m.state_dict(),x_mean=torch.zeros(58),x_std=torch.ones(58)),p)
    rng=np.random.default_rng(33);obs=rng.normal(size=(4,53)).astype(np.float32);masks=np.ones((4,5),bool)
    expected=m(torch.tensor(np.concatenate([obs,masks],-1).astype(np.float32)))[0].detach().numpy()
    actual=AdvantageSelector([p]).scores(obs,masks)
    np.testing.assert_allclose(actual,expected,atol=1e-7)

def test_guard_never_calls_model_when_abstaining():
    class MustNotRun:
        def select(self,*args):raise AssertionError('called a disabled predictor')
    g=GuardedSelector(MustNotRun(),True)
    np.testing.assert_array_equal(g.select(np.zeros((4,53)),np.ones((4,5))),np.full(4,WAIT))

def test_pairing_subtracts_same_random_repetition():
    u=np.array([[1,-1,0],[-1,0,1],[1,1,-1]],float)
    delta=u-u[WAIT:WAIT+1]
    np.testing.assert_array_equal(delta[WAIT],np.zeros(3))
    np.testing.assert_allclose(delta.mean(1),u.mean(1)-u[WAIT].mean())

def test_optimizer_restore_exact_next_step(tmp_path):
    torch.manual_seed(71);m=PairedAdvantageModel('latent_aux',auxiliary_dim=8)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    x=torch.randn(4,58);y=torch.randn(4,12);z=torch.randn(4,12,3,8)
    def step(mm,oo):
        a,b=mm(x);loss=(a-y).square().mean()+.1*(b-z).square().mean()
        oo.zero_grad();loss.backward();oo.step()
    step(m,opt)
    path=tmp_path/'checkpoint.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},path)
    ck=torch.load(path,weights_only=True)
    mm=PairedAdvantageModel('latent_aux',auxiliary_dim=8);mm.load_state_dict(ck['model'])
    oo=torch.optim.AdamW(mm.parameters(),lr=.001);oo.load_state_dict(ck['optimizer'])
    step(m,opt);step(mm,oo)
    for a,b in zip(m.parameters(),mm.parameters()):assert torch.equal(a,b)

def test_no_planner_may_inject_illegal_attack():
    from dino_duel.macro_planner import PlanningPolicy
    from dino_duel.env import DuelEnv,EnvConfig
    root=Path(__file__).resolve().parents[1];path=root/'elimination/runs/baseline_seed11/policy.pt'
    if not path.exists():pytest.skip('Bundled controller unavailable')
    p=PlanningPolicy(path,commit_opportunities=20)
    class Constant:
        def select(self,obs,masks):return np.full(len(obs),4,dtype=np.int64)
    p.selector=Constant()
    e=DuelEnv(EnvConfig(max_seconds=None,natural_obstacles=False,initial_coins=0,attack_every_n_steps=6))
    for t in range(20):
        obs=e.observe();mask=e.attack_masks();a=p.act(obs[:1],mask[:1])
        assert mask[0,a[0,1]]
        _,_,term,trunc,_=e.step(np.concatenate([a,np.zeros((1,2),dtype=np.int64)]))
        if term or trunc:break

def test_guarded_wait_matches_fixed_wait():
    from dino_duel.macro_planner import PlanningPolicy
    from dino_duel.env import DuelEnv,EnvConfig
    root=Path(__file__).resolve().parents[1];path=root/'elimination/runs/baseline_seed11/policy.pt'
    if not path.exists():pytest.skip('Bundled controller unavailable')
    a=PlanningPolicy(path,commit_opportunities=20,fixed_plan=WAIT)
    b=PlanningPolicy(path,commit_opportunities=20);b.selector=GuardedSelector(abstain=True)
    a.reset(33);b.reset(33);e=DuelEnv(EnvConfig(max_seconds=None,natural_obstacles=False,initial_coins=60))
    for t in range(140):
        obs=e.observe();mask=e.attack_masks();aa=a.act(obs[:1],mask[:1]);bb=b.act(obs[:1],mask[:1])
        np.testing.assert_array_equal(aa,bb)
        _,_,term,trunc,_=e.step(np.concatenate([aa,np.zeros((1,2),dtype=np.int64)]))
        if term or trunc:break

def test_tree_selector_public_features_only():
    from dino_duel.paired_control import TreeDiagnosticSelector
    class CheckInput:
        def predict(self,x):
            assert x.shape==(2,58)
            np.testing.assert_array_equal(x[:,:53],np.full((2,53),.5))
            np.testing.assert_array_equal(x[:,53:],np.ones((2,5)))
            q=np.zeros((2,12));q[:,8]=1;return q
    selector=TreeDiagnosticSelector.__new__(TreeDiagnosticSelector);selector.model=CheckInput()
    a=selector.select(np.full((2,53),.5),np.ones((2,5)))
    np.testing.assert_array_equal(a,np.full(2,8))

def test_bad_gate_and_input_rejected():
    with pytest.raises(ValueError):GuardedSelector(None,False)
    with pytest.raises(ValueError):PairedAdvantageModel()(torch.zeros(58))
