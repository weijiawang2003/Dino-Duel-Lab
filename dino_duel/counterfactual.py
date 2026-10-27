"""Public-observation, closed-loop counterfactual attack experiments.

Simulator copies are used ONLY to generate/score offline labels. Learned planners
receive (observation, current legal mask, intended own macro); never a future
opponent action, its policy ID, simulator state, or outcome. Greedy motors are
held fixed. Opponent attacks are sampled afresh from its branch-specific state.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .env import DuelEnv, OBS_DIM, RUNNER_DIM

# One token per purchase opportunity, 0.4s under the frozen protocol. -1 means
# consult the original PPO at that future state, NOT repeat a recorded action.
PLAN_NAMES = ('ppo', 'wait', 'spaced_cactus', 'double_cactus', 'wide_pair',
              'low_then_cactus', 'cactus_then_low', 'high_then_cactus',
              'delayed_double', 'delayed_wide', 'alternating', 'late_low')
PLANS = np.asarray([
    [-1,-1,-1,-1,-1,-1,-1,-1], [0,0,0,0,0,0,0,0],
    [1,0,1,0,1,0,1,0], [1,1,0,0,0,0,0,0],
    [2,0,0,2,0,0,0,0], [3,0,1,0,1,0,0,0],
    [1,0,3,0,0,0,0,0], [4,1,0,1,0,0,0,0],
    [0,0,1,1,0,0,0,0], [0,0,0,2,0,2,0,0],
    [1,3,1,3,0,0,0,0], [0,0,0,0,3,1,0,0]],dtype=np.int64)
HORIZONS = (30, 60, 120)


def atomic_json(path, data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(p)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PublicPolicy:
    """Inference-only NumPy copy. Fixed-index uniform tapes give paired noise.

    One draw exists per seat per step even when an action is forced. Thus a
    branch cannot shift later random draws just by changing the legal mask.
    """
    def __init__(self, path):
        from .learning import load_policy
        from .policies import StochasticAttackPolicy
        p=StochasticAttackPolicy(load_policy(path))
        self.layers=p.layers; self.head=p.head; self.motor=p.motor
        self.residual=p.motor_residual; self.scale=p.motor_logit_scale
        self.path=str(path); self.config=p.config

    def act(self, observations, masks, uniforms):
        x=np.asarray(observations,np.float32).reshape(-1,OBS_DIM)
        z=x
        for w,b in self.layers: z=np.tanh(z@w.T+b)
        w,b=self.head
        logits=np.where(masks,z@w.T+b,-np.inf)
        probs=np.exp((logits-logits.max(-1,keepdims=True)).astype(np.float64))
        probs/=probs.sum(-1,keepdims=True)
        # Avoid a possible roundoff sample of an illegal trailing action.
        cdf=np.cumsum(probs,-1); cdf[:,-1]=1.0
        attack=(np.asarray(uniforms)[:,None]>=cdf).sum(-1)
        motor=self.motor.logits(x[:,:RUNNER_DIM])*self.scale
        if self.residual is not None: motor=motor+z@self.residual[0].T+self.residual[1]
        return np.column_stack((motor.argmax(-1),attack)).astype(np.int64)


def outcome_code(env, seat):
    """0=ongoing at horizon, 1=win, 2=loss, 3=simultaneous collision.

    Ongoing is NOT a draw. Both yield zero signed *within-window* net wins, but
    remain separate classes in data and every report.
    """
    if not env._done: return 0
    if env.winner is None: return 3
    return 1 if env.winner==seat else 2


def choose_root(own, opponent, env_config, seed, warmup_steps=360):
    """One reservoir-sampled eligible snapshot per independent source episode."""
    env=DuelEnv(env_config); env.reset(seed)
    rng=np.random.default_rng(seed+1234567)
    tape=rng.random((warmup_steps,2)); seat=int(rng.integers(2))
    candidates=0; selected=None
    for t in range(warmup_steps):
        obs=env.observe(); masks=env.attack_masks()
        # Exclude trivial starting empty queues, retain all eligible phases.
        if t>=12 and t%env.config.attack_every_n_steps==0 and masks[seat,1:].any():
            candidates+=1
            if rng.integers(candidates)==0:
                selected={'state':env.get_state(),'seat':seat,'source_step':t,
                          'source_seed':seed,'source_candidates':candidates}
        acts=np.zeros((2,2),np.int64)
        acts[seat]=own.act(obs[seat:seat+1],masks[seat:seat+1],tape[t,seat:seat+1])[0]
        other=1-seat
        acts[other]=opponent.act(obs[other:other+1],masks[other:other+1],tape[t,other:other+1])[0]
        _,_,term,trunc,_=env.step(acts)
        if term or trunc: break
    if selected is None: raise RuntimeError(f'No eligible snapshot for seed {seed}')
    selected['source_candidates']=candidates
    selected['source_steps_simulated']=t+1
    return selected


def branch_rollouts(root, own, opponent, repeats=4, seed=0,
                    plans=PLANS, horizons=HORIZONS, capture=None):
    """Evaluate all candidate macros, reactively resampling both policies.

    capture=(candidate, repetition) optionally records that actual branch.
    Per-repetition tapes are shared ACROSS candidates (common random numbers).
    After the macro's 8 opportunities, the original PPO resumes reactively.
    Dead branches are absorbed, never reset or trained as subsequent episodes.
    """
    if repeats<1 or sorted(set(horizons))!=list(horizons) or min(horizons)<1:
        raise ValueError('Invalid repeats/horizons')
    C=len(plans); K=len(horizons); H=max(horizons); seat=root['seat']; other=1-seat
    cfg=root['state']['config']; every=cfg['attack_every_n_steps']
    if (root['state']['ticks']//cfg['action_repeat'])%every:
        raise ValueError('Root must be at a purchase decision boundary')
    envs=[]
    for _ in range(C*repeats):
        e=DuelEnv(cfg); e.set_state(root['state']); envs.append(e)
    xs=envs[0].observe()[seat].copy(); mask=envs[0].attack_masks()[seat].copy()
    tape=np.random.default_rng(seed).random((repeats,H,2))
    target=np.zeros((C,repeats,K,OBS_DIM),np.float32)
    labels=np.zeros((C,repeats,K),np.int64)
    elapsed=np.zeros((C,repeats,K),np.float32)
    attempted=np.zeros((C,repeats),np.int64); rejected=attempted.copy(); purchased=attempted.copy()
    frames=[]; capture_index=None if capture is None else capture[0]*repeats+capture[1]
    if capture_index is not None: frames.append(envs[capture_index].snapshot())
    active=list(range(C*repeats)); nsteps=0; react_hashes=hashlib.sha256()
    for t in range(H):
        if active:
            ob=np.stack([envs[i].observe() for i in active]); ma=np.stack([envs[i].attack_masks() for i in active])
            rep=np.asarray(active)%repeats
            own_a=own.act(ob[:,seat],ma[:,seat],tape[rep,t,seat])
            opp_a=opponent.act(ob[:,other],ma[:,other],tape[rep,t,other])
            react_hashes.update(opp_a.tobytes())
            new_active=[]
            for j,i in enumerate(active):
                c,r=divmod(i,repeats); a=np.zeros((2,2),np.int64)
                a[seat]=own_a[j]; a[other]=opp_a[j]
                if t%every==0 and t//every<plans.shape[1]:
                    wanted=int(plans[c,t//every])
                    if wanted>=0:
                        if wanted: attempted[c,r]+=1
                        if not ma[j,seat,wanted]: rejected[c,r]+=int(wanted>0); wanted=0
                        a[seat,1]=wanted
                before=envs[i].purchases[seat]
                _,_,term,trunc,_=envs[i].step(a); nsteps+=1
                purchased[c,r]+=envs[i].purchases[seat]-before
                if not (term or trunc): new_active.append(i)
                if capture_index==i: frames.append(envs[i].snapshot())
            active=new_active
        if t+1 in horizons:
            k=horizons.index(t+1)
            for i,e in enumerate(envs):
                c,r=divmod(i,repeats)
                target[c,r,k]=e.observe()[seat]
                labels[c,r,k]=outcome_code(e,seat)
                elapsed[c,r,k]=e.time-root['state']['time']
    return {'x':xs,'mask':mask,'target':target,'labels':labels,'elapsed':elapsed,
            'attempted':attempted,'rejected':rejected,'purchased':purchased,
            'simulated_steps':np.asarray(nsteps),'opponent_actions_hash':np.asarray(react_hashes.hexdigest()),
            'frames':frames}
