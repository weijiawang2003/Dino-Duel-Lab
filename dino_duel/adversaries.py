"""Evaluation-only attack programs, absent from training opponent sampling.

Both use the same distilled basic motor. No oracle observes model internals.
The public act interface supports a batch of one (evaluation usage).
"""
from __future__ import annotations
import numpy as np
from .policies import NumpyRunner

class ScriptedAdversary:
    def __init__(self,style,runner_path):
        if style not in {'wide','alternating'}:raise ValueError('Unknown adversary')
        self.style=style;self.runner=NumpyRunner(runner_path)
        self.name='heldout:'+style;self.reset()
    def reset(self,seed=0):
        self.index=0;self.rng=np.random.default_rng(seed)
    def act(self,obs,masks,runner_obs,deterministic=True):
        obs,masks,runner_obs=map(np.atleast_2d,(obs,masks,runner_obs))
        if len(obs)!=1:raise ValueError('Stateful evaluation adversary requires one seat')
        motor=int(self.runner(runner_obs)[0])
        desired=2 if self.style=='wide' else (2,3,1,3)[self.index%4]
        attack=desired if masks[0,desired] else 0
        if attack:self.index+=1
        return np.asarray([[motor,attack]],dtype=np.int64)


class NoAttackPolicy:
    """Evaluation intervention: keep a policy's motor decisions, suppress attacks."""
    def __init__(self,policy):
        self.policy=policy;self.name=policy.name+' [defense only]'
    def reset(self,seed=0):
        self.policy.reset(seed=seed)
    def act(self,obs,masks,runner_obs,deterministic=True):
        actions=self.policy.act(obs,masks,runner_obs,deterministic=deterministic).copy()
        actions[:,1]=0
        return actions
