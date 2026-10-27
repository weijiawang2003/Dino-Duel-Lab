"""Deployment adapter for the frozen paired-advantage experiment."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .paired_advantage import AdvantageSelector, WAIT
from .macro_planner import PlanningPolicy

class GuardedSelector:
    """Explicit abstention avoids treating a large threshold as a safety proof."""
    def __init__(self, selector=None, abstain=False):
        self.selector=selector; self.abstain=bool(abstain)
        if selector is None and not self.abstain:
            raise ValueError('A non-abstaining gate needs a selector')
    def select(self, observations, masks):
        if self.abstain:
            return np.full(len(np.atleast_2d(observations)), WAIT, dtype=np.int64)
        return self.selector.select(observations,masks)


def make_policy(project_root, experiment_out, method):
    root=Path(project_root);out=Path(experiment_out)
    controller=root/'elimination/runs/baseline_seed11/policy.pt'
    if method in ('ppo','fixed_wait'):
        p=PlanningPolicy(controller,commit_opportunities=20,fixed_plan=WAIT if method=='fixed_wait' else None)
    else:
        v=method.removesuffix('_raw').removesuffix('_gated')
        paths=[out/'models'/f'{v}_seed{s}/best.pt' for s in (101,202,303)]
        p=PlanningPolicy(controller,commit_opportunities=20)
        if method.endswith('_gated'):
            frozen=json.loads((out/'FROZEN_SELECTION.json').read_text())['selections'][v]
            abstain=frozen['threshold']==2.1
            selector=None if abstain else AdvantageSelector(paths,frozen['threshold'],frozen['penalty'])
            p.selector=GuardedSelector(selector,abstain)
        elif method.endswith('_raw'):
            p.selector=AdvantageSelector(paths)
        else:
            raise ValueError('Unknown method; choose explicit _raw or _gated')
    p.name=method
    return p

class TreeDiagnosticSelector:
    """Local trusted tree checkpoint; public features only, no simulation."""
    def __init__(self, path):
        import pickle
        with Path(path).open('rb') as f:
            self.model=pickle.load(f)
    def select(self, observations, masks):
        x=np.concatenate([np.atleast_2d(observations),np.atleast_2d(masks)],axis=-1).astype(np.float32)
        return self.model.predict(x).argmax(-1).astype(np.int64)
