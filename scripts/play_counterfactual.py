#!/usr/bin/env python3
"""Play one recorded match with a frozen controller and optional macro planner."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from dino_duel.macro_planner import PlanningPolicy
from dino_duel.evaluation import play_episode
from dino_duel.replay import write_replay

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'counterfactual')
    p.add_argument('--method',choices=['ppo','fixed_wait','latent_nonlinear','state_nonlinear','outcome_only'],default='latent_nonlinear')
    p.add_argument('--opponent',choices=['baseline_seed11','jepa_seed11','baseline_seed23','jepa_seed23','baseline_seed47','jepa_seed47'],default='baseline_seed23')
    p.add_argument('--commit',type=int,default=20)
    p.add_argument('--seed',type=int,default=2**49)
    p.add_argument('--swap',action='store_true')
    p.add_argument('--safety-seconds',type=float,default=300)
    p.add_argument('--replay',type=Path,default=ROOT/'counterfactual/live_replay.html')
    a=p.parse_args();torch.set_num_threads(1)
    weights=[] if a.method in ('ppo','fixed_wait') else [a.out/'models'/f'{a.method}_seed{s}'/'best.pt' for s in (101,202,303)]
    own=PlanningPolicy(ROOT/'elimination/runs/baseline_seed11/policy.pt',weights,
                       commit_opportunities=a.commit,fixed_plan=1 if a.method=='fixed_wait' else None)
    opponent=PlanningPolicy(ROOT/'elimination/runs'/a.opponent/'policy.pt')
    own.name=a.method;opponent.name=a.opponent
    cfg=json.loads((a.out/'PROTOCOL.json').read_text())['env']
    row,frames=play_episode(own,opponent,cfg,a.seed,left_seat=int(a.swap),record=True,safety_seconds=a.safety_seconds)
    write_replay(frames,row,a.replay)
    a.replay.with_suffix('.json').write_text(json.dumps(row,indent=2))
    print(json.dumps({'replay':str(a.replay),'episode':row}),flush=True)
if __name__=='__main__':main()
