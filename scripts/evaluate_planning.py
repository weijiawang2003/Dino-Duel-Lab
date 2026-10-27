#!/usr/bin/env python3
"""Fresh full-duel pilot. Horizon=8s is a prediction horizon, not a game timer."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from concurrent.futures import ProcessPoolExecutor,as_completed
from dino_duel.counterfactual import atomic_json,sha256

_CACHE={}
def task_worker(task):
    import torch
    torch.set_num_threads(1)
    from dino_duel.macro_planner import PlanningPolicy
    from dino_duel.evaluation import play_episode
    out=Path(task['out']);own=ROOT/'elimination/runs/baseline_seed11/policy.pt'
    key=task['method'];commit=task['commit']
    if key not in _CACHE:
        checkpoints=[] if key in ('ppo','fixed_wait') else [out/'models'/f'{key}_seed{s}/best.pt' for s in task['fit_seeds']]
        _CACHE[key]=PlanningPolicy(own,checkpoints,commit_opportunities=commit,
                                  fixed_plan=1 if key=='fixed_wait' else None)
    opp_key=task['opponent']
    if opp_key not in _CACHE:_CACHE[opp_key]=PlanningPolicy(ROOT/'elimination/runs'/opp_key/'policy.pt')
    left,right=_CACHE[key],_CACHE[opp_key]
    left.name=key;right.name=opp_key
    start=time.monotonic()
    episode,_=play_episode(left,right,task['env'],task['seed'],left_seat=int(task['swap']),
                           record=False,safety_seconds=task['safety_seconds'])
    return {'method':key,'opponent':opp_key,'environment_seed':task['seed'],'swap':task['swap'],
            'elapsed_wall_seconds':time.monotonic()-start,'planner_calls':left.selection_calls,
            'plan_counts':left.plan_counts.tolist(),**episode}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'counterfactual')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--num-seeds',type=int,default=8)
    p.add_argument('--safety-seconds',type=float,default=300);p.add_argument('--commit',type=int,default=1)
    p.add_argument('--methods',nargs='+',default=['ppo','latent_nonlinear','state_nonlinear','outcome_only'],
                   choices=['ppo','latent_nonlinear','state_nonlinear','outcome_only','fixed_wait'])
    p.add_argument('--seed-start',type=int,default=2**48)
    p.add_argument('--note',default='Prospective primary deployment test')
    p.add_argument('--fit-seeds',type=int,nargs='+',default=[101,202,303])
    a=p.parse_args();out=a.out.resolve();directory=out/f'online_commit{a.commit}';directory.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((out/'PROTOCOL.json').read_text())['env'];assert cfg['max_seconds'] is None
    methods=a.methods;opponents=['baseline_seed23','jepa_seed23','baseline_seed47','jepa_seed47']
    protocol={'methods':methods,'opponents':opponents,'environment_seed_start':a.seed_start,'num_seeds':a.num_seeds,
              'fit_seeds':a.fit_seeds,'commit_opportunities':a.commit,'safety_seconds':a.safety_seconds,
              'analysis_status':a.note,'game_time_limit':None,'env':cfg,'code_sha256':sha256(ROOT/'dino_duel/macro_planner.py'),
              'note':'New pilot on full duels; shared pretrained controller. Checkpoint selection fixed using offline validation only. External cutoff remains unresolved, not draw.'}
    atomic_json(directory/'PROTOCOL.json',protocol)
    tasks=[]
    for m in methods:
        for o in opponents:
            for i in range(a.num_seeds):
                for swap in (False,True):
                    tasks.append({'out':str(out),'method':m,'opponent':o,'seed':a.seed_start+i,'swap':swap,
                                  'env':cfg,'safety_seconds':a.safety_seconds,'commit':a.commit,'fit_seeds':a.fit_seeds})
    rows=[];t0=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(task_worker,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            rows.append(f.result())
            if i%16==0:print(json.dumps({'complete':i,'total':len(tasks),'elapsed_seconds':time.monotonic()-t0}),flush=True)
    rows.sort(key=lambda r:(r['method'],r['opponent'],r['environment_seed'],r['swap']))
    import csv
    with (directory/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for m in methods:
        rr=[r for r in rows if r['method']==m]
        summary[m]={'games':len(rr),'wins':sum(r['winner_policy']=='left' for r in rr),
                    'losses':sum(r['winner_policy']=='right' for r in rr),
                    'simultaneous':sum(r['winner_policy']=='draw' and not r['censored'] for r in rr),
                    'unresolved':sum(r['censored'] for r in rr),'planner_calls':sum(r['planner_calls'] for r in rr),
                    'wall_seconds_sum':sum(r['elapsed_wall_seconds'] for r in rr),
                    'attack_commands':sum(r['left_attack_commands'] for r in rr),
                    'steps':sum(r['steps'] for r in rr)}
    atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
