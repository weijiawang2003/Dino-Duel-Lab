#!/usr/bin/env python3
"""New full-duel diagnostic, choices frozen before fresh-test outcome inspection.

The raw neural policies were rejected by the validation gate. They are retained
here as explicit diagnostics, not deployed improvements. A separate tree model
is a non-JEPA sanity baseline. All candidates have the same motor and 20-slot
commitment. Unfinished games remain censored, not draws.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,argparse,json,time,csv
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from dino_duel.counterfactual import atomic_json,sha256
from dino_duel.paired_control import make_policy,TreeDiagnosticSelector
from dino_duel.macro_planner import PlanningPolicy
from dino_duel.evaluation import play_episode
from paired_advantage_suite import load_roots,bootstrap

METHODS=['ppo','fixed_wait','advantage_only_raw','latent_aux_raw','state_aux_raw','tree_diagnostic']
OPPONENTS=['baseline_seed23','jepa_seed23','baseline_seed47','jepa_seed47']
_CACHE={}

def worker(task):
    import torch
    torch.set_num_threads(1)
    out=Path(task['out']);directory=out/'online';method=task['method'];opp=task['opponent']
    path=directory/'episodes'/f"{method}__{opp}__{task['seed']}__{task['swap']}.json"
    if path.exists():
        row=json.loads(path.read_text())
        if row['protocol_sha256']!=task['protocol_sha256']:raise ValueError('Online cache mismatch')
        return row
    if method not in _CACHE:
        if method=='tree_diagnostic':
            policy=PlanningPolicy(ROOT/'elimination/runs/baseline_seed11/policy.pt',commit_opportunities=20)
            policy.selector=TreeDiagnosticSelector(out/'tree_diagnostic/model.pkl')
        else:policy=make_policy(ROOT,out,method)
        policy.name=method;_CACHE[method]=policy
    key='opponent:'+opp
    if key not in _CACHE:
        _CACHE[key]=PlanningPolicy(ROOT/'elimination/runs'/opp/'policy.pt')
    p=_CACHE[method];o=_CACHE[key];tick=time.monotonic()
    row,_=play_episode(p,o,task['env'],task['seed'],left_seat=task['swap'],safety_seconds=task['safety_seconds'])
    row.update(method=method,opponent=opp,environment_seed=task['seed'],swap=task['swap'],
        planner_calls=p.selection_calls,plan_counts=p.plan_counts.tolist(),
        wall_seconds=time.monotonic()-tick,protocol_sha256=task['protocol_sha256'])
    atomic_json(path,row);return row


def run(out,workers,num_seeds):
    directory=out/'online';directory.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((out/'PROTOCOL.json').read_text())['env']
    frozen=json.loads((out/'FROZEN_SELECTION.json').read_text())
    tree=json.loads((out/'tree_diagnostic/FROZEN.json').read_text())
    if sha256(out/'tree_diagnostic/model.pkl')!=tree['model_sha256']:raise ValueError('Tree checkpoint changed')
    prot=dict(methods=METHODS,opponents=OPPONENTS,num_environment_seeds=num_seeds,
        seed_start=2**52+2**30,swap_seats=True,safety_seconds=300,game_time_limit=None,
        commitment_opportunities=20,prediction_window_seconds=8,
        frozen_selection_hash=sha256(out/'FROZEN_SELECTION.json'),tree_selection_hash=sha256(out/'tree_diagnostic/FROZEN.json'),
        env=cfg,code_sha256=sha256(__file__),adapter_sha256=sha256(ROOT/'dino_duel/paired_control.py'),
        analysis='Prospective secondary diagnostic. Raw neural selectors rejected by validation gate. No learned weights selected from fresh tests.',
        uncertainty_unit='environment seed cluster with all opponent/seat cases; fixed pretrained population, not independent RL fits')
    pp=directory/'PROTOCOL.json'
    if pp.exists() and json.loads(pp.read_text())!=prot:raise ValueError('Online protocol mismatch')
    atomic_json(pp,prot)
    if not (directory/'START_TIME.json').exists():
        atomic_json(directory/'START_TIME.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),
          'fresh_offline_results_existed':(out/'OFFLINE_RESULTS.json').exists()})
    tasks=[dict(out=str(out),method=m,opponent=o,seed=prot['seed_start']+i,swap=swap,env=cfg,
           safety_seconds=300,protocol_sha256=sha256(pp)) for i in range(num_seeds)
           for o in OPPONENTS for swap in (0,1) for m in METHODS]
    rows=[];tick=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(worker,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            rows.append(f.result())
            if i%12==0:print(json.dumps(dict(online_done=i,total=len(tasks),wall_seconds=time.monotonic()-tick)),flush=True)
    rows.sort(key=lambda r:(r['method'],r['opponent'],r['environment_seed'],r['swap']))
    with (directory/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for m in METHODS:
        rr=[r for r in rows if r['method']==m]
        win=sum(r['winner_policy']=='left' for r in rr);loss=sum(r['winner_policy']=='right' for r in rr)
        censored=sum(r['censored'] for r in rr);draw=len(rr)-win-loss-censored
        summary[m]=dict(games=len(rr),wins=win,losses=loss,simultaneous=draw,unresolved=censored,
            full_game_score_bounds=[(win+.5*draw)/len(rr),(win+.5*draw+censored)/len(rr)],
            within_300s_net_win=(win-loss)/len(rr),steps=sum(r['steps'] for r in rr),
            planner_calls=sum(r['planner_calls'] for r in rr),
            plan_counts=np.sum([r['plan_counts'] for r in rr],axis=0).tolist())
        if m!='fixed_wait':
            diffs=[]
            for i in range(num_seeds):
                s=prot['seed_start']+i
                def v(method):
                    rrr=[r for r in rows if r['method']==method and r['environment_seed']==s]
                    return np.mean([int(r['winner_policy']=='left')-int(r['winner_policy']=='right') for r in rrr])
                diffs.append(v(m)-v('fixed_wait'))
            summary[m]['paired_net_gain_vs_wait']=bootstrap(diffs)
    atomic_json(directory/'SUMMARY.json',summary);print(json.dumps(summary),flush=True)


def tree_offline(out):
    tree=TreeDiagnosticSelector(out/'tree_diagnostic/model.pkl');summary={};rows=[]
    f=json.loads((out/'tree_diagnostic/FROZEN.json').read_text())
    if sha256(out/'tree_diagnostic/model.pkl')!=f['model_sha256']:raise ValueError('Tree changed')
    for split in ('fresh_id','fresh_ood'):
        d=load_roots(out/'data'/split,False);u=d['u'].mean(-1)
        a=tree.select(d['x'][:,:53],d['x'][:,53:]);value=u[np.arange(len(u)),a]
        summary[split]=dict(gain_vs_wait=bootstrap(value-u[:,1]),gain_vs_ppo=bootstrap(value-u[:,0]),
            switches=int((a!=1).sum()),roots=len(u),net_within_8s=float(value.mean()),
            plan_counts=np.bincount(a,minlength=12).tolist())
        for i in range(len(u)):
            rows.append(dict(split=split,root=i,source_seed=d['metadata'][i]['source_seed'],opponent=d['metadata'][i]['opponent'],
                method='tree_diagnostic',plan=int(a[i]),net_within_8s=float(value[i]),delta_wait=float(value[i]-u[i,1]),delta_ppo=float(value[i]-u[i,0])))
    with (out/'tree_diagnostic/per_root.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    atomic_json(out/'tree_diagnostic/OFFLINE_RESULTS.json',summary);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','tree-offline']);p.add_argument('--out',type=Path,default=ROOT/'paired_advantage')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--num-seeds',type=int,default=8)
    a=p.parse_args()
    if a.command=='run':run(a.out.resolve(),a.workers,a.num_seeds)
    else:tree_offline(a.out.resolve())
if __name__=='__main__':main()
