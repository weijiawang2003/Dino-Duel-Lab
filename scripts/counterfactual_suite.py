#!/usr/bin/env python3
"""Reproducible v1.3 branch dataset. Safe restart: each completed root is cached."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from dino_duel.counterfactual import *

_CACHE={}
def worker_init():
    import torch
    torch.set_num_threads(1)

def getpolicy(name):
    if name not in _CACHE: _CACHE[name]=PublicPolicy(ROOT/'elimination/runs'/name/'policy.pt')
    return _CACHE[name]

def collect_task(task):
    out=Path(task['out']); split=task['split']; index=task['index']
    folder=out/'data'/split;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'root_{index:05d}.npz'; meta=path.with_suffix('.json')
    signature={k:v for k,v in task.items() if k!='out'}
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved['signature']!=signature: raise ValueError(f'Cache mismatch {path}')
        return saved['summary']
    started=time.monotonic();seed=task['seed']; opponent=task['opponent']
    own=getpolicy('baseline_seed11'); opp=getpolicy(opponent)
    root=choose_root(own,opp,task['env'],seed,task['warmup_steps'])
    res=branch_rollouts(root,own,opp,task['repeats'],seed+2**40)
    frames=res.pop('frames')
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**res);tmp.replace(path)
    summary={'split':split,'index':index,'source_seed':seed,'opponent':opponent,
             'root_seconds':root['state']['time'],'repeats':task['repeats'],
             'simulated_steps':int(res['simulated_steps']),
             'source_steps_simulated':root['source_steps_simulated'],
             'seconds':time.monotonic()-started}
    atomic_json(meta,{'signature':signature,'root':root,'summary':summary})
    return summary

def collect(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/elimination.json').read_text())['env']
    sources=['baseline_seed11','jepa_seed11','baseline_seed23','jepa_seed23','baseline_seed47','jepa_seed47']
    hashes={n:sha256(ROOT/'elimination/runs'/n/'policy.pt') for n in sources}
    protocol={'schema':1,'source_archive_sha256':'b691398d11748cc28cbfa8cfafa971c6fe8629a39332ecf9c2cd223cb64eb9d7',
       'env':cfg,'controller':'baseline_seed11','model_inputs':['public_observation','current_legal_mask','intended_own_macro'],
       'opponent_reacts_in_each_branch':True,'oracle_future_actions':False,
       'motor_mode':'greedy','attack_mode':'sampled_common_random_tapes',
       'plan_names':PLAN_NAMES,'plans':PLANS.tolist(),'horizons':HORIZONS,
       'counts':{'train':args.train,'valid':args.valid,'test':args.test,'ood':args.ood},
       'repeats':{'train':args.train_repeats,'valid':args.eval_repeats,'test':args.eval_repeats,'ood':args.eval_repeats},
       'source_policy_hashes':hashes,'source_code_sha256':sha256(ROOT/'dino_duel/counterfactual.py'),
       'new_training_seeds':[101,202,303], 'warmup_steps':args.warmup_steps,
       'selection':'Minimum validation outcome NLL; test/ood never select epochs or hyperparameters.',
       'score':'P(win within 8s) - P(loss within 8s). Ongoing remains separately reported, NOT a draw.',
       'sampling_unit':'One snapshot per source episode, all its branches remain in one split.',
       'oracle':'Empirical oracle uses first half repeats to select; evaluate on second half. Full-max is only optimistic ceiling.',
       'scope':'Frozen trained-policy population, attack macros only. Downstream predictor restarts are NOT independent full RL training repetitions.'}
    p=out/'PROTOCOL.json'
    if p.exists() and json.loads(p.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Existing protocol differs; use a new output directory')
    atomic_json(p,protocol);tasks=[]
    for j,(split,n) in enumerate(protocol['counts'].items()):
        pool=sources[4:] if split=='ood' else sources[:4]
        for i in range(n):
            tasks.append({'out':str(out),'split':split,'index':i,'seed':2**44+j*2**28+i,
                'opponent':pool[i%len(pool)],'env':cfg,'repeats':protocol['repeats'][split],
                'warmup_steps':args.warmup_steps,'code':protocol['source_code_sha256'],
                'own_sha':hashes['baseline_seed11'],'opponent_sha':hashes[pool[i%len(pool)]]})
    t0=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as ex:
        futures=[ex.submit(collect_task,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            row=f.result();rows.append(row)
            if i%8==0 or i==len(tasks):
                print(json.dumps({'done':i,'total':len(tasks),'elapsed':round(time.monotonic()-t0,1),'last':row}),flush=True)
    rows.sort(key=lambda r:(r['split'],r['index']))
    atomic_json(out/'collection_summary.json',{'roots':len(rows),'branches':sum(len(PLANS)*r['repeats'] for r in rows),
        'simulated_steps':sum(r['simulated_steps'] for r in rows),
        'source_steps':sum(r['source_steps_simulated'] for r in rows),
        'wall_seconds':time.monotonic()-t0,'roots_detail':rows})

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('--out',type=Path,default=ROOT/'counterfactual');c.add_argument('--workers',type=int,default=3)
    for k,n in [('train',512),('valid',96),('test',128),('ood',128)]:c.add_argument('--'+k,type=int,default=n)
    c.add_argument('--train-repeats',type=int,default=3);c.add_argument('--eval-repeats',type=int,default=8)
    c.add_argument('--warmup-steps',type=int,default=360)
    a=p.parse_args()
    if any(v<1 for k,v in vars(a).items() if k in {'workers','train','valid','test','ood','train_repeats','eval_repeats','warmup_steps'}):p.error('Budgets must be positive')
    collect(a)
if __name__=='__main__':main()
