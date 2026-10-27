#!/usr/bin/env python3
"""Selected illustrations with exact paired resimulation checks, not extra tests."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dino_duel.counterfactual import PublicPolicy,branch_rollouts,PLANS,PLAN_NAMES,atomic_json
from dino_duel.replay import write_replay

def main():
    torch.set_num_threads(1);out=ROOT/'paired_advantage';folder=out/'cases';folder.mkdir(exist_ok=True)
    rows=list(csv.DictReader((out/'tree_diagnostic/per_root.csv').open()))
    own=PublicPolicy(ROOT/'elimination/runs/baseline_seed11/policy.pt');selected=[]
    for name,sign in [('tree_improves',1),('tree_worsens',-1)]:
        candidates=sorted([r for r in rows if r['split']=='fresh_id' and int(r['plan'])!=1 and sign*float(r['delta_wait'])>0],key=lambda r:int(r['root']))
        if not candidates:continue
        row=candidates[0];index=int(row['root']);candidate=int(row['plan'])
        stem=out/'data/fresh_id'/f'root_{index:05d}'
        meta=json.loads(stem.with_suffix('.json').read_text());root=meta['root']
        with np.load(stem.with_suffix('.npz')) as d:
            labels=d['labels'].copy();targets=d['target'].copy()
        utility=lambda a:(a==1).astype(int)-(a==2).astype(int)
        difference=utility(labels[candidate,:,-1])-utility(labels[1,:,-1])
        ix=np.flatnonzero(sign*difference>0);prefer=ix[ix>=len(difference)//2]
        rep=int(prefer[0] if len(prefer) else ix[0])
        opp=PublicPolicy(ROOT/'elimination/runs'/meta['summary']['opponent']/'policy.pt')
        selected_plans=PLANS[[1,candidate]];paths={}
        for pos,method in [(0,'fixed_wait'),(1,'tree')]:
            result=branch_rollouts(root,own,opp,labels.shape[1],root['source_seed']+2**40,plans=selected_plans,capture=(pos,rep))
            np.testing.assert_array_equal(result['labels'],labels[[1,candidate]])
            np.testing.assert_array_equal(result['target'],targets[[1,candidate]])
            plan=[1,candidate][pos];code=int(labels[plan,rep,-1]);frames=result['frames']
            ep=dict(left_name=method+' / '+PLAN_NAMES[plan],right_name=meta['summary']['opponent'],
                left_seat=root['seat'],seed=root['source_seed'],winner_policy={0:'censored',1:'left',2:'right',3:'draw'}[code],
                censored=code==0,reason='8s_prediction_window' if code==0 else 'collision',seconds=frames[-1]['time']-frames[0]['time'])
            path=folder/f'{name}_{method}.html';write_replay(frames,ep,path);paths[method]=path.name
        selected.append(dict(case=name,source_root=index,source_seed=root['source_seed'],plan=candidate,
            plan_name=PLAN_NAMES[candidate],repetition=rep,mean_delta_wait=float(row['delta_wait']),
            exact_resimulation=True,paths=paths))
    atomic_json(folder/'selection.json',dict(rule='Fresh-ID lowest root index with strictly positive/negative mean advantage, non-wait tree choice; first matching repeat in second half if available, otherwise first. Selected illustrations, NOT unbiased performance evidence.',cases=selected))
    print(json.dumps(selected),flush=True)
if __name__=='__main__':main()
