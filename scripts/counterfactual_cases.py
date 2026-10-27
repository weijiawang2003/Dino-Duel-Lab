#!/usr/bin/env python3
"""Re-simulate explicitly selected success AND failure cases, with paired tapes."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dino_duel.counterfactual import branch_rollouts,PublicPolicy,PLANS,PLAN_NAMES,atomic_json
from dino_duel.replay import write_replay


def generate(out):
    out=Path(out);folder=out/'cases';folder.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((out/'evaluation/root_selections.csv').open()))
    subset=[r for r in rows if r['split']=='test' and r['selector']=='latent_nonlinear_ensemble' and int(r['choice'])!=0]
    selections=[]
    for label,sign in [('success',1),('failure',-1)]:
        options=sorted([r for r in subset if sign*float(r['delta'])>0],key=lambda r:r['root_id'])
        if not options: continue
        record=options[0];p=out/'data/test'/record['root_id']
        meta=json.loads(p.with_suffix('.json').read_text());root=meta['root'];candidate=int(record['choice'])
        with np.load(p.with_suffix('.npz')) as d:
            labels=d['labels'][:,:,-1].copy();truth=d['target'].copy();repeats=labels.shape[1]
        utility=lambda a:(a==1).astype(int)-(a==2).astype(int)
        eligible=np.flatnonzero(sign*(utility(labels[candidate])-utility(labels[0]))>0)
        held=[int(i) for i in eligible if i>=repeats//2]
        if not held:raise AssertionError('Nonzero held-out mean has no matching repetition')
        rep=held[0];own=PublicPolicy(ROOT/'elimination/runs/baseline_seed11/policy.pt')
        opp=PublicPolicy(ROOT/'elimination/runs'/meta['summary']['opponent']/'policy.pt')
        paths={}
        for name,c in [('ppo',0),('planned',candidate)]:
            result=branch_rollouts(root,own,opp,repeats,root['source_seed']+2**40,capture=(c,rep))
            np.testing.assert_array_equal(result['labels'][:,:,-1],labels)
            np.testing.assert_array_equal(result['target'],truth)
            frames=result['frames'];code=int(labels[c,rep]);censored=code==0
            reason='branch_observation_horizon' if censored else 'simultaneous_collision' if code==3 else 'collision'
            ep={'left_name':'PPO / '+PLAN_NAMES[c],'right_name':meta['summary']['opponent'],
                'left_seat':root['seat'],'seed':root['source_seed'],
                'winner_policy':{0:'censored',1:'left',2:'right',3:'draw'}[code],
                'censored':censored,'reason':reason,'seconds':frames[-1]['time']-frames[0]['time']}
            path=folder/f'{label}_{name}.html';write_replay(frames,ep,path);paths[name]=path.name
        selections.append({'case':label,'root':record['root_id'],'candidate':candidate,'plan':PLAN_NAMES[candidate],
                           'repetition':rep,'source_seed':root['source_seed'],'paired_replay_exactly_reproduced':True,
                           'mean_heldout_delta':float(record['delta']),'ppo_class':int(labels[0,rep]),
                           'planned_class':int(labels[candidate,rep]),'paths':paths})
    info={'selection_rule':'ID test split only: lowest root ID with non-PPO ensemble choice and positive/negative held-out mean difference; then first held-out repetition with that sign. These are selected illustrations, NOT unbiased performance estimates.',
          'cases':selections}
    atomic_json(folder/'selection.json',info);print(json.dumps(info,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'counterfactual');a=p.parse_args();torch.set_num_threads(1);generate(a.out.resolve())
