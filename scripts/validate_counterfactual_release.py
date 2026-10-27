#!/usr/bin/env python3
"""Validate portable artifacts without retraining or inspecting test outcomes for tuning."""
from __future__ import annotations
import argparse,json,sys,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dino_duel.counterfactual import PLANS,HORIZONS,sha256,atomic_json
from dino_duel.counterfactual_models import MacroSelector

def validate(out):
    out=Path(out);p=json.loads((out/'PROTOCOL.json').read_text());seeds={};branches=0;steps=0;counts={}
    assert p['env']['max_seconds'] is None
    assert p['plans']==PLANS.tolist() and p['horizons']==list(HORIZONS)
    assert p['source_code_sha256']==sha256(ROOT/'dino_duel/counterfactual.py')
    for name,digest in p['source_policy_hashes'].items():
        assert sha256(ROOT/'elimination/runs'/name/'policy.pt')==digest,f'Source policy changed: {name}'
    for split,count in p['counts'].items():
        paths=sorted((out/'data'/split).glob('root_*.npz'));assert len(paths)==count
        seeds[split]=set();counts[split]=len(paths)
        for path in paths:
            meta=json.loads(path.with_suffix('.json').read_text());seed=meta['root']['source_seed']
            assert seed not in seeds[split];seeds[split].add(seed)
            with np.load(path,allow_pickle=False) as d:
                assert d['target'].shape==(12,p['repeats'][split],3,53)
                labels=d['labels'];y=d['target'];assert np.isfinite(y).all()
                own=y[...,5]>.5;opp=y[...,29]>.5
                expected=np.where(own,np.where(opp,0,1),np.where(opp,2,3))
                assert np.array_equal(expected,labels),f'Outcome mismatch: {path}'
                assert np.all(d['rejected']<=d['attempted'])
                branches+=12*p['repeats'][split];steps+=int(d['simulated_steps'])
    names=list(seeds)
    for i,name in enumerate(names):
        for other in names[i+1:]:assert not seeds[name]&seeds[other]
    rows=json.loads((out/'training_summary.json').read_text())
    weight_checks=[]
    for row in rows:
        path=out/'models'/f'{row["variant"]}_seed{row["seed"]}'/'best.pt'
        selector=MacroSelector([path]);probs=selector.probabilities(np.zeros((1,53),np.float32),np.ones((1,5),bool))
        assert np.isfinite(probs).all();np.testing.assert_allclose(probs.sum(-1),1,atol=1e-6)
        weight_checks.append({'variant':row['variant'],'seed':row['seed'],'best_sha256':sha256(path)})
    online_validation={}
    online_seeds=[]
    import csv
    for commitment in (1,20):
        directory=out/f'online_commit{commitment}'
        if not (directory/'summary.json').exists():continue
        protocol=json.loads((directory/'PROTOCOL.json').read_text())
        table=list(csv.DictReader((directory/'episodes.csv').open()))
        summary=json.loads((directory/'summary.json').read_text())
        assert protocol['game_time_limit'] is None
        for method,m in summary.items():
            rr=[r for r in table if r['method']==method]
            assert len(rr)==m['games']
            assert sum(r['winner_policy']=='left' for r in rr)==m['wins']
            assert sum(r['winner_policy']=='right' for r in rr)==m['losses']
            assert sum(r['censored']=='True' for r in rr)==m['unresolved']
            assert all(r['truncated']=='False' for r in rr)
            if commitment==1 and method!='ppo':
                assert sum(int(r['left_attack_commands']) for r in rr)==0
        current_seeds={int(r['environment_seed']) for r in table}
        for other in seeds.values():assert not current_seeds & other
        for other in online_seeds:assert not current_seeds & other
        online_seeds.append(current_seeds)
        online_validation[str(commitment)]={'games':len(table),'source_seeds_disjoint':True,
                                           'summary_matches_raw_rows':True}
    raw=json.loads((out/'collection_summary.json').read_text())
    assert branches==raw['branches'] and steps==raw['simulated_steps']
    result={'status':'passed','roots':sum(counts.values()),'counts':counts,'branches':branches,'simulated_steps':steps,
       'disjoint_source_episode_seeds':True,'future_alive_flags_match_outcome_labels':True,
       'all_source_policies_unchanged':True,'loaded_predictors':len(rows),'weight_checks':weight_checks,'online':online_validation,
       'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform()}
    atomic_json(out/'artifact_validation.json',result);print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'counterfactual');a=p.parse_args();torch.set_num_threads(1);validate(a.out.resolve())
