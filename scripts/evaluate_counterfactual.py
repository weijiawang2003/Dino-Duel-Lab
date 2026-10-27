#!/usr/bin/env python3
"""Held-out action-selection test; selectors never see branch outcomes.

This evaluates a local macro intervention, NOT complete-match win rates. Every
root's branches stay together. Bootstrap conditions on this frozen collection
of pretrained RL policies and the fitted downstream predictor ensemble.
"""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dino_duel.counterfactual import PLANS,PLAN_NAMES,HORIZONS,atomic_json,sha256
from dino_duel.counterfactual_models import MacroSelector,VARIANTS


def read_roots(folder):
    rows=[]
    for p in sorted(folder.glob('root_*.npz')):
        with np.load(p,allow_pickle=False) as d:
            row={k:d[k].copy() for k in ('x','mask','labels','target','elapsed','attempted','rejected','purchased')}
        meta=json.loads(p.with_suffix('.json').read_text())
        row.update(root_id=p.stem,meta=meta);rows.append(row)
    if not rows:raise ValueError(f'Empty split {folder}')
    return rows


def bootstrap_mean(a,seed=61025,samples=5000):
    a=np.asarray(a,dtype=float);rng=np.random.default_rng(seed)
    if len(a)<2:return [None,None]
    draws=a[rng.integers(len(a),size=(samples,len(a)))].mean(1)
    return np.quantile(draws,[.025,.975]).tolist()


def u(labels):return (labels==1).astype(float)-(labels==2).astype(float)

def outcome_rates(labels):
    return {name:float((labels==k).mean()) for k,name in enumerate(('ongoing','win','loss','simultaneous'))}


def rank_accuracy(scores,truth):
    vals=[]
    for p,q in zip(scores,truth):
        a,b=np.triu_indices(len(p),1);diff=q[a]-q[b];keep=np.abs(diff)>=.25
        if not keep.any(): vals.append(np.nan);continue
        correct=(np.sign(p[a]-p[b])==np.sign(diff)).astype(float)
        correct[np.abs(p[a]-p[b])<1e-8]=.5
        vals.append(float(correct[keep].mean()))
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else None


def evaluate(out,seeds):
    out=Path(out);outdir=out/'evaluation';outdir.mkdir(parents=True,exist_ok=True)
    va=read_roots(out/'data/valid')
    valid_values=np.stack([u(r['labels'][:,:,-1]).mean(-1) for r in va])
    fixed=int(valid_values.mean(0).argmax())
    selectors={};weight_hashes={}
    for v in VARIANTS:
        for seed in seeds:
            path=out/'models'/f'{v}_seed{seed}'/'best.pt'
            selectors[f'{v}_seed{seed}']=MacroSelector([path]);weight_hashes[str(path.relative_to(out))]=sha256(path)
        selectors[v+'_ensemble']=MacroSelector([out/'models'/f'{v}_seed{s}'/'best.pt' for s in seeds])
    report={'schema':1,'fixed_plan_from_validation':PLAN_NAMES[fixed],'weight_hashes':weight_hashes,
            'primary':'Same-root change in P(win within 8s)-P(loss within 8s), versus original PPO continuation.',
            'ongoing_is_not_draw':True,'evaluation_half':'Second half of independent branch tapes for all primary comparisons.',
            'ci':'Unadjusted paired root bootstrap; conditional on frozen source policies and fitted predictors. Not RL training-seed uncertainty.',
            'source':'Frozen v1.2 policies. Predictor restarts share training data.', 'splits':{}}
    all_rows=[]
    for split in ('test','ood'):
        rows=read_roots(out/'data'/split);x=np.stack([r['x'] for r in rows]);mask=np.stack([r['mask'] for r in rows])
        labels=np.stack([r['labels'][:,:,-1] for r in rows]);N,C,R=labels.shape
        if R<2: raise ValueError('At least two independent repeats required for cross-fitted oracle')
        half=R//2;hold=labels[:,:,half:];hold_u=u(hold).mean(-1);all_u=u(labels).mean(-1)
        base=u(hold[:,0]).mean(-1);cross_oracle=u(labels[:,:,:half]).mean(-1).argmax(-1)
        choices={'ppo':np.zeros(N,dtype=int),'best_fixed':np.full(N,fixed,dtype=int),'empirical_oracle':cross_oracle}
        prediction={};probabilities={}
        for name,selector in selectors.items():
            prob=selector.probabilities(x,mask);score=prob[:,:,1]-prob[:,:,2]
            choices[name]=score.argmax(-1);prediction[name]=score;probabilities[name]=prob
        stats={}
        for name,choice in choices.items():
            selected=hold[np.arange(N),choice];selected_u=u(selected).mean(-1);delta=selected_u-base
            all_selected=labels[np.arange(N),choice]
            metric={'net_win_within_horizon':float(selected_u.mean()),'delta_vs_ppo':float(delta.mean()),
                    'delta_ci95':bootstrap_mean(delta),'rates':outcome_rates(selected),
                    'roots':N,'independent_repeats_per_root':R-half,
                    'choice_counts':{PLAN_NAMES[c]:int((choice==c).sum()) for c in range(C)},
                    'delta_all_repeats_sensitivity':float((u(all_selected).mean(-1)-all_u[:,0]).mean())}
            if name in prediction:
                metric['pairwise_rank_accuracy']=rank_accuracy(prediction[name],hold_u)
                p=probabilities[name]; logp=np.log(np.maximum(p,1e-9))
                metric['outcome_nll']=float(-np.take_along_axis(np.repeat(logp[:,:,None,:],R-half,axis=2),hold[...,None],axis=3).mean())
            stats[name]=metric
            for i,r in enumerate(rows):
                all_rows.append({'split':split,'root_id':r['root_id'],'source_seed':r['meta']['summary']['source_seed'],
                                 'opponent':r['meta']['summary']['opponent'],'selector':name,
                                 'choice':int(choice[i]),'plan':PLAN_NAMES[choice[i]],
                                 'chosen_net':float(selected_u[i]),'ppo_net':float(base[i]),'delta':float(delta[i]),
                                 'predicted_score':None if name not in prediction else float(prediction[name][i,choice[i]])})
        random_net=hold_u.mean(-1);stats['uniform_random_macro']={'net_win_within_horizon':float(random_net.mean()),
           'delta_vs_ppo':float((random_net-base).mean()),'delta_ci95':bootstrap_mean(random_net-base),
           'rates':outcome_rates(hold),'note':'Exact expectation over a uniform macro choice; no cherry-picked random seed.'}
        contrasts={}
        for other in ('latent_additive_ensemble','state_nonlinear_ensemble','outcome_only_ensemble'):
            va=hold_u[np.arange(N),choices['latent_nonlinear_ensemble']]
            vb=hold_u[np.arange(N),choices[other]]
            diff=va-vb
            contrasts['latent_nonlinear_ensemble-minus-'+other]={'mean_delta':float(diff.mean()),'ci95':bootstrap_mean(diff)}
        report['splits'][split]={'root_count':N,'candidate_count':C,'repeats':R,'metrics':stats,'paired_model_contrasts':contrasts,
            'optimistic_full_sample_best_delta':float((hold_u.max(-1)-base).mean()),
            'candidate_effect_root_fraction':float((hold_u.max(-1)-hold_u.min(-1)>=.25).mean()),
            'root_time_median':float(np.median([r['meta']['root']['state']['time'] for r in rows])),
            'root_side_counts':{str(seat):sum(r['meta']['root']['seat']==seat for r in rows) for seat in (0,1)},
            'explicit_attempts':int(sum(r['attempted'].sum() for r in rows)),
            'explicit_rejections':int(sum(r['rejected'].sum() for r in rows))}
        predictions={'root_ids':np.array([r['root_id'] for r in rows]),'labels':labels,
                     **{name:choice for name,choice in choices.items()},
                     **{name+'_scores':v for name,v in prediction.items()}}
        np.savez_compressed(outdir/f'{split}_predictions.npz',**predictions)
        print(json.dumps({'split':split,'metrics':{n:stats[n] for n in ('ppo','best_fixed','empirical_oracle','latent_nonlinear_ensemble','state_nonlinear_ensemble','outcome_only_ensemble')}}),flush=True)
    atomic_json(outdir/'summary.json',report)
    with (outdir/'root_selections.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0]);w.writeheader();w.writerows(all_rows)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'counterfactual');p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    a=p.parse_args();torch.set_num_threads(1);evaluate(a.out.resolve(),a.seeds)
if __name__=='__main__':main()
