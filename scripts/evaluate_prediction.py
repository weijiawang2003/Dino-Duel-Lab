#!/usr/bin/env python3
"""Held-out latent-forecast check, NOT a planning or causal-intervention test.

Compare a JEPA predictor with target-space persistence and shuffled factual
future actions. Future opponent actions are privileged evaluation labels and
never supplied to the acting policy. Overlapping windows are not independent.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from dino_duel.env import DuelEnv
from dino_duel.learning import load_policy
from elimination_suite import policy


def evaluate_seed(out,seed,episodes=10,samples=512):
    folder=out/'prediction'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    jp=out/f'runs/jepa_seed{seed}/policy.pt';bp=out/f'runs/baseline_seed{seed}/policy.pt'
    trained=load_policy(jp);model=trained.model;model.eval()
    left=policy(str(jp),'joint');right=policy(str(bp),'joint')
    horizons=list(model.prediction_horizons);H=max(horizons)
    trajectories=[];candidates=[]
    for ep in range(episodes):
        e=DuelEnv(trained.config['env']);envseed=2**37+ep
        e.reset(envseed);left.reset(envseed*2+1009);right.reset(envseed*2+1010)
        observations=[e.observe()[0].copy()];actions=[];terminal=False
        for _ in range(int(120/(e.config.dt*e.config.action_repeat))):
            obs=e.observe();m=e.attack_masks();ro=e.runner_observations()
            a=np.concatenate([left.act(obs[:1],m[:1],ro[:1]),right.act(obs[1:],m[1:],ro[1:])])
            new,_,term,trunc,_=e.step(a)
            actions.append(a.reshape(4));observations.append(new[0].copy())
            if term or trunc:terminal=term;break
        index=len(trajectories);trajectories.append((np.asarray(observations),np.asarray(actions),terminal))
        candidates.extend((index,t) for t in range(max(0,len(actions)-min(horizons)+1)))
    if not candidates:raise RuntimeError('No sufficiently long forecast windows')
    rng=np.random.default_rng(2**38+seed)
    selected=[candidates[i] for i in rng.choice(len(candidates),min(samples,len(candidates)),replace=False)]
    n=len(selected);x=np.zeros((n,53),np.float32);a=np.zeros((n,H,4),np.int64)
    target=np.zeros((n,len(horizons),53),np.float32);valid=np.zeros((n,len(horizons)),bool)
    for i,(ep,t) in enumerate(selected):
        states,acts,terminal=trajectories[ep];x[i]=states[t]
        avail=min(H,len(acts)-t);a[i,:avail]=acts[t:t+avail]
        for k,h in enumerate(horizons):
            if t+h<=len(acts):target[i,k]=states[t+h];valid[i,k]=True
    x=torch.tensor(x);a=torch.tensor(a);target=torch.tensor(target)
    with torch.no_grad():
        z=model.encoder(x);truth=model.target_encoder(target)
        persist=model.target_encoder(x)[:,None,:].expand_as(truth)
        def predict(actions):
            encoded=torch.cat([F.one_hot(actions[...,0],3),F.one_hot(actions[...,1],5),
                               F.one_hot(actions[...,2],3),F.one_hot(actions[...,3],5)],-1).float()
            tr,_=model.predictor(encoded)
            context=tr[:,[h-1 for h in horizons]]
            return model.prediction_head(torch.cat([z[:,None,:].expand(-1,len(horizons),-1),context],-1))
        predictions={'learned':predict(a),'persistence':persist,
                     'shuffled_actions':predict(a[torch.tensor(rng.permutation(n))])}
        losses={name:F.smooth_l1_loss(pred,truth,reduction='none').mean(-1).numpy()
                for name,pred in predictions.items()}
    rows=[];summary=[]
    for k,h in enumerate(horizons):
        keep=valid[:,k];r={'horizon_steps':h,'horizon_seconds':h*trained.config['env']['dt']*trained.config['env']['action_repeat'],'windows':int(keep.sum())}
        for name,loss in losses.items():r[name+'_huber']=float(loss[keep,k].mean()) if keep.any() else None
        base=r['persistence_huber'];r['relative_improvement_vs_persistence']=1-r['learned_huber']/base if base is not None and base>0 else None
        summary.append(r)
        for i,(ep,t) in enumerate(selected):
            if keep[i]:rows.append({'episode':ep,'start_step':t,'horizon':h,**{name:float(loss[i,k]) for name,loss in losses.items()}})
    result={'seed':seed,'episodes':episodes,'window_candidates':len(candidates),'selected_windows':n,
            'horizons':summary,'target_latent_std':float(truth.std(0,unbiased=False).mean()),
            'policy_sha256':hashlib.sha256(jp.read_bytes()).hexdigest(),
            'environment_seed_start':2**37,'oracle_future_actions':True,
            'limitations':'Frozen forecast diagnostic with actual future joint actions. No future information enters acting policies. Shuffling tests action conditioning, not off-policy causal validity. Overlapping windows are correlated; scores are not independent trial statistics. Different models have different latent spaces.'}
    (folder/'summary.json').write_text(json.dumps(result,indent=2))
    with (folder/'windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'elimination')
    p.add_argument('--episodes',type=int,default=10);p.add_argument('--samples',type=int,default=512)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,47]);a=p.parse_args()
    if a.episodes<1 or a.samples<1:p.error('Positive episode/sample budgets required')
    torch.set_num_threads(1);results=[]
    for seed in a.seeds:
        r=evaluate_seed(a.out.resolve(),seed,a.episodes,a.samples);results.append(r);print(json.dumps(r),flush=True)
    all_results=[json.loads(f.read_text()) for f in sorted((a.out/'prediction').glob('seed*/summary.json'))]
    (a.out/'prediction/summary.json').write_text(json.dumps(all_results,indent=2))
if __name__=='__main__':main()
