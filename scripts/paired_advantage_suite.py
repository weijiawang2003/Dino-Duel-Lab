#!/usr/bin/env python3
"""v1.4 paired-advantage pilot. All work can run in the assistant's container.

Train/validation are reused from v1.3. New test roots are generated only after
fit and gate choices have been written to FROZEN_SELECTION.json. Raw selectors
are diagnostic; guarded ones may abstain to fixed wait. No test-based tuning.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse,copy,csv,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from concurrent.futures import ProcessPoolExecutor,as_completed
from dino_duel.counterfactual import PLANS, PLAN_NAMES, atomic_json, sha256, PublicPolicy, choose_root, branch_rollouts
from dino_duel.paired_advantage import PairedAdvantageModel, AdvantageSelector, VARIANTS, WAIT
BASE=ROOT/'counterfactual'
OUT=ROOT/'paired_advantage'
SEEDS=(101,202,303)


def save_pt(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.pt');torch.save(payload,tmp);tmp.replace(path)


def load_roots(folder, targets=True):
    xs=[]; utilities=[]; ys=[]; meta=[]
    for p in sorted(Path(folder).glob('root_*.npz')):
        with np.load(p,allow_pickle=False) as d:
            xs.append(np.concatenate([d['x'],d['mask'].astype(np.float32)]))
            ll=d['labels'][:,:,-1]
            utilities.append((ll==1).astype(np.float32)-(ll==2).astype(np.float32))
            if targets: ys.append(d['target'].copy())
        meta.append(json.loads(p.with_suffix('.json').read_text())['summary'])
    if not xs: raise ValueError(f'No roots in {folder}')
    return dict(x=np.asarray(xs,dtype=np.float32),u=np.stack(utilities),
                y=np.stack(ys) if targets else None,metadata=meta)


def protocol(epochs=140):
    return dict(schema=1, base_version='1.3.0',scope='paired baseline-relative selector, NOT new PPO training',
        source_archive_sha256=sha256(ROOT.parent.parent/'dino-duel-lab-v1.3.0-counterfactual.zip'),
        train_roots=512,validation_roots=96,train_repeats=3,validation_repeats=8,
        fitting_seeds=list(SEEDS),variants=list(VARIANTS),epochs_max=epochs,patience=30,
        learning_rate=0.001,weight_decay=0.0001,batch_roots=64,auxiliary_weight=0.1,
        auxiliary='paired future representation differences, averaged within root/candidate',
        primary_objective='mean squared error of E[within-8s net win(candidate)-net win(wait)]',
        checkpoint_selection='minimum validation paired-advantage MSE; no test/ood files opened',
        deploy='8 purchase opportunities of chosen macro + 12 opportunities of reactive PPO; commit=20',
        gate_thresholds=[0.0,0.05,0.1,0.2,0.3,0.5,2.1],gate_penalty=1.0,
        gate_selection='maximize validation root-bootstrap 10th-percentile gain; require positive lower bound and at least 10 switches, else abstain; not a safety guarantee',
        fresh_test_counts={'fresh_id':64,'fresh_ood':64},fresh_test_repeats=8,
        fresh_seed_start=2**51,source_episode_unit=True,env=json.loads((BASE/'PROTOCOL.json').read_text())['env'],
        score='P(win by 8s)-P(loss by 8s); ongoing is explicitly NOT a game draw',
        no_simulator_input=True,no_future_actions=True,
        auxiliary_target_encoder_sha256=sha256(ROOT/'elimination/runs/jepa_seed11/policy.pt'))


def prepare(out,epochs):
    out.mkdir(parents=True,exist_ok=True)
    p=protocol(epochs)
    if (out/'PROTOCOL.json').exists():
        if json.loads((out/'PROTOCOL.json').read_text())!=p: raise ValueError('Protocol mismatch; do not overwrite experiment')
    else: atomic_json(out/'PROTOCOL.json',p)
    if (out/'prepared.pt').exists(): return
    torch.set_num_threads(1)
    tr=load_roots(BASE/'data/train');va=load_roots(BASE/'data/valid')
    if len(tr['x'])!=512 or len(va['x'])!=96:raise ValueError('Unexpected training/validation size')
    prep_old=torch.load(BASE/'prepared/targets.pt',map_location='cpu',weights_only=True)
    encoder=torch.nn.Sequential(torch.nn.Linear(53,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh())
    encoder.load_state_dict(prep_old['encoder']);encoder.eval().requires_grad_(False)
    xmean=torch.as_tensor(tr['x']).mean(0);xstd=torch.as_tensor(tr['x']).std(0,unbiased=False).clamp_min(.1)
    data={};scales={}
    for name,d in [('train',tr),('valid',va)]:
        u=torch.as_tensor(d['u']);adv=(u-u[:,WAIT:WAIT+1]).mean(-1)
        target=torch.as_tensor(d['y'])
        with torch.no_grad():latent=encoder(target)
        pair_state=(target-target[:,WAIT:WAIT+1]).mean(2)
        pair_latent=(latent-latent[:,WAIT:WAIT+1]).mean(2)
        if name=='train':
            scales={'state_aux':pair_state.square().mean((0,1,2)).sqrt().clamp_min(.1),
                    'latent_aux':pair_latent.square().mean((0,1,2)).sqrt().clamp_min(.1)}
        data[name]=dict(x=(torch.as_tensor(d['x'])-xmean)/xstd,adv=adv,u=u,
                        state_aux=pair_state/scales['state_aux'],latent_aux=pair_latent/scales['latent_aux'])
    save_pt(out/'prepared.pt',dict(data=data,x_mean=xmean,x_std=xstd,scales=scales))
    atomic_json(out/'DATA_PROVENANCE.json',{'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'files_read':['counterfactual/data/train','counterfactual/data/valid','counterfactual/prepared/targets.pt'],
        'train_sources':[m['source_seed'] for m in tr['metadata']], 'valid_sources':[m['source_seed'] for m in va['metadata']],
        'train_metadata':tr['metadata'],'valid_metadata':va['metadata'],
        'prepared_sha256':sha256(out/'prepared.pt'),'code_sha256':sha256(__file__),
        'model_code_sha256':sha256(ROOT/'dino_duel/paired_advantage.py')})


def fit_task(task):
    torch.set_num_threads(1);out=Path(task['out']);variant=task['variant'];seed=task['seed']
    folder=out/'models'/f'{variant}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    sign={'variant':variant,'seed':seed,'prepared':sha256(out/'prepared.pt'),'code':sha256(__file__),
          'model':sha256(ROOT/'dino_duel/paired_advantage.py'),'epochs':task['epochs']}
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text())
        if summary['signature']!=sign: raise ValueError('Fit cache signature mismatch')
        return summary
    prep=torch.load(out/'prepared.pt',map_location='cpu',weights_only=True)
    tr,va=prep['data']['train'],prep['data']['valid']
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    dim=53 if variant=='state_aux' else 128
    m=PairedAdvantageModel(variant,auxiliary_dim=dim)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');history=[];best_epoch=0;start=0;elapsed_before=0
    if (folder/'latest.pt').exists():
        ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if ck['signature']!=sign:raise ValueError('Resume signature mismatch')
        m.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['torch_rng']);rng.bit_generator.state=ck['numpy_rng']
        start=ck['epoch'];best=ck['best_validation_mse'];best_epoch=ck['best_epoch'];history=ck['history'];elapsed_before=ck['elapsed_seconds']
    tick=time.monotonic()
    def payload(epoch):
        return dict(model_config=m.config,model_state=m.state_dict(),x_mean=prep['x_mean'],x_std=prep['x_std'],
            signature=sign,epoch=epoch,best_validation_mse=best,best_epoch=best_epoch,
            elapsed_seconds=elapsed_before+time.monotonic()-tick)
    for epoch in range(start,task['epochs']):
        m.train();losses=[]
        for ix in np.array_split(rng.permutation(len(tr['x'])),8):
            pred,aux=m(tr['x'][ix]);primary=F.mse_loss(pred,tr['adv'][ix])
            auxiliary=pred.sum()*0 if aux is None else F.mse_loss(aux,tr[variant][ix])
            loss=primary+.1*auxiliary
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
            losses.append(float(loss.detach()))
        m.eval()
        with torch.no_grad():
            pred,aux=m(va['x']);mse=float(F.mse_loss(pred,va['adv']))
            choices=pred.argmax(1);gain=float(va['adv'][torch.arange(len(choices)),choices].mean())
            switches=int((choices!=WAIT).sum())
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_mse=mse,
                            validation_raw_gain=gain,validation_raw_switches=switches))
        if mse<best:
            best=mse;best_epoch=epoch+1;save_pt(folder/'best.pt',payload(epoch+1))
        stop=(epoch+1>=40 and epoch+1-best_epoch>=30)
        if (epoch+1)%10==0 or epoch+1==task['epochs'] or stop:
            ck=payload(epoch+1);ck.update(optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,history=history)
            save_pt(folder/'latest.pt',ck);atomic_json(folder/'history.json',history)
        if stop:break
    summary=dict(variant=variant,seed=seed,epochs_ran=epoch+1,best_epoch=best_epoch,
        validation_mse=best,parameters=sum(p.numel() for p in m.parameters()),
        shared_parameters=sum(p.numel() for n,p in m.named_parameters() if not n.startswith('auxiliary.')),
        seconds=elapsed_before+time.monotonic()-tick,signature=sign,best_sha256=sha256(folder/'best.pt'))
    atomic_json(folder/'summary.json',summary)
    return summary


def bootstrap(values,seed=924,repeats=3000):
    v=np.asarray(values,dtype=np.float64)
    r=np.random.default_rng(seed).integers(len(v),size=(repeats,len(v)))
    means=v[r].mean(1)
    return dict(mean=float(v.mean()),ci95=[float(x) for x in np.quantile(means,[.025,.975])],
                lower90=float(np.quantile(means,.1)))


def freeze(out):
    d=load_roots(BASE/'data/valid',False);u=d['u'].mean(-1);choices={};details={}
    for variant in VARIANTS:
        paths=[out/'models'/f'{variant}_seed{s}/best.pt' for s in SEEDS]
        selector=AdvantageSelector(paths)
        scores=selector.member_scores(d['x'][:,:53],d['x'][:,53:])
        conservative=scores.mean(0)-scores.std(0)
        proposed=conservative.argmax(-1);peak=conservative[np.arange(len(u)),proposed]
        trials=[]
        for threshold in [0,.05,.1,.2,.3,.5,2.1]:
            a=np.where(peak>threshold,proposed,WAIT)
            gain=u[np.arange(len(u)),a]-u[:,WAIT]
            trials.append(dict(threshold=threshold,switches=int((a!=WAIT).sum()),**bootstrap(gain)))
        eligible=[t for t in trials if t['switches']>=10 and t['lower90']>0]
        choice=max(eligible,key=lambda t:t['lower90']) if eligible else trials[-1]
        raw=scores.mean(0).argmax(-1)
        choices[variant]=dict(threshold=choice['threshold'],penalty=1.0,validation=choice,
                             weights={str(p.relative_to(ROOT)):sha256(p) for p in paths})
        details[variant]=dict(gate_trials=trials,raw=bootstrap(u[np.arange(len(u)),raw]-u[:,WAIT]),
             raw_switches=int((raw!=WAIT).sum()),raw_plan_counts=np.bincount(raw,minlength=len(PLANS)).tolist())
    tr=load_roots(BASE/'data/train',False)
    global_values=tr['u'].mean((0,2));fixed_best=int(global_values.argmax())
    frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),selections=choices,
        validation_details=details,global_train_best=fixed_best,plan_names=PLAN_NAMES,
        note='All choices use train/validation only. Threshold selection is not a formal safe-policy guarantee.')
    atomic_json(out/'FROZEN_SELECTION.json',frozen)
    print(json.dumps({'frozen_choices':choices,'validation_details':details}),flush=True)


def train(out,workers,epochs):
    prepare(out,epochs)
    tasks=[dict(out=str(out),variant=v,seed=s,epochs=epochs) for v in VARIANTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(fit_task,t) for t in tasks]
        for f in as_completed(fs):print(json.dumps({'fit_finished':f.result()}),flush=True)
    freeze(out)

_CACHE={}
def policy(name):
    if name not in _CACHE:_CACHE[name]=PublicPolicy(ROOT/'elimination/runs'/name/'policy.pt')
    return _CACHE[name]

def collect_task(task):
    torch.set_num_threads(1);out=Path(task['out']);folder=out/'data'/task['split'];folder.mkdir(parents=True,exist_ok=True)
    p=folder/f"root_{task['index']:05d}.npz";meta=p.with_suffix('.json')
    sig={k:v for k,v in task.items() if k!='out'}
    if p.exists() and meta.exists():
        m=json.loads(meta.read_text())
        if m['signature']!=sig:raise ValueError('New data cache mismatch')
        return m['summary']
    tick=time.monotonic();own=policy('baseline_seed11');opp=policy(task['opponent'])
    root=choose_root(own,opp,task['env'],task['seed'],360)
    result=branch_rollouts(root,own,opp,task['repeats'],task['seed']+2**40);result.pop('frames')
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**result);tmp.replace(p)
    summary=dict(split=task['split'],index=task['index'],source_seed=task['seed'],opponent=task['opponent'],
        simulated_steps=int(result['simulated_steps']),source_steps_simulated=root['source_steps_simulated'],
        seconds=time.monotonic()-tick,repeats=task['repeats'])
    atomic_json(meta,dict(signature=sig,root=root,summary=summary))
    return summary


def collect(out,workers):
    frozen=out/'FROZEN_SELECTION.json'
    if not frozen.exists():raise ValueError('Freeze selections before generating fresh tests')
    p=json.loads((out/'PROTOCOL.json').read_text());tasks=[]
    for j,(split,n) in enumerate(p['fresh_test_counts'].items()):
        pool=['baseline_seed47','jepa_seed47'] if split=='fresh_ood' else ['baseline_seed11','jepa_seed11','baseline_seed23','jepa_seed23']
        for i in range(n):
            tasks.append(dict(out=str(out),split=split,index=i,seed=p['fresh_seed_start']+j*2**28+i,
                opponent=pool[i%len(pool)],env=p['env'],repeats=p['fresh_test_repeats'],
                frozen_hash=sha256(frozen),code=sha256(ROOT/'dino_duel/counterfactual.py')))
    rows=[];tick=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(collect_task,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            rows.append(f.result())
            if i%8==0:print(json.dumps(dict(collected=i,total=len(tasks),seconds=time.monotonic()-tick)),flush=True)
    rows.sort(key=lambda r:(r['split'],r['index']))
    atomic_json(out/'collection_summary.json',dict(roots=len(rows),branches=sum(len(PLANS)*r['repeats'] for r in rows),
        simulated_steps=sum(r['simulated_steps'] for r in rows),source_steps=sum(r['source_steps_simulated'] for r in rows),
        wall_seconds=time.monotonic()-tick,roots_detail=rows))


def evaluate(out):
    torch.set_num_threads(1);frozen=json.loads((out/'FROZEN_SELECTION.json').read_text());summaries={};allrows=[]
    for split in ['fresh_id','fresh_ood']:
        d=load_roots(out/'data'/split,False);u=d['u'].mean(-1);N=len(u)
        methods={'ppo':np.zeros(N,dtype=int),'fixed_wait':np.full(N,WAIT),
                 'train_best_constant':np.full(N,frozen['global_train_best'])}
        raw_scores={}
        for v in VARIANTS:
            paths=[out/'models'/f'{v}_seed{s}/best.pt' for s in SEEDS]
            for path in paths:
                if sha256(path)!=frozen['selections'][v]['weights'][str(path.relative_to(ROOT))]:raise ValueError('Weights changed after freeze')
            raw=AdvantageSelector(paths);raw_scores[v]=raw.member_scores(d['x'][:,:53],d['x'][:,53:])
            methods[v+'_raw']=raw_scores[v].mean(0).argmax(-1)
            g=frozen['selections'][v]
            methods[v+'_gated']=AdvantageSelector(paths,g['threshold'],g['penalty']).select(d['x'][:,:53],d['x'][:,53:])
        # Separate repetitions for choosing and scoring the simulator diagnostic.
        half=d['u'].shape[-1]//2
        oracle=d['u'][:,:,:half].mean(-1).argmax(-1)
        summaries[split]={}
        for method,a in methods.items():
            selected=u[np.arange(N),a];delta=selected-u[:,WAIT]
            summaries[split][method]=dict(roots=N,gain_vs_wait=bootstrap(delta),gain_vs_ppo=bootstrap(selected-u[:,0]),
                net_within_8s=float(selected.mean()),switches=int((a!=WAIT).sum()),
                plan_counts=np.bincount(a,minlength=len(PLANS)).tolist())
            for i in range(N):
                allrows.append(dict(split=split,root=i,source_seed=d['metadata'][i]['source_seed'],opponent=d['metadata'][i]['opponent'],
                    method=method,plan=int(a[i]),plan_name=PLAN_NAMES[a[i]],net_within_8s=float(selected[i]),
                    delta_wait=float(delta[i]),delta_ppo=float(selected[i]-u[i,0])))
        heldout=d['u'][:,:,half:].mean(-1)
        summaries[split]['simulator_split_repeat_diagnostic']=dict(
            gain_vs_wait=bootstrap(heldout[np.arange(N),oracle]-heldout[:,WAIT]),
            note='Diagnostic upper reference, not deployed; select using first four repeats, evaluate on remaining four.')
        np.savez_compressed(out/f'{split}_predictions.npz',**raw_scores,observed_utility=d['u'])
    with (out/'offline_per_root.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
    atomic_json(out/'OFFLINE_RESULTS.json',summaries)
    print(json.dumps(summaries),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','collect','evaluate']);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--epochs',type=int,default=140)
    a=p.parse_args();out=a.out.resolve()
    if a.command=='train':train(out,a.workers,a.epochs)
    elif a.command=='collect':collect(out,a.workers)
    else:evaluate(out)
if __name__=='__main__':main()
