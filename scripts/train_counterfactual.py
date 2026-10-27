#!/usr/bin/env python3
"""Train branch predictors; only train/validation splits are opened here."""
from __future__ import annotations
import argparse,copy,csv,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from dino_duel.counterfactual import PLANS,HORIZONS,atomic_json,sha256
from dino_duel.counterfactual_models import MacroWorldModel,OutcomeDecoder,VARIANTS
from dino_duel.learning import load_policy


def load_split(folder):
    xs=[];ys=[];ls=[];ps=[];roots=[]
    for p in sorted(folder.glob('root_*.npz')):
        with np.load(p,allow_pickle=False) as d:
            C,R,K,D=d['target'].shape
            x=np.concatenate([d['x'],d['mask'].astype(np.float32)])
            xs.append(np.broadcast_to(x,(C*R,len(x))).copy())
            ys.append(d['target'].reshape(C*R,K,D));ls.append(d['labels'].reshape(C*R,K))
            ps.append(np.repeat(PLANS,R,axis=0)); roots.extend([p.stem]*(C*R))
    if not xs: raise ValueError(f'No roots in {folder}')
    return {'x':torch.tensor(np.concatenate(xs)), 'y':torch.tensor(np.concatenate(ys)),
            'labels':torch.tensor(np.concatenate(ls)), 'plans':torch.tensor(np.concatenate(ps)),
            'roots':roots}


def checkpoint_save(path, ck):
    tmp=path.with_suffix('.tmp.pt');torch.save(ck,tmp);tmp.replace(path)


def prepare(out,decoder_epochs=35):
    out=Path(out);folder=out/'prepared';folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'elimination/runs/jepa_seed11/policy.pt'
    provenance={'source_encoder_sha256':sha256(source),'protocol_sha256':sha256(out/'PROTOCOL.json'),
                'decoder_epochs':decoder_epochs,'target_encoder_frozen':True,
                'training_code_sha256':sha256(__file__),'model_code_sha256':sha256(ROOT/'dino_duel/counterfactual_models.py')}
    if (folder/'targets.pt').exists():
        if json.loads((folder/'provenance.json').read_text())!=provenance: raise ValueError('Prepared cache mismatch')
        return
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    for split in ('train','valid'):
        if len(list((out/'data'/split).glob('root_*.npz')))!=protocol['counts'][split]:
            raise ValueError(f'Incomplete {split} data; finish collection before training')
    train=load_split(out/'data/train');valid=load_split(out/'data/valid')
    model=load_policy(source).model;encoder=model.encoder.eval().requires_grad_(False)
    with torch.no_grad():
        ztrain=encoder(train['y']);zvalid=encoder(valid['y'])
    stats={};decoders={};history=[]
    for kind,y,v in [('latent',ztrain,zvalid),('state',train['y'],valid['y'])]:
        mean=y.flatten(0,1).mean(0);std=y.flatten(0,1).std(0,unbiased=False).clamp_min(.1)
        stats[kind]={'mean':mean,'std':std}
        y=(y-mean)/std;v=(v-mean)/std
        torch.manual_seed(741)
        decoder=OutcomeDecoder(y.shape[-1]);opt=torch.optim.Adam(decoder.parameters(),lr=.002)
        rng=np.random.default_rng(741);x=y.flatten(0,1);l=train['labels'].flatten();vx=v.flatten(0,1);vl=valid['labels'].flatten()
        best=float('inf');best_state=None
        for epoch in range(decoder_epochs):
            decoder.train()
            for ids in np.array_split(rng.permutation(len(x)),max(1,(len(x)+1023)//1024)):
                loss=F.cross_entropy(decoder(x[ids]),l[ids]);opt.zero_grad();loss.backward();opt.step()
            decoder.eval()
            with torch.no_grad():
                logits=decoder(vx);nll=float(F.cross_entropy(logits,vl));acc=float((logits.argmax(-1)==vl).float().mean())
            history.append({'kind':kind,'epoch':epoch+1,'validation_nll':nll,'validation_accuracy':acc})
            if nll<best:best=nll;best_state=copy.deepcopy(decoder.state_dict())
        decoders[kind]=best_state
        print(json.dumps({'prepared':kind,'validation_best_nll':best,'validation_last_accuracy':acc}),flush=True)
    target={'normalization':stats,'decoders':decoders,'encoder':encoder.state_dict(),
            'encoder_config':{'input_dim':53,'hidden':128},
            'x_mean':train['x'].mean(0),'x_std':train['x'].std(0,unbiased=False).clamp_min(.1)}
    checkpoint_save(folder/'targets.pt',target);atomic_json(folder/'provenance.json',provenance)
    atomic_json(folder/'decoder_history.json',history)


def train_one(task):
    torch.set_num_threads(1)
    out=Path(task['out']);variant=task['variant'];seed=task['seed'];epochs=task['epochs']
    folder=out/'models'/f'{variant}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    sig={'variant':variant,'seed':seed,'batch_size':task['batch_size'],'lr':task['lr'],
         'prepared_sha256':sha256(out/'prepared/targets.pt'),'code_sha256':sha256(__file__),
         'model_sha256':sha256(ROOT/'dino_duel/counterfactual_models.py')}
    old=None
    if (folder/'latest.pt').exists():
        old=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if old['signature']!=sig:raise ValueError(f'Resume signature mismatch {folder}')
        if old['epoch']>=epochs and (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    t0=time.monotonic();tr=load_split(out/'data/train');va=load_split(out/'data/valid')
    prep=torch.load(out/'prepared/targets.pt',map_location='cpu',weights_only=True)
    kind='state' if variant=='state_nonlinear' else 'latent'
    mean=prep['normalization'][kind]['mean'];std=prep['normalization'][kind]['std']
    if kind=='latent':
        encoder=torch.nn.Sequential(torch.nn.Linear(53,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh())
        encoder.load_state_dict(prep['encoder']);encoder.eval().requires_grad_(False)
        with torch.no_grad(): tr['y']=encoder(tr['y']);va['y']=encoder(va['y'])
    for d in [tr,va]:
        d['y']=(d['y']-mean)/std;d['x']=(d['x']-prep['x_mean'])/prep['x_std']
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    m=MacroWorldModel(variant,tr['y'].shape[-1]);decoder=OutcomeDecoder(m.target_dim)
    decoder.load_state_dict(prep['decoders'][kind]);decoder.eval().requires_grad_(False)
    opt=torch.optim.Adam(m.parameters(),lr=task['lr'],weight_decay=1e-5)
    start=0;best=float('inf');hist=[];elapsed_before=0
    if old is not None:
        m.load_state_dict(old['model_state']);opt.load_state_dict(old['optimizer']);start=old['epoch'];best=old['best_validation_nll']
        torch.set_rng_state(old['torch_rng']);rng.bit_generator.state=old['numpy_rng'];hist=json.loads((folder/'history.json').read_text());elapsed_before=old['elapsed_seconds']
    def payload(epoch):
        return {'model_config':m.config,'model_state':m.state_dict(),'decoder_state':decoder.state_dict(),
                'x_mean':prep['x_mean'],'x_std':prep['x_std'],'target_mean':mean,'target_std':std,
                'signature':sig,'epoch':epoch,'best_validation_nll':best,
                'elapsed_seconds':elapsed_before+time.monotonic()-t0,'seed':seed,
                'target_kind':kind,'frozen_encoder':True}
    def evaluate():
        m.eval();sse=0;nll=0;correct=0;n=0
        with torch.no_grad():
            for i in range(0,len(va['x']),1024):
                sl=slice(i,i+1024);p=m(va['x'][sl],va['plans'][sl]);l=decoder(p[:,-1]);num=len(p)
                nll+=float(F.cross_entropy(l,va['labels'][sl,-1],reduction='sum'))
                sse+=float(F.smooth_l1_loss(p,va['y'][sl],reduction='none').mean((1,2)).sum())
                correct+=int((l.argmax(-1)==va['labels'][sl,-1]).sum());n+=num
        return nll/n,sse/n,correct/n
    for epoch in range(start,epochs):
        m.train();train_loss=0;count=0
        ids=rng.permutation(len(tr['x']))
        for begin in range(0,len(ids),task['batch_size']):
            ix=ids[begin:begin+task['batch_size']]
            pred=m(tr['x'][ix],tr['plans'][ix]);logits=decoder(pred)
            ce=F.cross_entropy(logits.flatten(0,1),tr['labels'][ix].flatten())
            reg=F.smooth_l1_loss(pred,tr['y'][ix])
            loss=ce+(0.0 if variant=='outcome_only' else 1.0)*reg
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step()
            train_loss+=float(loss.detach())*len(ix);count+=len(ix)
        nll,reg,acc=evaluate()
        hist.append({'epoch':epoch+1,'train_loss':train_loss/count,'validation_nll':nll,
                     'validation_huber':reg,'validation_accuracy':acc})
        if nll<best:
            best=nll;checkpoint_save(folder/'best.pt',payload(epoch+1))
        if (epoch+1)%10==0 or epoch+1==epochs:
            ck=payload(epoch+1);ck.update(optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state)
            checkpoint_save(folder/'latest.pt',ck);atomic_json(folder/'history.json',hist)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    summary={'variant':variant,'seed':seed,'epochs':epochs,'best_epoch':chosen['epoch'],
             'best_validation_nll':best,'parameters':sum(p.numel() for p in m.parameters()),
             'training_branches':len(tr['x']),'validation_branches':len(va['x']),
             'elapsed_seconds':elapsed_before+time.monotonic()-t0,'input_has_future_actions':False}
    atomic_json(folder/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'counterfactual')
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--decoder-epochs',type=int,default=35)
    p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303]);p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--workers',type=int,default=3);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--lr',type=float,default=.001)
    a=p.parse_args()
    if min(a.epochs,a.decoder_epochs,a.workers,a.batch_size)<1 or a.lr<=0:p.error('Positive budgets required')
    torch.set_num_threads(1);a.out=a.out.resolve();prepare(a.out,a.decoder_epochs)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    tasks=[{'out':str(a.out),'variant':v,'seed':s,'epochs':a.epochs,'batch_size':a.batch_size,'lr':a.lr} for v in a.variants for s in a.seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(train_one,t) for t in tasks]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted((a.out/'models').glob('*/summary.json'))]
    atomic_json(a.out/'training_summary.json',rows)
if __name__=='__main__':main()
