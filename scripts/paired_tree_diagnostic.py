#!/usr/bin/env python3
"""Non-JEPA diagnostic: can a simple state-conditioned model beat waiting?

Uses ONLY v1.3 train/validation and freezes one model before opening any new
held-out labels. Trees are an extra diagnostic, not the main architecture test.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,pickle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import sklearn
from paired_advantage_suite import load_roots,bootstrap
from dino_duel.counterfactual import atomic_json,sha256
from datetime import datetime,timezone

def main():
    base=ROOT/'counterfactual';out=ROOT/'paired_advantage/tree_diagnostic';out.mkdir(exist_ok=True)
    protocol={'source':'train/valid only; no new test labels read', 'random_state':741,
              'n_estimators':256,'candidate_min_samples_leaf':[4,8,16,32],
              'selection':'validation MSE of baseline-relative target',
              'sklearn_version':sklearn.__version__,'model':'ExtraTreesRegressor',
              'created_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(out/'INTENT.json',protocol)
    tr=load_roots(base/'data/train',False);va=load_roots(base/'data/valid',False)
    ty=tr['u'].mean(-1)-tr['u'][:,1:2].mean(-1)
    vy=va['u'].mean(-1)-va['u'][:,1:2].mean(-1)
    rows=[];best=None;best_score=float('inf')
    for leaf in protocol['candidate_min_samples_leaf']:
        model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=leaf,random_state=741,n_jobs=1)
        model.fit(tr['x'],ty);pred=model.predict(va['x']);choice=pred.argmax(1)
        mse=float(np.mean((pred-vy)**2));gain=vy[np.arange(len(vy)),choice]
        rows.append(dict(leaf=leaf,mse=mse,validation_gain=bootstrap(gain),
            switches=int((choice!=1).sum()),plan_counts=np.bincount(choice,minlength=12).tolist()))
        if mse<best_score:best_score=mse;best=model;best_row=rows[-1]
    with (out/'model.pkl').open('wb') as f:pickle.dump(best,f)
    atomic_json(out/'FROZEN.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),candidates=rows,selected=best_row,
        model_sha256=sha256(out/'model.pkl'),code_sha256=sha256(__file__),
        warning='Local trusted checkpoint only; pickle must not load untrusted files. Not a JEPA claim.'))
    print(json.dumps({'tree_validation':rows,'selected':best_row}),flush=True)
if __name__=='__main__':main()
