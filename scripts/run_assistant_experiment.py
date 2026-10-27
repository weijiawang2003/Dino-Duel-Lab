#!/usr/bin/env python3
"""Idempotent resume for this frozen experiment. Never overwrite a frozen gate.

No user-local execution is required: this entry is for a later assistant runtime.
For a new scientific experiment, use a new output directory instead of silently
changing the fixed evaluation protocol or refitting on the held-out data.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,subprocess,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paired_advantage'

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-only',action='store_true');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    frozen=OUT/'FROZEN_SELECTION.json'
    if frozen.exists():
        for selection in json.loads(frozen.read_text())['selections'].values():
            for name,h in selection['weights'].items():
                path=ROOT/name
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=h:
                    raise RuntimeError('Frozen checkpoint missing/changed; recover it from the archive rather than refit on a previously tested protocol.')
    elif not a.audit_only:
        run('paired_advantage_suite.py','train','--workers',str(a.workers))
    else:
        raise RuntimeError('Frozen selection missing')
    if not a.audit_only:
        if not (OUT/'tree_diagnostic/FROZEN.json').exists():run('paired_tree_diagnostic.py')
        run('paired_advantage_suite.py','collect','--workers',str(a.workers))
        run('paired_advantage_suite.py','evaluate')
        run('paired_online.py','tree-offline')
        run('paired_online.py','run','--workers',str(a.workers))
    run('validate_paired_run.py')
    print('Frozen experiment verified. User-local computation is not required.')
if __name__=='__main__':main()
