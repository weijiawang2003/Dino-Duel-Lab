#!/usr/bin/env python3
"""Run/resume the complete shipped counterfactual protocol with one command."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'counterfactual')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--skip-online',action='store_true',help='Skip the separate full-duel pilot')
    a=p.parse_args()
    if a.workers<1 or a.epochs<1:p.error('workers and epochs must be positive')
    out=str(a.out.resolve());w=str(a.workers)
    commands=[['counterfactual_suite.py','collect','--out',out,'--workers',w],
              ['train_counterfactual.py','--out',out,'--workers',w,'--epochs',str(a.epochs)],
              ['evaluate_counterfactual.py','--out',out],
              ['audit_counterfactual_rules.py',out],
              ['counterfactual_cases.py','--out',out]]
    if not a.skip_online:
        commands.append(['evaluate_planning.py','--out',out,'--workers',w])
        commands.append(['evaluate_planning.py','--out',out,'--workers',w,'--commit','20',
                         '--seed-start',str(2**49),'--methods','ppo','fixed_wait',
                         'latent_nonlinear','state_nonlinear','--note','Replication of posthoc commitment diagnostic'])
    commands.extend([['counterfactual_report.py',out],['validate_counterfactual_release.py','--out',out]])
    for command in commands:
        args=[sys.executable,str(ROOT/'scripts'/command[0]),*command[1:]]
        print('RUN', ' '.join(args),flush=True)
        subprocess.run(args,cwd=ROOT,check=True)
    print('Completed:',Path(out)/'report.html')
if __name__=='__main__':main()
