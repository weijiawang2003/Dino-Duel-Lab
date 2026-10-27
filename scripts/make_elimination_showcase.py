#!/usr/bin/env python3
"""Reproduce a duration-selected, not winner-selected, recorded main-suite match."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'elimination')
    parser.add_argument('--section',choices=['joint','extended_joint'],default='joint')
    args=parser.parse_args();out=args.out.resolve()
    candidates=[]
    for train_seed in (11,23,47):
        source=out/'evaluations'/f'h2h_{args.section}_seed{train_seed}'
        with (source/'episodes.csv').open() as f:
            for row in csv.DictReader(f):
                if row['censored']=='False' and row['left_score'] in ('0.0','1.0','0','1'):
                    candidates.append((float(row['seconds']),train_seed,int(row['seed']),int(row['left_seat']),row,source))
    if not candidates:raise RuntimeError('No resolved decisive match in this section')
    duration,train_seed,seed,seat,expected,source=max(candidates,key=lambda x:(x[0],-x[1],-x[2],-x[3]))
    import torch
    from elimination_suite import policy
    from dino_duel.evaluation import play_episode
    from dino_duel.replay import write_replay
    torch.set_num_threads(1)
    task=json.loads((source/'task.json').read_text())
    leftpath=out/'runs'/f'jepa_seed{train_seed}'/'policy.pt'
    rightpath=out/'runs'/f'baseline_seed{train_seed}'/'policy.pt'
    actual,frames=play_episode(policy(leftpath,'joint'),policy(rightpath,'joint'),task['env'],seed,
                              left_seat=seat,record=True,max_frames=6000,safety_seconds=task['safety_seconds'])
    for key in ('winner_policy','reason','left_score','left_purchases','right_purchases'):
        if key in expected and str(actual.get(key))!=expected[key]:
            raise AssertionError(f'Replay mismatch {key}: {actual.get(key)} != {expected[key]}')
    if abs(actual['seconds']-duration)>1e-9:raise AssertionError('Replay duration differs from source trial')
    dest=out/'showcase';dest.mkdir(exist_ok=True)
    path=write_replay(frames,actual,dest/'replay.html')
    doc=path.read_text()
    note=f'This example is the longest resolved decisive match in the {args.section} suite, selected by duration, not by winner. Training seed {train_seed}. It is not a representative win-rate sample.'
    doc=doc.replace('Recorded simulation · fully offline replay','Recorded simulation · fully offline replay<br><small>'+note+'</small>')
    path.write_text(doc)
    selection={'criterion':'longest resolved decisive match; ties broken by ascending train/environment seed and seat; not winner-selected',
               'section':args.section,'training_seed':train_seed,'source_csv':str(source.relative_to(out)/'episodes.csv'),
               'source_csv_sha256':hashlib.sha256((source/'episodes.csv').read_bytes()).hexdigest(),
               'replay_exactly_reproduced':True,'episode':actual,'frames':len(frames)}
    (dest/'selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps({'replay':str(path),'seconds':actual['seconds'],'winner':actual['winner_policy'],'frames':len(frames)}))
if __name__=='__main__':main()
