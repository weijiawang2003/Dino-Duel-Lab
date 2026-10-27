#!/usr/bin/env python3
"""Audit saved checkpoints and actual evaluation rows. Does not train models."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from dino_duel.learning import _load_checkpoint


def validate(root,updates=300,require_main=True):
    root=Path(root);models=[];initial_checks=[]
    for seed in (11,23,47):
        initials={}
        for variant in ('baseline','jepa'):
            run=root/f'runs/{variant}_seed{seed}'
            p=_load_checkpoint(run/'policy.pt');full=_load_checkpoint(run/'latest.pt')
            init=_load_checkpoint(run/'initial_policy.pt');initials[variant]=init
            assert p['update']==full['update']==updates
            assert p['global_step']==updates*p['config']['training']['num_envs']*p['config']['training']['rollout_steps']
            assert p['config']['env']['max_seconds'] is None
            for k,v in p['model_state'].items():
                assert bool(torch.isfinite(v).all()),(variant,seed,k)
                assert torch.equal(v,full['model_state'][k])
                if k.startswith('runner.'):
                    assert torch.equal(v,init['model_state'][k]),'Frozen common prior changed'
            residual=p['model_state']['motor_residual_head.weight']
            assert float(residual.abs().sum())>0
            if variant=='jepa':
                assert any(k.startswith('predictor.') for k in p['model_state'])
                assert p['model_config']['prediction_horizons']==[8,32,64] or tuple(p['model_config']['prediction_horizons'])==(8,32,64)
            for line in (run/'metrics.jsonl').read_text().splitlines():
                for key,value in json.loads(line).items():
                    if isinstance(value,float):assert math.isfinite(value),(run,key)
            models.append({'variant':variant,'seed':seed,'update':p['update'],'global_step':p['global_step'],
                           'policy_sha256':hashlib.sha256((run/'policy.pt').read_bytes()).hexdigest(),
                           'residual_weight_l1':float(residual.abs().sum()),'frozen_motor_verified':True})
        common=initials['baseline']['model_state']
        for key,value in common.items():assert torch.equal(value,initials['jepa']['model_state'][key]),(seed,key)
        initial_checks.append({'seed':seed,'common_parameter_tensors_equal':len(common)})
    if require_main:
        for mode in ('joint','attack','greedy'):
            for seed in (11,23,47):assert (root/f'evaluations/h2h_{mode}_seed{seed}/summary.json').exists()
    evaluation=[]
    for path in sorted((root/'evaluations').glob('*/summary.json')):
        s=json.loads(path.read_text())
        rows=list(csv.DictReader((path.parent/'episodes.csv').open()))
        assert len(rows)==s['n_games']
        assert s['wins']+s['draws']+s['losses']+s.get('censored',0)==len(rows)
        for row in rows:
            assert row['truncated']=='False'
            if row['censored']=='True':
                assert row['left_score']=='' and row['winner_policy']=='censored' and row['terminated']=='False'
            else:
                assert row['terminated']=='True' and row['left_score'] in {'0.0','0.5','1.0'}
            for side in ('left','right'):
                assert row[f'{side}_collision_source'] in {'','player_0','player_1'}
        evaluation.append({'task':path.parent.name,'games':len(rows),'censored':s.get('censored',0)})
    record={'models':models,'matched_initialization':initial_checks,'evaluation_rows_checked':sum(x['games'] for x in evaluation),
            'evaluation_tasks_checked':len(evaluation),'games_have_no_clock_termination':True,
            'all_observed_deaths_caused_by_purchased_obstacles':True,
            'censored_games_have_no_score_or_winner':True,'evaluation_tasks':evaluation}
    (root/'audit.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in {'models','evaluation_tasks'}},indent=2))
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'elimination')
    p.add_argument('--updates',type=int,default=300);p.add_argument('--allow-partial-evaluation',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);validate(a.out,a.updates,not a.allow_partial_evaluation)
