#!/usr/bin/env python3
"""Frozen-rule audit on TRAIN snapshots only; no proposed rule change."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from dino_duel.env import DuelEnv
from dino_duel.counterfactual import atomic_json

def audit(out):
    out=Path(out);reasons={k:0 for k in ('allowed','funds','cooldown','space','reachability')};rows=[]
    for p in sorted((out/'data/train').glob('root_*.json')):
        root=json.loads(p.read_text())['root'];e=DuelEnv(root['state']['config']);e.set_state(root['state'])
        seat=root['seat'];cfg=e.config;player=e.players[seat]
        for kind in range(1,5):
            if player['cooldown']>1e-9:reason='cooldown'
            elif player['coins']+1e-9<cfg.attack_costs[kind]:reason='funds'
            else:
                x=e._placement_x(1-seat,kind)
                reason=('space' if x is None else 'allowed' if e._placement_allowed(1-seat,kind,x) else 'reachability')
            reasons[reason]+=1
        rows.append({'root':p.stem,'queue_lengths':[len(t) for t in e.tracks],
                     'time':e.time,'own_wallet':player['coins'],'seat':seat})
    result={'scope':'Current decision boundary of TRAIN roots only; not all future branch attempts.',
            'roots':len(rows),'purchase_types_examined':len(rows)*4,'reasons':reasons,
            'rules_unchanged':True,'no_exhaustive_attack_search_claim':True,
            'queue_beyond_observation_fraction':float(np.mean([max(r['queue_lengths'])>3 for r in rows])),
            'details':rows}
    atomic_json(out/'rule_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='details'}))
if __name__=='__main__':audit(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'counterfactual')
