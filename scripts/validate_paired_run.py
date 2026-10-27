#!/usr/bin/env python3
"""Audit measured outputs without rerunning or tuning any training."""
import json,sys,re,zipfile,hashlib,platform,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dino_duel.counterfactual import sha256,atomic_json
OUT=ROOT/'paired_advantage'

def main():
    frozen=json.loads((OUT/'FROZEN_SELECTION.json').read_text());frozenhash=sha256(OUT/'FROZEN_SELECTION.json')
    provenance=json.loads((OUT/'DATA_PROVENANCE.json').read_text())
    old=set(provenance['train_sources']+provenance['valid_sources'])
    for split in ('test','ood'):
        for p in (ROOT/'counterfactual/data'/split).glob('root_*.json'):
            old.add(json.loads(p.read_text())['summary']['source_seed'])
    fresh=[];data_count=0
    for split in ('fresh_id','fresh_ood'):
        files=list((OUT/'data'/split).glob('root_*.json'));assert len(files)==64
        for p in files:
            d=json.loads(p.read_text());assert d['signature']['frozen_hash']==frozenhash
            seed=d['summary']['source_seed'];assert seed not in old;fresh.append(seed)
            assert p.with_suffix('.npz').exists();data_count+=1
    assert len(fresh)==len(set(fresh))==128
    for selection in frozen['selections'].values():
        for name,h in selection['weights'].items():assert sha256(ROOT/name)==h
    tree=json.loads((OUT/'tree_diagnostic/FROZEN.json').read_text())
    assert sha256(OUT/'tree_diagnostic/model.pkl')==tree['model_sha256']
    summary=json.loads((OUT/'online/SUMMARY.json').read_text())
    assert sum(r['games'] for r in summary.values())==384
    prot_hash=sha256(OUT/'online/PROTOCOL.json')
    for p in (OUT/'online/episodes').glob('*.json'):
        row=json.loads(p.read_text());assert row['protocol_sha256']==prot_hash
        if row['censored']:assert row['winner_policy']=='censored' and row['left_score'] is None
    assert len(list((OUT/'online/episodes').glob('*.json')))==384
    start=json.loads((OUT/'online/START_TIME.json').read_text());assert not start['fresh_offline_results_existed']
    tests=(OUT/'tests.log').read_text();count=int(re.search(r'(\d+) passed',tests).group(1))
    # Original source files were neither silently patched nor overwritten.
    archive=ROOT.parent.parent/'dino-duel-lab-v1.3.0-counterfactual.zip';changed=[];checked=0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            rel=Path(info.filename).relative_to('dino-duel-lab');p=ROOT/rel
            if not p.exists() or hashlib.sha256(z.read(info)).digest()!=hashlib.sha256(p.read_bytes()).digest():changed.append(str(rel))
            checked+=1
    assert not changed,changed
    cases=json.loads((OUT/'cases/selection.json').read_text());assert all(c['exact_resimulation'] for c in cases['cases'])
    result=dict(tests_passed=count,test_log_sha256=sha256(OUT/'tests.log'),fresh_roots=128,
        fresh_source_seeds_disjoint_from_all_v13_roots=True,weights_unchanged_since_freeze=True,
        complete_duels=384,source_archive_members_checked=checked,source_archive_files_modified=changed,
        online_protocol_frozen_before_new_offline_scores=True,selected_case_replays_exact=True,
        versions={k:importlib.metadata.version(k) for k in ['numpy','torch','scikit-learn','pytest']},
        python=platform.python_version(),device='CPU; no CUDA used',
        note='Bootstrap intervals are exploratory and conditional on this fixed pretrained population; no unbounded storage or background compute promise.')
    atomic_json(OUT/'validation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
