"""Internal worker; public entrypoint is elimination_suite.py."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dino_duel.learning import train
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('variant');p.add_argument('seed',type=int);p.add_argument('out',type=Path)
p.add_argument('updates',type=int);p.add_argument('--device',default=None,choices=['auto','cpu','cuda','mps'])
p.add_argument('--config',type=Path,default=ROOT/'configs/elimination.json')
a=p.parse_args()
c=json.loads(a.config.read_text());c['training']['updates']=a.updates
c['training']['warm_start']=str(ROOT/f'assets/warmstart/seed_{a.seed}.pt')
if a.device is not None:c['training']['device']=a.device
train(c,a.out,a.variant,a.seed,resume=(a.out/'latest.pt').exists(),runner_path=ROOT/'assets/runner_adversarial.npz')
