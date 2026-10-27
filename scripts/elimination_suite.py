#!/usr/bin/env python3
"""Reproduce v1.2 training and censor-aware evaluations without editing code.

Example: python scripts/elimination_suite.py train --out elimination --updates 500 --workers 3
         python scripts/elimination_suite.py evaluate --out elimination --workers 3
         python scripts/elimination_suite.py report --out elimination
All continued runs use embedded optimizer, RNG and live environment states.
"""
from __future__ import annotations
import argparse,concurrent.futures,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy(path, mode):
    from dino_duel.learning import load_policy
    from dino_duel.policies import StochasticAttackPolicy, TeacherPolicy
    if str(path).startswith(('bc:','strong:')):
        prefix,style=str(path).split(':',1)
        runner=ROOT/('assets/runner_bc.npz' if prefix=='bc' else 'assets/runner_adversarial.npz')
        pol=TeacherPolicy(style,runner_path=runner);pol.name=prefix+':'+style
        return pol
    if str(path).startswith('heldout:'):
        from dino_duel.adversaries import ScriptedAdversary
        return ScriptedAdversary(str(path).split(':',1)[1],ROOT/'assets/runner_adversarial.npz')
    p=StochasticAttackPolicy(load_policy(path));p.sampling_mode=mode
    p.name=Path(path).parent.name+' / '+mode
    return p


def evaluate_task(task):
    import torch
    from dino_duel.evaluation import evaluate
    torch.set_num_threads(1)
    directory=Path(task['directory']);directory.mkdir(parents=True,exist_ok=True)
    signature={**task,'source_hashes':{x:sha(ROOT/'dino_duel'/x) for x in ('env.py','evaluation.py','policies.py','models.py','teacher.py','adversaries.py','learning.py')}}
    for side in ('left','right'):
        if not str(task[side]).startswith(('bc:','strong:','heldout:')):
            signature[side+'_sha256']=sha(task[side])
        else:
            runner='runner_bc.npz' if str(task[side]).startswith('bc:') else 'runner_adversarial.npz'
            signature[side+'_sha256']=sha(ROOT/'assets'/runner)
    saved=directory/'task.json'
    if saved.exists() and (directory/'summary.json').exists() and json.loads(saved.read_text())==signature:
        return task['name']
    left,right=policy(task['left'],task['mode']),policy(task['right'],task['mode'])
    if task.get('disable_left_attack'):
        from dino_duel.adversaries import NoAttackPolicy
        left=NoAttackPolicy(left)
    summary=evaluate(left,right,task['env'],task['seeds'],out_dir=directory,
                     record_replay=task.get('record',False),safety_seconds=task['safety_seconds'])
    summary['suite_task']={k:v for k,v in task.items() if k not in ('env','seeds')}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    saved.write_text(json.dumps(signature,indent=2))
    return task['name']


def make_tasks(out, num_seeds=50, safety=300.0, section='main'):
    config=json.loads((ROOT/'configs/elimination.json').read_text())
    recorded=out/'runs/baseline_seed11/config.json'
    if recorded.exists():config=json.loads(recorded.read_text())
    env=config['env'];start=2**36
    for record in (out/'runs').glob('*/config.json'):
        if json.loads(record.read_text())['env']!=env:
            raise ValueError('Main suite requires a common frozen environment across runs')
    seeds=[11,23,47];tasks=[]
    def path(v,s,initial=False,update=None):
        base=out/'runs'/f'{v}_seed{s}'
        return str(base/(f'checkpoints/update_{update:06d}.pt' if update else 'initial_policy.pt' if initial else 'policy.pt'))
    def add(name,left,right,mode,count,group,record=False,environment=None):
        tasks.append({'name':name,'left':left,'right':right,'mode':mode,'group':group,
                      'env':environment or env,'seeds':list(range(start,start+count)),
                      'safety_seconds':safety,'record':record,'directory':str(out/'evaluations'/name)})
    if section in ('main','all'):
        for mode in ('joint','attack','greedy'):
            for s in seeds:
                add(f'h2h_{mode}_seed{s}',path('jepa',s),path('baseline',s),mode,num_seeds,'head_to_head',s==11)
        for v in ('baseline','jepa'):
            for s in seeds:
                for opponent in ('aggressive','burst','cycle'):
                    add(f'fixed_{v}_seed{s}_{opponent}',path(v,s),'strong:'+opponent,'joint',min(20,num_seeds),'fixed')
                for opponent in ('wide','alternating'):
                    add(f'unseen_{v}_seed{s}_{opponent}',path(v,s),'heldout:'+opponent,'joint',min(10,num_seeds),'unseen')
                add(f'old_{v}_seed{s}',path(v,s),str(ROOT/f'assets/warmstart/seed_{s}.pt'),'joint',min(20,num_seeds),'old_policy')
        for s in seeds:
            for opponent in ('aggressive','burst','cycle'):
                add(f'initial_seed{s}_{opponent}',path('baseline',s,True),'strong:'+opponent,'joint',min(20,num_seeds),'initial')
    if section in ('extended',):
        for s in seeds:
            add(f'h2h_extended_joint_seed{s}',path('jepa',s),path('baseline',s),'joint',num_seeds,'extended')
    if section in ('defense','all'):
        for v,initial in [('baseline',True),('baseline',False),('jepa',False)]:
            for s in ([11] if initial else seeds):
                for style in ('wide','alternating'):
                    add(f'defense_{"initial" if initial else v}_seed{s}_{style}',path(v,s,initial),'heldout:'+style,'greedy',min(5,num_seeds),'defense')
                    tasks[-1]['disable_left_attack']=True
    if section in ('cross','all'):
        for js in seeds:
            for bs in seeds:
                add(f'cross_j{js}_b{bs}',path('jepa',js),path('baseline',bs),'joint',min(20,num_seeds),'cross_seed')
    if section in ('history','all'):
        checkpoints=[]
        for v in ('baseline','jepa'):
            for u in (100,200,300,500):
                cp=path(v,11,update=u)
                if Path(cp).exists():checkpoints.append((v,u,cp))
        for i,(v,u,cp) in enumerate(checkpoints):
            for w,t,other in checkpoints[i+1:]:
                add(f'league_{v}{u}_{w}{t}',cp,other,'joint',min(10,num_seeds),'history')
    if section in ('stress','all'):
        for name,patch in [('faster',{'initial_speed':10.,'initial_speed_jitter':1.}),
                           ('richer',{'coins_per_meter':1.5}),
                           ('tight',{'min_gap_seconds':.2})]:
            e={**env,**patch}
            for s in seeds:
                add(f'stress_{name}_seed{s}',path('jepa',s),path('baseline',s),'joint',min(20,num_seeds),'stress',environment=e)
    return tasks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['train','evaluate','report','all'])
    parser.add_argument('--out',type=Path,default=ROOT/'elimination')
    parser.add_argument('--updates',type=int,default=300,help='Total updates, not additional updates')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],default=None)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/elimination.json')
    parser.add_argument('--num-seeds',type=int,default=50)
    parser.add_argument('--safety-seconds',type=float,default=300.)
    parser.add_argument('--section',choices=['main','cross','history','stress','defense','extended','all'],default='main')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.workers<1 or args.updates<1 or args.num_seeds<1:parser.error('budgets must be positive')
    if args.task in ('train','all'):
        calls=[]
        for s in (11,23,47):
            for v in ('baseline','jepa'):
                dest=out/'runs'/f'{v}_seed{s}'
                calls.append((v,s,dest))
        def run(item):
            v,s,dest=item
            from dino_duel.experiment import OutputLock
            with OutputLock(dest):
                with (out/f'{v}_seed{s}.log').open('a') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/train_elimination_worker.py'),v,str(s),str(dest),str(args.updates),'--config',str(args.config.resolve())]+(['--device',args.device] if args.device else []),
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            return v,s
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run,calls):print('trained',result,flush=True)
    if args.task in ('evaluate','all'):
        tasks=make_tasks(out,args.num_seeds,args.safety_seconds,args.section)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(evaluate_task,tasks):print('evaluated',result,flush=True)
    if args.task in ('report','all','evaluate'):
        from elimination_report import make_report
        print(make_report(out))

if __name__=='__main__':main()
