#!/usr/bin/env python3
"""Common offline defense warm-start; never a JEPA-only advantage.

The demonstration teacher enumerates legal motor trajectories for already
revealed obstacles. No future purchases are available. Its full visible queue
can contain more than the student's three slots (privileged offline teacher).
At evaluation the student uses only its 24 features, with NO oracle call.
"""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from dino_duel.env import DuelEnv
from dino_duel.models import RunnerNet,load_runner_npz,initialize_runner,export_runner_npz
from dino_duel.teacher import scripted_attacks

def bootstrap(out,rounds=3,steps_per_round=12000,seed=20261026,config=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    c=json.loads(Path(config or ROOT/'configs/defense_bootstrap.json').read_text())['env']
    src=load_runner_npz(ROOT/'assets/runner_bc.npz')
    model=RunnerNet(hidden=64);initialize_runner(model,src.weights)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
    xs=[];masks=[];round_log=[];beg=time.monotonic()
    for iteration in range(rounds):
        env=DuelEnv(c);env.reset(int(rng.integers(2**31)))
        styles=['random','burst','cycle','aggressive'];bad=0;labelled=0;resets=0
        student=load_runner_npz(export_runner_npz(model,out/'current.npz'))
        for step in range(steps_per_round):
            obs=env.observe();ro=env.runner_observations();am=env.attack_masks()
            a=np.zeros((2,2),dtype=int);a[:,0]=student(ro)
            for seat in (0,1):
                vm=env.viable_motor_mask(seat)
                if vm.any():
                    # Ignore all-safe states, which contribute exactly zero loss.
                    if not vm.all():xs.append(ro[seat].copy());masks.append(vm.copy());labelled+=1
                    if not vm[a[seat,0]]:
                        bad+=1;a[seat,0]=int(np.flatnonzero(vm)[0])
                style=styles[(resets+seat+iteration)%4]
                a[seat,1]=scripted_attacks(obs[seat],am[seat],rng,style)
            _,_,done,_,_=env.step(a)
            # Only data collection truncates, never a game outcome.
            if done or env.time>=80:
                env.reset(int(rng.integers(2**31)));resets+=1
        X=torch.tensor(np.array(xs),dtype=torch.float32);M=torch.tensor(np.array(masks),dtype=torch.bool)
        for epoch in range(18):
            order=torch.randperm(len(X))
            for batch in order.split(512):
                logp=F.log_softmax(model(X[batch]),-1)
                loss=-torch.logsumexp(logp.masked_fill(~M[batch],-1e9),-1).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        with torch.no_grad():
            pred=model(X).argmax(-1)
            bad_rate=float((~M[torch.arange(len(M)),pred]).float().mean())
        row={'round':iteration+1,'collection_steps':steps_per_round,'new_labelled_states':labelled,
             'teacher_interventions':bad,'cumulative_labels':len(X),'training_invalid_action_rate':bad_rate,
             'elapsed_seconds':time.monotonic()-beg}
        round_log.append(row);print(json.dumps(row),flush=True)
        export_runner_npz(model,out/f'runner_round{iteration+1}.npz',{'source':'offline viability demonstrations','seed':seed,'round':iteration+1})
    export_runner_npz(model,out/'runner_adversarial.npz',{'source':'offline viability demonstrations','seed':seed,'rounds':rounds})
    np.savez_compressed(out/'demonstrations.npz',observations=np.array(xs),viable_actions=np.array(masks))
    (out/'provenance.json').write_text(json.dumps({'seed':seed,'rounds':round_log,'env':c,
        'teacher':'exact feasible trajectories for current queue only; no unknown future purchases',
        'student_observation':'24 features, first three obstacles','note':'teacher is privileged offline, never invoked by evaluated student'},indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'runs/defense_bootstrap_reproduction')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--config',type=Path,default=ROOT/'configs/defense_bootstrap.json')
    a=p.parse_args()
    if (a.out/'provenance.json').exists():p.error('Output already contains results; choose a new --out directory')
    bootstrap(a.out,a.rounds,a.steps,config=a.config)
