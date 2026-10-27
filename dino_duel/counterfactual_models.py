"""Small macro world models and public-input attack planning.

The latent targets use a FROZEN encoder from the supplied JEPA checkpoint. This
is a controlled frozen-representation JEPA-style predictor experiment, not a
new end-to-end PPO/JEPA training claim. No inference call accepts future actions
or a simulator handle. All candidates use the same live PPO motor/controller.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .counterfactual import PLANS, PLAN_NAMES, HORIZONS

VARIANTS=('latent_additive','latent_nonlinear','state_nonlinear','outcome_only')

class OutcomeDecoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dimension,64),nn.Tanh(),nn.Linear(64,4))
    def forward(self,x): return self.net(x)

class MacroWorldModel(nn.Module):
    def __init__(self, variant, target_dim, input_dim=58, width=128, horizons=HORIZONS):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(f'Unknown variant {variant}')
        self.variant=variant;self.target_dim=target_dim;self.horizons=tuple(horizons)
        self.config={'variant':variant,'target_dim':target_dim,'input_dim':input_dim,'width':width,'horizons':self.horizons}
        self.state=nn.Sequential(nn.Linear(input_dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.actions=nn.GRU(6,32,batch_first=True)
        self.fusion=(nn.Linear(width+32,len(horizons)*target_dim) if variant=='latent_additive' else
                     nn.Sequential(nn.Linear(width+32,width),nn.Tanh(),nn.Linear(width,len(horizons)*target_dim)))
    def forward(self,x,plans):
        # -1..4 -> 0..5. -1 denotes the original reactive policy, not oracle acts.
        tokens=F.one_hot(plans.long()+1,6).to(x.dtype)
        _,h=self.actions(tokens)
        fused=torch.cat((self.state(x),h[-1]),-1)
        return self.fusion(fused).reshape(-1,len(self.horizons),self.target_dim)

class MacroSelector:
    """Inference API depends only on current public observation and legal mask."""
    def __init__(self, checkpoints):
        self.members=[]
        for path in checkpoints:
            ck=torch.load(path,map_location='cpu',weights_only=True)
            m=MacroWorldModel(**ck['model_config']);m.load_state_dict(ck['model_state']);m.eval()
            d=OutcomeDecoder(m.target_dim);d.load_state_dict(ck['decoder_state']);d.eval()
            self.members.append((m,d,ck['x_mean'],ck['x_std']))
        if not self.members: raise ValueError('At least one checkpoint is required')
        self.plans=torch.tensor(PLANS,dtype=torch.long)
    @torch.no_grad()
    def probabilities(self,observations,masks):
        x=np.concatenate((np.atleast_2d(observations),np.atleast_2d(masks)),axis=-1).astype(np.float32)
        B=len(x);C=len(PLANS);t=torch.tensor(x).repeat_interleave(C,0)
        plans=self.plans.repeat(B,1);results=[]
        for m,d,mean,std in self.members:
            y=m((t-mean)/std,plans)
            results.append(d(y[:,-1]).softmax(-1).reshape(B,C,4))
        return torch.stack(results).mean(0).numpy()
    def scores(self,observations,masks):
        p=self.probabilities(observations,masks)
        return p[:,:,1]-p[:,:,2]
    def select(self,observations,masks):
        return self.scores(observations,masks).argmax(-1)
