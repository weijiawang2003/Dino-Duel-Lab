"""Baseline-relative attack selector; no simulator/future-action input.

A candidate is evaluated against the SAME root's fixed wait+reactive-PPO plan.
The latent auxiliary target uses the frozen v1.2 JEPA encoder; this is not an
end-to-end JEPA or a new full-agent PPO training claim.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .counterfactual import PLANS

WAIT = 1
VARIANTS = ('advantage_only', 'latent_aux', 'state_aux')

class PairedAdvantageModel(nn.Module):
    def __init__(self, variant='advantage_only', input_dim=58, auxiliary_dim=128):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.config = dict(variant=variant, input_dim=int(input_dim), auxiliary_dim=int(auxiliary_dim))
        self.variant = variant
        self.register_buffer('plan_tokens', F.one_hot(torch.as_tensor(PLANS)+1, 6).float().flatten(1))
        self.state = nn.Sequential(nn.Linear(input_dim, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.action = nn.Sequential(nn.Linear(48, 96), nn.SiLU())
        self.interaction = nn.Sequential(nn.Linear(288, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.value = nn.Linear(64, 1)
        # Created AFTER the common modules, so shared initialization is exactly
        # matched across variants at a fixed fitting seed.
        self.auxiliary = (None if variant == 'advantage_only' else nn.Linear(64, 3*auxiliary_dim))

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.config['input_dim']:
            raise ValueError('Expected normalized public feature matrix [batch,58]')
        s = self.state(x)[:, None, :].expand(-1, len(PLANS), -1)
        a = self.action(self.plan_tokens)[None, :, :].expand(len(x), -1, -1)
        h = self.interaction(torch.cat((s, a, s*a), -1))
        raw = self.value(h).squeeze(-1)
        advantage = raw - raw[:, WAIT:WAIT+1]  # wait's advantage is identically 0
        aux = None
        if self.auxiliary is not None:
            raw_aux = self.auxiliary(h).reshape(len(x), len(PLANS), 3, self.config['auxiliary_dim'])
            aux = raw_aux - raw_aux[:, WAIT:WAIT+1]
        return advantage, aux

class AdvantageSelector:
    """Select a macro from public observation and current legal mask only.

    Gate thresholds are validation choices, NOT confidence guarantees.
    A high abstention threshold falls back to the fixed wait+PPO plan.
    """
    def __init__(self, paths, threshold=0.0, disagreement_penalty=0.0):
        if not paths:
            raise ValueError('At least one checkpoint is required')
        self.members = []
        for path in paths:
            ck = torch.load(Path(path), map_location='cpu', weights_only=True)
            model = PairedAdvantageModel(**ck['model_config'])
            model.load_state_dict(ck['model_state']); model.eval()
            self.members.append((model, ck['x_mean'], ck['x_std']))
        self.threshold = float(threshold)
        self.penalty = float(disagreement_penalty)

    @torch.no_grad()
    def member_scores(self, observations, masks):
        x = np.concatenate((np.atleast_2d(observations), np.atleast_2d(masks)), -1).astype(np.float32)
        t = torch.as_tensor(x)
        return np.stack([m((t-mean)/std)[0].numpy() for m,mean,std in self.members])

    def scores(self, observations, masks):
        scores = self.member_scores(observations, masks)
        return scores.mean(0) - self.penalty*scores.std(0)

    def select(self, observations, masks):
        scores = self.scores(observations, masks)
        selected = scores.argmax(-1)
        selected[scores[np.arange(len(selected)), selected] <= self.threshold] = WAIT
        return selected
