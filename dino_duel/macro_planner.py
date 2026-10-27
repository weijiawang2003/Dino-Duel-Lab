"""Public-observation macro attacker for a frozen PPO controller.

The default executes the first token and replans. ``commit_opportunities=20``
executes the 8-token macro and its 12-opportunity PPO tail (8 seconds total).
This optional commitment matches the continuation used for offline targets.
No simulator state or opponent object is accepted by the policy API.
"""
from __future__ import annotations
import numpy as np
from .counterfactual import PublicPolicy, PLANS
from .counterfactual_models import MacroSelector

class PlanningPolicy:
    def __init__(self, controller, checkpoints=None, attack_every=6,
                 commit_opportunities=1, fixed_plan=None):
        if attack_every < 1 or commit_opportunities < 1:
            raise ValueError('Invalid purchase/commit interval')
        if fixed_plan is not None and not 0 <= fixed_plan < len(PLANS):
            raise ValueError('fixed_plan must index a supported macro')
        if checkpoints and fixed_plan is not None:
            raise ValueError('Choose a predictor or a fixed plan, not both')
        self.controller = PublicPolicy(controller)
        self.selector = None if not checkpoints else MacroSelector(checkpoints)
        self.attack_every = int(attack_every)
        self.commit = int(commit_opportunities)
        self.fixed_plan = fixed_plan
        self.name = ('PPO' if self.selector is None and fixed_plan is None
                     else 'PPO + macro planner')
        self.reset(0)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.active_plan = None
        self.plan_step = 0
        self.plan_counts = np.zeros(len(PLANS), dtype=np.int64)
        self.selection_calls = 0

    def act(self, obs, masks, runner_obs=None, deterministic=True):
        obs = np.atleast_2d(obs)
        masks = np.atleast_2d(masks)
        if len(obs) != 1:
            raise ValueError('One policy instance per live duel is required')
        # Consume exactly one uniform even when the macro overrides the attack.
        action = self.controller.act(obs, masks, self.rng.random(1))
        has_planner = self.selector is not None or self.fixed_plan is not None
        if has_planner and self.t % self.attack_every == 0:
            if self.active_plan is None or self.plan_step >= self.commit:
                if masks[0, 1:].any():
                    self.active_plan = (int(self.fixed_plan) if self.fixed_plan is not None
                                        else int(self.selector.select(obs, masks)[0]))
                    if self.selector is not None:
                        self.selection_calls += 1
                    self.plan_counts[self.active_plan] += 1
                else:
                    # Do not commit a long default macro while no attack is legal.
                    # Try again at the next opportunity using its new public state.
                    self.active_plan = None
                self.plan_step = 0
            if self.active_plan is not None:
                # After the explicit 3.2s macro, use reactive PPO for the rest
                # of the commitment. No future controller actions are precomputed.
                requested = (int(PLANS[self.active_plan, self.plan_step])
                             if self.plan_step < PLANS.shape[1] else -1)
                if requested >= 0:
                    action[0, 1] = requested if masks[0, requested] else 0
                self.plan_step += 1
        self.t += 1
        return action
