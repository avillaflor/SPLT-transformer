import numpy as np
import torch


from src.policies.policy import Policy
from src.search.ma_trajectory_core import splt_trajectory_plan


class SPLTBTPolicy(Policy):
    def __init__(
            self,
            model,
            horizon,
            observation_dim,
            action_dim,
            discount,
            bs=1,
            max_history=0,
            max_context=0,
            device='cuda',
            agg='min',
    ):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount = discount
        self.bs = bs
        self.max_history = max_history
        self.max_context = max_context
        self.device = device
        self.agg = agg

        self.observations = []
        self.actions = []

    @torch.inference_mode()
    def forward(self, observation, max_horizon=None, return_plans=False):
        if len(observation.shape) == 1:
            assert(self.bs == 1)
            no_bs = True
            observation = observation[None]
        else:
            no_bs = False

        if max_horizon is not None:
            horizon = min(self.horizon, max_horizon)
        else:
            horizon = self.horizon

        self.observations.append(observation)
        self.observations = self.observations[-self.max_history:]

        self.actions.append(np.zeros((self.bs, self.action_dim)))
        self.actions = self.actions[-self.max_history:]

        history = {
                'observations': torch.Tensor(np.stack(self.observations, axis=1)).to(device=self.device),
                'actions': torch.Tensor(np.stack(self.actions, axis=1)).to(device=self.device),
            }

        ## sample sequence from model beginning with `prefix`
        sequence, candidates, world_index, policy_index = splt_trajectory_plan(
            self.model,
            history,
            horizon,
            self.observation_dim,
            self.action_dim,
            self.discount,
            max_context_transitions=self.max_context,
            device=self.device,
            agg=self.agg,
        )

        action = sequence['actions'][:, 0].cpu().numpy()
        self.actions[-1] = action

        if return_plans:
            sequence = torch.cat([
                    sequence['observations'],
                    sequence['actions'],
                ], dim=-1).cpu().numpy()

            candidates = torch.cat([
                    candidates['observations'],
                    candidates['actions'],
                ], dim=-1).cpu().numpy()

            world_index = world_index.cpu().numpy()
            policy_index = policy_index.cpu().numpy()

            if no_bs:
                return action[0], sequence[0], candidates[0], world_index[0], policy_index[0]
            else:
                return action, sequence, candidates, world_index, policy_index
        else:
            if no_bs:
                return action[0]
            else:
                return action

    @torch.inference_mode()
    def reset(self):
        self.observations = []
        self.actions = []
