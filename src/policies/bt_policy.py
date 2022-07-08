import numpy as np
import torch


from src.policies.policy import Policy


class BTPolicy(Policy):
    def __init__(
            self,
            model,
            observation_dim,
            action_dim,
            discount,
            bs=1,
            max_history=0,
            device='cuda',
    ):
        super().__init__()
        self.model = model
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount = discount
        self.bs = bs
        self.max_history = max_history
        self.device = device

        self.observations = []
        self.actions = []

    @torch.inference_mode()
    def forward(self, observation):
        if len(observation.shape) == 1:
            assert(self.bs == 1)
            no_bs = True
            observation = observation[None]
        else:
            no_bs = False

        self.observations.append(observation)
        self.observations = self.observations[-self.max_history:]

        self.actions.append(np.zeros((self.bs, self.action_dim)))
        self.actions = self.actions[-self.max_history:]

        history = {
                'observations': torch.Tensor(np.stack(self.observations, axis=1)).to(device=self.device),
                'actions': torch.Tensor(np.stack(self.actions, axis=1)).to(device=self.device),
            }

        outputs = self.model(history)

        action = outputs['actions'][:, -1].cpu().numpy()
        self.actions[-1] = action

        if no_bs:
            return action[0]
        else:
            return action

    @torch.inference_mode()
    def reset(self):
        self.observations = []
        self.actions = []
