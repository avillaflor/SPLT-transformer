import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import Independent


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            observation_dim,
            action_dim,
            embedding_dim,
            n_hidden_layers=1,
            min_log_std=None,
            max_log_std=None,
            **kwargs
    ):
        super().__init__()
        mlp = [
            nn.Linear(observation_dim, embedding_dim),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers):
            mlp.append(nn.Linear(embedding_dim, embedding_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(embedding_dim, action_dim))
        self.mlp = nn.Sequential(*mlp)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std_logits = nn.Parameter(
            torch.zeros(action_dim, requires_grad=True))

    def get_action(self, obs):
        return self.mlp(obs)

    def std(self):
        log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        return std

    def forward(self, obs):
        mean = self.mlp(obs)
        std = self.std()

        dist = Independent(Normal(mean, std), reinterpreted_batch_ndims=1)
        return dist
