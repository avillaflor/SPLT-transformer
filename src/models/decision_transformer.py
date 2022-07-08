import torch
import torch.nn as nn
from torch.nn import functional as F


from src.models.gpt import GPT


class DecisionTransformerGPT(GPT):
    def __init__(self, config):
        if not hasattr(config, 'output_dim'):
            config.output_dim = config.n_embd
        config.mask_values = False

        self.action_tanh = hasattr(config, 'action_tanh') and config.action_tanh

        super().__init__(config)

        self.observation_mean = nn.Parameter(config.observation_mean, requires_grad=False)
        self.observation_std = nn.Parameter(config.observation_std + 1.e-6, requires_grad=False)
        self.action_mean = nn.Parameter(config.action_mean, requires_grad=False)
        self.action_std = nn.Parameter(config.action_std + 1.e-6, requires_grad=False)
        self.return_mean = nn.Parameter(config.return_mean, requires_grad=False)
        self.return_std = nn.Parameter(config.return_std + 1.e-6, requires_grad=False)

    def create_layers(self, config):
        ### embedding layers
        self.return_embed = nn.Sequential(
            nn.Linear(1, self.embedding_dim)
        )
        self.observation_embed = nn.Sequential(
            nn.Linear(self.observation_dim, self.embedding_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.embedding_dim)
        )
        self.embed_ln = nn.LayerNorm(self.embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.embedding_dim))

        ### decoder layers
        self.action_decoder = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.action_dim)
        )
        super().create_layers(config)

    def pad_to_full_observation(self, x):
        x_view = x.view(-1, self.transition_dim, self.embedding_dim)
        return x_view, 0

    def embed_inputs(self, inputs):
        returns = (inputs['returns'] - self.return_mean) / self.return_std
        observations = (inputs['observations'] - self.observation_mean) / self.observation_std
        actions = (inputs['actions'] - self.action_mean) / self.action_std
        b, R_t, *_ = returns.shape
        _, obs_t, *_ = observations.shape
        _, act_t, *_ = actions.shape
        t = R_t + obs_t + act_t
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        return_embeddings = self.return_embed(returns)
        observation_embeddings = self.observation_embed(observations)
        action_embeddings = self.action_embed(actions)

        ## [ B x T x embedding_dim ]
        embeddings = torch.stack([return_embeddings, observation_embeddings, action_embeddings], dim=2).reshape((b, t, self.embedding_dim))
        embeddings = self.embed_ln(embeddings)

        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        if 'embedding_offset' in inputs:
            position_embeddings = position_embeddings + inputs['embedding_offset']
        return embeddings + position_embeddings

    def decode_outputs(self, outputs, inputs):
        #  return_outputs = outputs[:, ::3]
        observation_outputs = outputs[:, 1::3]
        #  action_outputs = outputs[:, 2::3]

        preds = {}

        if self.action_tanh:
            preds['actions'] = self.action_decoder(observation_outputs).tanh()
        else:
            preds['actions'] = self.action_std * self.action_decoder(observation_outputs) + self.action_mean
        return preds

    def compute_loss(self, outputs, inputs, targets, mask=None):
        if self.action_tanh:
            target_actions = targets['actions'].clamp(-0.999, 0.999)
        else:
            target_actions = targets['actions']
        action_error = F.mse_loss(outputs['actions'], target_actions, reduction='none')
        action_loss = torch.sum(action_error / (self.action_std ** 2), dim=-1, keepdims=True)
        loss = action_loss[mask[:, :-1]].mean()
        return loss
