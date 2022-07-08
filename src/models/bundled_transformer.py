import torch
import torch.nn as nn
from torch.nn import functional as F


from src.models.gpt import GPT


class BundledTransformerGPT(GPT):
    def __init__(self, config):
        self.observation_weight = config.observation_weight if hasattr(config, 'observation_weight') else 1
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight

        if not hasattr(config, 'output_dim'):
            config.output_dim = config.n_embd
        config.mask_values = False

        self.res = hasattr(config, 'res') and config.res
        self.action_tanh = hasattr(config, 'action_tanh') and config.action_tanh

        super().__init__(config)

        self.observation_mean = nn.Parameter(config.observation_mean, requires_grad=False)
        self.observation_std = nn.Parameter(config.observation_std + 1.e-6, requires_grad=False)
        if self.res:
            self.observation_diff_mean = nn.Parameter(config.observation_diff_mean, requires_grad=False)
            self.observation_diff_std = nn.Parameter(config.observation_diff_std + 1.e-6, requires_grad=False)
        self.action_mean = nn.Parameter(config.action_mean, requires_grad=False)
        self.action_std = nn.Parameter(config.action_std + 1.e-6, requires_grad=False)
        self.reward_mean = nn.Parameter(config.reward_mean, requires_grad=False)
        self.reward_std = nn.Parameter(config.reward_std + 1.e-6, requires_grad=False)
        self.value_mean = nn.Parameter(config.value_mean, requires_grad=False)
        self.value_std = nn.Parameter(config.value_std + 1.e-6, requires_grad=False)

    def create_layers(self, config):
        # embedding layers
        self.observation_embed = nn.Sequential(
            nn.Linear(self.observation_dim, self.embedding_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.embedding_dim)
        )
        self.embed_ln = nn.LayerNorm(self.embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.embedding_dim))

        # decoder layers
        self.observation_decoder = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.observation_dim)
        )
        self.action_decoder = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.action_dim)
        )
        self.reward_decoder = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, 1)
        )
        self.value_decoder = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, 1)
        )

        super().create_layers(config)

    def pad_to_full_observation(self, x):
        x_view = x.view(-1, self.transition_dim, self.embedding_dim)
        return x_view, 0

    def embed_inputs(self, inputs):
        observations = (inputs['observations'] - self.observation_mean) / self.observation_std
        actions = (inputs['actions'] - self.action_mean) / self.action_std
        b, obs_t, *_ = observations.shape
        _, act_t, *_ = actions.shape
        t = obs_t + act_t
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        observation_embeddings = self.observation_embed(observations)
        action_embeddings = self.action_embed(actions)

        # [ B x T x embedding_dim ]
        embeddings = torch.stack([observation_embeddings, action_embeddings], dim=2).reshape((b, t, self.embedding_dim))
        embeddings = self.embed_ln(embeddings)

        # [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector

        if 'embedding_offset' in inputs:
            position_embeddings = position_embeddings + inputs['embedding_offset']
        return embeddings + position_embeddings

    def decode_outputs(self, outputs, inputs):
        observation_outputs = outputs[:, ::2]
        action_outputs = outputs[:, 1::2]

        preds = {}

        if self.action_weight != 0.:
            if self.action_tanh:
                preds['actions'] = self.action_decoder(observation_outputs).tanh()
            else:
                preds['actions'] = self.action_std * self.action_decoder(observation_outputs) + self.action_mean

        if self.value_weight != 0.:
            preds['values'] = self.value_std * self.value_decoder(action_outputs) + self.value_mean

        if self.observation_weight != 0.:
            if self.res:
                inp_observations = inputs['observations']
                observation_preds = self.observation_diff_std * self.observation_decoder(action_outputs) + inp_observations + self.observation_diff_mean
            else:
                observation_preds = self.observation_std * self.observation_decoder(action_outputs) + self.observation_mean
            preds['observations'] = observation_preds

        if self.reward_weight != 0.:
            preds['rewards'] = self.reward_std * self.reward_decoder(action_outputs) + self.reward_mean

        return preds

    def compute_loss(self, outputs, inputs, targets, mask=None):
        loss = 0
        if self.observation_weight != 0.:
            observation_error = F.mse_loss(outputs['observations'], targets['observations'], reduction='none')
            if self.res:
                observation_loss = torch.sum(observation_error / (self.observation_diff_std ** 2), dim=-1, keepdims=True)
            else:
                observation_loss = torch.sum(observation_error / (self.observation_std ** 2), dim=-1, keepdims=True)
            loss = loss + self.observation_weight * observation_loss[mask[:, 1:]].mean()

        if self.action_weight != 0.:
            if self.action_tanh:
                target_actions = targets['actions'].clamp(-0.999, 0.999)
            else:
                target_actions = targets['actions']
            action_error = F.mse_loss(outputs['actions'], target_actions, reduction='none')
            action_loss = torch.sum(action_error / (self.action_std ** 2), dim=-1, keepdims=True)
            loss = loss + self.action_weight * action_loss[mask[:, :-1]].mean()

        if self.reward_weight != 0.:
            reward_error = F.mse_loss(outputs['rewards'], targets['rewards'], reduction='none')
            reward_loss = torch.sum(reward_error / (self.reward_std ** 2), dim=-1, keepdims=True)
            loss = loss + self.reward_weight * reward_loss[mask[:, :-1]].mean()

        if self.value_weight != 0.:
            value_error = F.mse_loss(outputs['values'], targets['values'], reduction='none')
            value_loss = torch.sum(value_error / (self.value_std ** 2), dim=-1, keepdims=True)
            loss = loss + self.value_weight * value_loss[mask[:, :-1]].mean()

        return loss
