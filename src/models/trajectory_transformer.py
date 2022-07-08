import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


from src.models.gpt import GPT


class TrajectoryTransformerGPT(GPT):
    def __init__(self, config):
        self.observation_weight = config.observation_weight if hasattr(config, 'observation_weight') else 1
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim

        if not hasattr(config, 'output_dim'):
            config.output_dim = config.vocab_size + 1
        config.mask_values = True

        super().__init__(config)

    def create_layers(self, config):
        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        super().create_layers(config)

    def offset_tokens(self, idx):
        _, t = idx.shape
        n_states = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(idx.device)
        offset_idx = idx + offsets[:t]
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_full_observation(self, x):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device)
        ## [ B x T' x embedding_dim ]
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
        return x_pad, n_pad

    def embed_inputs(self, inputs):
        idx = inputs['transitions']
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        offset_idx = self.offset_tokens(idx)
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        token_embeddings = self.tok_emb(offset_idx) # each index maps to a (learnable) vector
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        if 'embedding_offset' in inputs:
            position_embeddings = position_embeddings + inputs['embedding_offset']
        return token_embeddings + position_embeddings

    def decode_outputs(self, outputs, inputs):
        return {
            'transitions': outputs
        }

    def compute_loss(self, outputs, inputs, targets, mask=None):
        output_transitions = outputs['transitions']
        target_transitions = targets['transitions']
        b, t, *_ = output_transitions.shape
        loss = F.cross_entropy(output_transitions.reshape(-1, output_transitions.size(-1)), target_transitions.view(-1), reduction='none')
        if self.observation_weight != 1 or self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
            #### make weights
            n_states = int(np.ceil(t / self.transition_dim))
            weights = torch.cat([
                torch.ones(self.observation_dim, device=output_transitions.device) * self.observation_weight,
                torch.ones(self.action_dim, device=output_transitions.device) * self.action_weight,
                torch.ones(1, device=output_transitions.device) * self.reward_weight,
                torch.ones(1, device=output_transitions.device) * self.value_weight,
            ])
            ## [ t + 1]
            weights = weights.repeat(n_states)
            ## [ b x t ]
            weights = weights[1:].repeat(b, 1)
            ####
            loss = loss * weights.view(-1)
            mask = mask & (weights > 0.)
        loss = loss[mask.view(-1)].mean()
        return loss
