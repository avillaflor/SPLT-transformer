import torch
from torch import nn
from copy import copy, deepcopy


from src.models.ein import EinLinear


class SPLTTransformerGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.embedding_dim = config.n_embd

        self.policy_beta = config.beta
        self.world_beta = config.beta

        self.world_latent_dim = config.world_latent_dim
        self.policy_latent_dim = config.policy_latent_dim
        self.total_latent_dim = self.world_latent_dim + self.policy_latent_dim

        self.create_layers(config)

    def create_layers(self, config):
        self.create_encoders(config)
        self.create_decoders(config)

        self.world_latent_emb = nn.Embedding(2 * self.world_latent_dim, self.embedding_dim)
        self.combine_world_latent = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.policy_latent_emb = nn.Embedding(2 * self.policy_latent_dim, self.embedding_dim)
        self.combine_policy_latent = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

    def get_block_size(self):
        if hasattr(self.world_decoder, 'block_size'):
            return self.world_decoder.block_size
        else:
            raise NotImplementedError

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('emb') or pn.endswith('mean') or pn.endswith('std') or pn.endswith('logvar'):
                    # weights of embedding modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def create_encoders(self, config):
        world_encoder_config = deepcopy(config)
        if 'encoder_config' in world_encoder_config:
            world_encoder_config.update(world_encoder_config.encoder_config)
        world_encoder_config.output_dim = self.world_latent_dim
        self.world_encoder = config.encoder_class(world_encoder_config)

        policy_encoder_config = deepcopy(config)
        if 'encoder_config' in policy_encoder_config:
            policy_encoder_config.update(policy_encoder_config.encoder_config)
        policy_encoder_config.output_dim = self.policy_latent_dim
        self.policy_encoder = config.encoder_class(policy_encoder_config)

    def create_decoders(self, config):
        world_decoder_config = deepcopy(config)
        world_decoder_config.action_weight = 0.
        self.world_decoder = config.decoder_class(world_decoder_config)

        policy_decoder_config = deepcopy(config)
        policy_decoder_config.observation_weight = 0.
        policy_decoder_config.reward_weight = 0.
        policy_decoder_config.value_weight = 0.
        self.policy_decoder = config.decoder_class(policy_decoder_config)

    def encode(self, inputs):
        world_logits = self.world_encoder.process(inputs).mean(dim=1)
        world_p = torch.sigmoid(world_logits)
        world_z = torch.bernoulli(world_p)
        world_z = world_z.detach() + world_p - world_p.detach()

        policy_logits = self.policy_encoder.process(inputs).mean(dim=1)
        policy_p = torch.sigmoid(policy_logits)
        policy_z = torch.bernoulli(policy_p)
        policy_z = policy_z.detach() + policy_p - policy_p.detach()

        latents = {
            'world_z': world_z,
            'world_p': world_p,
            'policy_z': policy_z,
            'policy_p': policy_p,
        }
        return latents

    def world_embed_latents(self, latents):
        world_z = latents['world_z']
        world_w = self.world_latent_emb.weight.reshape(2, 1, self.world_latent_dim, self.embedding_dim)
        world_latent = world_z.unsqueeze(2) * world_w[0] + (1. - world_z.unsqueeze(2)) * world_w[1]
        world_latent = torch.mean(world_latent, dim=1)
        world_latent = self.combine_world_latent(world_latent).unsqueeze(1)
        return world_latent

    def policy_embed_latents(self, latents):
        policy_z = latents['policy_z']
        policy_w = self.policy_latent_emb.weight.reshape(2, 1, self.policy_latent_dim, self.embedding_dim)
        policy_latent = policy_z.unsqueeze(2) * policy_w[0] + (1. - policy_z.unsqueeze(2)) * policy_w[1]
        policy_latent = torch.mean(policy_latent, dim=1)
        policy_latent = self.combine_policy_latent(policy_latent).unsqueeze(1)
        return policy_latent

    def world_decode(self, inputs, latents):
        world_latent_embeddings = self.world_embed_latents(latents)

        # world decode
        world_inputs = copy(inputs)
        world_inputs['embedding_offset'] = world_latent_embeddings
        world_outputs = self.world_decoder.forward(world_inputs)
        return world_outputs

    def policy_decode(self, inputs, latents):
        policy_latent_embeddings = self.policy_embed_latents(latents)

        # policy decode
        policy_inputs = copy(inputs)
        policy_inputs['embedding_offset'] = policy_latent_embeddings
        policy_outputs = self.policy_decoder.forward(policy_inputs)
        return policy_outputs

    def latent_loss(self, outputs):
        world_p = outputs['world_p']
        world_latent_loss = self.world_beta * torch.mean(torch.sum(world_p * torch.log2(world_p) + (1. - world_p) * torch.log2(1. - world_p) + 1, dim=-1))

        policy_p = outputs['policy_p']
        policy_latent_loss = self.policy_beta * torch.mean(torch.sum(policy_p * torch.log2(policy_p) + (1. - policy_p) * torch.log2(1. - policy_p) + 1, dim=-1))

        return world_latent_loss + policy_latent_loss

    def compute_loss(self, outputs, inputs, targets, mask=None):
        world_loss = self.world_decoder.compute_loss(outputs['world_outputs'], inputs, targets, mask=mask)
        policy_loss = self.policy_decoder.compute_loss(outputs['policy_outputs'], inputs, targets, mask=mask)
        latent_loss = self.latent_loss(outputs['latents'])
        return policy_loss + world_loss + latent_loss

    def forward(self, inputs):
        latents = self.encode(inputs)
        world_outputs = self.world_decode(inputs, latents)
        policy_outputs = self.policy_decode(inputs, latents)
        outputs = {
            'world_outputs': world_outputs,
            'policy_outputs': policy_outputs,
            'latents': latents,
        }
        return outputs

    def get_all_latents(self):
        mesh = torch.meshgrid([torch.arange(2)] * self.total_latent_dim, indexing='ij')
        z = torch.stack(mesh, dim=0).reshape((self.total_latent_dim, 2 ** self.total_latent_dim)).T

        world_z = z[:, :self.world_latent_dim]
        policy_z = z[:, self.world_latent_dim:]

        latents = {
            'world_z': world_z,
            'policy_z': policy_z,
        }
        return latents
