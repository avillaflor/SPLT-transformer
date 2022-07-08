import torch
from torch import nn


from src.rl_models.gaussian_policy import GaussianPolicy


class IQLTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.embedding_dim = config.embedding_dim

        self.quantile = config.quantile
        self.discount = config.discount
        self.soft_target_tau = config.soft_target_tau
        self.alpha = config.alpha
        self.clip_score = config.clip_score
        self.reward_scale = config.reward_scale

        self.observation_mean = nn.Parameter(config.observation_mean, requires_grad=False)
        self.observation_std = nn.Parameter(config.observation_std + 1.e-6, requires_grad=False)
        self.action_mean = nn.Parameter(config.action_mean, requires_grad=False)
        self.action_std = nn.Parameter(config.action_std + 1.e-6, requires_grad=False)

        self.create_layers(config)

    def configure_optimizers(self, train_config):
        self.qf1_optimizer = torch.optim.AdamW(
            self.qf1.parameters(),
            weight_decay=train_config.weight_decay,
            lr=train_config.learning_rate,
        )
        self.qf2_optimizer = torch.optim.AdamW(
            self.qf2.parameters(),
            weight_decay=train_config.weight_decay,
            lr=train_config.learning_rate,
        )
        self.vf_optimizer = torch.optim.AdamW(
            self.vf.parameters(),
            weight_decay=train_config.weight_decay,
            lr=train_config.learning_rate,
        )
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            weight_decay=train_config.weight_decay,
            lr=train_config.learning_rate,
        )

        optimizers = [
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.vf_optimizer,
            self.policy_optimizer,
        ]
        return optimizers

    def create_layers(self, config):
        self.policy = GaussianPolicy(**config)

        qf1 = [
            nn.Linear(self.transition_dim, self.embedding_dim),
            nn.ReLU(),
        ]
        for _ in range(config.n_hidden_layers):
            qf1.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            qf1.append(nn.ReLU())
        qf1.append(nn.Linear(self.embedding_dim, 1))
        self.qf1 = nn.Sequential(*qf1)
        #  self.qf1 = nn.Sequential(
            #  nn.Linear(self.transition_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, 1)
        #  )

        qf2 = [
            nn.Linear(self.transition_dim, self.embedding_dim),
            nn.ReLU(),
        ]
        for _ in range(config.n_hidden_layers):
            qf2.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            qf2.append(nn.ReLU())
        qf2.append(nn.Linear(self.embedding_dim, 1))
        self.qf2 = nn.Sequential(*qf2)
        #  self.qf2 = nn.Sequential(
            #  nn.Linear(self.transition_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, 1)
        #  )

        target_qf1 = [
            nn.Linear(self.transition_dim, self.embedding_dim),
            nn.ReLU(),
        ]
        for _ in range(config.n_hidden_layers):
            target_qf1.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            target_qf1.append(nn.ReLU())
        target_qf1.append(nn.Linear(self.embedding_dim, 1))
        self.target_qf1 = nn.Sequential(*target_qf1)
        #  self.target_qf1 = nn.Sequential(
            #  nn.Linear(self.transition_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, 1)
        #  )

        target_qf2 = [
            nn.Linear(self.transition_dim, self.embedding_dim),
            nn.ReLU(),
        ]
        for _ in range(config.n_hidden_layers):
            target_qf2.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            target_qf2.append(nn.ReLU())
        target_qf2.append(nn.Linear(self.embedding_dim, 1))
        self.target_qf2 = nn.Sequential(*target_qf2)
        #  self.target_qf2 = nn.Sequential(
            #  nn.Linear(self.transition_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, 1)
        #  )

        vf = [
            nn.Linear(self.observation_dim, self.embedding_dim),
            nn.ReLU(),
        ]
        for _ in range(config.n_hidden_layers):
            vf.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            vf.append(nn.ReLU())
        vf.append(nn.Linear(self.embedding_dim, 1))
        self.vf = nn.Sequential(*vf)
        #  self.vf = nn.Sequential(
            #  nn.Linear(self.observation_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, self.embedding_dim),
            #  nn.ReLU(),
            #  nn.Linear(self.embedding_dim, 1)
        #  )

    def forward(self, inputs):
        obs = (inputs['observations'] - self.observation_mean) / self.observation_std
        actions = (inputs['actions'] - self.action_mean) / self.action_std
        transition = torch.cat([obs, actions], dim=-1)

        preds = {}
        preds['actions'] = (self.policy.get_action(obs) * self.action_std) + self.action_mean
        preds['qs'] = self.qf1(transition)
        preds['vs'] = self.vf(obs)

        return preds

    def compute_losses(self, inputs):
        rewards = inputs['rewards']
        terminals = inputs['terminals']
        obs = (inputs['observations'] - self.observation_mean) / self.observation_std
        actions = (inputs['actions'] - self.action_mean) / self.action_std
        next_obs = (inputs['next_observations'] - self.observation_mean) / self.observation_std
        transition = torch.cat([obs, actions], dim=-1)
        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)

        """
        QF Loss
        """
        q1_pred = self.qf1(transition)
        q2_pred = self.qf2(transition)
        target_vf_pred = self.vf(next_obs).detach()

        q_target = self.reward_scale * rewards + (1. - terminals.float()) * self.discount * target_vf_pred
        q_target = q_target.detach()
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        """
        VF Loss
        """
        q_pred = torch.min(
            self.target_qf1(transition),
            self.target_qf2(transition),
        ).detach()
        vf_pred = self.vf(obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        """
        Policy Loss
        """
        policy_logpp = dist.log_prob(actions)

        adv = q_pred - vf_pred
        exp_adv = torch.exp(adv / self.alpha)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        weights = exp_adv[..., 0].detach()
        policy_loss = (-policy_logpp * weights).mean()

        losses = [
            qf1_loss,
            qf2_loss,
            vf_loss,
            policy_loss
        ]
        return losses

    def target_updates(self):
        soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
