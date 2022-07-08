import torch


from src.policies.policy import Policy
from src.search.ma_core import (
    beam_plan,
    make_prefix,
    update_context,
)


class TTPolicy(Policy):
    def __init__(
            self,
            model,
            discretizer,
            horizon,
            beam_width,
            n_expand,
            value_fn,
            observation_dim,
            action_dim,
            discount,
            bs=1,
            verbose=False,
            k_obs=1,
            k_act=None,
            cdf_obs=None,
            cdf_act=None,
            prefix_context=False,
            max_history=0,
            max_context=0,
            device='cuda',
    ):
        super().__init__()
        self.model = model
        self.discretizer = discretizer
        self.horizon = horizon
        self.beam_width = beam_width
        self.n_expand = n_expand
        self.value_fn = value_fn
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount = discount
        self.bs = bs
        self.verbose = verbose
        self.k_obs = k_obs
        self.k_act = k_act
        self.cdf_obs = cdf_obs
        self.cdf_act = cdf_act
        self.prefix_context = prefix_context
        self.max_history = max_history
        self.max_context = max_context
        self.device = device

        self.context = []

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

        ## concatenate previous transitions and current observations to input to model
        prefix = make_prefix(self.discretizer, self.context, observation, self.prefix_context)

        ## sample sequence from model beginning with `prefix`
        sequence, candidates = beam_plan(
            self.model, self.value_fn, prefix,
            horizon, self.beam_width, self.n_expand, self.observation_dim, self.action_dim,
            self.discount, self.max_context, verbose=self.verbose,
            k_obs=self.k_obs, k_act=self.k_act, cdf_obs=self.cdf_obs, cdf_act=self.cdf_act,
        )

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = self.discretizer.reconstruct(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        candidates_recon = self.discretizer.reconstruct(candidates.reshape(-1, candidates.shape[-1])).reshape(candidates.shape)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = sequence_recon[:, 0, self.observation_dim:self.observation_dim+self.action_dim]

        if return_plans:
            if no_bs:
                return action[0], sequence_recon[0], candidates_recon[0]
            else:
                return action, sequence_recon, candidates_recon
        else:
            if no_bs:
                return action[0]
            else:
                return action

    @torch.inference_mode()
    def update_context(self, observation, action, reward):
        if len(observation.shape) == 1:
            assert(self.bs == 1)
            #  no_bs = True
            observation = observation[None]
            action = action[None]
            reward = reward[None]
        # observation (N, obs_dim)
        assert((len(observation.shape) == 2) and (observation.shape[0] == self.bs))
        # action (N, act_dim)
        assert((len(action.shape) == 2) and (action.shape[0] == self.bs))
        # reward (N, )
        assert((len(reward.shape) == 1) and (observation.shape[0] == self.bs))
        self.context = update_context(self.context, self.discretizer, observation, action, reward, self.max_history)

    @torch.inference_mode()
    def reset(self):
        self.context = []
