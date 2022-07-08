import torch
import numpy as np


from src.search.sampling import sample_n
from src.utils.arrays import to_torch


REWARD_DIM = VALUE_DIM = 1
VALUE_PLACEHOLDER = 1e6


@torch.inference_mode()
def make_prefix(discretizer, context, obs, prefix_context=True):
    bs, observation_dim = obs.shape
    obs_discrete = discretizer.discretize(obs, subslice=[0, observation_dim])
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)

    if prefix_context:
        prefix = torch.cat(context + [obs_discrete], dim=-1)
    else:
        prefix = obs_discrete

    return prefix


@torch.inference_mode()
def update_context(context, discretizer, observation, action, reward, max_context_transitions):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    transition = np.concatenate([
        observation,
        action,
        reward[:, None],
        np.ones((observation.shape[0], 1)) * VALUE_PLACEHOLDER,
    ], axis=1)

    ## discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = to_torch(transition_discrete, dtype=torch.long)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]

    return context


@torch.inference_mode()
def beam_plan(
    model, value_fn, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None,
):
    '''
        x : tensor[N x input_sequence_length ]
    '''

    bs = x.shape[0]

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    ## repeat input for search
    x = x.repeat(beam_width, 1)

    ## construct reward and discount tensors for estimating values
    rewards = torch.zeros(bs * beam_width, n_steps + 1, device=x.device)
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)

    ## logging
    #  progress = utils.Progress(n_steps) if verbose else utils.Silent()

    for t in range(n_steps):
        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        ## sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_probs)

        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        x = x.reshape((bs, beam_width * n_expand, -1))
        rewards = rewards.reshape((bs, beam_width * n_expand, -1))
        values = values.reshape((bs, beam_width * n_expand))

        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[torch.arange(bs).reshape((bs, 1)), inds]
        rewards = rewards[torch.arange(bs).reshape((bs, 1)), inds]

        x = x.reshape((bs * beam_width, -1))
        rewards = rewards.reshape((bs * beam_width, n_steps+1))
        values = values.reshape((bs * beam_width))

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(bs, beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, :, -n_steps:]

    ## return best sequence
    argmax = values.reshape((bs, -1)).argmax(dim=-1)
    best_sequence = x[torch.arange(bs), argmax]

    return best_sequence, x
