import torch
from copy import deepcopy


@torch.no_grad()
def splt_trajectory_plan(
    model,
    inp,
    n_steps,
    observation_dim,
    action_dim,
    discount=0.99,
    max_context_transitions=0,
    device='cpu',
    agg='min',
):
    x = deepcopy(inp)

    bs = x['observations'].shape[0]

    latents = model.get_all_latents()

    beam_width, world_dim = latents['world_z'].shape
    policy_dim = latents['policy_z'].shape[1]

    latents['world_z'] = latents['world_z'].repeat(bs, 1)
    latents['policy_z'] = latents['policy_z'].repeat(bs, 1)

    ## repeat input for search
    for k in x:
        x[k] = x[k].repeat_interleave(beam_width, 0)

    ## construct reward and discount tensors for estimating values
    rewards = torch.zeros(beam_width * bs, n_steps + 1, device=device)
    discounts = discount ** torch.arange(n_steps + 1, device=device)

    for t in range(n_steps):
        ## sample actions
        inp_x = {}
        for k in x:
            inp_x[k] = x[k][:, -max_context_transitions:]
        policy_outputs = model.policy_decode(inp_x, latents)
        actions = policy_outputs['actions'][:, -1]

        x['actions'] = torch.cat([x['actions'][:, :-1], actions.unsqueeze(1)], dim=1)
        inp_x = {}
        for k in x:
            inp_x[k] = x[k][:, -max_context_transitions:]
        world_outputs = model.world_decode(inp_x, latents)

        r_t = world_outputs['rewards'][:, -1, 0]
        V_t = world_outputs['values'][:, -1, 0]
        observations = world_outputs['observations'][:, -1]

        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x['observations'] = torch.cat([x['observations'], observations.unsqueeze(1)], dim=1)
            x['actions'] = torch.cat([x['actions'], torch.zeros_like(x['actions'])[:, :1]], dim=1)

    for k in x:
        ## [ batch_size x beam_width x (n_context + n_steps) x dim ]
        x[k] = x[k].view(bs, beam_width, -1, x[k].shape[-1])

        ## crop out context transitions
        ## [ batch_size x beam_width x n_steps x dim ]
        x[k] = x[k][:, :, -n_steps:]

    values = values.reshape(bs, 2**world_dim, -1)
    if agg == 'min':
        world_values, world_indices = values.min(dim=1)
    elif agg == 'max':
        world_values, world_indices = values.max(dim=1)
    elif agg == 'mean':
        _, world_indices = values.min(dim=1)
        world_values = values.mean(dim=1)
    else:
        raise NotImplementedError
    policy_index = world_values.argmax(dim=1)
    world_index = world_indices[torch.arange(bs), policy_index]

    picked_sequence = {}
    for k in x:
        picked_sequence[k] = x[k].reshape(bs, 2**world_dim, 2**policy_dim, n_steps, -1)[torch.arange(bs), world_index, policy_index]

    return picked_sequence, x, world_index, policy_index
