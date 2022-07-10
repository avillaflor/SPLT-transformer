import numpy as np
import torch


from src.utils import discretization
from src.utils.arrays import to_torch
from src.data.d4rl import load_environment, qlearning_dataset_with_timeouts, qlearning_dataset


def segment(observations, terminals, max_path_length):
    """
        segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    curr_len = 0
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        curr_len += 1
        if term.squeeze() or (curr_len >= max_path_length):
            trajectories.append([])
            curr_len = 0

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1

    return trajectories_pad, early_termination, path_lengths


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env=None, sequence_length=250, step=10, discount=0.99, max_path_length=1000, target_offset=1, penalty=None, device='cuda:0', dataset=None, timeouts=True, **kwargs):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device

        self.target_offset = target_offset

        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        if timeouts:
            dataset = qlearning_dataset_with_timeouts(env=env.unwrapped if env else env, dataset=dataset, terminate_on_end=True)
        else:
            dataset = qlearning_dataset(env=env.unwrapped if env else env, dataset=dataset)
        print('✓')

        observations = dataset['observations']
        actions = dataset['actions']
        next_observations = dataset['next_observations']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        realterminals = dataset['realterminals']

        self.observations_raw = observations
        self.actions_raw = actions
        self.next_observations_raw = next_observations
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.rewards_raw = rewards
        self.terminals_raw = terminals

        ## terminal penalty
        if penalty is not None:
            terminal_mask = realterminals.squeeze()
            self.rewards_raw[terminal_mask] = penalty

        ## segment
        print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        self.next_observations_segmented, *_ = segment(self.next_observations_raw, terminals, max_path_length)
        self.terminals_segmented, *_ = segment(self.terminals_raw, terminals, max_path_length)
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length)
        print('✓')

        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)
        self.returns_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
            self.values_segmented[:,t] = V

            R = (self.rewards_segmented[:,t:] * self.discounts[:self.max_path_length-t]).sum(axis=1)
            self.returns_segmented[:,t] = R


        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)

        returns_raw = self.returns_segmented.squeeze(axis=-1).reshape(-1)
        returns_mask = ~self.termination_flags.reshape(-1)
        self.returns_raw = returns_raw[returns_mask, None]

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            end = length
            for i in range(end):
                indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.n_trajectories = n_trajectories
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.next_observations_segmented = np.concatenate([
            self.next_observations_segmented,
            np.zeros((n_trajectories, sequence_length-1, self.observation_dim)),
        ], axis=1)
        self.terminals_segmented = np.concatenate([
            self.terminals_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.returns_segmented = np.concatenate([
            self.returns_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool),
        ], axis=1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]

        ## [ (sequence_length / skip) x observation_dim]
        joined = to_torch(joined, device='cpu', dtype=torch.float).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0
        # TODO mask is problematic if not predicting terminals

        ## flatten everything
        joined = joined.view(-1)
        mask = mask.view(-1)

        X = {
            'transitions': joined[:-1],
        }
        Y = {
            'transitions': joined[self.target_offset:],
        }
        mask = mask[self.target_offset:]

        return X, Y, mask

    def get_stats(self):
        unfilt_diffs = self.observations_raw[1:] - self.observations_raw[:-1]
        diffs = unfilt_diffs[~self.terminals_raw[:-1, 0].astype(bool)]
        return {
            'observation_mean': self.observations_raw.mean(axis=0),
            'observation_std': self.observations_raw.std(axis=0),
            'action_mean': self.actions_raw.mean(axis=0),
            'action_std': self.actions_raw.std(axis=0),
            'reward_mean': self.rewards_raw.mean(axis=0),
            'reward_std': self.rewards_raw.std(axis=0),
            'value_mean': self.values_raw.mean(axis=0),
            'value_std': self.values_raw.std(axis=0),
            'observation_diff_mean': diffs.mean(axis=0),
            'observation_diff_std': diffs.std(axis=0),
            'return_mean': self.returns_raw.mean(axis=0),
            'return_std': self.returns_raw.std(axis=0),

            'observation_max': self.observations_raw.max(axis=0),
            'observation_min': self.observations_raw.min(axis=0),
            'action_max': self.actions_raw.max(axis=0),
            'action_min': self.actions_raw.min(axis=0),
            'reward_max': self.rewards_raw.max(axis=0),
            'reward_min': self.rewards_raw.min(axis=0),
            'value_max': self.values_raw.max(axis=0),
            'value_min': self.values_raw.min(axis=0),
            'observation_diff_max': diffs.max(axis=0),
            'observation_diff_min': diffs.min(axis=0),
            'return_max': self.returns_raw.max(axis=0),
            'return_min': self.returns_raw.min(axis=0),
        }

    def get_max_return(self):
        return self.returns_segmented[:, 0, 0].max()

    def get_min_return(self):
        return self.returns_segmented[:, 0, 0].min()


class DiscretizedDataset(SequenceDataset):

    def __init__(self, *args, N=50, discretizer='QuantileDiscretizer', **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        discretizer_class = getattr(discretization, discretizer)
        self.discretizer = discretizer_class(self.joined_raw, N)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]

        joined_discrete = self.discretizer.discretize(joined)

        ## replace with termination token if the sequence has ended
        assert (joined[terminations] == 0).all(), \
                f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
        # TODO this is the problem
        joined_discrete[terminations] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(joined_discrete, device='cpu', dtype=torch.long).contiguous()

        joined_rounded = self.discretizer.reconstruct(joined_discrete)
        joined_rounded = to_torch(joined_rounded, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0

        ## flatten everything
        joined_discrete = joined_discrete.view(-1)
        joined_rounded = joined_rounded.view(-1)
        mask = mask.view(-1)

        X = {
            'transitions': joined_discrete[:-1],
        }
        Y = {
            'transitions': joined_discrete[self.target_offset:],
            'transitions_rounded': joined_rounded[self.target_offset:],
            'mask_rounded': mask[self.target_offset:],
        }
        mask = mask[:-1]

        return X, Y, mask
