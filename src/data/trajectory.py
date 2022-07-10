import torch


from src.utils.arrays import to_torch
from src.data.sequence import SequenceDataset


class TrajectoryDataset(SequenceDataset):
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        rewards = joined[:, -2:-1]
        values = joined[:, -1:]

        returns = to_torch(returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(1)
        mask[traj_inds > self.max_path_length - self.step] = 0

        X = {
            'observations': observations[:-1],
            'next_observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'traj_indices': path_ind,
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
        }
        return X, Y, mask
