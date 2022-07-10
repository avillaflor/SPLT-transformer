import os
import numpy as np
import torch


from src.utils.arrays import to_torch
import src.utils as utils
from src.data.trajectory import TrajectoryDataset
from src.rl_models.iql import IQLTrainer


class Parser(utils.Parser):
    dataset: str = 'random-ttc'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('train')
args.exp_name = 'iql/' + args.exp_name
split_path = args.savepath.split('/')
args.savepath = '/'.join(split_path[:1] + ['iql'] + split_path[1:])
utils.serialization.mkdir(args.savepath)

#######################
####### dataset #######
#######################

data_file = 'datasets/nocrash/{0}/data.npz'.format(args.dataset.replace('-', '_'))
with open(data_file, 'rb') as f:
    data = dict(np.load(f))

args.subsampled_sequence_length = 2
sequence_length = args.subsampled_sequence_length * args.step

dataset_config = utils.Config(
    TrajectoryDataset,
    savepath=(args.savepath, 'data_config.pkl'),
    env=None,
    dataset=data,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    max_path_length=5000,
    timeouts=False,
)

dataset = dataset_config()
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = obs_dim + act_dim
stats = dataset.get_stats()
reward_scale = 1.

#######################
######## model ########
#######################

block_size = args.subsampled_sequence_length * transition_dim - 1
print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)

model_config = utils.Config(
    IQLTrainer,
    action_mean=to_torch(stats['action_mean'], device=args.device),
    action_std=to_torch(stats['action_std'], device=args.device),
    observation_mean=to_torch(stats['observation_mean'], device=args.device),
    observation_std=to_torch(stats['observation_std'], device=args.device),
    savepath=(args.savepath, 'model_config.pkl'),
    ## architecture
    block_size=block_size,
    n_hidden_layers=1,
    embedding_dim=256,
    reward_scale=reward_scale,
    quantile=0.7,
    discount=args.discount,
    soft_target_tau=5.e-3,
    alpha=1./3.,
    clip_score=100.,
    max_log_std=0,
    min_log_std=-6,
    ## dimensions
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
)

model = model_config()
model.to(args.device)

#######################
####### trainer #######
#######################

trainer_config = utils.Config(
    utils.RLTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=3.e-4,
    betas=(0.9, 0.95),
    weight_decay=0., # only applied on matmul weights
    ## dataloader
    num_workers=4,
    device=args.device,
)

trainer = trainer_config()

#######################
###### main loop ######
#######################

## scale number of epochs to keep number of updates constant
n_epochs = int((1e6 / len(dataset) * args.n_epochs_ref))
for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}')

    trainer.train(model, dataset)

    statepath = os.path.join(args.savepath, f'state_{epoch}.pt')
    print(f'Saving model to {statepath}')

    ## save state to disk
    state = model.state_dict()
    torch.save(state, statepath)
