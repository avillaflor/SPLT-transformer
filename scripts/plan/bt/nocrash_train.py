import os
import numpy as np
import torch

from src.utils.arrays import to_torch
import src.utils as utils
from src.datasets.trajectory import TrajectoryDataset
from src.models.bundled_transformer import BundledTransformerGPT


class Parser(utils.Parser):
    dataset: str = 'random-ttc'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('bc_train')
args.exp_name = 'bt/' + args.exp_name
split_path = args.savepath.split('/')
args.savepath = '/'.join(split_path[:1] + ['bt'] + split_path[1:])
utils.serialization.mkdir(args.savepath)

#######################
####### dataset #######
#######################

data_file = 'datasets/nocrash/{0}/data.npz'.format(args.dataset.replace('-', '_'))
with open(data_file, 'rb') as f:
    data = dict(np.load(f))

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
transition_dim = 2
stats = dataset.get_stats()

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
    BundledTransformerGPT,
    observation_mean=to_torch(stats['observation_mean'], device=args.device),
    observation_std=to_torch(stats['observation_std'], device=args.device),
    action_mean=to_torch(stats['action_mean'], device=args.device),
    action_std=to_torch(stats['action_std'], device=args.device),
    reward_mean=to_torch(stats['reward_mean'], device=args.device),
    reward_std=to_torch(stats['reward_std'], device=args.device),
    value_mean=to_torch(stats['value_mean'], device=args.device),
    value_std=to_torch(stats['value_std'], device=args.device),
    res=False,
    action_tanh=True,
    savepath=(args.savepath, 'model_config.pkl'),
    ## architecture
    block_size=block_size,
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
    ## dimensions
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
    ## loss weighting
    action_weight=1.0,
    observation_weight=0.,
    reward_weight=0.,
    value_weight=0.,
    ## dropout probabilities
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
)

model = model_config()
model.to(args.device)

#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
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

    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
    statepath = os.path.join(args.savepath, f'state_{epoch}.pt')
    print(f'Saving model to {statepath}')

    ## save state to disk
    state = model.state_dict()
    torch.save(state, statepath)
