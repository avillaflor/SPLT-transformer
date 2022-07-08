import json
from os.path import join
from tqdm import tqdm
import numpy as np


from src.policies.splt_bt_policy import SPLTBTPolicy
import src.utils as utils
from src.utils.serialization import mkdir
from src.envs.toy_car.toy_car import ToyCar

class Parser(utils.Parser):
    dataset: str = 'idm-uniform07'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

utils.set_device(args.device)

#######################
####### models ########
#######################

args.logbase = args.logbase + 'splt_bt/'
args.exp_name = args.gpt_loadpath + '/' + args.exp_name
args.savepath = join(args.logbase, args.dataset, args.exp_name)
dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)

#######################
####### dataset #######
#######################

env = ToyCar()

discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#######################
###### main loop ######
#######################

returns = []
successes = []

T = 40
num_episodes = 100
max_history = args.max_context_transitions

gpt.eval()

policy = SPLTBTPolicy(
    gpt,
    args.horizon,
    observation_dim,
    action_dim,
    discount,
    max_history=max_history,
    device=args.device,
    agg='min',
)

for i in tqdm(range(num_episodes)):
    traj_path = join(args.savepath, 'traj_{0:04d}'.format(i))
    mkdir(traj_path)
    observation = env.reset(testing=True)
    policy.reset()
    total_reward = 0

    for t in range(T):

        action, sequence, candidates, world_index, policy_index = policy(observation, max_horizon=T-t, return_plans=True)

        ## execute action in environment
        next_observation, reward, terminal, info = env.step(action)

        # saving predictions for plotting
        gt_sequence = np.concatenate([observation, action, [reward]])
        filename = join(traj_path, '{0:04d}.npz'.format(t))
        np.savez(
            filename,
            plan=sequence,
            candidates=candidates,
            gt=gt_sequence,
            world_index=world_index,
            policy_index=policy_index,
            world_dim=gpt.world_latent_dim,
            policy_dim=gpt.policy_latent_dim)

        ## update return
        total_reward += reward

        if terminal:
            returns.append(total_reward)
            successes.append(info['success'])
            print(total_reward, t)
            break

        observation = next_observation

print('Mean Return {0}'.format(np.mean(returns)))
print('Std Return {0}'.format(np.std(returns)))
print('Mean Success Rate {0}'.format(np.mean(successes)))

utils.serialization.mkdir(args.savepath)
json_path = join(args.savepath, 'rollout.json')
json_data = {
    'Mean Return': np.mean(returns),
    'Std Return': np.std(returns),
    'Mean Success Rate': np.mean(successes),
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
