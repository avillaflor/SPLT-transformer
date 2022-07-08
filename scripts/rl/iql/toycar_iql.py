import json
from os.path import join
from tqdm import tqdm
import numpy as np


from src.policies.bt_policy import BTPolicy
import src.utils as utils
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

args.logbase = args.logbase + 'iql/'
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
max_history = 1

gpt.eval()

policy = BTPolicy(
    gpt,
    observation_dim,
    action_dim,
    discount,
    max_history=max_history,
    device=args.device)

for i in tqdm(range(num_episodes)):
    observation = env.reset(testing=True)
    policy.reset()
    total_reward = 0

    for t in range(T):

        action = policy(observation)

        ## execute action in environment
        next_observation, reward, terminal, info = env.step(action)


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
