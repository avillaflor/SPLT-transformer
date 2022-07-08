import json
from os.path import join
from tqdm import tqdm
import numpy as np


from src.datasets.d4rl import load_environment
from src.policies.splt_bt_policy import SPLTBTPolicy
import src.utils as utils


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
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

env = load_environment(args.dataset)

discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#######################
###### main loop ######
#######################

returns = []
successes = []

T = 1000
num_episodes = 10
max_history = args.max_context_transitions
max_context = args.max_context_transitions

gpt.eval()

policy = SPLTBTPolicy(
    gpt,
    args.horizon,
    observation_dim,
    action_dim,
    discount,
    max_history=max_history,
    max_context=max_context,
    device=args.device)

for i in tqdm(range(num_episodes)):
    observation = env.reset()
    policy.reset()
    total_reward = 0

    for t in range(T):

        action, sequence, candidates, world_index, policy_index = policy(observation, max_horizon=T-t, return_plans=True)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        ## update return
        total_reward += reward

        if terminal or (t == (T - 1)):
            total_reward = 100. * env.get_normalized_score(total_reward)
            returns.append(total_reward)
            successes.append(t == (T - 1))
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
