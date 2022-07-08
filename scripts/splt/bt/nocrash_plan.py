import json
from os.path import join
from tqdm import tqdm
import numpy as np


import src.utils as utils
from src.policies.splt_bt_policy import SPLTBTPolicy
from src.envs.nocrash.environment.config.config import DefaultMainConfig
from src.envs.nocrash.environment.multi_agent.carla_env import CarlaEnv
from src.envs.nocrash.environment.config.scenario_configs import NoCrashDenseTown02Config
from src.envs.nocrash.environment.config.observation_configs import LowerDimPIDObservationConfig
from src.envs.nocrash.environment.config.action_configs import MergedSpeedTanhConfig
from src.envs.nocrash.environment.config.reward_configs import SpeedRewardConfig


class Parser(utils.Parser):
    dataset: str = 'random-ttc'
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

config = DefaultMainConfig()
config.populate_config(
    observation_config=LowerDimPIDObservationConfig(),
    action_config=MergedSpeedTanhConfig(),
    reward_config=SpeedRewardConfig(),
    scenario_config=NoCrashDenseTown02Config(fps=5),
    carla_gpu=int(args.device[-1]),
    testing=True,
)
num_agents = 1
env = CarlaEnv(config=config, num_agents=num_agents, like_gym=False)

discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#######################
###### main loop ######
#######################

returns = []
traj_speeds = []

successes = 0
fails = 0

T = 5000
num_trials = 4
num_episodes = int((num_trials * 25) / num_agents)
max_history = args.max_context_transitions

gpt.eval()

policy = SPLTBTPolicy(
    gpt,
    args.horizon,
    observation_dim,
    action_dim,
    discount,
    bs=num_agents,
    max_history=max_history,
    device=args.device,
    )

try:
    for ep in tqdm(range(num_episodes)):
        actives = np.ones(num_agents, dtype=bool)
        observation = env.reset(random_spawn=False, index=((num_agents * ep) % 25))
        policy.reset()
        total_reward = np.zeros(num_agents, dtype=np.float32)

        for t in range(T):

            action, sequence, candidates, world_index, policy_index = policy(observation, max_horizon=T-t, return_plans=True)

            ## execute action in environment
            next_observation, reward, terminal, info = env.step(action)

            for i in range(num_agents):
                if actives[i]:
                    ## update return
                    total_reward[i] += reward[i]

                    traj_speeds.append(info[i]['speed'])

                    if terminal[i]:
                        returns.append(total_reward[i])
                        status = info[i]['termination_state']
                        success = (status == 'success')
                        successes += int(success)
                        fails += int(not success)

                        actives[i] = False
                        print(total_reward[i], t, i)

            if ~actives.any():
                break

            observation = next_observation

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    mean_speed = float(np.mean(traj_speeds))
    num_successes = float(successes)
    success_rate = float(successes) / float(successes + fails)
    print('Mean Return', mean_return)
    print('Std Return', std_return)
    print('Mean Speed', mean_speed)
    print('Num Succeses', num_successes)
    print('Success Rate', success_rate)

    utils.serialization.mkdir(args.savepath)
    json_path = join(args.savepath, 'rollout.json')
    json_data = {
        'Mean Return': mean_return,
        'Std Return': std_return,
        'Mean Speed': mean_speed,
        'Num Succeses': num_successes,
        'Success Rate': success_rate,
    }
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
finally:
    env.close()
