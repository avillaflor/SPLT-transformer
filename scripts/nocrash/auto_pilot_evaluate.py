from tqdm import tqdm
import numpy as np


from src.envs.nocrash.environment.config.config import DefaultMainConfig
from src.envs.nocrash.environment.multi_agent.carla_env import CarlaEnv
from src.envs.nocrash.environment.config.scenario_configs import NoCrashDenseTown02Config
from src.envs.nocrash.environment.config.observation_configs import LowerDimPIDObservationConfig
from src.envs.nocrash.environment.config.action_configs import MergedSpeedTanhConfig
from src.envs.nocrash.environment.config.reward_configs import SpeedRewardConfig


#######################
######## setup ########
#######################

config = DefaultMainConfig()
config.populate_config(
    observation_config=LowerDimPIDObservationConfig(),
    action_config=MergedSpeedTanhConfig(),
    reward_config=SpeedRewardConfig(),
    scenario_config=NoCrashDenseTown02Config(fps=5),
    carla_gpu=0,
    testing=True,
)
num_agents = 1
env = CarlaEnv(config=config, num_agents=num_agents, like_gym=False, behavior='normal')

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

try:
    for ep in tqdm(range(num_episodes)):

        actives = np.ones(num_agents, dtype=bool)
        observation = env.reset(random_spawn=False, index=((num_agents * ep) % 25))
        total_reward = np.zeros(num_agents, dtype=np.float32)

        for t in range(T):

            action = env.get_autopilot_action()

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
finally:
    env.close()
