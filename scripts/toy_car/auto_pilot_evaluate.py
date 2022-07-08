from tqdm import tqdm
import numpy as np


from src.envs.toy_car.toy_car import ToyCar

#######################
####### dataset #######
#######################

env = ToyCar()

#######################
###### main loop ######
#######################

returns = []
successes = []

T = 40
num_episodes = int(1e2)

for i in tqdm(range(num_episodes)):
    observation = env.reset(testing=True)
    total_reward = 0

    for t in range(T):

        action = env.autopilot()

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

json_data = {
    'Mean Return': np.mean(returns),
    'Std Return': np.std(returns),
    'Mean Success Rate': np.mean(successes),
}
print(json_data)
