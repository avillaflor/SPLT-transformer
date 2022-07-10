import os
import json
import datetime
import argparse
from tqdm import tqdm
import numpy as np


# Environment
from src.envs.nocrash.environment.config.config import DefaultMainConfig
from src.envs.nocrash.environment.multi_agent.carla_env import CarlaEnv
from src.envs.nocrash.environment.config.scenario_configs import NoCrashDenseTown01Config
from src.envs.nocrash.environment.config.observation_configs import LowerDimPIDObservationConfig
from src.envs.nocrash.environment.config.action_configs import MergedSpeedTanhConfig
from src.envs.nocrash.environment.config.reward_configs import SpeedRewardConfig
# Policy
from src.envs.nocrash.policies.auto_pilot_policies import AutopilotNoisePolicy


def collect_trajectories(env, save_dir, policy, num_agents, fps, max_path_length=5000):
    now = datetime.datetime.now()
    salt = np.random.randint(100)
    save_paths = []
    for i in range(num_agents):
        fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
        fname = '{0}_{1:03d}'.format(fname, i)
        save_path = os.path.join(save_dir, fname)
        measurements_path = os.path.join(save_path, 'measurements')

        # make directories
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        os.mkdir(measurements_path)
        save_paths.append(save_path)

    obses = env.reset(random_spawn=True)
    actives = np.ones(num_agents, dtype=bool)

    total_steps = 0
    for step in tqdm(range(max_path_length)):

        actions = policy(obses)

        next_obses, rewards, dones, infos = env.step(actions)
        for i in range(num_agents):
            if actives[i]:
                total_steps += 1
                experience = {
                    'obs': obses[i].tolist(),
                    'next_obs': next_obses[i].tolist(),
                    'action': actions[i].tolist(),
                    'reward': float(rewards[i]),
                    'done': bool(dones[i]),
                }
                experience.update(infos[i])

                save_env_state(experience, save_paths[i], step)
                if dones[i]:
                    actives[i] = False

        if not actives.any():
            break

        obses = next_obses

    return total_steps


def save_env_state(measurements, save_path, idx):
    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def main(args):

    config = DefaultMainConfig()
    config.populate_config(
        observation_config=LowerDimPIDObservationConfig(),
        action_config=MergedSpeedTanhConfig(),
        reward_config=SpeedRewardConfig(),
        scenario_config=NoCrashDenseTown01Config(fps=args.fps),
        testing=False,
        carla_gpu=args.gpu
    )
    config.server_fps = args.fps

    with CarlaEnv(config=config, num_agents=args.num_agents, behavior=args.behavior) as env:
        # Create the policy
        policy = AutopilotNoisePolicy(env, steer_noise_std=1.e-2, speed_noise_std=1.e-2, clip=True)

        total_samples = 0
        while total_samples < args.n_samples:
            traj_length = collect_trajectories(env, args.path, policy, args.num_agents, args.fps, max_path_length=args.max_path_length)
            total_samples += traj_length
            print(float(total_samples) / args.n_samples)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_samples', type=int, default=int(3.e5))
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--num_agents', type=int, default=22)
    parser.add_argument('--max_path_length', type=int, default=5000)
    parser.add_argument('--behavior', type=str, default='random')
    args = parser.parse_args()
    main(args)
