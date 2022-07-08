import json
import glob
import argparse
import numpy as np
import os


def main(args):
    trajectory_paths = glob.glob('{}/*'.format(args.data_path))
    assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(args.data_path)
    obs = []
    actions = []
    rewards = []
    dones = []
    next_obs = []
    speeds = []
    successes = []
    for trajectory_path in trajectory_paths:
        json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
        if len(json_paths) > 0:
            traj_obs = []
            traj_actions = []
            traj_rewards = []
            traj_dones = []
            traj_next_obs = []
            for json_path in json_paths:
                with open(json_path) as f:
                    sample = json.load(f)
                    traj_obs.append(sample['obs'])
                    traj_actions.append(sample['action'])
                    traj_rewards.append(sample['reward'])
                    traj_dones.append(sample['done'])
                    traj_next_obs.append(sample['next_obs'])

                    speeds.append(sample['speed'])
                    if sample['done']:
                        successes.append(sample['termination_state'] == 'success')

            traj_obs = np.array(traj_obs)
            traj_actions = np.array(traj_actions)
            traj_rewards = np.array(traj_rewards)
            traj_dones = np.array(traj_dones)
            traj_next_obs = np.array(traj_next_obs)

            traj_dones[-1] = True

            obs.append(traj_obs)
            actions.append(traj_actions)
            rewards.append(traj_rewards)
            dones.append(traj_dones)
            next_obs.append(traj_next_obs)

    obs = np.concatenate(obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    dones = np.concatenate(dones, axis=0)
    next_obs = np.concatenate(next_obs, axis=0)

    print(np.mean(speeds), np.std(speeds))
    print(np.mean(successes), np.std(successes))

    f = os.path.join(args.data_path, 'data.npz')
    np.savez(
        f,
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=dones,
        next_observations=next_obs,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to trajectory path')
    args = parser.parse_args()
    main(args)
