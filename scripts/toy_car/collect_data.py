import argparse
import numpy as np
import os


from src.envs.toy_car.toy_car import ToyCar


def main(args):
    env = ToyCar()
    o = env.reset()
    total_steps = 0
    obses = []
    actions = []
    rewards = []
    dones = []
    next_obses = []
    while total_steps <= args.num_steps:
        a = env.autopilot()
        noise = 0.01 * np.random.randn()
        a = np.clip(a + noise, a_min=-0.99, a_max=0.99)
        next_o, r, done, _ = env.step(a)
        obses.append(o)
        actions.append(a)
        rewards.append(r)
        dones.append(done)
        next_obses.append(next_o)
        if done:
            o = env.reset()
        else:
            o = next_o
        total_steps += 1
    obses = np.stack(obses, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0)
    dones = np.stack(dones, axis=0)
    next_obses = np.stack(next_obses, axis=0)
    data_path = os.path.join('datasets/toycar', args.exp_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_file = os.path.join(data_path, 'data.npz')
    np.savez(
        data_file,
        observations=obses,
        actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=next_obses,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='data',
                        type=str, help='name of experiment')
    parser.add_argument('--num_steps', default=100000, type=int,
                        help='gpu device to run experiment')
    args = parser.parse_args()
    main(args)
