import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob


def main(args):
    plt.figure(figsize=(4, 3))
    files = sorted(glob.glob(args.data_path + '/*'))
    real_data = []
    real_time = []
    cand_color = 'r'
    plan_color = 'b'
    real_color = 'k'
    labels = {}
    for i, f in enumerate(files):
        if i < args.start or i > args.end:
            continue
        data = np.load(f)
        gt = data['gt']
        real_data.append(gt[args.index])
        real_time.append(0.25 * i)

        candidates = data['candidates']
        if (i % candidates.shape[1]) == 0:
            for cand in candidates:
                cand_line = plot(cand, i, cand_color, args.index)

        plan_line = plot(data['plan'], i, plan_color, args.index)

    labels['plan'] = plan_line
    labels['candidates'] = cand_line
    plt.legend(labels)
    plt.scatter(real_time, real_data, c=real_color, marker='.')
    plt.show()


def plot(data, i, color, index):
    mask = ~((np.cumsum(data, axis=0) == np.inf).any(axis=1))
    timesteps = 0.25 * (i + np.arange(data.shape[0]))
    return plt.plot(
        timesteps[mask],
        data[:, index][mask],
        color=color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to trajectory path')
    parser.add_argument('--index', type=int, default=0, help='which index to plot')
    parser.add_argument('--start', type=int, default=0, help='when to start plot')
    parser.add_argument('--end', type=int, default=100, help='when to end plot')
    args = parser.parse_args()
    main(args)
