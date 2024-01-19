import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import argparse
import csv
from pathlib import Path
import itertools
import time
from copy import deepcopy

from general_config import GeneralConfig

# GAME_NAME = '27m_vs_30m'
# GAME_NAME = '3s5z_vs_3s6z'
# GAME_NAME = '6h_vs_8z'
# GAME_NAME = 'corridor'
# GAME_NAME = 'MMM2'


def csv_has_header(filename):
    with open(filename, 'r') as f:
        first_line = f.readline()
    # if 'mean_reward' in first_line:
    if 'Value' in first_line:
        return True
    else:
        return False


def creat_csv(file_path, csv_head):
    with open(file_path, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)


def write_csv(file_path, data_row):
    with open(file_path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


def moving_average(x, w):
    '''
    Args:
        ndarray or list
    Returns:
        smoothed ndarray or list
    '''
    assert isinstance(x, (list, np.ndarray))
    if isinstance(x, np.ndarray):
        length = x.shape[0]
    else:
        length = len(x)
    # print(f"w: {w}, length: {length}")
    # assert w // 2 <= length
    if w // 2 > length:
        print(f"Warning: Skipping smoothing because data length ({length}) is too short for window size ({w}).")
        return x  
    result = []
    last_left = 0
    last_right = w // 2 - 1
    window_sum = np.sum(x[last_left:last_right + 1])
    for i in range(length):
        left = max(0, i - w // 2)
        right = min(length - 1, i + w // 2)  # include
        if right > last_right:
            window_sum += x[right]
        if left > last_left:
            window_sum -= x[last_left]
        count = right - left + 1
        result.append(window_sum / count)

        last_left = left
        last_right = right

    if isinstance(x, np.ndarray):
        return np.array(result, dtype='float32')
    else:
        return result


def get_single_csv_data(csv_filename):
    has_header = csv_has_header(csv_filename)
    if has_header:
        df = pd.read_csv(csv_filename, sep=',')
    else:
        df = pd.read_csv(
            csv_filename,
            header=None,
            names=['Wall time', 'Step', 'Value'],
            sep=',')
    return df


def get_smoothed_csv_data(csv_filename, point_interval, reference_window_size, window_size_increase_rate):
    ''' Sample smoothed points according to the interval
    Args:
        csv_filename: csv file
        smoothing_window_size: window size to do a moving average.
    Returns:
        Steps (list): smoothed step
        value (list): smoothed value
    '''

    has_header = csv_has_header(csv_filename)
    if has_header:
        df = pd.read_csv(csv_filename, sep=',')
    else:
        df = pd.read_csv(
            csv_filename,
            header=None,
            names=['Wall time', 'Step', 'Value'],
            sep=',')
    #print('origin df shape : {}'.format(df.shape))
    # df['Step'] = (df['num_timesteps'].values // point_interval * point_interval).astype('int64')
    df['Step'] = (df['Step'].values // point_interval * point_interval).astype('int64')
    new_df = df.groupby('Step').mean()
    #print('new_df.shape:{}'.format(new_df.shape))

    steps = new_df.index.values.astype('int64').tolist()
    values = new_df['Value'].values.tolist()
    point_dense = len(steps) / steps[-1]
    relative_point_dense = point_dense * 1000 * window_size_increase_rate
    smoothing_window_size = int(relative_point_dense * reference_window_size)
    # print('window_size: {}'.format(smoothing_window_size))
    smoothed_values = moving_average(values, smoothing_window_size)
    return steps, smoothed_values


def calc_mean_and_confidence(csv_file_list,
                             reference_window_size, window_size_increase_rate):
    # print('num experients: {}'.format(len(csv_file_list)))
    steps_all = []
    values_all = []

    first_df = get_single_csv_data(csv_file_list[0])
    first_steps = first_df['Step'].values
    point_interval = first_steps[-1] // first_steps.shape[0]
    # print('first_steps =', first_steps)
    # print('first_steps[-1] =', first_steps[-1])
    # print('first_steps.shape[0] =', first_steps.shape[0])
    # print('point_interval =', point_interval)
    for csv_file_name in csv_file_list:
        steps, values = get_smoothed_csv_data(csv_file_name, point_interval,
                                              reference_window_size, window_size_increase_rate)
        steps_all.append(steps)
        values_all.append(values)
        print('len steps : {}, values {}'.format(len(steps), len(values)))

    # make sure that all csv files has the same steps
    selected_steps = []
    selected_values = [[] for _ in range(len(steps_all))]
    while True:
        if np.any([len(steps) == 0 for steps in steps_all]):
            break
        max_step = np.max([steps[0] for steps in steps_all])
        select = True
        for i in range(len(steps_all)):
            if steps_all[i][0] < max_step:
                steps_all[i].pop(0)
                values_all[i].pop(0)
                select = False
        if select == True:
            selected_steps.append(max_step)
            for i in range(len(steps_all)):
                selected_values[i].append(values_all[i][0])
                steps_all[i].pop(0)
                values_all[i].pop(0)

    std = []
    last = []
    for i in range(len(selected_values[0])):
        std.append(np.std([values[i] for values in selected_values]))
    assert len(std) == len(selected_steps)
    std = np.array(std)

    selected_values_all = np.array(selected_values)
    values_mean = np.mean(selected_values_all, axis=0)
    values_ub = values_mean + std
    values_lb = values_mean - std
    selected_steps = np.array(selected_steps)
    last.append(values_mean[-1])
    last.append(values_ub[-1])
    last.append(values_lb[-1])

    print('steps: {}, mean: {}, ub: {}, lb: {}'.format(
        selected_steps.shape, values_mean.shape, values_ub.shape,
        values_lb.shape))
    return selected_steps, values_mean, values_ub, values_lb, last


def plot_mean_and_CI(ax,
                     x,
                     mean,
                     ub,
                     lb,
                     color_mean=None,
                     color_shading=None,
                     line=None):
    ax.fill_between(x, ub, lb, color=color_shading, alpha=0.03, lw=0.0)
    line = ax.plot(x, mean, color=color_mean, lw=1.0)
    ax.grid(visible=True, alpha=0.2)
    # ax.set_facecolor('plum')
    ax.patch.set_alpha(0.2)


def draw_single_env(all_csv_file_list, ax, task_config, env_config, max_step):
    for i in range(len(task_config['algorithms'])):
        # print(task_config['algorithms'][i])
        x, mean, ub, lb, last_value = calc_mean_and_confidence(
            all_csv_file_list[i], task_config['reference_window_size'], task_config['window_size_increase_rate'])

        file_name = task_config['line_labels'][i] + '_' + env_config['name']
        file_path = task_config['root_path'] + task_config['algorithms'][i] + '/' + file_name
        print('+++ file_path =', file_path)
        csv_head = ['last_mean', 'last_ub', 'last_lb']
        creat_csv(file_path, csv_head)
        write_csv(file_path, last_value)

        plot_mean_and_CI(
            ax,
            x,
            mean,
            ub,
            lb,
            color_mean=env_config['pic_color'][i],
            color_shading=env_config['pic_color'][i],
            line=None)

    ax.set_title(env_config['fig_title'], fontsize=12)
    ax.set_xlabel(env_config['x_label'], fontsize=10)
    ax.set_ylabel(env_config['y_label'], fontsize=10)

    ax.set_xlim(0, max_step)
    ax.set_ylim(env_config['min_y'], env_config['max_y'])

    xmajorLocator = tick.MultipleLocator(env_config['x_major_loc'])
    ymajorLocator = tick.MultipleLocator(env_config['y_major_loc'])
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.tick_params(labelsize=12)

    xminorLocator = tick.MultipleLocator(env_config['x_minor_loc'])
    yminorLocator = tick.MultipleLocator(env_config['y_minor_loc'])
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(12)


def get_all_csv_filename(path):
    result = list(
        map(lambda x: str(x.resolve()),
            list(Path(path).rglob("*.[c][s][v]"))))
    return result


def main(task_config, general_config):
    path = task_config['root_path']
    all_csv_files = get_all_csv_filename(path)
    # fig, axs = plt.subplots(task_config['subplot'][0], task_config['subplot'][1], figsize=(10, 6))
    fig, axs = plt.subplots(task_config['subplot'][0], task_config['subplot'][1], figsize=(4, 3))
    # axes = list(itertools.chain.from_iterable(axs))
    axes = axs
    # axes[-1].remove()
    for i, env in enumerate(task_config['envs']):
        print('-----------env: {} -----------'.format(env))
        all_algs_files = []
        for alg_name in task_config['algorithms']:
            all_algs_files.append(
                list(
                    filter(lambda x: alg_name + '/' + env in x,
                           all_csv_files)))
        draw_single_env(all_algs_files, axes, task_config,
                        general_config[env], task_config['env_steps'][i])
        handles = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'], task_config['line_labels'])]
        leg = fig.legend(
            # labels=task_config['line_labels'],
            handles=handles,
            # bbox_to_anchor=(0.2, 0.87),
            # loc='upper left',
            bbox_to_anchor=(0.93, 0.2),
            loc='lower right',
            fontsize=9)
    fig.tight_layout(pad=0.4)
    for legobj in leg.legend_handles:
        legobj.set_linewidth(4.0)

    plt.savefig(task_config['output_name'], dpi=600)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN agent')
    parser.add_argument('--game', type=str, default='Pong', help="Atari game names")
    args = parser.parse_args()

    GAME_NAME = args.game

    start_time = time.time()
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["font.family"] = "Times New Roman"
    general_config = deepcopy(GeneralConfig)

    if GAME_NAME == '27m_vs_30m':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['qmix', 'action_ranked_qmix'],
            'envs': ['27m_vs_30m'],
            'line_labels': ['QMIX', 'AR QMIX'],
            'reference_window_size': 1,
            'window_size_increase_rate': 1,
            'env_steps': [10000000, 10000000, 10000000, 10000000, 10000000],
            'output_name': './results/27m_vs_30m.pdf'
        }
    elif GAME_NAME == '3s5z_vs_3s6z':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['qmix', 'action_ranked_qmix'],
            'envs': ['3s5z_vs_3s6z'],
            'line_labels': ['QMIX', 'AR QMIX'],
            'reference_window_size': 7,
            'window_size_increase_rate': 7,
            'env_steps': [10000000, 10000000, 10000000, 10000000, 10000000],
            'output_name': './results/3s5z_vs_3s6z.pdf'
        }
    elif GAME_NAME == '6h_vs_8z':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['qmix', 'action_ranked_qmix'],
            'envs': ['6h_vs_8z'],
            'line_labels': ['QMIX', 'AR QMIX'],
            'reference_window_size': 7,
            'window_size_increase_rate': 7,
            'env_steps': [10000000, 10000000, 10000000, 10000000, 10000000],
            'output_name': './results/6h_vs_8z.pdf'
        }
    elif GAME_NAME == 'corridor':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['qmix', 'action_ranked_qmix'],
            'envs': ['corridor'],
            'line_labels': ['QMIX', 'AR QMIX'],
            'reference_window_size': 7,
            'window_size_increase_rate': 7,
            'env_steps': [10000000, 10000000, 10000000, 10000000, 10000000],
            'output_name': './results/corridor.pdf'
        }
    elif GAME_NAME == 'MMM2':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['qmix', 'action_ranked_qmix'],
            'envs': ['MMM2'],
            'line_labels': ['QMIX', 'AR QMIX'],
            'reference_window_size': 7,
            'window_size_increase_rate': 7,
            'env_steps': [10000000, 10000000, 10000000, 10000000, 10000000],
            'output_name': './results/MMM2.pdf'
        }
    elif GAME_NAME == 'CartPole_avgreturn':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['adaptive', 'loge', 'log2', 'log10'],
            'envs': ['CartPole_avgreturn'],
            'line_labels': ['adaptive', 'loge', 'log2', 'log10'],
            'reference_window_size': 0.3,
            'window_size_increase_rate': 0.3,
            'env_steps': [3000, 3000, 3000, 3000, 3000],
            'output_name': './results/CartPole_avgreturn.pdf'
        }
    elif GAME_NAME == 'CartPole_loss':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['adaptive', 'loge', 'log2', 'log10'],
            'envs': ['CartPole_loss'],
            'line_labels': ['adaptive', 'loge', 'log2', 'log10'],
            'reference_window_size': 0.3,
            'window_size_increase_rate': 0.3,
            'env_steps': [3000, 3000, 3000, 3000, 3000],
            'output_name': './results/CartPole_loss.pdf'
        }
    elif GAME_NAME == 'CartPole_ratio':
        task_config = {
            'subplot': (1, 1),
            'root_path': './',
            'algorithms': ['adaptive', 'loge', 'log2', 'log10'],
            'envs': ['CartPole_ratio'],
            'line_labels': ['adaptive', 'loge', 'log2', 'log10'],
            'reference_window_size': 0.3,
            'window_size_increase_rate': 0.3,
            'env_steps': [3000, 3000, 3000, 3000, 3000],
            'output_name': './results/CartPole_ratio.pdf'
        }

    main(task_config, general_config)
    print('finished! total time: {} s'.format(time.time() - start_time))
