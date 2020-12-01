import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np
import itertools
import sys
import  re

import matplotlib
# Uncomment for remote
# matplotlib.use('nbAgg')
DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

min_entry_train = sys.maxsize
min_entry = sys.maxsize
cut_off_plot = False


def plot_data(data,
              xaxis='Epoch',
              value="AverageEpRet",
              condition="Experiment",
              smooth=1,
              fresh_plot=True,
              **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            if not isinstance(datum, str):
                x = np.asarray(datum[value])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x, y, 'same') / np.convolve(
                    z, y, 'same')
                datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)

    # sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    sns.lineplot(data=data, x=xaxis, y=value, ci=68, **kwargs)
    plt.legend(loc='lower center').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in
        # scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None, eval_exps=False, eval_seen=False):
    """
    Recursively look through logdir for output files.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    per_env_datasets = []
    env_dataframes = []

    # train_time = time.time()
    for root, _, files in os.walk(logdir):
        # Keep the train and eval data separated
        if (not eval_exps):
            if 'train.csv' in files:
                # Modify these if the name of experiments change
                env_name = logdir.split('runs')[1].split('/')[2].split('_')[0]
                # Try this if running fb_servers/
                # env_name = logdir.split('runs')[1].split('/')[4].split('_')[0]
                # Default agent exp name
                if 'agent.cls' not in logdir:
                    agent_name = 'DBC'
                else:
                    agent_name = logdir.split('agent.name=')[-1].split(',')[0]

                exp_name = env_name + '_' + agent_name

                try:
                    penalty_name = logdir.split('agent.name=')[-1].split(
                        'penalty_type')[1].split('=')[1].split('/')[0].split(
                            ',')[0]
                except:
                    penalty_name = None

                try:
                    penalty_weight = logdir.split('agent.name=')[-1].split(
                        'penalty_weight')[1].split('=')[1].split('/')[0].split(
                            ',')[0]
                except:
                    penalty_weight = None

                try:
                    env_resample_rate = logdir.split('agent.name=')[-1].split(
                        'env_resample_rate')[1].split('=')[1].split(
                            '/')[0].split(',')[0]
                except:
                    env_resample_rate = None

                try:
                    batch_size = logdir.split('agent.name=')[-1].split(
                        'batch_size')[1].split('=')[1].split(
                            '/')[0].split(',')[0]
                except:
                    batch_size = None

                try:
                    learning_rate_arg = logdir.split('agent.name=')[-1].split('lr')[-1].split('=')[1]
                    learning_rate_arg = re.findall(r'\d+', learning_rate_arg)
                    learning_rate = learning_rate_arg[0] + '.' + learning_rate_arg[1]
                except:
                    learning_rate = None


                if agent_name != 'drq' and penalty_name is not None:
                    exp_name += '_' + 'penalty_' + penalty_name

                if agent_name != 'drq' and penalty_weight is not None:
                    exp_name += '_' + 'pw_' + penalty_weight

                if env_resample_rate is not None:
                    exp_name += '_' + 'esr_' + env_resample_rate

                if batch_size is not None:
                    exp_name += '_' + 'bs_' + batch_size

                if learning_rate is not None:
                    exp_name += '_' + 'lr_' + learning_rate



                condition1 = condition or exp_name or 'exp'
                condition2 = condition1 + '-' + str(exp_idx)
                exp_idx += 1
                if condition1 not in units:
                    units[condition1] = 0
                unit = units[condition1]
                units[condition1] += 1

                try:
                    exp_data = pd.read_csv(os.path.join(root, 'train.csv'))
                except:
                    print('Could not read from %s' %
                          os.path.join(root, 'train.csv'))
                    continue
                performance = 'episode_reward'
                # Just merge all the data
                exp_data.insert(len(exp_data.columns), 'Unit', unit)
                exp_data.insert(len(exp_data.columns), 'Condition1',
                                condition1)
                exp_data.insert(len(exp_data.columns), 'Condition2',
                                condition2)
                exp_data.insert(len(exp_data.columns), 'Average Return',
                                exp_data[performance])
                datasets.append(exp_data)
                # Take average over all envs
                num_of_envs = 0
                idx = 0
                e1 = exp_data['episode'][0]
                e2 = exp_data['episode'][idx]
                while (e1 == e2):
                    num_of_envs += 1
                    idx += 1
                    e2 = exp_data['episode'][idx]
                idx = 0
                entries = len(exp_data)
                env_smooth_rewards = pd.DataFrame(
                    columns=['episode', 'Average Return', 'step'])
                while ((idx + num_of_envs) <= entries):
                    env_avg_rew = np.mean(
                        exp_data['episode_reward'][idx:idx + num_of_envs])
                    env_smooth_rewards = env_smooth_rewards.append(
                        {
                            'episode': exp_data['episode'][idx],
                            'Average Return': env_avg_rew,
                            'step': exp_data['step'][idx],
                        },
                        ignore_index=True)
                    idx += num_of_envs

                env_smooth_rewards.insert(len(env_smooth_rewards.columns),
                                          'Unit', unit)
                env_smooth_rewards.insert(len(env_smooth_rewards.columns),
                                          'Condition1', condition1)
                env_smooth_rewards.insert(len(env_smooth_rewards.columns),
                                          'Condition2', condition2)
                per_env_datasets.append(env_smooth_rewards)

                # Take separate env data
                # idx = 0
                # entries = len(exp_data)
                # for env in range(num_of_envs):
                #     env_dataframes.append(
                #         pd.DataFrame(
                #             columns=['episode', 'Average Return', 'step']))
                # while ((idx + num_of_envs) <= entries):
                #     env_rewards = exp_data['episode_reward'][
                #         idx:idx + num_of_envs].reset_index(drop=True)
                #     for env in range(num_of_envs):
                #         env_dataframes[env] = env_dataframes[env].append(
                #             {
                #                 'episode': exp_data['episode'][idx],
                #                 'Average Return': env_rewards[env],
                #                 'step': exp_data['step'][idx],
                #             },
                #             ignore_index=True)
                #     idx += num_of_envs

        # print('Train files processing took ', time.time() - train_time)

        eval_seen_res = 'eval_seen.csv' in files
        eval_unseen_res = 'eval_unseen.csv' in files
        if eval_exps and (eval_seen_res or eval_unseen_res):
            # Modify these if the name of experiments change
            env_name = logdir.split('runs')[1].split('/')[2].split('_')[0]

            if 'agent.cls' not in logdir:
                agent_name = 'DBC'
            else:
                agent_name = logdir.split('agent.name=')[-1].split(',')[0]

            exp_name = env_name + '_' + agent_name

            try:
                penalty_name = logdir.split('agent.name=')[-1].split(
                    'penalty_type')[1].split('=')[1].split('/')[0].split(
                        ',')[0]
            except:
                penalty_name = None

            try:
                penalty_weight = logdir.split('agent.name=')[-1].split(
                    'penalty_weight')[1].split('=')[1].split('/')[0].split(
                        ',')[0]
            except:
                penalty_weight = None

            try:
                env_resample_rate = logdir.split('agent.name=')[-1].split(
                    'env_resample_rate')[1].split('=')[1].split(
                        '/')[0].split(',')[0]
            except:
                env_resample_rate = None

            try:
                batch_size = logdir.split('agent.name=')[-1].split(
                    'batch_size')[1].split('=')[1].split(
                        '/')[0].split(',')[0]
            except:
                batch_size = None

            try:
                learning_rate_arg = logdir.split('agent.name=')[-1].split('lr')[-1].split('=')[1]
                learning_rate_arg = re.findall(r'\d+', learning_rate_arg)
                learning_rate = learning_rate_arg[0] + '.' + learning_rate_arg[1]
            except:
                learning_rate = None

            if agent_name != 'drq' and penalty_name is not None:
                exp_name += '_' + 'penalty_' + penalty_name

            if agent_name != 'drq' and penalty_weight is not None:
                exp_name += '_' + 'pw_' + penalty_weight

            if env_resample_rate is not None:
                exp_name += '_' + 'esr_' + env_resample_rate

            if batch_size is not None:
                exp_name += '_' + 'bs_' + batch_size

            if learning_rate is not None:
                exp_name += '_' + 'lr_' + learning_rate


            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                if eval_seen and eval_seen_res:
                    exp_data = pd.read_csv(os.path.join(root, 'eval_seen.csv'))
                elif (not eval_seen) and eval_unseen_res:
                    exp_data = pd.read_csv(
                        os.path.join(root, 'eval_unseen.csv'))
            except:
                print('Could not read from %s' %
                      os.path.join(root, 'eval_x.csv'))
                continue
            performance = 'episode_reward'
            if cut_off_plot:
                global min_entry
                data_points = len(exp_data)
                if data_points <= min_entry:
                    min_entry = data_points
                    print('Min entry updated to ', min_entry)
                    exp_data = exp_data[0:min_entry]
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Average Return',
                            exp_data[performance])
            datasets.append(exp_data)

    return datasets, per_env_datasets, env_dataframes


def get_all_datasets(all_logdirs,
                     legend=None,
                     select=None,
                     exclude=None,
                     eval_exps=False,
                     multi_seed=False,
                     eval_seen=False):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    main_dir = os.getcwd()

    if not multi_seed:

        for logdir in all_logdirs:
            exp_dir = main_dir + '/runs/' + logdir
            if osp.isdir(exp_dir) and exp_dir[-1] == os.sep:
                logdirs += [exp_dir]
            else:
                basedir = osp.dirname(exp_dir)
                fulldir = lambda x: osp.join(basedir, x)
                prefix = exp_dir.split(os.sep)[-1]
                listdir = os.listdir(basedir)
                logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    else:
        for logdir in all_logdirs:
            exp_top_dir = main_dir + '/runs/' + logdir
            exp_seed_dir = os.listdir(exp_top_dir)

            for exp_seed in exp_seed_dir:
                exp_dir = exp_top_dir + exp_seed
                if osp.isdir(exp_dir) and exp_dir[-1] == os.sep:
                    logdirs += [exp_dir]
                else:
                    basedir = osp.dirname(exp_dir)
                    fulldir = lambda x: osp.join(basedir, x)
                    prefix = exp_dir.split(os.sep)[-1]
                    listdir = os.listdir(basedir)
                    logdirs += sorted(
                        [fulldir(x) for x in listdir if prefix in x])
    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [
            log for log in logdirs if all(not (x in log) for x in exclude)
        ]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    all_data = []
    all_per_env_data = []
    all_sep_env_data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data, per_env_data, sep_env_data = get_datasets(
                log, leg, eval_exps=eval_exps, eval_seen=eval_seen)
            all_data.append(data)
            all_per_env_data.append(per_env_data)
            all_sep_env_data.append(sep_env_data)
    else:
        for log in logdirs:
            data, per_env_data, sep_env_data = get_datasets(
                log, eval_exps=eval_exps, eval_seen=eval_seen)
            all_data.append(data)
            all_per_env_data.append(per_env_data)
            all_sep_env_data.append(sep_env_data)

    return all_data, all_per_env_data, all_sep_env_data


def make_plots(all_logdirs,
               legend=None,
               xaxis=None,
               values=None,
               count=True,
               font_scale=1.5,
               smooth=1,
               select=None,
               exclude=None,
               estimator='mean',
               multi_seed=False):

    data, per_env_data, sep_env_data = get_all_datasets(all_logdirs,
                                                        legend,
                                                        select,
                                                        exclude,
                                                        multi_seed=multi_seed)
    if not multi_seed:
        data = data[0][0]
        per_env_data = per_env_data[0][0]
        if len(sep_env_data) > 0:
            sep_env_data = sep_env_data[0]

        legend_label = per_env_data['Condition1'][0]
    else:
        data = list(itertools.chain(*data))
        per_env_data = list(itertools.chain(*per_env_data))
        if len(sep_env_data) > 0:
            sep_env_data = list(itertools.chain(*sep_env_data))

        legend_label = per_env_data[0]['Condition1'][0]

    values = values if isinstance(values, list) else [values]
    condition = 'Experiment2' if count else 'Experiment'
    estimator = getattr(np, estimator)

    for value in values:
        fig = plt.gcf()
        fig.set_size_inches(12.5, 8.5)
        plot_data(data,
                  xaxis=xaxis,
                  value=value,
                  condition=condition,
                  smooth=smooth,
                  estimator=estimator,
                  label=legend_label)
    main_dir = os.getcwd()
    exp_dir = main_dir + '/runs/' + all_logdirs[0]
    plot_name = exp_dir + 'train_performance.png'
    plt.savefig(plot_name)

    plt.cla()
    legend_label_avg = legend_label + '_' + 'avg_envs'
    for value in values:
        fig = plt.gcf()
        fig.set_size_inches(12.5, 8.5)
        plot_data(per_env_data,
                  xaxis=xaxis,
                  value=value,
                  condition=condition,
                  smooth=smooth,
                  estimator=estimator,
                  label=legend_label_avg)
    main_dir = os.getcwd()
    exp_dir = main_dir + '/runs/' + all_logdirs[0]
    plot_name = exp_dir + 'train_smooth_per_env_performance.png'
    plt.savefig(plot_name)

    if len(sep_env_data) > 0:
        for env_idx, env in enumerate(sep_env_data):
            plt.cla()
            legend_label_env = legend_label + '_env_' + str(env_idx)
            for value in values:
                fig = plt.gcf()
                fig.set_size_inches(12.5, 8.5)
                plot_data(env,
                          xaxis=xaxis,
                          value=value,
                          condition=condition,
                          smooth=smooth,
                          estimator=estimator,
                          label=legend_label_env)
            main_dir = os.getcwd()
            exp_dir = main_dir + '/runs/' + all_logdirs[0]
            plot_name = exp_dir + 'train_env_' + str(
                env_idx) + '_performance.png'
            plt.savefig(plot_name)


def make_comparison_plots(all_logdirs,
                          legend=None,
                          xaxis=None,
                          values=None,
                          count=True,
                          font_scale=1.5,
                          smooth=1,
                          select=None,
                          exclude=None,
                          estimator='mean',
                          multi_seed=False,
                          eval_seen=False):

    estimator = getattr(np, estimator)

    # Assuming we have enough colors per data entry
    colors = [
        'violet', 'orange', 'green', 'deeppink', 'm', 'mediumslateblue', 'c',
        'y', 'gold', 'olive', 'seagreen'
    ]

    all_train_data = []
    all_train_avg_data = []
    for idx, exp_data in enumerate(all_logdirs):
        if '.png' in exp_data: continue
        data, avg_env_data, _ = get_all_datasets([exp_data],
                                                 legend,
                                                 select,
                                                 exclude,
                                                 multi_seed=multi_seed)

        all_train_data.append(data)
        all_train_avg_data.append(avg_env_data)

    for idx, data in enumerate(all_train_data):
        if not multi_seed:
            data = data[0][0]
            legend_label = data['Condition1'][0]
        else:
            data = list(itertools.chain(*data))
            legend_label = data[0]['Condition1'][0]

        values = values if isinstance(values, list) else [values]
        condition = 'Experiment2' if count else 'Experiment'

        for value in values:
            fig = plt.gcf()
            fig.set_size_inches(12.5, 8.5)

            plot_data(data,
                      xaxis=xaxis,
                      color=colors[idx],
                      value=value,
                      condition=condition,
                      smooth=smooth,
                      estimator=estimator,
                      label=legend_label)
    main_dir = os.getcwd()
    # Store the plots under the "first" experiment dir
    exp_dir = main_dir + '/runs/' + all_logdirs[0].split('/')[0]
    plot_name = exp_dir + '/train_performance.png'
    plt.savefig(plot_name)

    plt.cla()

    for idx, avg_env_data in enumerate(all_train_avg_data):
        if not multi_seed:
            avg_env_data = avg_env_data[0][0]
            legend_label = avg_env_data['Condition1'][0]
        else:
            avg_env_data = list(itertools.chain(*avg_env_data))
            legend_label = avg_env_data[0]['Condition1'][0]

        values = values if isinstance(values, list) else [values]
        condition = 'Experiment2' if count else 'Experiment'

        for value in values:
            fig = plt.gcf()
            fig.set_size_inches(12.5, 8.5)

            plot_data(avg_env_data,
                      xaxis=xaxis,
                      color=colors[idx],
                      value=value,
                      condition=condition,
                      smooth=smooth,
                      estimator=estimator,
                      label=legend_label)
    main_dir = os.getcwd()
    # Store the plots under the "first" experiment dir
    exp_dir = main_dir + '/runs/' + all_logdirs[0].split('/')[0]
    plot_name = exp_dir + '/train_performance_smooth_over_envs.png'
    plt.savefig(plot_name)

    # Plot the evaluation results
    try:
        plt.cla()
        make_eval_plots(all_logdirs,
                        legend,
                        xaxis,
                        values,
                        count,
                        font_scale,
                        smooth,
                        select,
                        exclude,
                        compare=True,
                        multi_seed=multi_seed,
                        eval_seen=True)
    except:
        print('Did not find eval_seen.csv results.')

    try:
        plt.cla()
        make_eval_plots(all_logdirs,
                        legend,
                        xaxis,
                        values,
                        count,
                        font_scale,
                        smooth,
                        select,
                        exclude,
                        compare=True,
                        multi_seed=multi_seed,
                        eval_seen=False)
    except:
        print('Did not find eval_unseen.csv results.')


def make_eval_plots(all_logdirs,
                    legend=None,
                    xaxis=None,
                    values=None,
                    count=True,
                    font_scale=1.5,
                    smooth=1,
                    select=None,
                    exclude=None,
                    estimator='mean',
                    compare=False,
                    multi_seed=False,
                    eval_seen=False):
    values = values if isinstance(values, list) else [values]
    estimator = getattr(np, estimator)
    # Assuming we have enough colors per data entry
    colors = [
        'crimson', 'orange', 'green', 'deeppink', 'm', 'mediumslateblue', 'c',
        'y', 'gold', 'olive', 'seagreen'
    ]

    for idx, exp_data in enumerate(all_logdirs):
        if '.png' in exp_data: continue
        data, _, _ = get_all_datasets([exp_data],
                                      legend,
                                      select,
                                      exclude,
                                      eval_exps=True,
                                      multi_seed=multi_seed,
                                      eval_seen=eval_seen)
        if not multi_seed:
            data = data[0][0]
            legend_label = data['Condition1'][0]

        else:
            data = list(itertools.chain(*data))
            legend_label = data[0]['Condition1'][0]

        values = values if isinstance(values, list) else [values]
        condition = 'Experiment2' if count else 'Experiment'
        for value in values:
            fig = plt.gcf()
            fig.set_size_inches(12.5, 8.5)

            plot_data(data,
                      xaxis=xaxis,
                      color=colors[idx],
                      value=value,
                      condition=condition,
                      smooth=smooth,
                      estimator=estimator,
                      label=legend_label)
    main_dir = os.getcwd()

    if eval_seen:
        eval_tag = 'eval_seen_performance'
    else:
        eval_tag = 'eval_unseen_performance'
    if not compare:
        exp_dir = main_dir + '/runs/' + all_logdirs[0]
        plot_name = exp_dir + '/' + eval_tag + '.png'
    else:
        # Store in the outter dir
        exp_dir = main_dir + '/runs/' + all_logdirs[0].split('/')[0]
        plot_name = exp_dir + '/' + eval_tag + '_comparison.png'
    plt.savefig(plot_name)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--style', type=str, default='single_seed')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='episode')
    parser.add_argument('--value', '-y', default='Average Return', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=12)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args:
        logdir (strings): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.

    """
    if args.style == "single_seed":
        make_plots(args.logdir,
                   args.legend,
                   args.xaxis,
                   args.value,
                   args.count,
                   smooth=args.smooth,
                   select=args.select,
                   exclude=args.exclude,
                   estimator=args.est)

        try:
            plt.cla()
            make_eval_plots(args.logdir,
                            args.legend,
                            args.xaxis,
                            args.value,
                            args.count,
                            smooth=args.smooth,
                            select=args.select,
                            exclude=args.exclude,
                            estimator=args.est,
                            eval_seen=True)
        except:
            print('Not enough seen evaluation data point.')
        try:
            plt.cla()
            make_eval_plots(args.logdir,
                            args.legend,
                            args.xaxis,
                            args.value,
                            args.count,
                            smooth=args.smooth,
                            select=args.select,
                            exclude=args.exclude,
                            estimator=args.est,
                            eval_seen=False)
        except:
            print('Not enough unseen evaluation data point.')

    if args.style == "multi_seed":
        make_plots(args.logdir,
                   args.legend,
                   args.xaxis,
                   args.value,
                   args.count,
                   smooth=args.smooth,
                   select=args.select,
                   exclude=args.exclude,
                   estimator=args.est,
                   multi_seed=True)

        try:
            plt.cla()
            make_eval_plots(args.logdir,
                            args.legend,
                            args.xaxis,
                            args.value,
                            args.count,
                            smooth=args.smooth,
                            select=args.select,
                            exclude=args.exclude,
                            estimator=args.est,
                            eval_seen=True,
                            multi_seed=True)
        except:
            print('Not enough seen evaluation data point.')
        try:
            plt.cla()
            make_eval_plots(args.logdir,
                            args.legend,
                            args.xaxis,
                            args.value,
                            args.count,
                            smooth=args.smooth,
                            select=args.select,
                            exclude=args.exclude,
                            estimator=args.est,
                            eval_seen=False,
                            multi_seed=True)
        except:
            print('Not enough unseen evaluation data point.')

    elif args.style == "compare":
        make_comparison_plots(args.logdir,
                              args.legend,
                              args.xaxis,
                              args.value,
                              args.count,
                              smooth=args.smooth,
                              select=args.select,
                              exclude=args.exclude,
                              estimator=args.est)

    elif args.style == "compare_seeds":
        make_comparison_plots(args.logdir,
                              args.legend,
                              args.xaxis,
                              args.value,
                              args.count,
                              smooth=args.smooth,
                              select=args.select,
                              exclude=args.exclude,
                              estimator=args.est,
                              multi_seed=True)


if __name__ == "__main__":
    main()
