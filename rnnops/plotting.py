import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from itertools import product

from rnnops import Trial

def plot_all_conditions(
        trials,
        which_data='outputs',
        num_rows=2,
        fig=None,
        axes=None,
        num_trials=None,
        num_neurons=None,
        cmap=None,
        trial_summary_fn=None,
        **plotting_args
):
    """
    Generates plots of trial activity for each condition, in a grid.
    Assumes that conditions are along index 2 of the trial data.
    num_trials: number of trials to plot
    """

    if isinstance(trials, Trial):
        trials = [trials]

    # process plotting arguments
    n = trials[0].data[which_data].shape[1]
    num_conditions = trials[0].num_conditions
    num_cols = int(num_conditions / num_rows)
    num_trials = trials[0].num_trials if num_trials is None else num_trials
    if num_neurons is None:
        num_neurons = n

    # if subselecting neurons, randomly choose neurons to plot
    if num_neurons < n:
        nidx = np.random.choice(n, num_neurons, replace=False)
    else:
        nidx = range(num_neurons)

    # assign neuron colors
    cmap = cm.get_cmap('tab10') if cmap is None else cmap
    if len(trials) == 1:
        colors = cmap.colors * int(np.ceil(num_neurons/len(cmap.colors)))
    else:
        colors = cmap.colors * int(np.ceil(len(trials)/len(cmap.colors)))

    # create plotting elements
    if fig is None and axes is None:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            sharex='all',
            sharey='all',
            **plotting_args,
        )

    def plot_one(trial, i):
        data = trial.data[which_data]
        for (cond_num, ax), trial_num, (nrn_num, nrn) in product(
                enumerate(axes.ravel()), range(num_trials), enumerate(nidx)
        ):
            if trial_summary_fn is None:
                plot_data = data[:, nrn, cond_num, trial_num]
            else:
                plot_data = trial_summary_fn(
                    data[:, nrn, cond_num, :],
                    axis=-1,
                    keepdims=True
                )
            ax.plot(
                trial.tt,
                plot_data,
                c= colors[nrn_num] if i is None else colors[i]
            )
            ax.set_title(
                '{}'.format(trial.targets[0, :, cond_num, 0].astype(int))
            )

    for i, trial in enumerate(trials):
        if len(trials) == 1:
            i = None
        plot_one(trial, i)

    plt.tight_layout()

    return fig, axes


