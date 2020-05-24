import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from itertools import product


def plot_all_conditions(
        trial,
        which_data='outputs',
        num_rows=2,
        num_trials=None,
        num_neurons=None,
        cmap=None,
        **plotting_args
):
    """
    Generates plots of trial activity for each condition, in a grid.
    num_trials: number of trials to plot
    """
    # process plotting arguments
    data = trial.data[which_data]
    n = data.shape[1]
    num_conditions = trial.num_conditions
    num_cols = int(num_conditions / num_rows)
    num_trials = trial.num_trials if num_trials is None else num_trials
    if num_neurons is None:
        num_neurons = n

    # if subselecting neurons, randomly choose neurons to plot
    if num_neurons < n:
        nidx = np.random.choice(n, num_neurons, replace=False)
    else:
        nidx = range(num_neurons)

    # assign neuron colors
    cmap = cm.get_cmap('tab10') if cmap is None else cmap
    colors = cmap.colors * int(np.ceil(num_neurons/len(cmap.colors)))

    # create plotting elements
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        sharex='all',
        sharey='all',
        **plotting_args,
    )

    for (cond_num, ax), trial_num, (nrn_num, nrn) in product(
            enumerate(axes.ravel()), range(num_trials), enumerate(nidx)
    ):
        ax.plot(
            trial.tt,
            data[:, nrn, cond_num, trial_num],
            c=colors[nrn_num]
        )
        ax.set_title('{}'.format(trial.targets[0, :, cond_num, 0].astype(int)))
    plt.tight_layout()
    plt.show()

    return


