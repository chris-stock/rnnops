import os

import numpy as np
from matplotlib import pyplot as plt

from rnnops.plotting import plot_all_conditions
from rnnops.ops.construction import construct_xor_rnn
from rnnops.tasks import xor_integration_task
from rnnops.trial import run_neural_dynamics
from rnnops.ops.balancing import robustness_cost_fn, solve_balancing


def create_network(n_rec, expand_targets):
    init_args = {
        'n_rec': n_rec,
        'nonlinearity': 'relu',
        'with_bias': False,
        'expand_inputs': True,
        'expand_targets': expand_targets,
        'trim_zeros': False,
    }
    return construct_xor_rnn(**init_args)


def create_cost_fn(rnn, u_noise_std, x_noise_std,
                   expand_targets, trial_len, num_trials):
    # draw a "training trial" of inputs
    train_trial_args = {
        'noise_std': u_noise_std,
        'trial_len': trial_len,
        'num_trials': num_trials,
        'expand_inputs': True,
        'expand_targets': expand_targets,
    }
    train_trial = xor_integration_task(**train_trial_args)

    # run the network on the training trial
    response_args = {
        'rnn': rnn,
        'trial': train_trial,
        'noise_std': x_noise_std,
    }
    response = run_neural_dynamics(**response_args)

    # make cost function
    return robustness_cost_fn(response, rnn.nonlinearity)


def create_balanced_network(rnn, cost_fn):
    solve_args = {
        'how': 'cvx',
        'cost_fn': cost_fn,
        'verbose': True,
        'solver': 'ECOS',
        'max_iters': 1000,
        'feastol': 1e-5,
        'reltol': 1e-5,
        'abstol': 1e-5,
    }

    balanced_rnn, opt_results = solve_balancing(rnn, **solve_args)
    print(opt_results)
    return balanced_rnn


def create_antibalanced_network(rnn, cost_fn):
    solve_args = {
        'how': 'odeint',
        'cost_fn': cost_fn,
        'method': 'DOP853',
        'T_max': 5.,  # todo: create termination condition (when cost doubles)
        'tau': -1.,
    }

    antibalanced_rnn, opt_results = solve_balancing(rnn, **solve_args)
    print_opt_results(opt_results)
    return antibalanced_rnn


def draw_test_trial(u_noise_std, trial_len, num_trials, noise_levels,
                    expand_targets):
    # draw a sequence of trials with varying noise levels

    test_trial_args = {
        'noise_std': u_noise_std,
        'trial_len': trial_len,
        'num_trials': num_trials,
        'expand_inputs': True,
        'expand_targets': expand_targets,
    }
    train_trial = xor_integration_task(**train_trial_args)
    pass


# def run_rnns(rnns, trial, x_noise_stds):
#     # run some rnns on some trials

def print_opt_results(opt_results):
    for k in ['c0', 'cf', 'delta_c', 'lb', 'ub']:
        print('{:8s} {:.4f}'.format(k, opt_results[k]))
    print('|J0|     {:.4f}'.format(np.linalg.norm(opt_results['J0'])))
    print('|Jf|     {:.4f}'.format(np.linalg.norm(opt_results['Jf'])))


from os import path


def run_networks_with_noise(rnns, test_stim, noise_std):
    response_args = {
        'trial': test_stim,
        'noise_std': noise_std,
    }

    responses = [
        run_neural_dynamics(network, **response_args) for network in rnns
    ]

    return responses


def plot_trial_average_across_conditions(
        responses,
        fig_dir,
        fig_id,
        comparison_trial=None,
        **plot_args
):

    trial_averaged_outputs = np.array([
        np.mean(trial.outputs, axis=-1) for trial in responses
    ])

    if comparison_trial is not None:
        comparison_trial_averaged_outputs = \
            np.mean(comparison_trial.outputs, axis=-1)
    else:
        comparison_trial_averaged_outputs = None

    fig, axes = plt.subplots(
        2,
        2,
        sharex='all',
        sharey='all',
        **plot_args,
    )

    for i, ax in enumerate(axes.ravel()):
        if comparison_trial_averaged_outputs is not None:
            ax.plot(
                comparison_trial_averaged_outputs[:, 0, i],
                comparison_trial_averaged_outputs[:, 1, i],
                color=(0.3, 0.3, 0.3),
                lw=2,
                linestyle='dotted',
            )
        for j, tr in enumerate(responses):
            ax.plot(
                trial_averaged_outputs[j, :, 0, i],
                trial_averaged_outputs[j, :, 1, i],
                color=plt.get_cmap('Dark2').colors[j],
                lw=2,
            )

        ax.scatter(
            responses[0].targets[0, 0, i, 0],
            responses[0].targets[0, 1, i, 0],
            c=[(.5, .5, .5)],

        )

        ax.set_xlim(-.1, 1.1)
        ax.set_ylim(-.1, 1.1)
        plt.tight_layout()

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    figname = 'output_trajectories_by_condition_{:0.2f}.png'.format(fig_id)
    plt.savefig(path.join(fig_dir, figname), dpi=350)




if __name__ == 'main':
    # parse args

    expand_inputs, expand_targets = True, True

    response_args = {
        'trial': stim,
        'noise_std': 0.2,
    }

    transformed_response_cvx = run_neural_dynamics(
        transformed_rnn_cvx, **response_args)
    transformed_response = run_neural_dynamics(
        transformed_rnn, **response_args)
    transformed_response_reverse = run_neural_dynamics(
        transformed_rnn_reverse, **response_args)
    response = run_neural_dynamics(rnn, **response_args)

    create_network()
    create_cost_fn()
    create_balanced_network()
    create_antibalanced_network()

    trials = draw_trials()

    pass
