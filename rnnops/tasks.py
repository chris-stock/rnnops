"""
tasks.py
Functions which produce trials of specific tasks.
"""
import numpy as np
from .trial import Trial
__all__ = [
    'boolean_integration_task',
    'xor_integration_task',
    'expand_condition_inputs'
]


"""
Boolean integration tasks. Includes context dependent integration.
"""
XOR_conditions = [
    ((0, 0), (0,)),
    ((0, 1), (1,)),
    ((1, 0), (1,)),
    ((1, 1), (0,))
]


def boolean_integration_task(
        conditions,
        num_trials: int = 1,
        trial_len: float = 1.,
        dt: float = 0.05,
        noise_std: float = None,
        name: float = 'boolean_integration',
        expand_inputs: bool = False,
        **trial_args
):
    """
    Generate trials from a Boolean function. Inputs and targets are constant,
    taking values in {0, 1}. For example, XOR would have two input signals,
    one output, and four conditions.

    arguments:
    conditions: list of (input, output), where input and output
    are each tuples of boolean values of len n_in and n_out respectively.
    For example, for XOR this would be
        [ ((0, 0), (0,)),
          ((0, 1), (1,)),
          ((1, 0), (1,)),
          ((1, 1), (0,)) ]

    num_trials: number of independent trials to generate
    noise_std: standard deviation of Gaussian white noise added to inputs,
    if not None

    returns a Trial instance
    """

    # process conditions
    if expand_inputs:
        conditions = expand_condition_inputs(conditions)

    inputs, targets = zip(*conditions)
    if len(inputs) != len(set(inputs)):
        raise ValueError('Inputs in conditions must be unique')

    # create time and num_trial axes
    tt = np.arange(trial_len, step=dt)
    time_axis = np.zeros((len(tt), 1, 1, 1))
    trial_axis = np.zeros((1, 1, 1, num_trials))

    def prepare_data(cond, noise=None):
        cond = np.array(cond).T  # make (n_neurons x conditions)
        cond_with_axes = trial_axis + time_axis + cond[None, :, :, None]
        if noise is None:
            noise = 0.
        else:
            noise = dt * noise * np.random.randn(*cond_with_axes.shape)
        return cond_with_axes + noise

    # assemble trial arguments
    trial_args.update({
        'inputs': prepare_data(inputs, noise_std),
        'targets': prepare_data(targets),
        'trial_len': trial_len,
        'dt': dt,
        'name': name,
    })
    return Trial(**trial_args)


def xor_integration_task(*args, **kwargs):
    """
    Generates trials of XOR function.
    """
    return boolean_integration_task(XOR_conditions, *args, **kwargs)


def expand_condition_inputs(conditions):
    """
    Given conditions, expands inputs such that for each original input i,
    a new input of NOT i is added. This is equivalent to adding a
    linear layer of neurons (with biases) in between the input and recurrent
    layer.
    """

    # flip inputs
    inputs, targets = zip(*conditions)
    flipped_inputs = list(tuple(1 - i for i in ii) for ii in inputs)

    # interleave original input and flipped inputs
    expanded_inputs = []
    for ii in zip(inputs, flipped_inputs):
        expanded_inputs.append(tuple(j for jj in zip(*ii) for j in jj))

    return list(zip(expanded_inputs, targets))
