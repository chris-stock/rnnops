"""
tasks.py
Functions which produce trials of specific tasks.
"""
import numpy as np
from .trial import Trial
__all__ = [
    'boolean_integration_task',
    'xor_integration_task',
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
    inputs, targets = zip(*conditions)
    if len(inputs) != len(set(inputs)):
        raise ValueError('Inputs in conditions must be unique')

    # create time and num_trial axes
    tt = np.arange(trial_len, step=dt)
    time_axis = np.zeros((len(tt), 1, 1, 1))
    trial_axis = np.zeros((1, 1, 1, num_trials))

    # create inputs and targets
    trial_data = []
    for cond in zip(*conditions):
        cond = np.array(cond).T  # make (n_neurons x conditions)
        condition_axes = trial_axis + time_axis + cond[None, :, :, None]
        if noise_std is None:
            noise = 0.
        else:
            noise = dt * noise_std * np.random.randn(*condition_axes.shape)
        trial_data.append(condition_axes + noise)

    # assemble trial arguments
    trial_args.update({
        'inputs': trial_data[0],
        'targets': trial_data[1],
        'trial_len': trial_len,
        'dt': dt,
        'name': name,
    })
    return Trial(**trial_args)


def xor_integration_task(*args, **kwargs):
    kwargs['name'] = 'XOR_integration'
    return boolean_integration_task(XOR_conditions, *args, **kwargs)