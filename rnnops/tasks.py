"""
tasks.py
Functions which produce trials of specific tasks.
"""
import numpy as np
from .trial import Trial
__all__ = [
    'boolean_integration_task',
    'xor_integration_task',
    'expand_condition_variables'
]


"""
Boolean integration tasks. Includes context dependent integration.
"""


XOR_CONDITIONS = [
    ((0, 0), (0,)),
    ((0, 1), (1,)),
    ((1, 0), (1,)),
    ((1, 1), (0,))
]


CDI_CONDITIONS = [
    ((0, 0, 0), (0,)),
    ((0, 0, 1), (0,)),
    ((0, 1, 0), (1,)),
    ((0, 1, 1), (1,)),
    ((1, 0, 0), (0,)),
    ((1, 0, 1), (1,)),
    ((1, 1, 0), (0,)),
    ((1, 1, 1), (1,))
]


def task_from_functions(
        input_function,
        target_function,
        trial_len: float = 1,
        num_trials: int = 1,
        dt: float = 0.05,
        name: str = '',
):
    """
    Construct a trial with functions of time determining inputs and targets
    from
    :param input_function: an n_in-valued function of time
    :param target_function: an n_out-valued function of time
    :param trial_args:
    :return:
    """
    tt = np.arange(trial_len, step=dt)

    trial_args = {
        'trial_len': trial_len,
        'dt': dt,
        'name': name,
        'inputs': np.array([input_function(t) for t in tt]),
        'targets': np.array([target_function(t) for t in tt])
    }
    return Trial(**trial_args)


def boolean_integration_task(
        conditions,
        num_trials: int = 1,
        trial_len: float = 1.,
        dt: float = 0.05,
        noise_std: float = None,
        name: float = 'boolean_integration',
        expand_inputs: bool = False,
        expand_targets: bool = False,
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
    conditions = expand_condition_variables(
        conditions, expand_inputs, expand_targets
    )

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
    return boolean_integration_task(XOR_CONDITIONS, *args, **kwargs)


def cdi_integration_task(*args, **kwargs):
    """
    Generates trials of CDI function.
    """
    return boolean_integration_task(XOR_CONDITIONS, *args, **kwargs)


def expand_condition_variables(
        conditions,
        expand_inputs=False,
        expand_outputs=False
):
    """
    Given conditions, expands inputs such that for each original input i,
    a new input of NOT i is added. This is equivalent to adding a
    linear layer of neurons (with biases) in between the input and recurrent
    layer.
    """

    def expand_variable(var):
        # interleave original input and flipped inputs
        flipped_var = list(tuple(1 - i for i in ii) for ii in var)
        expanded_var = []
        for ii in zip(var, flipped_var):
            expanded_var.append(tuple(j for jj in zip(*ii) for j in jj))
        return expanded_var

    inputs, targets = zip(*conditions)
    new_inputs = expand_variable(inputs) if expand_inputs else inputs
    new_targets = expand_variable(targets) if expand_outputs else targets

    return list(zip(new_inputs, new_targets))


"""
Various loss functions for Boolean outputs.
"""


def l2_loss(trial):
    #  average l2 distance of network from target during trial
    return np.mean((trial.outputs - trial.targets)**2)


def exp_loss(trial):
    #  exponentiated average  l2 distance of network from target
    return np.exp(-l2_loss(trial))


def logistic_loss(trial):
    # logistic loss averaged over neurons
    # clip outputs to (0,1)
    eps = 0.01
    outputs = np.minimum(np.maximum(trial.outputs, 0.+eps), 1.-eps)

    logistic_loss = trial.targets * np.log(outputs) + \
        (1 - trial.targets) * np.log(1 - outputs)

    return np.mean(logistic_loss)
