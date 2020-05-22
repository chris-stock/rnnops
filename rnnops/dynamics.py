"""
Methods to implement RNN dynamics, both in firing rates and weights.
"""
__all__ = [
    'run_neural_dynamics',
    'run_weight_dynamics',
]

from . import RNN, Trial
import numpy as np
from tqdm import tqdm


def run_neural_dynamics(
        rnn: RNN,
        trial: Trial,
        x0=None,
        noise_std=None,
):
    """
    Run neural dynamics via forward Euler-Maruyama integration.
    rnn: the weight configuration to use for the simulation
    inputs: (trial_len x n_in x [...]) array of inputs to network
    tt: (trial_len, ) array marking time points to simulate
    update_fns: list of m update functions (J -> dJdt)
    update_rates: list of m learning rates to scale the update functions
    x0: initial condition of firing rates. Set to the origin by default.
    noise_std: std deviation of additive Gaussian white noise, if not None
    # Todo: use einsum for array multiplication
    # Todo: implement this with jax.lax.scan() and jax pseudorandomness
    """
    # set default value of x0 based on shape of inputs
    if x0 is None:
        x0 = np.zeros(trial.shape(t=0, n=rnn.n_rec))
    xx = [x0]  # initialize neural state

    # iterate through inputs
    for _, u, _ in tqdm(trial):
        x = xx[-1]  # most recent value of x
        dxdt = _calc_dxdt(x, u, rnn)  # time derivative of x
        if noise_std is None:
            noise = 0.
        else:
            noise = noise_std * np.random.randn(*dxdt.shape)
        x_ = x + trial.dt * (dxdt + noise)  # forward Euler update step
        xx.append(x_)
    xx = xx[:len(trial.tt)]  # trim to trial length
    yy = [_adot(rnn.w_out, x) for x in xx]  # calculate network outputs
    return {'x': np.array(xx), 'y': np.array(yy)}


def _adot(
        a: np.ndarray,
        b: np.ndarray,
):
    """
    Accordioning np.dot. Dot multiplication between last axis of first
    argument and first axis of second argument.
    a:  ndarray of shape (*shape_1, n)
    b:  ndarray of shape (n, *shape_2)
    output: ndarray of shape  (*shape_1, *shape_2)
    """
    return np.tensordot(a, b, axes=(-1, 0))


def _calc_dxdt(
        x: np.ndarray,
        u: np.ndarray,
        rnn: RNN,
):
    """
    Compute the time derivative of neural firing rates.
    x: array with shape (n_rec, [...])
    u: array with shape (n_in, [...])
    rnn: the network configuration
    """
    a = _adot(rnn.w_rec, rnn.phi(x)) + _adot(rnn.w_in, u)
    b = (np.ones(
        [1] * a.ndim) * rnn.b).T  # make sure to broadcast biases along axis 0
    return a + b


def run_weight_dynamics(
        rnn,
        update_fns,
        update_rates,
        dt=0.05,
        save_weights=False,
        tol=None,
        t_final=None,
):
    """
    Run weight dynamics using Euler-Maruyama integration.
    rnn is the the initial condition
    update_fns is a list of m update functions (J -> dJdt)
    update_rates is a list of m learning rates to scale the update functions
    dt is the step size to take with Euler-Maruyama integration
    if save_weights is True, return a list of the trajectory taken
    if tol is given, stop when average change in weight converges to tol
    if t_final is given, do not run past time t_final
    # Todo: implement this with jax.lax.scan()
    """
    pass
