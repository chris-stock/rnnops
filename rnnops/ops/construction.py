"""
Methods to construct recurrent networks that perform specific tasks. Todo.
"""
__all__ = [
    'pseudoinverse_rule',
    'construct_boolean_integration_rnn',
    'construct_xor_rnn'
]
import numpy as np
from itertools import product
from scipy.linalg import lstsq
from rnnops.weights import nonlinearities, iid_gaussian_init, zeros_init
from rnnops.tasks import XOR_conditions, expand_condition_inputs
from rnnops import RNN


def pseudoinverse_rule(
        x_fixed: np.ndarray,
        u_fixed: np.ndarray,
        nonlinearity: str,
):
    """
    from a specified list of fixed points, use the pseudoinverse rule
    to construct an RNN that matches those fixed points under given inputs.
    In particular, given points in the columns of x_fixed, and nonlinearity
    phi, find the smallest-norm J such that J phi(x_fixed) + u = x_out.

    x_fixed: (n, p) ndarray
    u_fixed: (n, p) ndarray
    nonlinearity: string
    """
    # set up problem
    phi = nonlinearities[nonlinearity]
    A = phi(x_fixed)
    B = x_fixed - u_fixed

    # solve J * A = B
    J = lstsq(A.T, B.T)[0].T
    return J


def construct_boolean_integration_rnn(
        conditions,
        n_rec,
        nonlinearity,
        with_bias=False,
        expand_inputs=False,
):
    """
    from a specified boolean function, use the pseudoinverse rule
    to construct an RNN that calculates that function
    """

    # process conditions
    if expand_inputs:
        conditions = expand_condition_inputs(conditions)
    inputs, targets = zip(*conditions)
    if len(inputs) != len(set(inputs)):
        raise ValueError('Inputs in conditions must be unique')

    n_in = len(inputs[0])
    n_out = len(targets[0])

    # draw (random) input and output matrices
    init = iid_gaussian_init(normalize_by=n_rec)
    zeros = zeros_init()
    w_in = init((n_rec, n_in))
    w_out = init((n_out, n_rec))

    # draw bias, if desired
    b = init((n_rec,)) if with_bias else zeros((n_rec,))

    # create matrices U and Y (n_neurons x num_conditions)
    U = w_in.dot(np.array(inputs).T) + b[:, None]  # (n_rec x num_conditions)
    Y = np.array(targets).T  # (n_out x num_conditions)

    # solve for input-adjusted fixed points B in w_out * B = Y
    B = lstsq(w_out, Y)[0]

    # solve for fixed points X in B = X - U
    X = B + U

    # construct recurrent matrix via pseudoinverse rule
    w_rec = pseudoinverse_rule(X, U, nonlinearity)

    # return rnn object
    rnn_args = {
        'w_in': w_in,
        'w_rec': w_rec,
        'w_out': w_out,
        'b': b,
        'nonlinearity': nonlinearity
    }
    return RNN(**rnn_args)


def construct_xor_rnn(*args, **kwargs):
    """
    construct an RNN that performs the XOR task
    """
    return construct_boolean_integration_rnn(XOR_conditions, *args, **kwargs)


def construct_fsm_rnn(args):
    """
    construct an RNN that performs a finite state computation task
    """
    pass