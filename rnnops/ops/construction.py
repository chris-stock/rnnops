"""
Methods to construct recurrent networks that perform specific tasks.
"""
__all__ = [
    'pseudoinverse_rule',
    'construct_boolean_integration_rnn',
    'construct_xor_rnn'
]
import numpy as np
from scipy.linalg import lstsq

from rnnops import RNN
from rnnops.weights import nonlinearities, iid_gaussian_init, zeros_init
from rnnops.tasks import XOR_CONDITIONS, CDI_CONDITIONS
from rnnops.tasks import expand_condition_variables
from rnnops.factored_matrix import FactoredMatrix


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
    from scipy.linalg import lstsq
    from sklearn.utils.extmath import randomized_svd

    # set up problem
    phi = nonlinearities[nonlinearity]
    A = phi(x_fixed)
    B = x_fixed - u_fixed

    # solve J * A = B and determine rank of resulting matrix
    J = lstsq(A.T, B.T)[0].T
    k = np.minimum(np.linalg.matrix_rank(A), np.linalg.matrix_rank(B))

    # get low-rank factorization
    U, Sigma, VT = randomized_svd(J, k)
    U = U * np.sqrt(Sigma[None, :])
    V = VT.T * np.sqrt(Sigma[None, :])
    # J = U.dot(V.T)
    J = FactoredMatrix(U, V)

    # A_pinv = np.linalg.pinv(A)
    # factors = (B, A_pinv.T)
    # J = B.dot(A_pinv)

    return J, (U, V)


def construct_boolean_integration_rnn(
        conditions,
        n_rec,
        nonlinearity,
        with_bias=False,
        expand_inputs=False,
        expand_targets=False,
        trim_zeros = False,
):
    """
    from a specified boolean function, use the pseudoinverse rule
    to construct an RNN that calculates that function
    """

    # process conditions
    conditions = expand_condition_variables(
        conditions, expand_inputs, expand_targets
    )

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

    # initialize bias
    b = init((n_rec,)) if with_bias else zeros((n_rec,))

    # create matrices U and Y (n_neurons x num_conditions)
    U = w_in.dot(np.array(inputs).T) + b[:, None]  # (n_rec x num_conditions)
    Y = np.array(targets).T  # (n_out x num_conditions)

    # solve for input-adjusted fixed points B in w_out * B = Y
    B = lstsq(w_out, Y)[0]

    # solve for fixed points X in B = X - U
    X = B + U

    # construct recurrent matrix via pseudoinverse rule
    w_rec, factors = pseudoinverse_rule(X, U, nonlinearity)

    # trim zeros from matrix by removing dead-end neurons
    if trim_zeros:
        is_fully_connected = np.sum(np.abs(factors[1]), axis=1) > 0.
        w_in = w_in[is_fully_connected, :]
        w_rec = np.array(w_rec)[:, is_fully_connected][is_fully_connected, :]
        b = b[is_fully_connected]

        # rescale w_out to keep the norm of the output weights the same
        new_w_out = w_out[:, is_fully_connected]
        w_out_norm = np.linalg.norm(w_out, axis=1, keepdims=True)
        new_w_out_norm = np.linalg.norm(new_w_out, axis=1, keepdims=True)

    # return rnn object
    rnn_args = {
        'w_in': w_in,
        'w_rec': w_rec,
        'w_out': w_out,
        'b': b,
        'nonlinearity': nonlinearity,
        'rank': factors[0].shape[1]
    }
    return RNN(**rnn_args)


def construct_xor_rnn(*args, **kwargs):
    """
    construct an RNN that performs the XOR task
    """
    return construct_boolean_integration_rnn(XOR_CONDITIONS, *args, **kwargs)


def construct_cdi_rnn(*args, **kwargs):
    """
    construct an RNN that performs the XOR task
    """
    return construct_boolean_integration_rnn(CDI_CONDITIONS, *args, **kwargs)


def construct_fsm_rnn(args):
    """
    construct an RNN that performs a finite state computation task
    """
    pass