"""
Functionality for specifying and initializing RNNs,
running neural dynamics, and running
weight dynamics of a recurrent network
"""
__all__ = [
    'RNN',
    'initialize_rnn',
    'update_rnn',
]


import numpy as np
from copy import deepcopy
from .components import nonlinearities, zeros_init, iid_gaussian_init


class RNN(object):
    """
    A container for RNN weights, biases, and nonlinearity.
    This class is "read-only:" new set of weights gets a new instance.
    """

    def __init__(
            self,
            w_in,
            w_rec,
            w_out,
            b,
            nonlinearity,
            check_dims=True,
    ):

        self._params = {
            'w_in': np.array(w_in, dtype=float),
            'w_rec': np.array(w_rec, dtype=float),
            'w_out': np.array(w_out, dtype=float),
            'b': np.array(b, dtype=float),
        }

        # check nonlinearity
        self.nonlinearity = nonlinearity
        try:
            self.phi = nonlinearities[nonlinearity]
        except:
            raise ValueError(
                "Nonlinearity must be one of {}".format(nonlinearities.keys())
            )

        # check weights and biases for dimension compatibility
        if check_dims:
            try:
                assert (self.w_in.ndim == 2)
                _ = self.w_rec @ self.w_in
            except:
                raise ValueError("w_in and w_rec are not compatible.")
            try:
                assert (self.w_rec.ndim == 2)
                _ = self.w_rec @ self.w_rec
            except:
                raise ValueError("w_rec is not square")
            try:
                assert (self.w_out.ndim == 2)
                _ = self.w_out @ self.w_rec
            except:
                raise ValueError("w_rec and w_out are not compatible")
            try:
                assert (self.b.ndim == 1)
                _ = self.w_rec + self.b[:, None]
            except:
                raise ValueError("w_rec and b are not compatible")

    @property
    def params(self):
        return self._params

    @property
    def w_in(self):
        return self.params['w_in']

    @property
    def w_rec(self):
        return self.params['w_rec']

    @property
    def w_out(self):
        return self.params['w_out']

    @property
    def b(self):
        return self.params['b']

    @property
    def n_in(self):
        return self.w_in.shape[1]

    @property
    def n_rec(self):
        return self.w_rec.shape[0]

    @property
    def n_out(self):
        return self.w_out.shape[0]

    @property
    def signature(self):
        """
        Return the input and output dimensions of the function computed
        """
        return self.n_in, self.n_out


def initialize_rnn(
        n_rec,
        signature=None,
        n_in=None,
        n_out=None,
        nonlinearity=None,
        initializers=None,
        check_dims=True,
    ):
    """
    initialize an RNN according to the parameters specified.
    """

    if initializers is None:
        initializers = {}
    if signature is not None:
        n_in, n_out = signature
    elif n_in is None or n_out is None:
        raise ValueError(
            'signature must be provided if either n_in or n_out is None'
        )
    if nonlinearity is None:
        nonlinearity = 'linear'

    # set parameter shapes
    param_shapes = {
        'w_in': (n_rec, n_in),
        'w_rec': (n_rec, n_rec),
        'w_out': (n_out, n_rec),
        'b': (n_rec,),
    }

    # specify initializers
    default_initializers = {}
    for k in param_shapes.keys():
        if k[0] == 'b':
            init = zeros_init() # initialize biases at zero
        else:
            init = iid_gaussian_init() # initialize weights as iid gaussian
        default_initializers[k] = init
    if initializers is None:
        initializers = {}
    default_initializers.update(initializers)

    # build arguments to RNN class
    rnn_args = {
        'nonlinearity': nonlinearity,
        'check_dims': check_dims,
    }

    # initialize
    for k, init in default_initializers.items():
        rnn_args[k] = init(param_shapes[k])

    return RNN(**rnn_args)


def update_rnn(rnn: RNN, **update_args):
    """
    Given an existing rnn instance and new parameter values, create a new
    rnn instance with updated parameters.
    rnn: instance of RNN class
    update_args: dict with new values to update
    """
    params = deepcopy(rnn.params)
    for k in params.keys():
        if k in update_args.keys():
            params[k] = update_args[k]
    return RNN(nonlinearity=rnn.nonlinearity, **params)

