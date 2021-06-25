"""
Functionality for specifying and initializing RNNs, running neural dynamics,
and running weight dynamics of a recurrent network
"""
__all__ = [
    'nonlinearities',
    'nonlinearity_derivs',
    'RNN',
    'initialize_rnn',
    'update_rnn',
    'iid_gaussian_init',
    'mvn_rows_init',
    'zeros_init',
    'run_weight_dynamics'
]

import numpy as np
from copy import deepcopy

nonlinearities = {
    'linear': lambda x: x,
    'relu': lambda x: np.maximum(x, 0.),
    'tanh': lambda x: np.tanh(x),
}

nonlinearity_derivs = {
    'linear': lambda x: 1.,
    'relu': lambda x: x > 0.,
    'tanh': lambda x: 1 - np.tanh(x) ** 2,
}


class RNN(object):
    """
    Data type containing RNN weights, biases, and nonlinearity.
    """

    def __init__(
            self,
            w_in,
            w_rec,
            w_out,
            b,
            nonlinearity: str,
            rank=None,
            check_dims: bool = True,
            name: str = '',
    ):
        """
        Initialize an rnn with weights and nonlinearity.
        w_in, w_rec, w_out, b must be castable to numpy arrays.
        nonlinearity must be among supported nonlinearities.
        rank is the rank of the recurrent weight matrix.
        if check_dims is True, check that the dimensions of the weight
        matrices agree
        """

        # if w_rec_factors is provided, make sure it agrees with w_rec
        # U, V = w_rec_factors
        # if w_rec is None:
        #     try:
        #         w_rec = U.dot(V.T)
        #     except:
        #         raise ValueError(
        #             'either w_rec or w_rec_factors must be provided')
        # elif w_rec_factors is not None and U is not None and V is not None:
        #     try:
        #         assert (w_rec == U.dot(V.T)).all()
        #     except AssertionError:
        #         raise ValueError('w_rec and w_rec_factors do not agree.')

        self.name = name
        self._params = {
            'w_in': w_in,
            'w_rec': w_rec,
            'w_out': w_out,
            'b': b,
            'rank': rank,
            # 'w_rec_factors': tuple(w_rec_factors)
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

    def __str__(self):
        s = 'RNN object \n' + \
            ' signature: {}\n'.format(self.signature) + \
            ' n_rec: {}\n'.format(self.n_rec) + \
            ' nonlinearity: {}\n'.format(self.nonlinearity) + \
            ' name: {}'.format(self.name)
        return s

    @property
    def params(self):
        return self._params

    @property
    def w_in(self):
        """
        Exposes the input weight matrix as an ndarray.
        """
        return np.array(self.params['w_in'])

    @property
    def w_rec(self):
        """
        Exposes the recurrent weight matrix as an ndarray.
        """
        return np.array(self.params['w_rec'])

    @property
    def w_out(self):
        """
        Exposes the output weight matrix as an ndarray.
        """
        return np.array(self.params['w_out'])

    @property
    def b(self):
        """
        Exposes the bias as an ndarray.
        """
        return np.array(self.params['b'])

    @property
    def rank(self):
        """
        Exposes the rank of the recurrent weight matrix as an int.
        """
        try:
            return int(self.params['rank'])
        except TypeError:
            return None

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
    def factors(self):
        fs = {}
        for (k, w) in self.params.items():
            try:
                f = w.factors
            except AttributeError:
                f = None
            fs[k] = f
        return fs

    @property
    def signature(self):
        """
        Return the input and output dimensions of the function computed
        """
        return self.n_in, self.n_out


"""
Create and update weights.
"""


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
            init = zeros_init()  # initialize biases at zero
        else:
            init = iid_gaussian_init()  # initialize weights as iid gaussian
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
    Given an existing RNN instance and new parameter values, create a new
    RNN instance with updated parameters.
    rnn: instance of RNN class
    update_args: dict with new values to update
    """
    args = deepcopy(rnn.params)
    args.update({
        'nonlinearity': rnn.nonlinearity
    })
    args.update(update_args)
    return RNN(**args)


def iid_gaussian_init(std=1., mean=0., normalize_by=None):
    """
    randomly sample an ndarray with iid Gaussian elements
    normalize_by : scale output by 1/np.sqrt(normalize_by). default: n
    todo: change to use jax pseudorandomness
    """

    def init(shape):
        n = normalize_by if normalize_by is not None else shape[0]
        return std * np.random.randn(*shape) / np.sqrt(n) + mean

    return init


def mvn_rows_init(mean, cov):
    """
    draw an (m,n) matrix whose rows are independent samples from multivariate
    Gaussians with different means and a shared covariance matrix
    mean: (m,n) array
    cov: (n,n) symmetric PSD covariance matrix
    todo: change to use jax pseudorandomness
    """

    def init(shape):
        m, n = shape
        return mean + np.random.multivariate_normal(np.zeros(n), cov, m)

    return init


def zeros_init():
    """
    initialize a matrix with all zero elements
    """

    def init(shape):
        return np.zeros(shape)

    return init


"""
Update weights according to dynamical rules.
"""


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
