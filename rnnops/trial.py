"""
trial.py
Methods to construct and simulate trials.
"""

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from . import RNN

__all__ = [
    'Trial',
    'run_neural_dynamics'
]


class Trial(object):
    """
    Data type containing trial clock, inputs and target.
    """

    def __init__(
            self,
            trial_len: float = 1.,
            dt: float = 0.05,
            name: str = '',
            **data,
    ):
        """
        arguments:
        trial_len: length of trial, in units of time
        dt: simulation step size, in units of time
        inputs: ndarray shape (T, n_in, [...])
        targets: ndarray shape (T, n_out, [...])
        hiddens: ndarray shape (T, n_rec, [...])
        outputs: ndarray shape (T, n_out, [...])
        task_name: string (optional)
        """

        self.dt = dt
        self.trial_len = trial_len
        self.tt = np.arange(self.trial_len, step=self.dt)
        self.name = name

        # check that data shapes agree except on axis 1 (#neurons)
        data_dims = list(zip(*[d.shape for d in data.values()]))
        ignore_axes = [1]
        for i, dims in enumerate(data_dims):
            if i not in ignore_axes:
                try:
                    assert all([d == dims[0] for d in dims])
                except:
                    raise ValueError("not all dimensions match along axis "
                                     "{}".format(i))

        # check that data time axis agrees with clock
        try:
            assert (list(data_dims)[0][0] == len(self.tt))
        except:
            raise ValueError('clock and time axis of data do not not agree')

        # define template shape
        _data_shape = [dims[0] for dims in data_dims]
        _data_shape[1] = 0
        self._data_shape = tuple(_data_shape)

        # create template data and update with provided arguments
        data_keys = ['inputs', 'targets', 'hiddens', 'outputs']
        data_dict = {k: np.zeros(self._data_shape) for k in data_keys}
        data_dict.update({k: v for (k, v) in data.items() if k in data_keys})
        self.data = data_dict

        # create state variables for each data point
        self.inputs = self.data['inputs']
        self.targets = self.data['targets']
        self.hiddens = self.data['hiddens']
        self.outputs = self.data['outputs']

    def __iter__(self):
        """
        Define  such that next(trial) steps through time and
        returns a tuple of (t, input, target)
        """
        return iter(zip(
            self.tt,
            self.inputs,
            self.targets,
            self.hiddens,
            self.outputs,
        ))

    def __next__(self):
        return next(self)

    def __len__(self):
        return len(self.tt)

    def __str__(self):
        if self.name:
            name = '\n name: {}'.format(self.name)
        else:
            name = ''
        s = 'Trial object ' + \
            '\n signature: {}'.format(self.signature) + \
            '\n trial_len: {}'.format(self.trial_len) + \
            '\n dt: {}'.format(self.dt) + \
            name
        return s

    def shape(self,
              n: int,
              t: int = None,
              squeeze: bool = True,
              ):
        """
        Create a shape matching the specified dimensions.
        n: dimension of neron axis.
        t: dimension of time axis. If not given, default to _data_shape.
        squeeze: if True, drop any axes with dimension zero.
        """
        # start with base shape
        shape = list(self._data_shape)

        # fill in with given dimensions
        shape[1] = n
        if t is not None:
            shape[0] = t
        if squeeze:
            shape = [_ for _ in shape if _ != 0]  # drop any axes that are zero
        return tuple(shape)

    @property
    def n_in(self):
        return self.inputs.shape[1]

    @property
    def n_out(self):
        return max(self.targets.shape[1], self.outputs.shape[1])

    @property
    def n_rec(self):
        return self.hiddens.shape[1]

    @property
    def num_conditions(self):
        return self.hiddens.shape[2]

    @property
    def num_trials(self):
        return self.hiddens.shape[3]

    @property
    def signature(self):
        return self.n_in, self.n_out


def update_trial(trial: Trial, **data):
    """
    Given an existing Trial instance and new data, create a new Trial
    instance with updated data.
    trial: instance of original Trial class
    data: dict with new values to update
    """
    args = deepcopy(trial.data)  # start with arguments from original trial
    args.update({
        'dt': trial.dt,
        'trial_len': trial.trial_len,
        'name': trial.name
    })
    args.update(data)  # update arguments with the new ones
    return Trial(**args)


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
        x0 = np.zeros(trial.shape(n=rnn.n_rec, t=0))
    xx = [x0]  # initialize neural state

    # iterate through inputs
    for u in tqdm(trial.inputs):
        x = xx[-1]  # most recent value of x
        dxdt = _calc_dxdt(x, u, rnn)  # time derivative of x
        if noise_std is None:
            noise = 0.
        else:
            noise = noise_std * np.random.randn(*dxdt.shape)
        x_ = x + trial.dt * (dxdt + noise)  # forward Euler update step
        xx.append(x_)
    xx = xx[:len(trial)]  # trim to trial length
    yy = [_adot(rnn.w_out, x) for x in xx]  # calculate network outputs

    response_data = {
        'hiddens': np.array(xx),
        'outputs': np.array(yy),
    }
    return update_trial(trial, **response_data)


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
    a = -x + _adot(rnn.w_rec, rnn.phi(x)) + _adot(rnn.w_in, u)
    # make sure to broadcast biases along axis 0
    b = (np.ones([1] * a.ndim) * rnn.b).T
    return a + b
