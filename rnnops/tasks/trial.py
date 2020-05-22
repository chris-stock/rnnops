"""
trial.py
Base data type for trials
"""
import numpy as np


class Trial(object):
    """
    Data type containing trial clock, inputs, and target.
    """

    def __init__(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            dt: float,
            trial_len: float,
            name: str = '',
    ):
        """
        Initialize a Trial class.
        arguments:
        dt: step size of simulation
        trial_len: length of simulation
        inputs: ndarray shape (T, n_in, [...])
        outputs: ndarray shape (T, n_out, [...])
        task_name: string (optional)
        """

        self.dt = dt
        self.trial_len = trial_len
        self.tt = list(np.arange(self.trial_len, step=self.dt))
        self.name = name

        # check that inputs and outputs match on important dimensions
        try:
            assert (inputs.shape[0] == len(self.tt))
        except:
            raise ValueError('time axis of inputs does not match tt')
        try:
            assert (targets.shape[0] == len(self.tt))
        except:
            raise ValueError('time axis of outputs does not match tt')
        try:
            assert (list(targets.shape[2:]) == list(targets.shape)[2:])
        except:
            raise ValueError('additional axes of inputs and targets do not '
                             'match')
        self.inputs = inputs
        self.targets = targets

    def __iter__(self):
        """
        Define the iterator such that next(trial)
        iterates through time steps and returns a tuple of
        (t, input, target)
        """
        return iter(zip(self.tt, self.inputs, self.targets))

    def __next__(self):
        return next(self)

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
              t: int = None,
              n: int = None,
              ):
        """
        Create a shape matching the specified dimensions.
        t: dimension of time axis. If zero, drop axis in result.
        n: dimension of neron axis. If zero, drop axis in result.
        Default: return the shape of target array.
        """
        shape = list(np.array(self.targets).shape)
        # fill in with given dimensions
        if t is not None:
            shape[0] = t
        if n is not None:
            shape[1] = n
        shape = [_ for _ in shape if _ != 0]  # drop any axes that are zero
        return tuple(shape)

    @property
    def n_in(self):
        # return self.inputs.shape[1]
        return self.inputs[0].shape[0]

    @property
    def n_out(self):
        # return self.targets.shape[1]
        return self.targets[0].shape[0]

    @property
    def signature(self):
        return self.n_in, self.n_out
