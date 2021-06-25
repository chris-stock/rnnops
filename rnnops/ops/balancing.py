"""
Methods to implement local synaptic balancing, a form of Lax dynamics on the
recurrent weight matrix of the RNN.
"""

from copy import deepcopy

import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.integrate import solve_ivp
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from rnnops import RNN, Trial
from rnnops.weights import update_rnn, nonlinearity_derivs
from rnnops.factored_matrix import FactoredMatrix


class CostFunction(object):
    """
    a container for the weighted power law cost matrix with power p
    and weights alpha_ij:
    c_ij = alpha_ij |J_ij|^p
    """

    def __init__(
            self,
            p,
            alpha,
            rank=None,
    ):
        """
        arguments:
        p: exponent, float.
        alpha: weights on synaptic costs, ndarray of floats.
        alpha_factors: low rank factorization (A, B) such that alpha = A * B.T
        """

        # catch rank-one cases
        if isinstance(alpha, float) or isinstance(alpha, int):
            rank = 1
        elif isinstance(alpha, np.ndarray) and alpha.ndim < 2:
            rank = 1

        self._params = {
            'p': p,
            'alpha': alpha,
            'rank': rank,
        }

        try:
            assert (self.alpha >= 0).all()
        except:
            raise ValueError('cost weights must be non-negative')

    @property
    def params(self):
        return self._params

    @property
    def p(self):
        """
        Exposes the cost exponent as a float.
        """
        return float(self.params['p'])

    @property
    def rank(self):
        """
        Exposes the rank of the cost function weights as an int.
        """
        try:
            return int(self.params['rank'])
        except TypeError:
            return None

    @property
    def alpha(self):
        """
        Exposes the cost weights alpha as an ndarray.
        """
        return np.array(self.params['alpha'], dtype=float, ndmin=2)

    def matrix(self, J):
        """
        Return the cost matrix given connectivity J.
        arguments:
        J: ndarray.
        """
        return eval_cost(self, J)

    def total(self, J):
        """
        Return the total cost (sum of elements of cost matrix) given
        connectivity J.
        J: ndarray
        """
        return np.sum(self.matrix(J))

    def states(self, J):
        """
        Return the somatic state vector given connectivity J.
        arguments:
        J: ndarray.
        """
        return calc_somatic_states(self.matrix(J))

    def lax_update(self, J):
        """
        Return the time derivative of the Lax dynamics given connectivity J.
        J: ndarray.
        """
        return calc_lax_update(J, self.states)


"""
Methods for calculating the matrix of costs under different assumptions.
These functions generate instances of CostFunction.
"""


def eval_cost(cost_fn, J):
    """
    Return the cost matrix given cost function C and connectivity J.
    c_ij = alpha_ij |J_ij|^p
    """
    return cost_fn.alpha * (np.abs(J) ** cost_fn.p)


def calc_gains(
        trial: Trial,
        nonlinearity: str,
        weight_by_velocity: bool
):
    """
    Calculate the average gain of a neuron across time, trials and conditions.
    """
    x = trial.hiddens

    # check inputs
    if x.size == 0:
        raise ValueError('Please provide a trial with hidden activity from '
                         'which to calculate gains')

    # averaging over all but axis 1
    dims = list(range(trial.ndim))
    dims.pop(1)

    # calculate instantaneous gains
    gain = nonlinearity_derivs[nonlinearity](x)

    # calculate velocity of neural activity
    if weight_by_velocity:
        v = np.repeat(
            np.linalg.norm(x[1:] - x[:-1], axis=1, keepdims=True),
            gain.shape[1],
            axis=1
        )
    else:
        v = None

    # calculate neural gains on trial
    gains = np.average(
        gain[:-1],
        axis=tuple(dims),
        weights=v
    )

    return gains


def robustness_cost_fn(
        trial: Trial,
        nonlinearity: str,
        weight_by_velocity=False,
):
    """
    calculate the robustness cost matrix.
    c_ij = r_j |J_ij|^2 where r_j is average gain of neuron j
    """

    # calculate neural gains on trial
    gains = calc_gains(
        trial,
        nonlinearity,
        weight_by_velocity
    )
    alpha = gains[None, :]

    # use the weighted power law cost function with gains as weights
    return CostFunction(p=2., alpha=alpha, rank=1)


def frobenius_cost_fn():
    """
    calculate the squared Frobenius norm cost matrix
    c_ij = |J_ij|^2
    """
    return CostFunction(p=2., alpha=np.array(1.), rank=1)


def power_law_cost_fn(p):
    """
    calculate the power law cost matrix with power p:
    c_ij = |J_ij|^p
    """
    return CostFunction(p=p, alpha=np.array(1.), rank=1)


"""
Methods to simulate a Lax dynamical system  
"""


def run_lax_dynamics_odeint(
        J0,
        g_fn,
        tau=1.,
        T_max=1.,
        factors=None,
        terminal_fn=None,
        t_eval=None,
        **solve_args
):
    """
    Solve a Lax dynamical system starting at J0, via scipy's Runge-Kutta solver
    arguments:
    J0 is initial N x N matrix
    g_fn: function taking J to g
    factors: (U0, V0) where J0 = U0.dot(V0.T). Assumed to be same shape
    tau: timescale of dynamics (inverse of "learning rate")
    dt: Euler step size
    tol: (relative) tolerance for termination. If None, set to 0.
    T_max: max length of simulation. if None, set to 1.
    factors: not used
    terminal_J_fn: a function f(t, J, h) which will cause the
    integration to terminate when it hits zero.
    #
    returns:
    Jf: final state of J matrix
    h: coordinates of Jf such that Jf = e^diag(-h) J0 e^diag(h)
    """

    if 'method' not in solve_args.keys():
        solve_args['method'] ='DOP853'
    N = J0.shape[0]

    # set terminal event
    def terminal_z_fn(t, z):
        J, h = split_z(z)
        return terminal_fn(t, J, h)
    terminal_z_fn.terminal = True
    if terminal_fn is None:
        events = None
    else:
        events = [terminal_z_fn]

    def make_z(J, h):
        z = np.concatenate((J.ravel(), h))
        return z

    def split_z(z):
        J = np.reshape(z[:N**2], (N, N))
        h = z[-N:]
        return J, h

    def dzdt_fn(t, z):
        J, h = split_z(z)
        g = g_fn(J)
        J_dot = calc_lax_update(J, g) / tau
        h_dot = g / tau
        return make_z(J_dot, h_dot)

    t_span = (0, T_max)
    z0 = make_z(deepcopy(J0), np.zeros((N,)))

    res = solve_ivp(
        dzdt_fn,
        t_span,
        z0,
        events=events,
        t_eval=t_eval,
        **solve_args)

    if t_eval is None:        
        return split_z(res.y[:, -1])
    else:
        return list(zip(*[split_z(y) for y in res.y.T]))


def run_lax_dynamics_euler(
        J0,
        g_fn,
        factors=None,
        tau=1.,
        dt=0.05,
        tol=0.,
        T_max=1.):
    """
    Run a Lax dynamical system starting at J0, given a function that produces
    values of g at each time step, via explicit forward Euler integration.
    arguments:
    J0 is initial N x N matrix
    g_fn: function taking J to g
    factors: (U0, V0) where J0 = U0.dot(V0.T). Assumed to be same shape
    tau: timescale of dynamics (inverse of "learning rate")
    dt: Euler step size
    tol: (relative) tolerance for termination. If None, set to 0.
    T_max: max length of simulation. if None, set to 1.
    #
    returns:
    Jf: final state of J matrix
    h: coordinates of Jf such that Jf = e^diag(-h) J0 e^diag(h)
    """
    # forward Euler integration
    # TODO: implement this with jax.lax.scan()

    def _euler_step(*args):
        return (x + (dt / tau) * dxdt for (x, dxdt) in args)

    def _run_euler_factored(U0, V0):
        U, V = deepcopy(U0), deepcopy(V0)
        h = np.zeros((J0.shape[0],))
        tt = np.arange(0, T_max, step=dt)

        for _ in tqdm(tt):
            J = U @ V.T
            g = g_fn(J)
            U_dot, V_dot = calc_factored_lax_update((U, V), g)
            h, U, V = _euler_step(
                (h, g), (U, U_dot), (V, V_dot)
            )
            if np.max(np.abs(U_dot)) <= tol or np.max(np.abs(V_dot)) <= tol:
                break
        return (U, V), h

    def _run_euler(J0):
        J = deepcopy(np.array(J0))
        h = np.zeros((J0.shape[0],))
        tt = np.arange(0, T_max, step=dt)

        for _ in tqdm(tt):
            # import pdb; pdb.set_trace()
            g = g_fn(J)
            J_dot = calc_lax_update(J, g)
            h, J = _euler_step((h, g), (J, J_dot))
            if np.max(np.abs(J_dot)) <= tol:
                break
        return J, h

    if factors is not None:
        U0, V0 = factors
        (U, V), h = _run_euler_factored(U0, V0)
        J = FactoredMatrix(U, V)
    else:
        J, h = _run_euler(J0)

    return J, h


def calc_lax_update(J, g):
    """
    calculate the time derivative at J given a vector g
    arguments:
    J is (N, N) arrray
    g is (N,) array
    returns:
    dJ/dt = J * diag(g) - diag(g) * J
    """
    return J * g[None, :] - g[:, None] * J


def calc_factored_lax_update(factors, g):
    """
    calculate the Lax time derivative of U and V if J = U.dot(V^T), given g
    U, V are (N, M) arrrays
    g is (N,) array
    returns:
    (dU/dt, dV/dt) where dU/dt = - diag(g) x U and dV/dt = diag(g) x V
    """
    U, V = factors
    return - g[:, None] * U, g[:, None] * V


def _task_preserving_transform_J(
        J: np.ndarray,
        h: np.ndarray,
):
    """
    Given a recurrent weight matrix J and transform vector h, compute the
    task-preserving transform on the recurrent matrix.
    """
    return np.exp(-h[:, None] + h[None, :]) * J


def task_preserving_transform(
        rnn: RNN,
        h: np.ndarray
):
    """
    Given an initial rnn and transform vector h, compute the task-preserving
    transform.
    """
    exphp, exphn = np.exp(h), np.exp(-h)
    transformed_params = {
        'w_in': exphn[:, None] * rnn.w_in,
        'w_rec': _task_preserving_transform_J(rnn.w_rec, h),
        'w_out': rnn.w_out * exphp[None, :],
        'b': exphn * rnn.b,
    }

    return update_rnn(rnn, **transformed_params)



"""
Methods to extract information from the current cost matrix. Includes derived
cost matrices and lower/upper bounds on the final cost.  
"""


def calc_somatic_states(C):
    """
    Calculate the somatic states, g, given cost matrix C.
    The somatic state of neuron i is:
    g_i = sum(ith row of C) - sum(ith column of C)
    arguments:
    C: cost matrix.
    """
    return np.sum(C - C.T, axis=1)


def calc_symmetrized_cost_matrix(C):
    """
    Calculate the symmetrized cost matrix. This is the weighted
    adjacency matrix of the graph with respect to which the Jacobian of
    the balancing dynamics is the graph Laplacian.
    We adopt the convention of zeroing out the diagonal.
    arguments:
    C: cost matrix.
    """
    C_bar = C - np.diag(np.diag(C))
    return C_bar + C_bar.T


def calc_graph_laplacian(C):
    """
    Calculate the graph Laplacian derived from the symmetrized cost matrix.
    arguments:
    C: cost matrix.
    """
    C_bar = calc_symmetrized_cost_matrix(C)
    return laplacian(C_bar)


def calc_balancing_lb(C):
    """
    calculate a lower bound on |balancing_cost_final - balancing_cost_initial|.
    arguments:
    C: cost matrix.
    """
    C_bar = calc_symmetrized_cost_matrix(C)
    M = 2 * np.max(np.sum(C_bar, axis=0))
    g0 = calc_somatic_states(C)
    lb = np.linalg.norm(g0) ** 2 / (2 * M)
    return float(lb)


def calc_balancing_ub(C):
    """
    calculate an upper bound on |balancing_cost_final - balancing_cost_initial|
    arguments:
    C: cost matrix.
    """
    # calculate a lower bound on the the equilibrium cost matrix
    C_geometric_mean = np.sqrt(C * C.T)

    # return the difference between current cost and the lower bound
    return float(np.sum(C) - np.sum(C_geometric_mean))


"""
Methods for solving for the equilibrium state. 
"""


def _solve_balancing_rank_one(J0, cost_fn, factors=None, warn=True, **kwargs):
    """
    Implement the explicit expression for the minimizer of the cost function in
    the case that J0 is rank-one:
    |J_ij| = GeoMean( alpha_ij^(-1/p) |J0_ij|, alpha_ji^(1/p) |J0_ji| )
    arguments:
    factors = (U, V) is low rank factorization of J0 such that J0 = U * V.T
    cost_fn is CostFunction instance
    returns: summarized transform
    """

    if warn:
        raise Warning("Current implementation of rank-one solver is "
                      "numerically imprecise. Set warn=False to suppress "
                      "this warning.")

    def factor_rank_one(mat):
        u, s, v = randomized_svd(mat, n_components=1)
        u, v = np.abs(np.sqrt(s) * u[:, 0]), np.abs(np.sqrt(s) * v[0, :])
        return u, v

    # solve rank-one factorization of J0 and alpha
    u0, v0 = factor_rank_one(J0)
    a, b = factor_rank_one(cost_fn.alpha)

    # calculate J0 and the optimal h-transform
    J0 = u0[:, None] * v0[None, :]
    h = np.log(a / b) / (2. * cost_fn.p) + np.log(u0 / v0) / 2.
    # exph = (a/b)**(1/(2. * cost_fn.p)) * (u0/v0)**2

    # report results
    return summarize_transform(J0, cost_fn, h)


def _solve_balancing_cvx(J0, cost_fn, factors=None, **solve_args):
    """
    Use CVXPY to solve the synaptic balancing problem.
    This is very fast on small matrices (<100 neurons), but quite slow on
    larger matrices.
    arguments:
    J0 is N x N numpy array
    arguments:
    J0 is N x N numpy array
    cost_fn is CostFunction instance
    returns: summarized transform
    """
    import cvxpy as cvx

    # define constant element-wise weights
    J0 = np.array(J0).astype(float)
    weights = cost_fn.alpha * (np.abs(J0) ** cost_fn.p)

    # define variable (state, h, and the log ratio matrix derived from it)
    N = J0.shape[0]
    h = cvx.Variable((N, 1), value=np.zeros((N, 1)))
    log_ratios = -h * np.ones((1, N)) + np.ones((N, 1)) * h.T
    C = cvx.multiply(weights, cvx.exp(cost_fn.p * log_ratios))

    # solve problem and get final transform vector
    problem = cvx.Problem(cvx.Minimize(cvx.sum(C)))
    problem.solve(**solve_args)
    hf = h.value[:, 0]

    # report results
    return summarize_transform(J0, cost_fn, hf)


def _solve_balancing_euler(J0, cost_fn, **solve_args):
    """
    Euler integration of gradient descent dynamics to minimize a cost function.
    """
    _, h = run_lax_dynamics_euler(J0, cost_fn.states, **solve_args)
    return summarize_transform(J0, cost_fn, h)


def _solve_balancing_odeint(J0, cost_fn, **solve_args):
    """
    Runge-Kutta integration of gradient descent dynamics to minimize a cost
    function.
    """
    _, h = run_lax_dynamics_odeint(J0, cost_fn.states, **solve_args)
    return summarize_transform(J0, cost_fn, h)


def solve_balancing(
        rnn: RNN,
        cost_fn: CostFunction,
        how: str,
        **opt_args
):
    """
    Given an RNN and cost parameters, solve the balancing problem and return
    a balanced RNN.
    arguments:
    rnn: the initial state of the rnn to be optimized
    p, alpha: parameters of the cost function
    solver: {'cvx', 'rank_one', 'euler', 'odeint'}.
    returns: the transformed RNN and a summary of the optimization
    """

    # set optimizer
    if how == 'cvx':
        optimizer = _solve_balancing_cvx
    elif how == 'rank_one':
        optimizer = _solve_balancing_rank_one
    elif how == 'euler':
        optimizer = _solve_balancing_euler
    else:
        optimizer = _solve_balancing_odeint

    # set optimization arguments
    default_opt_args = {
        'factors': rnn.factors['w_rec']
    }
    default_opt_args.update(opt_args)

    # run optimizer
    results = optimizer(rnn.w_rec, cost_fn, **default_opt_args)
    transformed_rnn = task_preserving_transform(rnn, results['h'])
    return transformed_rnn, results


"""
Methods to calculate and report summary statistics. 
"""


def summarize_transform(
        J0,
        cost_fn,
        h: np.array,
):
    """
    Given an initial recurrent weight matrix, a transform vector h, and a cost
    function, calculate some useful statistics about the change in synaptic
    state before and after the task-preserving transform.
    """
    Jf = _task_preserving_transform_J(J0, h)
    C0 = cost_fn.matrix(J0)
    Cf = cost_fn.matrix(Jf)
    c0 = np.sum(C0)
    cf = np.sum(Cf)

    return {
        'J0': np.array(J0, dtype=float),
        'Jf': np.array(Jf, dtype=float),
        'h': np.array(h, dtype=float),
        'C0': np.array(C0, dtype=float),
        'Cf': np.array(Cf, dtype=float),
        'c0': float(c0),
        'cf': float(cf),
        'delta_c': float(c0 - cf),
        'lb': calc_balancing_lb(C0),
        'ub': calc_balancing_ub(C0),
        'deltaJ': float(np.linalg.norm(J0 - Jf)),
    }
