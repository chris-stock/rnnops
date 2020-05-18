"""
Methods to specify and initialize recurrent networks.
"""
__all__ = [
    'nonlinearities',
    'iid_gaussian_init',
    'mvn_rows_init',
    'zeros_init',
]


import numpy as np


nonlinearities = {
    'linear': lambda x: x,
    'relu': lambda x: np.maximum(x, 0.),
    'tanh': lambda x: np.tanh(x),
}


def iid_gaussian_init(std=1., mean=0.):
    """
    randomly sample a matrix with iid Gaussian elements
    todo: change to use jax pseudorandomness
    """
    def init(shape):
        m, n = shape
        return std*np.random.randn(*shape)/np.sqrt(n) + mean
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



