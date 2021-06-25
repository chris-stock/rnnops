import numpy as np


class FactoredMatrix(object):
    """
    Data type for a matrix along with its factors.
    """

    def __init__(
            self,
            U: np.ndarray,
            V: np.ndarray,
            check_rank: bool = True,
    ):
        """
        Initialize a matrix A = U * V.T with the factors U and V.
        U: (m,  r) array. Must have full column rank.
        V (n, r) array. Must have full column rank.
        check_rank: if True, check that U and V have full column rank (default
        True)
        """

        self._rank = U.shape[1]
        if check_rank:
            try:
                assert (np.linalg.matrix_rank(U) == self._rank)
            except AssertionError:
                raise ValueError("Matrix U must have full column rank")
            try:
                assert (np.linalg.matrix_rank(V) == self._rank)
            except AssertionError:
                raise ValueError("Matrix V must have full column rank")

        self._U = U
        self._V = V
        self._value = self._U.dot(self._V.T)

    def __array__(self):
        """
        Return the full matrix.
        """
        return self._value

    def __repr__(self):
        return str(self.__class__.__name__)

    @property
    def factors(self):
        """
        Return the left and right factors of the matrix.
        """
        return self._U, self._V

    @property
    def rank(self):
        """
        Return the rank of the matrix.
        """
        return self._rank


def init_factored_matrix(mat):
    """
    given a 2-dim numpy array, create a FactoredMatrix instance.
    """
    pass


def truncate_factored_matrix(fm: FactoredMatrix, k: int):
    """
    Given a factored matrix and a rank k, return the k-truncated factored
    matrix.
    arguments:
    fm: FactoredMatrix instance
    k: int. Keep the first k columns of matrix factors.
    returns:
    a truncated FactoredMatrix.
    """
    U, V = fm.factors
    return FactoredMatrix(U[:, :k], V[:, :k], check_rank=False)
