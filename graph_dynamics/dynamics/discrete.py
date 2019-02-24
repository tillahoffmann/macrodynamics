import numpy as np
from scipy import sparse

from ..util import lazy_property, to_array, is_homogeneous, add_leading_dims
from .operator import Operator


class DiscreteOperator(Operator):
    """
    Differential operator for linear dynamics on graphs.

    Parameters
    ----------
    matrix : np.ndarray
        flattened evolution tensor with shape `(k * n, k * n)` for vector dynamics, where `n` is
        the number of nodes and `k` is the number of state variables
    shape : np.ndarray
        shape `(k, n)` of the state vector

    Notes
    -----
    If `matrix` is a sparse matrix, it is recommended to use the `csr_matrix` format because it is
    particularly efficient for the dot product used in the calculation of the gradient. See
    https://docs.scipy.org/doc/scipy/reference/sparse.html#usage-information for details.
    """
    def __init__(self, matrix, shape):
        self.matrix = matrix
        self._shape = shape
        assert len(self.shape) == 2, "shape must have length two but got %d" % len(self.shape)
        matrix_rank = np.prod(self.shape)
        assert self.matrix.shape == (matrix_rank, matrix_rank), \
            "expected matrix shape %s but got %s" % ((matrix_rank, matrix_rank), self.matrix.shape)

    @classmethod
    def from_tensor(cls, tensor):
        """
        Create a differential operator for vector dynamics from an evolution tensor.

        Parameters
        ----------
        tensor : np.ndarray
            evolution tensor with shape `(k, k, n, n)` for vector dynamics, where `n` is the
            number of nodes and `k` is the number of state variables
        """
        tensor = np.asarray(tensor)
        assert tensor.ndim == 4, "expected 4D tensor but got %dD" % tensor.ndim
        assert is_homogeneous(tensor.shape[:2]) and is_homogeneous(tensor.shape[2:]), \
            "expected a tensor with shape (k, k, n, n) but got %s" % (tensor.shape, )
        state_shape = tensor.shape[1:3]
        # Flatten the tensor
        matrix_rank = np.prod(state_shape)
        matrix = np.rollaxis(tensor, 2, 1).reshape((matrix_rank, matrix_rank))
        return cls(matrix, state_shape)

    @classmethod
    def from_matrix(cls, matrix):
        """
        Create a differential operator for scalar dynamics from an evolution matrix.

        Parameters
        ----------
        matrix : np.ndarray
            evolution matrix with shape `(n, n)` for scalar dynamics, where `n` is the number of
            nodes
        """
        return cls(matrix, (1, ) + matrix.shape[:1])

    @lazy_property
    def issparse(self):
        return sparse.issparse(self.matrix)

    @lazy_property
    def shape(self):
        return self._shape

    @lazy_property
    def eig(self):
        """tuple : eigenvalues and eigenvectors of the evolution matrix"""
        if self.issparse:
            sparse.linalg.eigs(self.matrix)
        else:
            return np.linalg.eig(self.matrix)

    @property
    def evals(self):
        """np.ndarray : eigenvalues of the evolution matrix"""
        return self.eig[0]

    @property
    def evecs(self):
        """np.ndarray : eigenvectors of the evolution matrix"""
        return self.eig[1]

    @lazy_property
    def ievecs(self):
        """np.ndarray : inverse of the eigenvector matrix of the evolution matrix"""
        return np.linalg.inv(self.evecs)

    def integrate_analytic(self, z, t):
        z = self._assert_valid_shape(z)
        # Project into the diagonal basis
        z = np.dot(self.ievecs, z.ravel())
        # Evolve the state (which has shape (number of time steps, number of state variables))
        z = np.exp(self.evals * np.reshape(t, (-1, 1))) * z
        # Project back into the real space
        z = np.einsum('ij,tj->ti', self.evecs, z)
        z = np.reshape(z, (-1, *self.shape))
        return z[-1] if np.isscalar(t) else z

    def evaluate_gradient(self, z, t=None):
        z = self._assert_valid_shape(z)
        grad = self.matrix.dot(z.ravel())
        return np.reshape(grad, self.shape)

    @property
    def has_analytic_solution(self):
        return not self.issparse
