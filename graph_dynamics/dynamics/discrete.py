import numpy as np
import scipy.sparse

from ..util import lazy_property, to_array, is_homogeneous, add_leading_dims
from .operator import Operator


class DiscreteOperator(Operator):
    """
    Differential operator for linear dynamics on graphs.

    Parameters
    ----------
    tensor : np.ndarray
        evolution tensor (with shape `(k, k, n, n)`) for vector dynamics, where `n` is the number
        of nodes and `k` is the number of state variables
    """
    def __init__(self, tensor, sparse=False):
        self.tensor = np.asarray(tensor)
        assert self.tensor.ndim == 4, "expected 4D tensor but got %dD" % self.tensor.ndim
        assert is_homogeneous(self.tensor.shape[:2]) and is_homogeneous(self.tensor.shape[2:]), \
            "expected a tensor with shape (n, n, k, k) but got %s" % (self.tensor.shape, )
        # Flatten the tensor
        matrix_rank = np.prod(self.shape)
        self.matrix = np.rollaxis(self.tensor, 2, 1).reshape((matrix_rank, matrix_rank))
        if sparse:
            self.matrix = scipy.sparse.csr_matrix(self.matrix)

    @lazy_property
    def shape(self):
        return self.tensor.shape[1:3]

    @lazy_property
    def eig(self):
        """tuple : eigenvalues and eigenvectors of the evolution matrix"""
        return np.linalg.eig(to_array(self.matrix))

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
        return True

    @classmethod
    def from_scalar(cls, tensor):
        return cls(add_leading_dims(tensor, 2))
