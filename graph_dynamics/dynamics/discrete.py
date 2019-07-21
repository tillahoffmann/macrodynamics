import numpy as np
from scipy import sparse

from ..util import lazy_property, to_array, is_homogeneous, add_leading_dims, nexpm1
from .operator import Operator


class DiscreteOperator(Operator):
    """
    Differential operator for linear dynamics on graphs.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Flattened evolution tensor with shape `(k * n, k * n)` for vector dynamics, where `n` is
        the number of nodes and `k` is the number of state variables.
    shape : numpy.ndarray
        Shape `(k, n)` of the state vector.
    control : numpy.ndarray
        Static control vector to apply to the dynamics.

    Notes
    -----
    If `matrix` is a sparse matrix, it is recommended to use the `csr_matrix` format because it is
    particularly efficient for the dot product used in the calculation of the gradient. See
    https://docs.scipy.org/doc/scipy/reference/sparse.html#usage-information for details.
    """
    def __init__(self, matrix, shape, control=None):
        self.matrix = matrix
        self._shape = shape
        assert len(self.shape) == 2, "shape must have length two but got %d" % len(self.shape)
        matrix_rank = np.prod(self.shape)
        assert self.matrix.shape == (matrix_rank, matrix_rank), \
            "expected matrix shape %s but got %s" % ((matrix_rank, matrix_rank), self.matrix.shape)
        super(DiscreteOperator, self).__init__(control)

    @classmethod
    def from_tensor(cls, tensor):
        """
        Create a differential operator for vector dynamics from an evolution tensor.

        Parameters
        ----------
        tensor : list
            List of lists of matrices constituting an evolution tensor with shape `(k, k, n, n)` for
            vector dynamics, where `n` is the number of nodes and `k` is the number of state
            variables.

        Returns
        -------
        operator : DiscreteOperator
            Operator encoding the dynamics.
        """
        # Ensure the tensor appears as a list of lists with the right shapes
        k = len(tensor)
        n = None
        blocks = []
        issparse = False
        for row in tensor:
            assert len(row) == k
            tmp = []
            for block in row:
                if block is not None:
                    if n is None:
                        n = block.shape[0]
                    assert block.shape == (n, n)
                    issparse |= sparse.issparse(block)
                tmp.append(block)
            blocks.append(tmp)

        # Construct the evoluation matrix
        if issparse:
            matrix = sparse.bmat(blocks).tocsr()
        else:
            matrix = np.block(blocks)
        return cls(matrix, (k, n))

    @classmethod
    def from_matrix(cls, matrix):
        """
        Create a differential operator for scalar dynamics from an evolution matrix.

        Parameters
        ----------
        matrix : numpy.ndarray
            Evolution matrix with shape `(n, n)` for scalar dynamics, where `n` is the number of
            nodes.

        Returns
        -------
        operator : DiscreteOperator
            Operator encoding the dynamics.
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
        """numpy.ndarray : eigenvalues of the evolution matrix"""
        return self.eig[0]

    @property
    def evecs(self):
        """numpy.ndarray : eigenvectors of the evolution matrix"""
        return self.eig[1]

    @lazy_property
    def ievecs(self):
        """numpy.ndarray : inverse of the eigenvector matrix of the evolution matrix"""
        return np.linalg.inv(self.evecs)

    def integrate_analytic(self, z, t):
        z = self._assert_valid_shape(z)
        # Project into the diagonal basis
        z = np.dot(self.ievecs, z.ravel())
        # Evolve the state (which has shape (number of time steps, number of state variables))
        t_vector = np.reshape(t, (-1, 1))
        z = np.exp(self.evals * t_vector) * z
        if self.control is not None:
            # Project into the diagonal basis
            control = np.dot(self.ievecs, self.control.ravel())
            # Evolve the state
            z += control * nexpm1(self.evals, t_vector)
        # Project back into the real space
        z = np.einsum('ij,tj->ti', self.evecs, z)
        z = np.reshape(z, (-1, *self.shape))
        return z[-1] if np.isscalar(t) else z

    def evaluate_gradient(self, z, t=None):
        z = self._assert_valid_shape(z)
        grad = self.matrix.dot(z.ravel())
        grad = np.reshape(grad, self.shape)
        if self.control is not None:
            grad += self.control
        return grad

    @property
    def has_analytic_solution(self):
        return not self.issparse
