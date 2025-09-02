import numpy as np
from scipy import sparse
from typing import Self, cast

from ..util import lazy_property, nexpm1
from .operator import Operator


class DiscreteOperator(Operator):
    """Differential operator for linear dynamics on graphs.

    Args:
        matrix: Flattened evolution tensor with shape `(k * n, k * n)` for vector dynamics,
            where `n` is the number of nodes and `k` is the number of state variables.
        shape: Shape `(k, n)` of the state vector.

    Note:
        If `matrix` is a sparse matrix, it is recommended to use the `csr_matrix` format
        because it is particularly efficient for the dot product used in the calculation of
        the gradient. See
        https://docs.scipy.org/doc/scipy/reference/sparse.html#usage-information for
        details.
    """

    def __init__(
        self, matrix: np.ndarray | sparse.spmatrix, shape: tuple[int, ...]
    ) -> None:
        self.matrix = matrix
        self._shape = shape
        assert len(self.shape) == 2, "shape must have length two but got %d" % len(
            self.shape
        )
        matrix_rank = np.prod(self.shape)
        assert self.matrix.shape == (
            matrix_rank,
            matrix_rank,
        ), "expected matrix shape %s but got %s" % (
            (matrix_rank, matrix_rank),
            self.matrix.shape,
        )

    @classmethod
    def from_tensor(cls, tensor: np.ndarray) -> Self:
        """Create a differential operator for vector dynamics from an evolution tensor.

        Args:
            tensor: List of lists of matrices constituting an evolution tensor with shape
                `(k, k, n, n)` for vector dynamics, where `n` is the number of nodes and `k`
                is the number of state variables.

        Returns:
            Operator encoding the dynamics.
        """
        # Ensure the tensor appears as a list of lists with the right shapes
        k = len(tensor)
        n: int | None = None
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

        assert n, "The number of nodes `n` could not be inferred. Is the tensor empty?"

        # Construct the evoluation matrix
        if issparse:
            matrix = cast(sparse.csr_matrix, sparse.bmat(blocks).tocsr())
        else:
            matrix = np.block(blocks)
        return cls(matrix, (k, n))

    @classmethod
    def from_matrix(cls, matrix) -> Self:
        """Create a differential operator for scalar dynamics from an evolution matrix.

        Args:
            matrix: Evolution matrix with shape `(n, n)` for scalar dynamics, where `n` is the
                number of nodes.

        Returns:
            Operator encoding the dynamics.
        """
        return cls(matrix, (1,) + matrix.shape[:1])

    @lazy_property
    def issparse(self) -> bool:
        return sparse.issparse(self.matrix)

    @lazy_property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @lazy_property
    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """tuple : eigenvalues and eigenvectors of the evolution matrix"""
        if self.issparse:
            return sparse.linalg.eigs(self.matrix)
        else:
            return np.linalg.eig(self.matrix)  # pyright: ignore[reportArgumentType,reportCallIssue]

    @property
    def evals(self) -> np.ndarray:
        """numpy.ndarray : eigenvalues of the evolution matrix"""
        return self.eig[0]

    @property
    def evecs(self) -> np.ndarray:
        """numpy.ndarray : eigenvectors of the evolution matrix"""
        return self.eig[1]

    @lazy_property
    def ievecs(self) -> np.ndarray:
        """numpy.ndarray : inverse of the eigenvector matrix of the evolution matrix"""
        return np.linalg.inv(self.evecs)

    def integrate_analytic(
        self, z: np.ndarray, t: float | np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        z = self._assert_valid_shape(z)
        control = self._assert_valid_shape(control, strict=False)
        # Project into the diagonal basis.
        z = np.dot(self.ievecs, z.ravel())
        # Evolve the state (which has shape
        # .(number of time steps, number of state variables))
        t_vector = np.reshape(t, (-1, 1))
        z = np.exp(self.evals * t_vector) * z
        if control is not None:
            # Project into the diagonal basis.
            control = np.dot(self.ievecs, control.ravel())
            # Evolve the state
            z += control * nexpm1(self.evals, t_vector)
        # Project back into the real space.
        z = np.einsum("ij,tj->ti", self.evecs, z)
        z = np.reshape(z, (-1, *self.shape))
        return z[-1] if np.isscalar(t) else z

    def evaluate_gradient(
        self, z: np.ndarray, t=None, control: np.ndarray | None = None
    ) -> np.ndarray:
        z = self._assert_valid_shape(z)
        control = self._assert_valid_shape(control, strict=False)
        grad = self.matrix.dot(z.ravel())  # pyright: ignore[reportAttributeAccessIssue]
        grad = np.reshape(grad, self.shape)
        if control is not None:
            grad += control
        return grad

    @property
    def has_analytic_solution(self) -> bool:
        return not self.issparse
