import numpy as np
from scipy import sparse

from ..structure import evaluate_expected_degree
from .discrete import DiscreteOperator
from .continuous import ContinuousOperator
from ..util import _ignore_scipy_issue_9093


@_ignore_scipy_issue_9093
def evaluate_discrete_operator(adjacency):
    """
    Evaluate the differential operator for opinion averaging on a graph.

    Parameters
    ----------
    adjacency : numpy.ndarray or scipy.sparse.spmatrix
        Adjacency matrix.

    Returns
    -------
    operator : DiscreteOperator
        Differential operator for opinion averaging.
    """
    n = adjacency.shape[0]
    in_degree = adjacency.sum(axis=1)
    # Replace zeros by ones because the corresponding row will only contain zeros anyway and we don't want nans
    in_degree = np.where(in_degree > 0, in_degree, 1).astype(float)
    if sparse.issparse(adjacency):
        matrix = adjacency / in_degree - sparse.spdiags(np.ones(n), 0, n, n)
    else:
        matrix = adjacency / in_degree[:, None] - np.eye(n)
    return DiscreteOperator.from_matrix(matrix)


def evaluate_continuous_operator(connectivity, density, dx, **kwargs):
    """
    Evaluate the differential operator for opinion averaging on a graph.

    Parameters
    ----------
    connectivity : numpy.ndarray
        Evaluated connectivity kernel.
    density : numpy.ndarray or float
        Density of nodes (use a scalar for uniform densities).
    dx : numpy.ndarray or float
        Spacing of sample points.
    **kwargs : dict
        Keyword arguments passed to `ContinuousOperator.from_matrix`.

    Returns
    -------
    operator : ContinuousOperator
        Differential operator for opinion averaging.
    """
    # The contributions are down-weighted by the average degree of the nodes
    kernel_weight_x = 1 / evaluate_expected_degree(connectivity, density, dx)
    # And contributions are due to the connectivity kernel weighted by the density
    kernel = connectivity
    kernel_weight_y = density
    # The elementwise weight is -1 because the gradient is proportional to the difference
    # between the average and the current position
    weight = - np.ones_like(density)

    return ContinuousOperator.from_matrix(weight, kernel, kernel_weight_x, kernel_weight_y, dx,
                                          **kwargs)
