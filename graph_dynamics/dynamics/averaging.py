import numpy as np
import scipy.sparse

from ..structure import evaluate_expected_degree
from .discrete import DiscreteOperator
from .continuous import ContinuousOperator


def evaluate_discrete_operator(adjacency):
    """
    Evaluate the differential operator for opinion averaging on a graph.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix

    Returns
    -------
    operator : DiscreteOperator
        differential operator for opinion averaging
    """
    n = adjacency.shape[0]
    in_degree = adjacency.sum(axis=1)
    # Replace zeros by ones because the corresponding row will only contain zeros anyway and we don't want nans
    in_degree = np.where(in_degree > 0, in_degree, 1).astype(float)
    if scipy.sparse.issparse(adjacency):
        matrix = adjacency / in_degree - scipy.sparse.spdiags(np.ones(n), 0, n, n)
    else:
        matrix = adjacency / in_degree[:, None] - np.eye(n)
    return DiscreteOperator.from_scalar(matrix)


def evaluate_continuous_operator(connectivity, density, dx):
    """
    Evaluate the differential operator for opinion averaging on a graph.

    Parameters
    ----------
    connectivity : np.ndarray
        evaluated connectivity kernel
    density : np.ndarray or float
        density of nodes (use a scalar for uniform densities)
    dx : np.ndarray or float
        spacing of sample points

    Returns
    -------
    operator : ContinuousOperator
        differential operator for opinion averaging
    """
    # The contributions are down-weighted by the average degree of the nodes
    kernel_weight_x = 1 / evaluate_expected_degree(connectivity, density, dx)
    # And contributions are due to the connectivity kernel weighted by the density
    kernel = connectivity
    kernel_weight_y = density
    # The elementwise weight is -1 because the gradient is proportional to the difference
    # between the average and the current position
    weight = - np.ones_like(density)

    return ContinuousOperator.from_scalar(weight, kernel, kernel_weight_x, kernel_weight_y, dx)
