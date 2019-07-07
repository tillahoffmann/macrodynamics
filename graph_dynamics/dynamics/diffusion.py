import numpy as np
from scipy import sparse

from ..structure import evaluate_expected_degree
from .discrete import DiscreteOperator
from .continuous import ContinuousOperator
from ..util import ignore_scipy_issue_9093


@ignore_scipy_issue_9093
def evaluate_discrete_operator(adjacency):
    """
    Evaluate the differential operator for diffusion on a graph.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix

    Returns
    -------
    operator : DiscreteOperator
        differential operator for diffusion
    """
    out_degree = adjacency.sum(axis=0)
    if sparse.issparse(adjacency):
        diag = sparse.spdiags(out_degree, 0, adjacency.shape[0], adjacency.shape[1])
    else:
        diag = np.diag(out_degree)
    return DiscreteOperator.from_matrix(adjacency - diag)


def evaluate_continuous_operator(connectivity, density, dx):
    """
    Evaluate the differential operator for diffusion on a graph.

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
        differential operator for diffusion
    """
    # The nodes lose walkers proportional to their degree
    weight = - evaluate_expected_degree(connectivity, density, dx)
    # And gain walkers proportional to the density of nodes in the underlying space weighted by the kernel
    kernel = connectivity
    kernel_weight_y = density
    # The elementwise kernel weight is not relevant
    kernel_weight_x = np.ones_like(density)

    return ContinuousOperator.from_matrix(weight, kernel, kernel_weight_x, kernel_weight_y, dx)
