import numpy as np
from scipy import sparse

from ..structure import evaluate_expected_degree
from .discrete import DiscreteOperator
from .continuous import ContinuousOperator
from ..util import _ignore_scipy_issue_9093


@_ignore_scipy_issue_9093
def evaluate_discrete_operator(adjacency):
    """
    Evaluate the differential operator for diffusion on a graph.

    Parameters
    ----------
    adjacency : numpy.ndarray or scipy.sparse.spmatrix
        Adjacency matrix.

    Returns
    -------
    operator : DiscreteOperator
        Differential operator for diffusion.
    """
    out_degree = adjacency.sum(axis=0)
    if sparse.issparse(adjacency):
        diag = sparse.spdiags(out_degree, 0, adjacency.shape[0], adjacency.shape[1])
    else:
        diag = np.diag(out_degree)
    return DiscreteOperator.from_matrix(adjacency - diag)


def evaluate_continuous_operator(connectivity, density, dx, **kwargs):
    """
    Evaluate the differential operator for diffusion on a graph.

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
        Differential operator for diffusion.
    """
    # The nodes lose walkers proportional to their degree
    weight = - evaluate_expected_degree(connectivity, density, dx)
    # And gain walkers proportional to the density of nodes in the underlying space weighted by the kernel
    kernel = connectivity
    kernel_weight_y = density
    # The elementwise kernel weight is not relevant
    kernel_weight_x = np.ones_like(density)

    return ContinuousOperator.from_matrix(weight, kernel, kernel_weight_x, kernel_weight_y, dx,
                                          **kwargs)
