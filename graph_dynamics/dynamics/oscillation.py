import numpy as np
from scipy import sparse

from ..structure import evaluate_expected_degree
from .discrete import DiscreteOperator
from .continuous import ContinuousOperator
from ..util import ignore_scipy_issue_9093


@ignore_scipy_issue_9093
def evaluate_discrete_operator(adjacency, angular_frequency=1, **kwargs):
    """
    Evaluate the differential operator for coupled oscillators.

    Parameters
    ----------
    adjacency : np.ndarray
        adjacency matrix
    natural_frequency : np.ndarray
        angular frequency of each individual oscillator

    Returns
    -------
    operator : DiscreteOperator
        differential operator for coupled oscillators
    """
    n = adjacency.shape[0]
    in_degree = adjacency.sum(axis=1)
    if sparse.issparse(adjacency):
        tensor = [
            (None, sparse.eye(n)),  # displacement
            (-sparse.spdiags(in_degree.T + angular_frequency ** 2, 0, n, n) + adjacency, None)  # velocity
        ]
    else:
        tensor = [
            (np.zeros((n, n)), np.eye(n)),  # displacement
            (-np.diag(in_degree + angular_frequency ** 2) + adjacency, np.zeros((n, n)))  # velocity
        ]

    return DiscreteOperator.from_tensor(tensor, **kwargs)


def evaluate_continuous_operator(connectivity, density, dx, angular_frequency=1, **kwargs):
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
    weight = [
        (
            np.zeros_like(density),
            np.ones_like(density)
        ),
        (
            - angular_frequency ** 2 - evaluate_expected_degree(connectivity, density, dx),
            np.zeros_like(density)
        )
    ]

    shape = tuple([2, 2] + [1 for _ in range(np.ndim(density))])
    kernel = np.eye(2).reshape(shape) * connectivity
    kernel_weight_x = np.eye(2).reshape(shape) * np.ones_like(density)
    kernel_weight_y = [
        (
            np.zeros_like(density),
            np.zeros_like(density),
        ),
        (
            density,
            np.zeros_like(density)
        )
    ]

    return ContinuousOperator(weight, kernel, kernel_weight_x, kernel_weight_y, dx, **kwargs)
