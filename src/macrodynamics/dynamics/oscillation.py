import numpy as np
from scipy import sparse

from ..structure import evaluate_expected_degree
from .discrete import DiscreteOperator
from .continuous import ContinuousOperator

sparse.csr_matrix


def evaluate_discrete_operator(
    adjacency: np.ndarray | sparse.spmatrix,
    angular_frequency: np.ndarray | float = 1,
    **kwargs,
) -> DiscreteOperator:
    """Evaluate the differential operator for coupled oscillators.

    Args:
        adjacency: Adjacency matrix.
        angular_frequency: Angular frequency of each individual oscillator.
        **kwargs: Keyword arguments passed to `DiscreteOperator.from_tensor`.

    Returns:
        Differential operator for coupled oscillators.
    """
    n = adjacency.shape[0]
    in_degree = adjacency.sum(axis=1)  # pyright: ignore[reportAttributeAccessIssue]
    if sparse.issparse(adjacency):
        tensor = [
            (None, sparse.eye(n)),  # displacement
            (
                -sparse.spdiags(in_degree.T + angular_frequency**2, 0, n, n)
                + adjacency,
                None,
            ),  # velocity
        ]
    else:
        tensor = [
            (np.zeros((n, n)), np.eye(n)),  # displacement
            (
                -np.diag(in_degree + angular_frequency**2) + adjacency,
                np.zeros((n, n)),
            ),  # velocity
        ]

    tensor = np.asarray(tensor)
    return DiscreteOperator.from_tensor(tensor, **kwargs)


def evaluate_continuous_operator(
    connectivity: np.ndarray,
    density: np.ndarray,
    dx: np.ndarray | float,
    angular_frequency: np.ndarray | float = 1,
    **kwargs,
) -> ContinuousOperator:
    """Evaluate the differential operator for opinion averaging on a graph.

    Args:
        connectivity: Evaluated connectivity kernel.
        density: Density of nodes (use a scalar for uniform densities).
        dx: Spacing of sample points.
        angular_frequency: Angular frequency of each individual oscillator.
        **kwargs: Keyword arguments passed to `ContinuousOperator`.

    Returns:
        Differential operator for coupled oscillators.
    """
    weight = np.asarray(
        [
            (np.zeros_like(density), np.ones_like(density)),
            (
                -(angular_frequency**2)
                - evaluate_expected_degree(connectivity, density, dx),
                np.zeros_like(density),
            ),
        ]
    )

    shape = tuple([2, 2] + [1 for _ in range(np.ndim(density))])
    kernel = np.eye(2).reshape(shape) * connectivity
    kernel_weight_x = np.eye(2).reshape(shape) * np.ones_like(density)
    kernel_weight_y = np.asarray(
        [
            (
                np.zeros_like(density),
                np.zeros_like(density),
            ),
            (density, np.zeros_like(density)),
        ]
    )

    return ContinuousOperator(
        weight, kernel, kernel_weight_x, kernel_weight_y, dx, **kwargs
    )
