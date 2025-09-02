import numpy as np
from scipy.spatial.distance import squareform
from .util import convolve


def evaluate_distance(
    x: np.ndarray | float,
    y: np.ndarray | float,
    cov: np.ndarray | float = 1.0,
    domain: np.ndarray | None | float = 1.0,
) -> np.ndarray:
    """Evaluate the elementwise Mahalanobis distance between x and y.

    Args:
        x: Coordinates of shape `(..., k)`.
        y: Coordinates of shape `(..., k)`.
        cov: Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
        domain: Domain for periodic boundary conditions. If `None`, non-periodic boundary
            conditions are used.

    Returns:
        Non-negative distance array of shape `(...)`.
    """
    # Evaluate the difference
    delta = np.asarray(x - y)
    if domain is not None:
        domain = np.asarray(domain)
        # If x - y is smaller than -0.5, then x is very far left of y. But in the next
        # repetition of the space, x lies to the right of y such that the appropriate
        # distance is 1 + x - y = 1 + delta.
        delta += (delta < -0.5 * domain) * domain
        # If x - y is larger than 0.5, then x is very far right of y. But in the next
        # repetition of the space, y lies to the right of x such that the appropriate
        # distance is x - (1 + y) = delta - 1.
        delta -= (delta > 0.5 * domain) * domain

    # Use a diagonal scale
    if np.ndim(cov) < 2:
        cov = np.eye(delta.shape[-1]) * cov
    elif np.ndim(cov) > 2:  # pragma: no cover
        raise ValueError("cov must be a scalar, vector or matrix")

    # Evaluate the distance by contracting indices
    return np.sqrt(np.einsum("...i,ij,...j->...", delta, np.linalg.inv(cov), delta))


def evaluate_gaussian_kernel(
    x: np.ndarray | float,
    y: np.ndarray | float,
    norm: float,
    cov: np.ndarray | float,
    domain: float | np.ndarray | None = 1.0,
) -> np.ndarray:
    """Evaluate a Gaussian kernel.

    Args:
        x: Coordinates of shape `(..., k)`.
        y: Coordinates of shape `(..., k)`.
        norm: Normalisation of the kernel.
        cov: Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
        domain: Domain for periodic boundary conditions. If `None`, non-periodic boundary
            conditions are used.

    Returns:
        Non-negative kernel array of shape `(...)`.
    """
    distance = evaluate_distance(x, y, cov, domain)
    return norm * np.exp(-0.5 * distance * distance)


def evaluate_tophat_kernel(x, y, norm, cov, domain=1.0):
    """Evaluate a top hat (hard) kernel.

    Args:
        x: Coordinates of shape `(..., k)`.
        y: Coordinates of shape `(..., k)`.
        norm: Normalisation of the kernel.
        cov: Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
        domain: Domain for periodic boundary conditions. If `None`, non-periodic boundary
            conditions are used.

    Returns:
        Non-negative kernel array of shape `(...)`.
    """
    distance = evaluate_distance(x, y, cov, domain)
    return np.where(distance < 1, norm, 0)


def evaluate_laplace_kernel(x, y, norm, cov, domain=1.0):
    """Evaluate a Laplace (exponential) kernel.

    Args:
        x: Coordinates of shape `(..., k)`.
        y: Coordinates of shape `(..., k)`.
        norm: Normalisation of the kernel.
        cov: Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
        domain: Domain for periodic boundary conditions. If `None`, non-periodic boundary
            conditions are used.

    Returns:
        Non-negative kernel array of shape `(...)`.
    """
    distance = evaluate_distance(x, y, cov, domain)
    return norm * np.exp(-distance)


def evaluate_uniform_kernel(x, y, norm, cov=None, domain=1.0):
    """Evaluate a uniform kernel.

    Args:
        x: Coordinates of shape `(..., k)`.
        y: Coordinates of shape `(..., k)`.
        norm: Normalisation of the kernel.
        cov: Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
        domain: Domain for periodic boundary conditions. If `None`, non-periodic boundary
            conditions are used.

    Returns:
        Non-negative kernel array of shape `(...)`.
    """
    return norm * np.ones(np.broadcast(x, y).shape[:-1])


def sample_adjacency(coordinates, kernel, condensed=False, distribution="bernoulli"):
    """Sample an adjacency matrix.

    Args:
        coordinates: Coordinates of shape `(n, k)`.
        kernel: Kernel function.
        condensed: Whether to return a condensed adjacency matrix.
        distribution: Distribution to draw adjacency samples from.

    Returns:
        Sampled adjacency matrix.
    """
    i, j = np.triu_indices(coordinates.shape[0], 1)
    kernel = kernel(coordinates[i], coordinates[j])
    if distribution == "bernoulli":
        assert np.all((kernel >= 0) & (kernel <= 1)), (
            "kernel values must be in the interval [0, 1]"
        )
        adjacency = np.random.uniform(0, 1, kernel.shape) < kernel
    else:
        raise KeyError(distribution)

    if not condensed:
        adjacency = squareform(adjacency)

    return adjacency


def evaluate_expected_degree(connectivity, density, dx):
    """Evaluate the expected degree.

    Args:
        connectivity: Evaluated connectivity kernel.
        density: Density of nodes (use a scalar for uniform densities).
        dx: Spacing of sample points.

    Returns:
        Expected degree of nodes (returns a scalar if `density` is a scalar).
    """
    return convolve(density, connectivity, dx)
