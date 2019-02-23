import numpy as np
from scipy.spatial.distance import squareform
from .util import convolve


def evaluate_distance(x, y, cov=1.0, domain=1.0):
    """
    Evaluate the elementwise Mahalanobis distance between x and y.

    Parameters
    ----------
    x : np.ndarray
        coordinates of shape `(..., k)`
    y : np.ndarray
        coordinates of shape `(..., k)`
    cov : np.ndarray
        covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix)
    domain : array_like
        domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are used.

    Returns
    -------
    distance : np.ndarray
        non-negative distance array of shape `(...)`
    """
    # Evaluate the difference
    delta = x - y
    if domain is not None:
        domain = np.asarray(domain)
        # If x - y is smaller than -0.5, then x is very far left of y. But in the next repetition of the space,
        # x lies to the right of y such that the appropriate distance is 1 + x - y = 1 + delta.
        delta += (delta < -0.5 * domain) * domain
        # If x - y is larger than 0.5, then x is very far right of y. But in the next repetition of the space,
        # y lies to the right of x such that the appropriate distance is x - (1 + y) = delta - 1.
        delta -= (delta > 0.5 * domain) * domain

    # Use a diagonal scale
    if np.ndim(cov) < 2:
        cov = np.eye(delta.shape[-1]) * cov
    elif np.ndim(cov) > 2:  # pragma: no cover
        raise ValueError("cov must be a scalar, vector or matrix")

    # Evaluate the distance by contracting indices
    return np.sqrt(np.einsum('...i,ij,...j->...', delta, np.linalg.inv(cov), delta))


def evaluate_gaussian_kernel(x, y, norm, cov, domain=1.0):
    """
    Evaluate a Gaussian kernel.

    Parameters
    ----------
    x : np.ndarray
        coordinates of shape `(..., k)`
    y : np.ndarray
        coordinates of shape `(..., k)`
    norm : float
        normalisation of the kernel
    cov : np.ndarray
        covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix)
    domain : array_like
        domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are used.

    Returns
    -------
    kernel : np.ndarray
        non-negative kernel array of shape `(...)`
    """
    distance = evaluate_distance(x, y, cov, domain)
    return norm * np.exp(-0.5 * distance * distance)


def evaluate_tophat_kernel(x, y, norm, cov, domain=1.0):
    """
    Evaluate a top hat (hard) kernel.

    Parameters
    ----------
    x : np.ndarray
        coordinates of shape `(..., k)`
    y : np.ndarray
        coordinates of shape `(..., k)`
    norm : float
        normalisation of the kernel
    cov : np.ndarray
        covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix)
    domain : array_like
        domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are used.

    Returns
    -------
    kernel : np.ndarray
        non-negative kernel array of shape `(...)`
    """
    distance = evaluate_distance(x, y, cov, domain)
    return np.where(distance < 1, norm, 0)


def evaluate_laplace_kernel(x, y, norm, cov, domain=1.0):
    """
    Evaluate a Laplace (exponential) kernel.

    Parameters
    ----------
    x : np.ndarray
        coordinates of shape `(..., k)`
    y : np.ndarray
        coordinates of shape `(..., k)`
    norm : float
        normalisation of the kernel
    cov : np.ndarray
        covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix)
    domain : array_like
        domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are used.

    Returns
    -------
    kernel : np.ndarray
        non-negative kernel array of shape `(...)`
    """
    distance = evaluate_distance(x, y, cov, domain)
    return norm * np.exp(-distance)


def evaluate_uniform_kernel(x, y, norm, cov=None, domain=1.0):
    """
    Evaluate a uniform kernel.

    Parameters
    ----------
    x : np.ndarray
        coordinates of shape `(..., k)`
    y : np.ndarray
        coordinates of shape `(..., k)`
    norm : float
        normalisation of the kernel
    cov : np.ndarray
        covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix)
    domain : array_like
        domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are used.

    Returns
    -------
    kernel : np.ndarray
        non-negative kernel array of shape `(...)`
    """
    return norm * np.ones(np.broadcast(x, y).shape[:-1])


def sample_adjacency(coordinates, kernel, condensed=False, distribution='bernoulli'):
    """
    Sample an adjacency matrix.

    Parameters
    ----------
    coordinates : np.ndarray
        coordinates of shape `(n, k)`
    kernel : callable
        kernel function
    condensed : bool
        whether to return a condensed adjacency matrix
    distribution : str
        distribution to draw adjacency samples from

    Returns
    -------
    adjacency : np.ndarray
        sampled adjacency matrix
    """
    i, j = np.triu_indices(coordinates.shape[0], 1)
    kernel = kernel(coordinates[i], coordinates[j])
    if distribution == 'bernoulli':
        assert np.all((kernel >= 0) & (kernel <= 1)), "kernel values must be in the interval [0, 1]"
        adjacency = np.random.uniform(0, 1, kernel.shape) < kernel
    else:
        raise KeyError(distribution)

    if not condensed:
        adjacency = squareform(adjacency)

    return adjacency


def evaluate_expected_degree(connectivity, density, dx):
    """
    Evaluate the expected degree.

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
    expected_degree : np.ndarray or float
        expected degree of nodes (returns a scalar if `density` is a scalar)
    """
    return convolve(density, connectivity, dx)
