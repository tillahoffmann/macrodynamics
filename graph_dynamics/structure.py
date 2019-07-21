import numpy as np
from scipy.spatial.distance import squareform
from .util import convolve


def evaluate_distance(x, y, cov=1.0, domain=1.0):
    """
    Evaluate the elementwise Mahalanobis distance between x and y.

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates of shape `(..., k)`..
    y : numpy.ndarray
        Coordinates of shape `(..., k)`..
    cov : numpy.ndarray
        Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
    domain : array_like
        Domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are
        used.

    Returns
    -------
    distance : numpy.ndarray
        Non-negative distance array of shape `(...)`.
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
    x : numpy.ndarray
        Coordinates of shape `(..., k)`..
    y : numpy.ndarray
        Coordinates of shape `(..., k)`..
    norm : float
        Normalisation of the kernel..
    cov : numpy.ndarray
        Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
    domain : array_like
        Domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are
        used.

    Returns
    -------
    kernel : numpy.ndarray
        Non-negative kernel array of shape `(...)`.
    """
    distance = evaluate_distance(x, y, cov, domain)
    return norm * np.exp(-0.5 * distance * distance)


def evaluate_tophat_kernel(x, y, norm, cov, domain=1.0):
    """
    Evaluate a top hat (hard) kernel.

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates of shape `(..., k)`..
    y : numpy.ndarray
        Coordinates of shape `(..., k)`..
    norm : float
        Normalisation of the kernel..
    cov : numpy.ndarray
        Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
    domain : array_like
        Domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are
        used.

    Returns
    -------
    kernel : numpy.ndarray
        Non-negative kernel array of shape `(...)`.
    """
    distance = evaluate_distance(x, y, cov, domain)
    return np.where(distance < 1, norm, 0)


def evaluate_laplace_kernel(x, y, norm, cov, domain=1.0):
    """
    Evaluate a Laplace (exponential) kernel.

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates of shape `(..., k)`.
    y : numpy.ndarray
        Coordinates of shape `(..., k)`.
    norm : float
        Normalisation of the kernel.
    cov : numpy.ndarray
        Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
    domain : array_like
        Domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are
        used.

    Returns
    -------
    kernel : numpy.ndarray
        Non-negative kernel array of shape `(...)`.
    """
    distance = evaluate_distance(x, y, cov, domain)
    return norm * np.exp(-distance)


def evaluate_uniform_kernel(x, y, norm, cov=None, domain=1.0):
    """
    Evaluate a uniform kernel.

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates of shape `(..., k)`.
    y : numpy.ndarray
        Coordinates of shape `(..., k)`.
    norm : float
        Normalisation of the kernel.
    cov : numpy.ndarray
        Covariance matrix (can be a scalar, length-`k` vector or `(k, k)` matrix).
    domain : array_like
        Domain for periodic boundary conditions. If `None`, non-periodic boundary conditions are
        used.

    Returns
    -------
    kernel : numpy.ndarray
        Non-negative kernel array of shape `(...)`.
    """
    return norm * np.ones(np.broadcast(x, y).shape[:-1])


def sample_adjacency(coordinates, kernel, condensed=False, distribution='bernoulli'):
    """
    Sample an adjacency matrix.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinates of shape `(n, k)`.
    kernel : callable
        Kernel function.
    condensed : bool
        Whether to return a condensed adjacency matrix.
    distribution : str
        Distribution to draw adjacency samples from.

    Returns
    -------
    adjacency : numpy.ndarray
        Sampled adjacency matrix.
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
    connectivity : numpy.ndarray
        Evaluated connectivity kernel.
    density : numpy.ndarray or float
        Density of nodes (use a scalar for uniform densities).
    dx : numpy.ndarray or float
        Spacing of sample points.

    Returns
    -------
    expected_degree : numpy.ndarray or float
        Expected degree of nodes (returns a scalar if `density` is a scalar).
    """
    return convolve(density, connectivity, dx)
