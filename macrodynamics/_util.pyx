# cython: boundscheck=False

from libc.math cimport exp
import numpy as np

__GLOBAL_DUMMY = "https://stackoverflow.com/a/13976504/1150961"


def smoothed_sum(points, double[:, :] data, double[:] values, precision):
    """
    smooth_points(double[:, :] data, points, double[:] values, precision)

    Evaluate the smoothed sum of `values` located at `data` over `points`.

    Parameters
    ----------
    points : np.ndarray
        points at which to evaluate the sum with shape `(..., d)`
    data : np.ndarray
        sample coordinates with shape `(n, d)`
    values : np.ndarray
        values associated with the data of shape `n`
    precision : np.ndarray
        precision (matrix) for the Gaussian kernel
    """
    cdef:
        int i, j, k, l, n = data.shape[0], d = data.shape[1]
        double[:, :] _points = np.reshape(points, (-1, d)), _precision
        int m = _points.shape[0]
        double[:] result = np.zeros(m)
        double chi2, norm

    # Make sure the points are valid
    shape = np.shape(points)
    assert shape[-1] == d, "data have dimension %d but points have shape %d" % (d, points.shape[1])

    # Construct scalar values
    if values is None:
        values = np.ones(n)
    assert values.shape[0] == n, "coordinates have length %d but values have length %d" % (n, values.shape[0])

    # Construct a precision matrix
    if np.isscalar(precision):
        _precision = precision * np.eye(d)
    elif np.ndim(precision) == 1:
        _precision = np.diag(precision)
    elif np.ndim(precision) == 2:
        _precision = np.copy(precision)
    else:
        raise ValueError("not a valid precision: %s" % precision)

    # Evaluate the normalisation
    norm = np.sqrt(np.linalg.det(_precision)) / (2 * np.pi) ** (d / 2)

    # Divide the diagonal by two and we can omit the 0.5 in the exponential
    for k in range(d):
        _precision[k, k] /= 2

    # Iterate over points, data, and dimensions
    for j in range(m):
        for i in range(n):
            chi2 = 0
            for k in range(d):
                for l in range(k, d):
                    chi2 += (data[i, k] - _points[j, k]) * _precision[k, l] * (data[i, l] - _points[j, l])

            result[j] += values[i] * exp(-chi2)

    return np.asarray(result).reshape(shape[:-1]) * norm
