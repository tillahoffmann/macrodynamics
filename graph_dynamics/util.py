import functools as ft
import warnings
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import collections as mcollections
import numpy as np
from scipy.fftpack.helper import next_fast_len
from scipy import sparse
from ._util import smoothed_sum


def smoothed_statistic(points, data, values, precision, statistic='mean'):
    """
    Evaluate a smoothed statistic similar to `scipy.stats.binned_statistic_dd`.

    Parameters
    ----------
    points : np.ndarray
        points at which to evaluate the statistic with shape `(..., d)`
    data : np.ndarray
        sample coordinates with shape `(n, d)`
    values : np.ndarray
        values associated with the data of shape `n`
    precision : np.ndarray
        precision (matrix) for the Gaussian kernel
    statistic : np.ndarray
        statistic to evaluate (one of 'count', 'sum', 'mean', 'var')
    """
    if statistic == 'count':
        return smoothed_sum(points, data, np.ones(data.shape[0]), precision)
    elif statistic == 'sum':
        return smoothed_sum(points, data, values, precision)
    elif statistic == 'mean':
        return smoothed_statistic(points, data, values, precision, 'sum') / \
            smoothed_statistic(points, data, values, precision, 'count')
    elif statistic == 'var':
        _count = smoothed_statistic(points, data, values, precision, 'count')
        _sum = smoothed_statistic(points, data, values, precision, 'sum')
        _mean = _sum / _count
        _sum2 = smoothed_statistic(points, data, values * values, precision, 'sum')
        return _sum2 / _count - _mean * _mean
    else:
        raise KeyError(statistic)


def symmetric_vminmax(*x):
    """
    Evaluate keyword arguments for plotting functions taking `vmin` and `vmax` arguments.
    """
    # Map the maximum of the absolute value over the arguments
    vmax = max([np.max(np.abs(_x)) for _x in x])
    return {
        'vmax': vmax,
        'vmin': -vmax,
    }


def coordinate_tensor(*xi, roll=True, **kwargs):
    """
    Return a coordinate tensor from coordinate vectors.

    Parameters
    ----------
    xi : array_like
        one-dimensional coordinate vectors
    roll : bool
        whether to roll the dimension axis to the last position
    kwargs : dict
        keyword arguments passed to `np.meshgrid`

    Returns
    -------
    tensor : np.ndarray
        coordinate tensor
    """
    xi = np.asarray(np.meshgrid(*xi, **kwargs))
    if roll:
        xi = np.rollaxis(xi, 0, np.ndim(xi))
    return xi


def convolve(a, b, dx=1):
    """
    Convolve two inputs using periodic boundary conditions.

    Parameters
    ----------
    a : np.ndarray
        first input
    b : np.ndarray
        second input
    dx : np.ndarray or float
        spacing of sample points

    Returns
    -------
    c : np.ndarray
        convolved inputs
    """
    # Compute the differential volume element
    dV = np.prod(np.ones(np.ndim(a)) * dx)
    np.testing.assert_array_less(a.shape, np.asarray(b.shape) + 1, "expected first shape to be smaller or equal to the "
                                 "second but got %s > %s" % (a.shape, b.shape))
    # Take the transform
    fft_a = np.fft.rfftn(a, b.shape)
    fft_b = np.fft.rfftn(b, b.shape)
    # Take the inverse transform and normalise
    c = np.fft.irfftn(fft_a * fft_b, b.shape) * dV
    return origin_array(c, a.shape)


def map_colors(x, cmap=None, norm=None, alpha=None, **kwargs):
    """
    Apply a color map to the input.

    Parameters
    ----------
    x : np.ndarray
        input array with shape `(...)`
    cmap : str
        color map
    norm : type or matplotlib.colors.Normalize
        normalization
    alpha : np.ndarray
        opacity values with shape `(...)`
    kwargs : dict
        keyword arguments passed to `norm`

    Returns
    -------
    colors : np.ndarray
        array of RGBA colors with shape `(..., 4)`
    """
    if norm is None:
        norm = mcolors.Normalize
    if isinstance(norm, type):
        norm = norm(**kwargs)

    mappable = cm.ScalarMappable(norm, cmap)
    mappable.set_array(x)
    colors = mappable.to_rgba(x)
    if alpha is not None:
        colors[..., 3] = alpha / np.max(alpha)
    return colors, mappable


def lazy_property(func):
    """
    Lazy property decorator.
    """
    @property
    @ft.wraps(func)
    def _wrapper(self):
        # Try to get the value
        _name = '__lazy_%s' % func.__name__
        if getattr(self, _name, None) is None:
            # Evaluate if necessary
            setattr(self, _name, func(self))
        return getattr(self, _name)
    return _wrapper


def list_fixture(params, ids=None, **kwargs):
    """
    Shorthand for creating fixtures that return multiple values.
    """
    import pytest

    @pytest.fixture(params=params, ids=ids, **kwargs)
    def _wrapper(request):
        return request.param
    return _wrapper


def origin_array(arr, newshape, axes=None):
    """
    Return the subarray closest to the origin along the specified axes.

    Parameters
    ----------
    arr : np.ndarray
        array to extract the center from
    newshape : tuple
        shape of the region to extract
    axes : tuple
        axes along which to extract the region (default is all axes)

    Returns
    -------
    origin : np.ndarray
        array extracted from `arr` in the vicinity of the origin
    """
    # Get a vector of axes
    if axes is None:
        axes = np.arange(arr.ndim)
    else:
        axes = np.asarray(axes)
    assert len(axes) == len(newshape), "`newshape` has length %d but there are %d axes" % (len(newshape), len(axes))
    # Determine the shape
    shape = np.asarray(arr.shape)
    shape[axes] = newshape
    # Make sure the slice is sensible
    assert np.all(shape <= arr.shape), "expected the output shape to be smaller than or equal to the input shape %s " \
        "but got %s" % (arr.shape, shape)
    # Create the slices
    myslice = [slice(0, s) for s in shape]
    return arr[tuple(myslice)]


def next_fast_shape(shape):
    """
    Compute the next fast FFT shape.

    Parameters
    ----------
    shape : tuple
        shape to start searching from

    Returns
    -------
    fast_shape : tuple
        the first 5-smooth shape greater than or equal to the input shape
    """
    return tuple(map(next_fast_len, shape))


def first_element(arr, axis=None, squeeze=False):
    """
    Return the first element of an array.

    Parameters
    ----------
    arr : np.ndarray
        array for which to return the first element
    axis : tuple
        axis along which to return the first element
    """
    arr = np.asarray(arr)
    if axis is None:
        axis = np.arange(arr.ndim)
    else:
        axis = np.atleast_1d(axis)
    tup = tuple([slice(0, 1) if i in axis else slice(None) for i in range(arr.ndim)])
    arr = arr[tup]
    if squeeze is True:
        arr = np.squeeze(arr)
    elif squeeze is not False:
        squeeze = tuple(np.atleast_1d(squeeze))
        arr = np.squeeze(arr, squeeze)
    return arr


def is_homogeneous(arr, axis=None, *args, **kwargs):
    """
    Check whether all elements of an array are (almost) equal along the given dimensions.
    """
    if axis is not None:
        axis = tuple(np.atleast_1d(axis))
    return np.all(np.isclose(arr, first_element(arr, axis), *args, **kwargs), axis=axis)


def coordinate_tensors(*xi, periodic, next_fast_len=True, **kwargs):
    """
    Return coordinate tensors for the field and kernel.

    Parameters
    ----------
    xi : array_like
        one-dimensional coordinate vectors
    periodic : bool
        whether to assume periodic boundary conditions
    next_fast_len : bool
        whether to use the next best FFT size
    kwargs : dict
        keyword arguments passed to `coordinate_tensor`

    Returns
    -------
    coordinate_tensor : np.ndarray
        coordinate tensor for the field
    kernel_coordinate_tensor : np.ndarray
        coordinate tensor for the kernel (identical to `coordinate_tensor` if `periodic == True`)
    domain : np.ndarray
        domain for the periodicity of the kernel with one entry for each dimension
    """
    dxi = []
    lengths = []
    # Iterate over the different 1D grids
    for i, x in enumerate(xi):
        dx = np.diff(x)
        assert is_homogeneous(dx), "the sample spacing along axis %d is not homogeneous" % i
        dxi.append(dx[0])
        lengths.append(len(x))

    # Convert to arrays
    lengths = np.asarray(lengths)
    dxi = np.asarray(dxi)

    # Expand the domain
    if not periodic:
        lengths = 2 * (lengths - 1)
        # Use a size that is good for FFTs
        if next_fast_len:
            lengths = next_fast_shape(lengths)

    # Compute the domain size
    domain = lengths * dxi

    # Construct the tensors
    kernel_xi = [dx * np.arange(length) for length, dx in zip(lengths, dxi)]

    return coordinate_tensor(*xi, **kwargs), coordinate_tensor(*kernel_xi, **kwargs), domain


def to_array(arr):
    """
    Convert to a dense array.
    """
    if sparse.issparse(arr):
        return arr.toarray()
    return arr


def edgelist_to_sparse(edgelist, shape=None, weight=None):
    """
    Convert an edgelist to a sparse adjacency matrix.

    Parameters
    ----------
    edgelist : np.ndarray
        list of edges with shape `(m, 2)` where `m` is the number of edges
    shape : int or tuple
        shape of the adjacency matrix
    weight : np.ndarray
        weights associated with the edges

    Returns
    -------
    sparse : csr_matrix
        sparse adjacency matrix
    """
    if isinstance(shape, int):
        shape = (shape, shape)
    if weight is None:
        weight = np.ones(len(edgelist))
    adjacency = sparse.coo_matrix((weight, np.transpose(edgelist)), shape)
    return adjacency.tocsr()


def add_leading_dims(x, n):
    """
    Add `n` leading dimensions of size `1` to `x`.
    """
    x = np.asarray(x)
    return x.reshape((1,) * n + x.shape)


def plot_edges(x, edges, ax=None, **kwargs):
    """
    Plot edges.

    Parameters
    ----------
    x : np.ndarray
        positions of nodes
    edges : np.ndarray
        edgelist
    **kwargs : dict
        parameters passed to the `LineCollection` created by this function

    Returns
    -------
    collection : LineCollection
        collection of edges
    """
    default_kwargs = {
        'color': 'k',
        'alpha': 0.5,
        'zorder': 1,
    }
    default_kwargs.update(kwargs)
    ax = ax or plt.gca()
    segments = [[x[i], x[j]] for i, j in edges]
    collection = mcollections.LineCollection(segments, **default_kwargs)
    ax.add_collection(collection)
    return collection


def label_axes(*axes, x=0.05, y=0.95, va='top', offset=0, labels=None, **kwargs):
    """
    Attach alphabetical labels to a sequence of axes.
    """
    labels = labels or 'abcdefghijklmnopqrstuvwxyz'
    elements = []
    for i, ax in enumerate(np.ravel(axes)):
        elements.append(ax.text(x, y, f'({labels[i + offset]})', va=va, transform=ax.transAxes,
                        **kwargs))
    return elements


def ignore_scipy_issue_9093(function):
    """
    Ignore warnings generated by https://github.com/scipy/scipy/issues/9093.
    """
    @ft.wraps(function)
    def _wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "the matrix subclass is not the recommended way")
            return function(*args, **kwargs)
    return _wrapper
