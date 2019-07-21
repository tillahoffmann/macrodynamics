import functools as ft
import warnings
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import collections as mcollections
import numpy as np
from scipy.fftpack.helper import next_fast_len
from scipy import sparse, special
from ._util import smoothed_sum


def smoothed_statistic(points, data, values, precision, statistic='mean'):
    """
    Evaluate a smoothed statistic similar to `scipy.stats.binned_statistic_dd`.

    Parameters
    ----------
    points : numpy.ndarray
        Points at which to evaluate the statistic with shape `(..., d)`.
    data : numpy.ndarray
        Sample coordinates with shape `(n, d)`.
    values : numpy.ndarray
        Values associated with the data of shape `n`.
    precision : numpy.ndarray
        Precision (matrix) for the Gaussian kernel.
    statistic : numpy.ndarray
        Statistic to evaluate (one of 'count', 'sum', 'mean', 'var').

    Returns
    -------
    statistic : numpy.ndarray
        Statistic evaluated at the desired points.
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


def symmetric_vminmax(*xs):
    """
    Evaluate keyword arguments for plotting functions taking `vmin` and `vmax` arguments such that
    `vmax` is the maximum of the absolute value of `*xs` and `vmin = -vmax`.

    Parameters
    ----------
    *xs : list
        Value sequence for which to evaluate the maximum absolute value.

    Returns
    -------
    kwargs : dict
        Keyword arguments to pass to a plotting function.
    """
    # Map the maximum of the absolute value over the arguments
    vmax = max([np.max(np.abs(x)) for x in xs])
    return {
        'vmax': vmax,
        'vmin': -vmax,
    }


def coordinate_tensor(*xi, roll=True, **kwargs):
    """
    Return a coordinate tensor from coordinate vectors.

    Parameters
    ----------
    *xi : array_like
        One-dimensional coordinate vectors.
    roll : bool
        Whether to roll the dimension axis to the last position.
    **kwargs : dict
        Keyword arguments passed to `np.meshgrid`.

    Returns
    -------
    tensor : numpy.ndarray
        Coordinate tensor.
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
    a : numpy.ndarray
        First input array.
    b : numpy.ndarray
        Second input array.
    dx : numpy.ndarray or float
        Spacing of sample points.

    Returns
    -------
    c : numpy.ndarray
        Convolved inputs.
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
    x : numpy.ndarray
        Input array with shape `(...)`.
    cmap : str
        Color map.
    norm : type or matplotlib.colors.Normalize
        Normalization.
    alpha : numpy.ndarray
        Opacity values with shape `(...)`.
    **kwargs : dict
        Keyword arguments passed to `norm`.

    Returns
    -------
    colors : numpy.ndarray
        Array of RGBA colors with shape `(..., 4)`.
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

    Parameters
    ----------
    func : callable
        Function to decorate as a lazy property.

    Returns
    -------
    decorated : property
        Function acting as a lazy property.
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

    Parameters
    ----------
    params : list
        List of parameters which will cause multiple invocations of the fixture function and all of
        the tests using it.
    ids : list
        List of string ids each corresponding to the `params` so that they are part of the test id.
        If no `ids` are provided they will be generated automatically from the `params`.
    **kwargs : dict
        Keyword arguments passed to `pytest.fixture`.

    Returns
    -------
    fixture : pytest.fixture
        Fixture with multiple parameter values.
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
    arr : numpy.ndarray
        Array to extract the center from.
    newshape : tuple
        Shape of the region to extract.
    axes : tuple
        Axes along which to extract the region (default is all axes).

    Returns
    -------
    origin : numpy.ndarray
        Array extracted from `arr` in the vicinity of the origin.
    """
    # Get a vector of axes
    if axes is None:
        axes = np.arange(arr.ndim)
    else:
        axes = np.asarray(axes)
    assert len(axes) == len(newshape), "`newshape` has length %d but there are %d axes" % \
        (len(newshape), len(axes))
    # Determine the shape
    shape = np.asarray(arr.shape)
    shape[axes] = newshape
    # Make sure the slice is sensible
    assert np.all(shape <= arr.shape), "expected the output shape to be smaller than or equal to " \
        "the input shape %s but got %s" % (arr.shape, shape)
    # Create the slices
    myslice = [slice(0, s) for s in shape]
    return arr[tuple(myslice)]


def next_fast_shape(shape):
    """
    Compute the next fast FFT shape.

    Parameters
    ----------
    shape : tuple
        Shape to start searching from.

    Returns
    -------
    fast_shape : tuple
        The first 5-smooth shape greater than or equal to the input shape.
    """
    return tuple(map(next_fast_len, shape))


def first_element(arr, axis=None, squeeze=False):
    """
    Return the first element of an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to return the first element.
    axis : tuple
        Axis along which to return the first element (default is all axes).
    squeeze : tuple or bool
        Axis along which to squeeze the first element if possible or `True` to squeeze along all
        axes (default is no axes).

    Returns
    -------
    element : numpy.ndarray
        First element along the specified `axes`.
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
    Check whether all elements of an array are (almost) equal along the given axes.

    Parameters
    ----------
    arr : array_like
        Array to check for homogeneity.
    axis : tuple
        Axes along which to check for homogeneity (default is all axes).
    *args : list
        Positional arguments passed to `np.isclose`.
    **kwargs : dict
        Keyword arguments passed to `np.isclose`.

    Returns
    -------
    homogeneous : bool
        Whether the array is homogeneous along all the specified axes.
    """
    if axis is not None:
        axis = tuple(np.atleast_1d(axis))
    return np.all(np.isclose(arr, first_element(arr, axis), *args, **kwargs), axis=axis)


def coordinate_tensors(*xi, periodic, next_fast_len=True, **kwargs):
    """
    Return coordinate tensors for the field and kernel.

    Parameters
    ----------
    *xi : array_like
        One-dimensional coordinate vectors.
    periodic : bool
        Whether to assume periodic boundary conditions.
    next_fast_len : bool
        Whether to use the next best FFT size.
    **kwargs : dict
        Keyword arguments passed to `coordinate_tensor`.

    Returns
    -------
    coordinate_tensor : numpy.ndarray
        Coordinate tensor for the field.
    kernel_coordinate_tensor : numpy.ndarray
        Coordinate tensor for the kernel (identical to `coordinate_tensor` if `periodic == True`).
    domain : numpy.ndarray
        Domain for the periodicity of the kernel with one entry for each dimension.
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

    Parameters
    ----------
    arr : array_like
        Sparse or dense array to convert to a dense array.

    Returns
    -------
    dense : numpy.ndarray
        Dense array.
    """
    if sparse.issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def edgelist_to_sparse(edgelist, num_nodes, weight=None):
    """
    Convert an edgelist to a sparse adjacency matrix.

    Parameters
    ----------
    edgelist : numpy.ndarray
        List of edges with shape `(m, 2)` where `m` is the number of edges.
    num_nodes : int
        Number of nodes.
    weight : numpy.ndarray
        Weights associated with the edges.

    Returns
    -------
    sparse : csr_matrix
        Sparse adjacency matrix.
    """
    if weight is None:
        weight = np.ones(len(edgelist))
    adjacency = sparse.coo_matrix((weight, np.transpose(edgelist)), (num_nodes, num_nodes))
    return adjacency.tocsr()


def add_leading_dims(x, n):
    """
    Add `n` leading dimensions of size `1` to `x`.

    Parameters
    ----------
    x : array_like
        Array to which to add leading dimensions.
    n : int
        Number of leading dimensions to add.

    Returns
    -------
    y : numpy.ndarray
        Array with `n` leading dimensions added.
    """
    x = np.asarray(x)
    return x.reshape((1,) * n + x.shape)


def plot_edges(x, edges, ax=None, **kwargs):
    """
    Plot edges.

    Parameters
    ----------
    x : numpy.ndarray
        Positions of nodes.
    edges : numpy.ndarray
        Edgelist.
    ax : matplotlib.axes.Axes
        Axes in which to plot the edges.
    **kwargs : dict
        Keyword arguments passed to the `LineCollection` created by this function.

    Returns
    -------
    collection : LineCollection
        Collection of edges.
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


def label_axes(*axes, x=0.05, y=0.95, offset=0, labels=None, **kwargs):
    """
    Attach alphabetical labels to a sequence of axes.

    Parameters
    ----------
    *axes : list
        Sequence of axes to label.
    x : float
        Horizontal label position.
    y : float
        Vertical label position.
    offset : int
        Offset for axis labels.
    labels : iterable
        Sequence of axis labels.
    **kwargs : dict
        Keyword arguments passed to `matplotlib.axes.Axes.text`.

    Returns
    -------
    elements : list
        List of text elements representing labels.
    """
    labels = labels or 'abcdefghijklmnopqrstuvwxyz'
    elements = []
    va = kwargs.pop('va', kwargs.pop('verticalalignment', 'top'))
    for i, ax in enumerate(np.ravel(axes)):
        elements.append(ax.text(x, y, f'({labels[i + offset]})', va=va, transform=ax.transAxes,
                        **kwargs))
    return elements


def _ignore_scipy_issue_9093(function):
    """
    Ignore warnings generated by https://github.com/scipy/scipy/issues/9093.
    """
    @ft.wraps(function)
    def _wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "the matrix subclass is not the recommended way")
            return function(*args, **kwargs)
    return _wrapper


def nexpm1(a, x):
    r"""
    Evaluates :math:`\frac{\exp(a x) - 1}{a}` safely.

    Parameters
    ----------
    a : numpy.ndarray
        Scale factors.
    x : numpy.ndarray
        Input values.

    Returns
    -------
    y : numpy.ndarray
        Elementwise evaluation of the desired function.
    """
    fltr = a == 0
    return np.where(fltr, x, special.expm1(a * x) / np.where(fltr, 1, a))


def assert_correlated(actual, desired, tol=1e-3):
    """
    Raises an AssertionError if two objects are not sufficiently linearly correlated.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    tol : float
        Tolerance for the correlation coefficient.

    Raises
    ------
    AssertionError
        If actual and desired are not sufficiently linearly correlated.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    assert actual.shape == desired.shape, "`actual` has shape %s but `desired` has shape %s" % \
        (actual.shape, desired.shape)
    corrcoef = np.corrcoef(actual.ravel(), desired.ravel())[0, 1]
    delta = 1 - corrcoef
    assert delta < tol, "correlation coefficient %f differs from 1 by %f, exceeding tolerance %f" % \
        (corrcoef, delta, tol)
