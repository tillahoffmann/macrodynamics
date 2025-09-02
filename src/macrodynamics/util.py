import functools as ft
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import collections as mcollections
import numpy as np
from scipy.fftpack import next_fast_len
from scipy import sparse, special, spatial


def smoothed_sum(points, data, values, precision):
    """Evaluate the smoothed sum of `values` located at `data` over `points`.

    Args:
        points: Points at which to evaluate the sum with shape `(*spatial_dims, d)`.
        data: Sample coordinates with shape `(n, d)`.
        values: Values associated with the data of shape `(*state_dims, n)`.
        precision: Precision (matrix) for the Gaussian kernel.

    Returns:
        Sum evaluated at the desired points with shape `(*state_dims, *spatial_dims)`.
    """
    points = np.asarray(points)
    data = np.asarray(data)
    values = np.asarray(values)
    precision = np.asarray(precision)

    n, d = np.shape(data)
    *spatial_shape, d_ = np.shape(points)
    *state_shape, n_ = np.shape(values)
    assert d == d_, "data have dimension %d but points have dimension %d" % (d, d_)
    assert n == n_, "data have %d items but values have %d items" % (n, n_)

    if np.ndim(precision) == 0:
        precision = np.eye(d) * precision
    elif np.ndim(precision) == 1:
        precision = np.diag(precision)
    assert np.shape(precision) == (
        d,
        d,
    ), "data have dimension %d but precision has shape %s" % (d, precision.shape)

    # Evaluate the weight with shape (n, np.prod(spatial_shape)) and normalisation
    # constant.
    weight = np.exp(
        -np.square(
            spatial.distance.cdist(
                data,
                np.reshape(points, (-1, d)),
                metric="mahalanobis",
                VI=precision / 2,
            )
        )
    )
    norm = np.sqrt(np.linalg.det(precision / (2 * np.pi)))

    # Evaluate the weighted sum of the state variables with shape
    # (*state_shape, np.prod(spatial_shape))
    result = np.dot(values, weight)
    # Reshape and normalise
    return result.reshape((*state_shape, *spatial_shape)) * norm


def smoothed_statistic(points, data, values, precision, statistic="mean"):
    """Evaluate a smoothed statistic similar to `scipy.stats.binned_statistic_dd`.

    Args:
        points: Points at which to evaluate the statistic with shape `(*spatial_dims, d)`.
        data: Sample coordinates with shape `(n, d)`.
        values: Values associated with the data of shape `(*state_dims, n)`.
        precision: Precision (matrix) for the Gaussian kernel.
        statistic: Statistic to evaluate (one of 'count', 'sum', 'mean', 'var').

    Returns:
        Statistic evaluated at the desired points with shape `(*state_dims, *spatial_dims)`.
    """
    if statistic == "count":
        return smoothed_sum(points, data, np.ones(data.shape[0]), precision)
    elif statistic == "sum":
        return smoothed_sum(points, data, values, precision)
    elif statistic == "mean":
        return smoothed_statistic(
            points, data, values, precision, "sum"
        ) / smoothed_statistic(points, data, values, precision, "count")
    elif statistic == "var":
        _count = smoothed_statistic(points, data, values, precision, "count")
        _sum = smoothed_statistic(points, data, values, precision, "sum")
        _mean = _sum / _count
        _sum2 = smoothed_statistic(points, data, values * values, precision, "sum")
        return _sum2 / _count - _mean * _mean
    else:
        raise KeyError(statistic)


def symmetric_vminmax(*xs):
    """Evaluate keyword arguments for plotting functions taking `vmin` and `vmax` arguments
    such that `vmax` is the maximum of the absolute value of `*xs` and `vmin = -vmax`.

    Args:
        *xs: Value sequence for which to evaluate the maximum absolute value.

    Returns:
        Keyword arguments to pass to a plotting function.
    """
    # Map the maximum of the absolute value over the arguments
    vmax = max([np.max(np.abs(x)) for x in xs])
    return {
        "vmax": vmax,
        "vmin": -vmax,
    }


def coordinate_tensor(*xi, roll=True, **kwargs):
    """Return a coordinate tensor from coordinate vectors.

    Args:
        *xi: One-dimensional coordinate vectors.
        roll: Whether to roll the dimension axis to the last position.
        **kwargs: Keyword arguments passed to `np.meshgrid`.

    Returns:
        Coordinate tensor.
    """
    xi = np.asarray(np.meshgrid(*xi, **kwargs))
    if roll:
        xi = np.rollaxis(xi, 0, np.ndim(xi))
    return xi


def convolve(a: np.ndarray, b: np.ndarray, dx: np.ndarray | float = 1) -> np.ndarray:
    """Convolve two inputs using periodic boundary conditions.

    Args:
        a: First input array.
        b: Second input array.
        dx: Spacing of sample points.

    Returns:
        Convolved inputs.
    """
    # Compute the differential volume element
    dV = np.prod(np.ones(np.ndim(a)) * dx)
    np.testing.assert_array_less(
        a.shape,
        np.asarray(b.shape) + 1,
        "expected first shape to be smaller or equal to the "
        "second but got %s > %s" % (a.shape, b.shape),
    )
    # Take the transform
    axes = tuple(range(b.ndim))
    fft_a = np.fft.rfftn(a, b.shape, axes=axes)
    fft_b = np.fft.rfftn(b, b.shape, axes=axes)
    # Take the inverse transform and normalise
    c = np.fft.irfftn(fft_a * fft_b, b.shape, axes=axes) * dV
    return origin_array(c, a.shape)


def map_colors(x, cmap=None, norm=None, alpha=None, **kwargs):
    """Apply a color map to the input.

    Args:
        x: Input array with shape `(...)`.
        cmap: Color map.
        norm: Normalization type or matplotlib.colors.Normalize instance.
        alpha: Opacity values with shape `(...)`.
        **kwargs: Keyword arguments passed to `norm`.

    Returns:
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
    """Lazy property decorator.

    Args:
        func: Function to decorate as a lazy property.

    Returns:
        Function acting as a lazy property.
    """

    @property
    @ft.wraps(func)
    def _wrapper(self):
        # Try to get the value
        _name = "__lazy_%s" % func.__name__
        if getattr(self, _name, None) is None:
            # Evaluate if necessary
            setattr(self, _name, func(self))
        return getattr(self, _name)

    return _wrapper


def list_fixture(params, ids=None, **kwargs):
    """Shorthand for creating fixtures that return multiple values.

    Args:
        params: List of parameters which will cause multiple invocations of the fixture function
            and all of the tests using it.
        ids: List of string ids each corresponding to the `params` so that they are part of
            the test id. If no `ids` are provided they will be generated automatically from
            the `params`.
        **kwargs: Keyword arguments passed to `pytest.fixture`.

    Returns:
        Fixture with multiple parameter values.
    """
    import pytest

    @pytest.fixture(params=params, ids=ids, **kwargs)
    def _wrapper(request):
        return request.param

    return _wrapper


def origin_array(arr, newshape, axes=None):
    """Return the subarray closest to the origin along the specified axes.

    Args:
        arr: Array to extract the center from.
        newshape: Shape of the region to extract.
        axes: Axes along which to extract the region (default is all axes).

    Returns:
        Array extracted from `arr` in the vicinity of the origin.
    """
    # Get a vector of axes
    if axes is None:
        axes = np.arange(arr.ndim)
    else:
        axes = np.asarray(axes)
    assert len(axes) == len(newshape), (
        "`newshape` has length %d but there are %d axes" % (len(newshape), len(axes))
    )
    # Determine the shape
    shape = np.asarray(arr.shape)
    shape[axes] = newshape
    # Make sure the slice is sensible
    assert np.all(shape <= arr.shape), (
        "expected the output shape to be smaller than or equal to "
        "the input shape %s but got %s" % (arr.shape, shape)
    )
    # Create the slices
    myslice = [slice(0, s) for s in shape]
    return arr[tuple(myslice)]


def next_fast_shape(shape):
    """Compute the next fast FFT shape.

    Args:
        shape: Shape to start searching from.

    Returns:
        The first 5-smooth shape greater than or equal to the input shape.
    """
    return tuple(map(next_fast_len, shape))


def first_element(
    arr: np.ndarray,
    axis: np.ndarray | tuple[int, ...] | int | None = None,
    squeeze: bool | tuple[int, ...] = False,
) -> np.ndarray:
    """Return the first element of an array.

    Args:
        arr: Array for which to return the first element.
        axis: Axis along which to return the first element (default is all axes).
        squeeze: Axis along which to squeeze the first element if possible or `True` to squeeze
            along all axes (default is no axes).

    Returns:
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
    """Check whether all elements of an array are (almost) equal along the given axes.

    Args:
        arr: Array to check for homogeneity.
        axis: Axes along which to check for homogeneity (default is all axes).
        *args: Positional arguments passed to `np.isclose`.
        **kwargs: Keyword arguments passed to `np.isclose`.

    Returns:
        Whether the array is homogeneous along all the specified axes.
    """
    if axis is not None:
        axis = tuple(np.atleast_1d(axis))
    return np.all(np.isclose(arr, first_element(arr, axis), *args, **kwargs), axis=axis)


def coordinate_tensors(
    *xi: np.ndarray, periodic: bool, next_fast_len: bool = True, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate tensors for the field and kernel.

    Args:
        *xi: One-dimensional coordinate vectors.
        periodic: Whether to assume periodic boundary conditions.
        next_fast_len: Whether to use the next best FFT size.
        **kwargs: Keyword arguments passed to `coordinate_tensor`.

    Returns:
        Tuple containing:
            - Coordinate tensor for the field.
            - Coordinate tensor for the kernel (identical to coordinate_tensor if periodic == True).
            - Domain for the periodicity of the kernel with one entry for each dimension.
    """
    dxi = []
    lengths = []
    # Iterate over the different 1D grids
    for i, x in enumerate(xi):
        dx = np.diff(x)
        assert is_homogeneous(dx), (
            "the sample spacing along axis %d is not homogeneous" % i
        )
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
            lengths = np.asarray(next_fast_shape(lengths))

    # Compute the domain size
    domain = lengths * dxi

    # Construct the tensors
    kernel_xi = [dx * np.arange(length) for length, dx in zip(lengths, dxi)]

    return (
        coordinate_tensor(*xi, **kwargs),
        coordinate_tensor(*kernel_xi, **kwargs),
        domain,
    )


def to_array(arr):
    """Convert to a dense array.

    Args:
        arr: Sparse or dense array to convert to a dense array.

    Returns:
        Dense array.
    """
    if sparse.issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def edgelist_to_sparse(edgelist, num_nodes, weight=None):
    """Convert an edgelist to a sparse adjacency matrix.

    Args:
        edgelist: List of edges with shape `(m, 2)` where `m` is the number of edges.
        num_nodes: Number of nodes.
        weight: Weights associated with the edges.

    Returns:
        Sparse adjacency matrix.
    """
    if weight is None:
        weight = np.ones(len(edgelist))
    adjacency = sparse.coo_matrix(
        (weight, np.transpose(edgelist)), (num_nodes, num_nodes)
    )
    return adjacency.tocsr()


def add_leading_dims(x, n):
    """Add `n` leading dimensions of size `1` to `x`.

    Args:
        x: Array to which to add leading dimensions.
        n: Number of leading dimensions to add.

    Returns:
        Array with `n` leading dimensions added.
    """
    x = np.asarray(x)
    return x.reshape((1,) * n + x.shape)


def plot_edges(x, edges, ax=None, **kwargs):
    """Plot edges.

    Args:
        x: Positions of nodes.
        edges: Edgelist.
        ax: Axes in which to plot the edges.
        **kwargs: Keyword arguments passed to the `LineCollection` created by this function.

    Returns:
        Collection of edges.
    """
    default_kwargs = {
        "color": "k",
        "alpha": 0.5,
        "zorder": 1,
    }
    default_kwargs.update(kwargs)
    ax = ax or plt.gca()
    segments = [[x[i], x[j]] for i, j in edges]
    collection = mcollections.LineCollection(segments, **default_kwargs)
    ax.add_collection(collection)
    return collection


def label_axes(*axes, x=0.05, y=0.95, offset=0, labels=None, **kwargs):
    """Attach alphabetical labels to a sequence of axes.

    Args:
        *axes: Sequence of axes to label.
        x: Horizontal label position.
        y: Vertical label position.
        offset: Offset for axis labels.
        labels: Sequence of axis labels.
        **kwargs: Keyword arguments passed to `matplotlib.axes.Axes.text`.

    Returns:
        List of text elements representing labels.
    """
    labels = labels or "abcdefghijklmnopqrstuvwxyz"
    elements = []
    va = kwargs.pop("va", kwargs.pop("verticalalignment", "top"))
    for i, ax in enumerate(np.ravel(axes)):
        elements.append(
            ax.text(
                x, y, f"({labels[i + offset]})", va=va, transform=ax.transAxes, **kwargs
            )
        )
    return elements


def nexpm1(a, x):
    r"""Evaluates :math:`\frac{\exp(a x) - 1}{a}` safely.

    Args:
        a: Scale factors.
        x: Input values.

    Returns:
        Elementwise evaluation of the desired function.
    """
    fltr = a == 0
    return np.where(fltr, x, special.expm1(a * x) / np.where(fltr, 1, a))


def assert_correlated(actual, desired, tol=1e-3):
    """Raises an AssertionError if two objects are not sufficiently linearly correlated.

    Args:
        actual: Array obtained.
        desired: Array desired.
        tol: Tolerance for the correlation coefficient.

    Raises:
        AssertionError: If actual and desired are not sufficiently linearly correlated.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    assert actual.shape == desired.shape, (
        "`actual` has shape %s but `desired` has shape %s"
        % (
            actual.shape,
            desired.shape,
        )
    )
    corrcoef = np.corrcoef(actual.ravel(), desired.ravel())[0, 1]
    delta = 1 - corrcoef
    assert delta < tol, (
        "correlation coefficient %f differs from 1 by %f, exceeding tolerance %f"
        % (
            corrcoef,
            delta,
            tol,
        )
    )


def augment_with_periodic_bc(points, values, domain):
    """Augment the data to create periodic boundary conditions.

    Args:
        points: Tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values: The data on the regular grid in n dimensions.
        domain: The size of the domain along each of the n dimensions or a uniform domain size
            along all dimensions if a scalar. Using None specifies aperiodic boundary
            conditions.

    Returns:
        Tuple containing:
            - The points defining the regular grid in n dimensions with periodic boundary conditions.
            - The data on the regular grid in n dimensions with periodic boundary conditions.
    """
    # Validate the domain argument
    n = len(points)
    if np.ndim(domain) == 0:
        domain = [domain] * n
    if np.shape(domain) != (n,):
        raise ValueError(
            "`domain` must be a scalar or have the same length as `points`"
        )

    # Pre- and append repeated points
    points = [
        x if d is None else np.concatenate([x - d, x, x + d])
        for x, d in zip(points, domain)
    ]

    # Tile the values as necessary
    reps = [1 if d is None else 3 for d in domain]
    values = np.tile(values, reps)

    return points, values
