import itertools as it
import pytest
import graph_dynamics as gd
import numpy as np
from matplotlib import colors as mcolors
from scipy import sparse


@pytest.mark.parametrize('statistic', ['count', 'sum', 'mean', 'var'])
def test_smoothed_statistic(statistic):
    # Generate data
    data = np.random.normal(0, 1, (1000, 2))
    values = np.random.normal(0, 1, 1000)
    x, dx = np.linspace(-3, 3, retstep=True)
    xx, yy = np.meshgrid(x, x)
    points = np.transpose((xx, yy))
    # Evaluate the statistic and check its shape
    actual = gd.smoothed_statistic(points, data, values, 100, statistic)
    assert actual.shape == xx.shape, "unexpected shape"
    if statistic == 'count':
        np.testing.assert_allclose(np.sum(actual) * dx ** 2, len(data), 0.01)


@pytest.mark.parametrize('num_dims, roll', it.product(
    [1, 2, 3],
    [True, False],
))
def test_coordinate_tensor(num_dims, roll):
    x = np.linspace(0, 1, 50)
    tensor = gd.coordinate_tensor(*[x] * num_dims, roll=roll)
    if roll:
        expected_shape = (50, ) * num_dims + (num_dims, )
    else:
        expected_shape = (num_dims, ) + (50, ) * num_dims
    assert tensor.shape == expected_shape, "unexpected shape"


@pytest.mark.parametrize('shape, norm, alpha', it.product(
    [10, (10, 10)],
    [None, mcolors.Normalize, mcolors.Normalize()],
    [True, False]
))
def test_map_colors(shape, norm, alpha):
    x = np.random.uniform(0, 1, shape)
    colors, _ = gd.map_colors(x, norm=norm, alpha=x if alpha else None)
    assert colors.shape == x.shape + (4,), "unexpected shape"
    if not alpha:
        np.testing.assert_equal(colors[..., 3], 1, "unexpected alpha values")
    else:
        assert np.min(colors[..., 3]) != np.max(colors[..., 3]), "unexpected alpha values"


@pytest.mark.parametrize('newshape, axes', [
    ((10, 20, 30), None),
     ((11, 13), (1, 2))
])
def test_extract_array(newshape, axes):
    arr = np.random.uniform(0, 1, (20, 30, 40))
    # Make sure we have a valid shape
    if axes is None:
        expected_shape = newshape
    else:
        expected_shape = np.array(arr.shape)
        expected_shape[list(axes)] = newshape

    # Extract the array
    actual = gd.origin_array(arr, newshape, axes)
    np.testing.assert_equal(actual.shape, expected_shape, "unexpected new shape")


@pytest.mark.parametrize('shape, desired', [
    [(10,), (10,)],
    [(11, 13), (12, 15)],
])
def test_next_fast_shape(shape, desired):
    actual = gd.next_fast_shape(shape)
    assert actual == desired, "expected %s but got %s for next fast shape" % (desired, actual)


@pytest.mark.parametrize('x, axes, homogeneous', [
    ([1, 1, 1, 2], None, False),
    (np.diff(np.linspace(0, 1)), None, True),
    ([1], None, True),
    ([[1, 1], [2, 2]], None, False),
    ([[1, 1], [2, 2]], 1, True),
    ([[1, 1], [2, 2]], 0, False),
    ([[1, 1], [2, 3]], 1, [True, False]),
])
def test_is_uniform(x, axes, homogeneous):
    np.testing.assert_equal(gd.is_homogeneous(x, axes), homogeneous)


@pytest.mark.parametrize('next_fast_len', [True, False])
def test_coordinate_tensors(num_lin, num_dims, periodic, next_fast_len):
    if periodic and next_fast_len:
        pytest.skip()
    xi = [np.linspace(0, 1, num_lin, endpoint=not periodic)] * num_dims
    coordinate_tensor, kernel_coordinate_tensor, domain = gd.coordinate_tensors(
        *xi, periodic=periodic, next_fast_len=next_fast_len
    )

    assert len(domain) == num_dims, "expected domain vector of size %d but got %d" % (num_dims, len(domain))
    if periodic:
        np.testing.assert_almost_equal(domain, 1, err_msg="expected domain equal to 1")
    else:
        assert np.all(domain >= 2 * (1 - 1e-7)), "expected domain greater than 2"


@pytest.mark.parametrize('weight', [None, np.ones, lambda x: np.random.uniform(0, 1, x)])
def test_edgelist_to_sparse(weight):
    num_nodes = 1000
    edgelist = np.random.randint(0, num_nodes, (10, 2))
    if callable(weight):
        weight = weight(len(edgelist))
    dense = np.zeros((num_nodes, num_nodes))
    i, j = edgelist.T
    dense[i, j] = 1 if weight is None else weight
    sparse_ = gd.edgelist_to_sparse(edgelist, num_nodes, weight)
    assert sparse.issparse(sparse_)
    np.testing.assert_allclose(sparse_.toarray(), dense, err_msg='unexpected sparse adjacency')


def test_add_leading_dims():
    x = np.random.normal(0, 1, (3, 4, 5, 6))
    y = gd.add_leading_dims(x, 3)
    assert y.shape[:3] == (1, 1, 1)
    assert y.shape[3:] == x.shape
    np.testing.assert_equal(x, y[0, 0, 0])


def test_symmetric_vminmax():
    x = np.random.normal(0, 1, (30, 40))
    kwargs = gd.symmetric_vminmax(*x)
    assert kwargs['vmax'] == np.abs(x).max()
