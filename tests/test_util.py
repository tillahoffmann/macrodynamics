import itertools as it
import pytest
import macrodynamics as md
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy import sparse


@pytest.mark.parametrize(
    "statistic, precision",
    [("count", 100), ("sum", [100, 50]), ("mean", [[10, 2], [2, 3]]), ("var", 100)],
)
def test_smoothed_statistic(statistic, precision):
    # Generate data
    data = np.random.normal(0, 1, (1000, 2))
    values = np.random.normal(0, 1, 1000)
    x, dx = np.linspace(-3, 3, retstep=True)
    xx, yy = np.meshgrid(x, x)
    points = np.transpose((xx, yy))
    # Evaluate the statistic and check its shape
    actual = md.smoothed_statistic(points, data, values, precision, statistic)
    assert actual.shape == xx.shape, "unexpected shape"
    if statistic == "count":
        np.testing.assert_allclose(np.sum(actual) * dx**2, len(data), 0.01)


@pytest.mark.parametrize(
    "num_dims, roll",
    it.product(
        [1, 2, 3],
        [True, False],
    ),
)
def test_coordinate_tensor(num_dims, roll):
    x = np.linspace(0, 1, 50)
    tensor = md.coordinate_tensor(*[x] * num_dims, roll=roll)
    if roll:
        expected_shape = (50,) * num_dims + (num_dims,)
    else:
        expected_shape = (num_dims,) + (50,) * num_dims
    assert tensor.shape == expected_shape, "unexpected shape"


@pytest.mark.parametrize(
    "shape, norm, alpha",
    it.product(
        [10, (10, 10)], [None, mcolors.Normalize, mcolors.Normalize()], [True, False]
    ),
)
def test_map_colors(shape, norm, alpha):
    x = np.random.uniform(0, 1, shape)
    colors, _ = md.map_colors(x, norm=norm, alpha=x if alpha else None)
    assert colors.shape == x.shape + (4,), "unexpected shape"
    if not alpha:
        np.testing.assert_equal(colors[..., 3], 1, "unexpected alpha values")
    else:
        assert np.min(colors[..., 3]) != np.max(
            colors[..., 3]
        ), "unexpected alpha values"


@pytest.mark.parametrize("newshape, axes", [((10, 20, 30), None), ((11, 13), (1, 2))])
def test_extract_array(newshape, axes):
    arr = np.random.uniform(0, 1, (20, 30, 40))
    # Make sure we have a valid shape
    if axes is None:
        expected_shape = newshape
    else:
        expected_shape = np.array(arr.shape)
        expected_shape[list(axes)] = newshape

    # Extract the array
    actual = md.origin_array(arr, newshape, axes)
    np.testing.assert_equal(actual.shape, expected_shape, "unexpected new shape")


@pytest.mark.parametrize(
    "shape, desired",
    [
        [(10,), (10,)],
        [(11, 13), (12, 15)],
    ],
)
def test_next_fast_shape(shape, desired):
    actual = md.next_fast_shape(shape)
    assert actual == desired, "expected %s but got %s for next fast shape" % (
        desired,
        actual,
    )


@pytest.mark.parametrize(
    "x, axes, homogeneous",
    [
        ([1, 1, 1, 2], None, False),
        (np.diff(np.linspace(0, 1)), None, True),
        ([1], None, True),
        ([[1, 1], [2, 2]], None, False),
        ([[1, 1], [2, 2]], 1, True),
        ([[1, 1], [2, 2]], 0, False),
        ([[1, 1], [2, 3]], 1, [True, False]),
    ],
)
def test_is_uniform(x, axes, homogeneous):
    np.testing.assert_equal(md.is_homogeneous(x, axes), homogeneous)


@pytest.mark.parametrize("next_fast_len", [True, False])
def test_coordinate_tensors(num_lin, num_dims, periodic, next_fast_len):
    if periodic and next_fast_len:
        pytest.skip()
    xi = [np.linspace(0, 1, num_lin, endpoint=not periodic)] * num_dims
    coordinate_tensor, kernel_coordinate_tensor, domain = md.coordinate_tensors(
        *xi, periodic=periodic, next_fast_len=next_fast_len
    )

    assert len(domain) == num_dims, "expected domain vector of size %d but got %d" % (
        num_dims,
        len(domain),
    )
    if periodic:
        np.testing.assert_almost_equal(domain, 1, err_msg="expected domain equal to 1")
    else:
        assert np.all(domain >= 2 * (1 - 1e-7)), "expected domain greater than 2"


@pytest.mark.parametrize(
    "weight", [None, np.ones, lambda x: np.random.uniform(0, 1, x)]
)
def test_edgelist_to_sparse(weight):
    num_nodes = 1000
    edgelist = np.random.randint(0, num_nodes, (10, 2))
    if callable(weight):
        weight = weight(len(edgelist))
    dense = np.zeros((num_nodes, num_nodes))
    i, j = edgelist.T
    dense[i, j] = 1 if weight is None else weight
    sparse_ = md.edgelist_to_sparse(edgelist, num_nodes, weight)
    assert sparse.issparse(sparse_)
    np.testing.assert_allclose(
        sparse_.toarray(), dense, err_msg="unexpected sparse adjacency"
    )


def test_add_leading_dims():
    x = np.random.normal(0, 1, (3, 4, 5, 6))
    y = md.add_leading_dims(x, 3)
    assert y.shape[:3] == (1, 1, 1)
    assert y.shape[3:] == x.shape
    np.testing.assert_equal(x, y[0, 0, 0])


def test_symmetric_vminmax():
    x = np.random.normal(0, 1, (30, 40))
    kwargs = md.symmetric_vminmax(*x)
    assert kwargs["vmax"] == np.abs(x).max()


def test_use_bmat_for_matrix_construction():
    """
    This test does not test any code in the module but is retained to ensure that code
    changes as a result of
    https://stackoverflow.com/questions/55081721/stacking-sparse-matrices do not
    introduce regressions.
    """
    tensor = np.random.normal(0, 1, (5, 5, 10, 10))
    # Use our custom legacy implementation
    assert tensor.ndim == 4, "expected 4D tensor but got %dD" % tensor.ndim
    state_shape = tensor.shape[1:3]
    # Flatten the tensor
    matrix_rank = np.prod(state_shape)
    matrix = np.rollaxis(tensor, 2, 1).reshape((matrix_rank, matrix_rank))
    # Use bmat instead
    blocks = list(map(list, tensor))
    matrix2 = np.block(blocks)
    np.testing.assert_equal(matrix, matrix2)
    # Try the same with a sparse matrix
    blocks = [[sparse.csr_matrix(block) for block in row] for row in blocks]
    matrix3 = sparse.bmat(blocks).toarray()
    np.testing.assert_equal(matrix3, matrix2)


def test_plot_edges():
    plt.subplots()
    x = np.random.normal(0, 1, (100, 2))
    edgelist = np.random.randint(x.shape[0], size=(50, 2))
    collection = md.plot_edges(x, edgelist)
    assert len(collection.get_segments()) == 50


def test_label_axes():
    fig, axes = plt.subplots(2, 2)
    assert len(md.label_axes(axes)) == 4
    assert len(md.label_axes(*axes)) == 4


def test_first_element():
    x = np.random.normal(0, 1, (10, 10))
    np.testing.assert_array_equal(x[:, 0], md.first_element(x, 1, True))
