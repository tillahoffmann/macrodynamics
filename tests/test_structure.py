import itertools as it
import functools as ft
import macrodynamics as md
import pytest
import numpy as np

# Define global parameters
eps = np.finfo(np.float32).eps


@pytest.mark.parametrize('num_dims, kernel, periodic, condensed', it.product(
    [1, 3],
    [md.evaluate_gaussian_kernel, md.evaluate_laplace_kernel, md.evaluate_tophat_kernel, md.evaluate_uniform_kernel],
    [True, False],
    [True, False],
))
def test_sample_adjacency(num_dims, kernel, periodic, condensed):
    num_nodes = 100
    coordinates = np.random.uniform(0, 1, (num_nodes, num_dims))
    kernel = ft.partial(kernel, norm=0.5, cov=1, domain=1.0 if periodic else None)
    adjacency = md.sample_adjacency(coordinates, kernel, condensed=condensed)
    expected_shape = (num_nodes * (num_nodes - 1) // 2,) if condensed else (num_nodes, num_nodes)
    assert adjacency.shape == expected_shape, "unexpected shape for adjacency matrix"


@pytest.mark.parametrize('num_dims, periodic', it.product(
    [1, 2, 3],
    [True, False, 3]
))
def test_evaluate_distance(num_dims, periodic):
    # Create a grid
    x = np.linspace(0, max(periodic, 1), endpoint=False)
    xx = md.coordinate_tensor(*[x] * num_dims)
    # Evaluate the distance to the origin
    distance = md.evaluate_distance(0, xx, domain=(periodic * np.ones(num_dims)) if periodic else None)
    # Check basic properties
    assert distance.shape == (len(x), ) * num_dims, "unexpected shape"
    np.testing.assert_array_less(-eps, distance, "distance is negative")
    # Test the maximum distance
    scale = (0.5 * periodic) if periodic else 1
    max_distance = scale * np.sqrt(num_dims)
    np.testing.assert_array_less(distance, max_distance + eps,
                                 "distance is too large (%f > %f)" % (np.max(distance), max_distance))
