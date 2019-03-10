import pytest
import numpy as np
from scipy import sparse
import graph_dynamics as gd

from _test_common import _test_integration, _test_integration_shape


# Define global parameters
num_nodes = 100
eps = np.finfo(np.float32).eps

is_sparse = gd.list_fixture([False, True])

STATE = {}

def get_state(request, fixtures, key, default):
    """
    Get shared state across multiple runs of a test with different fixtures.

    Parameters
    ----------
    request :
        request associated with the test run.
    fixtures : list[str]
        the fixture names to key on. If a fixture name does not appear in this list, state will be
        shared across different values of the fixture.
    key : str
        name of the state
    default :
        value to use if no shared state is available
    """
    key = (request.function.__name__, key) + \
        tuple(request.getfixturevalue(fixture) for fixture in fixtures)
    return STATE.setdefault(key, default)


@pytest.fixture
def adjacency(num_dims, kernel, is_sparse):
    # Sample coordinates
    coordinates = np.random.uniform(0, 1, (num_nodes, num_dims))
    adjacency = gd.sample_adjacency(coordinates, kernel)
    if is_sparse:
        adjacency = sparse.csr_matrix(adjacency)
    return adjacency


@pytest.fixture
def discrete_diffusion_operator(adjacency):
    return gd.diffusion.evaluate_discrete_operator(adjacency)


@pytest.fixture
def discrete_averaging_operator(adjacency):
    return gd.averaging.evaluate_discrete_operator(adjacency)


@pytest.fixture
def discrete_oscillation_operator(adjacency):
    return gd.oscillation.evaluate_discrete_operator(adjacency)


def test_discrete_diffusion_operator(discrete_diffusion_operator, request):
    matrix = gd.to_array(discrete_diffusion_operator.matrix)
    # Evaluate the diffusion Laplacian and test its fundamental properties
    np.testing.assert_array_less(np.diag(matrix), eps, "diagonal entries must be non-positive")
    np.testing.assert_array_less(-eps, np.triu(matrix, 1), "upper diagonal must be non-negative")
    np.testing.assert_allclose(matrix, matrix.T, err_msg='operator is not symmetric')

    # Get a cached state if available
    z0 = get_state(request, ['num_dims', 'kernel'], 'z0', np.random.normal(0, 1, (1, num_nodes)))
    grad = discrete_diffusion_operator.evaluate_gradient(z0)
    # Check that the average gradient is close to zero (total is conserved for diffusion)
    assert np.abs(np.mean(grad)) < 1e-10, "average gradient is not close to zero"
    other_grad = get_state(request, ['num_dims', 'kernel'], 'grad', grad)
    np.testing.assert_allclose(grad, other_grad)


def test_discrete_diffusion_integration(discrete_diffusion_operator):
    # Run the simulator
    dt = 0.0001
    num_steps = 10000
    z0, z_actual, _ = _test_integration(discrete_diffusion_operator, time=dt * num_steps)
    # Evolve the state in small steps
    z_desired = np.ravel(z0.copy())
    for _ in range(num_steps):
        z_desired += discrete_diffusion_operator.matrix.dot(z_desired) * dt
    # Ensure conservation
    np.testing.assert_allclose(np.sum(z_desired), np.sum(z0), 1e-5, err_msg="number of walkers not conserved")
    assert np.corrcoef(z_actual.ravel(), z_desired)[0, 1] > 0.99, "naive and `scipy.integrate.odeint` differ"


def test_discrete_diffusion_integration_shape(discrete_diffusion_operator, integration_method, time):
    _test_integration_shape(discrete_diffusion_operator, integration_method, time)


def test_discrete_averaging_operator(discrete_averaging_operator, request):
    matrix = gd.to_array(discrete_averaging_operator.matrix)
    np.testing.assert_allclose(np.diag(matrix), -1, err_msg="diagonal is not minus one")
    np.testing.assert_array_less(-eps, np.triu(matrix, 1), err_msg="upper diagonal must be non-negative")
    np.testing.assert_array_less(-eps, np.tril(matrix, -1), err_msg="lower diagonal must be non-negative")

    # Get a cached state if available
    z0 = get_state(request, ['num_dims', 'kernel'], 'z0', np.random.normal(0, 1, (1, num_nodes)))
    grad = discrete_averaging_operator.evaluate_gradient(z0)
    other_grad = get_state(request, ['num_dims', 'kernel'], 'grad', grad)
    np.testing.assert_allclose(grad, other_grad)


def test_discrete_oscillation_operator(discrete_oscillation_operator, request):
    # Get a cached state if available
    z0 = get_state(request, ['num_dims', 'kernel'], 'z0', np.random.normal(0, 1, (2, num_nodes)))
    grad = discrete_oscillation_operator.evaluate_gradient(z0)
    other_grad = get_state(request, ['num_dims', 'kernel'], 'grad', grad)
    np.testing.assert_allclose(grad, other_grad)
