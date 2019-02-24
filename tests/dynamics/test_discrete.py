import pytest
import numpy as np
from scipy import sparse
import graph_dynamics as gd

from _test_common import _test_integration, _test_integration_shape


# Define global parameters
num_nodes = 100
eps = np.finfo(np.float32).eps


@pytest.fixture
def adjacency(num_dims, kernel):
    # Sample coordinates
    coordinates = np.random.uniform(0, 1, (num_nodes, num_dims))
    return gd.sample_adjacency(coordinates, kernel)


@pytest.fixture
def discrete_diffusion_operator(adjacency):
    return gd.diffusion.evaluate_discrete_operator(adjacency)


@pytest.fixture
def discrete_averaging_operator(adjacency):
    return gd.averaging.evaluate_discrete_operator(adjacency)


@pytest.fixture
def sparse_discrete_diffusion_operator(adjacency):
    return gd.diffusion.evaluate_discrete_operator(sparse.csr_matrix(adjacency))


@pytest.fixture
def sparse_discrete_averaging_operator(adjacency):
    return gd.averaging.evaluate_discrete_operator(sparse.csr_matrix(adjacency))


def test_discrete_diffusion_operator(discrete_diffusion_operator, sparse_discrete_diffusion_operator):
    for op in [discrete_diffusion_operator, sparse_discrete_diffusion_operator]:
        matrix = gd.to_array(op.matrix)
        # Evaluate the diffusion Laplacian and test its fundamental properties
        np.testing.assert_array_less(np.diag(matrix), eps, "diagonal entries must be non-positive")
        np.testing.assert_array_less(-eps, np.triu(matrix, 1), "upper diagonal must be non-negative")
        np.testing.assert_allclose(matrix, matrix.T, err_msg='operator is not symmetric')


def test_discrete_averaging_operator(discrete_averaging_operator, sparse_discrete_averaging_operator):
    for op in [discrete_averaging_operator, sparse_discrete_averaging_operator]:
        matrix = gd.to_array(op.matrix)
        np.testing.assert_allclose(np.diag(matrix), -1, err_msg="diagonal is not minus one")
        np.testing.assert_array_less(-eps, np.triu(matrix, 1), err_msg="upper diagonal must be non-negative")
        np.testing.assert_array_less(-eps, np.tril(matrix, -1), err_msg="lower diagonal must be non-negative")


def test_discrete_diffusion_gradient(discrete_diffusion_operator, sparse_discrete_diffusion_operator):
    z0 = np.random.normal(0, 1, (1, num_nodes))
    grads = []
    for op in [discrete_diffusion_operator, sparse_discrete_diffusion_operator]:
        grad = op.evaluate_gradient(z0)
        # Check that the average gradient is close to zero (total is conserved for diffusion)
        assert np.abs(np.mean(grad)) < 1e-10, "average gradient is not close to zero"
        grads.append(grad)

    np.testing.assert_allclose(*grads, err_msg="sparse and dense gradient mismatch")
    assert sparse.issparse(sparse_discrete_diffusion_operator.matrix)


def test_discrete_diffusion_integration(discrete_diffusion_operator, sparse_discrete_diffusion_operator):
    for op in [discrete_diffusion_operator, sparse_discrete_diffusion_operator]:
        # Run the simulator
        dt = 0.0001
        num_steps = 10000
        z0, z_actual, _ = _test_integration(op, time=dt * num_steps)
        # Evolve the state in small steps
        z_desired = np.ravel(z0.copy())
        for _ in range(num_steps):
            z_desired += op.matrix.dot(z_desired) * dt
        # Ensure conservation
        np.testing.assert_allclose(np.sum(z_desired), np.sum(z0), 1e-5, err_msg="number of walkers not conserved")
        assert np.corrcoef(z_actual.ravel(), z_desired)[0, 1] > 0.99, "naive and `scipy.integrate.odeint` differ"


def test_discrete_diffusion_integration_shape(discrete_diffusion_operator, integration_method, time):
    _test_integration_shape(discrete_diffusion_operator, integration_method, time)
