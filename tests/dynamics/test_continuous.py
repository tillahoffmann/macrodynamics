import pytest
import numpy as np
import graph_dynamics as gd

from _test_common import _test_integration, _test_integration_shape

# Declare constants
num_nodes = 100
eps = np.finfo(np.float32).eps


# Declare fixtures
homogeneous = gd.list_fixture([True, False], ['homogeneous', 'inhomogeneous'])


@pytest.fixture
def lin_x(periodic, num_lin):
    return np.linspace(0, 1, num_lin, endpoint=not periodic)


@pytest.fixture
def lin_dx(lin_x):
    return lin_x[1] - lin_x[0]


@pytest.fixture
def coordinate_tensor(lin_x, num_dims):
    return gd.coordinate_tensor(*[lin_x] * num_dims)


@pytest.fixture
def connectivity(kernel, lin_x, periodic, num_dims):
    _, kernel_coordinate_tensor, domain = gd.coordinate_tensors(*[lin_x] * num_dims, periodic=periodic)
    return kernel(0, kernel_coordinate_tensor, domain=domain)


@pytest.fixture
def density(num_dims, coordinate_tensor, homogeneous, lin_dx):
    if homogeneous:
        density = np.ones(coordinate_tensor.shape[:-1])
    else:
        # Evaluate the distance from the origin
        distance = gd.evaluate_distance(0, coordinate_tensor, domain=1.0)
        # Create a density
        density = num_dims / 2 - distance ** 2
    # Normalise it
    density *= num_nodes / (np.sum(density) * lin_dx ** num_dims)
    return density


@pytest.fixture
def initial_conditions(coordinate_tensor, num_dims):
    center = np.random.uniform(0, 1, num_dims)
    ic = gd.evaluate_gaussian_kernel(coordinate_tensor, center, 1, 0.5 ** 2, domain=None)
    return ic[None]  # Add the first dimension


@pytest.fixture
def continuous_diffusion_operator(connectivity, density, lin_dx):
    return gd.diffusion.evaluate_continuous_operator(connectivity, density, lin_dx)


@pytest.fixture
def continuous_averaging_operator(connectivity, density, lin_dx):
    return gd.averaging.evaluate_continuous_operator(connectivity, density, lin_dx)


@pytest.fixture
def continuous_oscillation_operator(connectivity, density, lin_dx):
    return gd.oscillation.evaluate_continuous_operator(connectivity, density, lin_dx)


def test_operator_shape_validation():
    with pytest.raises(ValueError):
        gd.ContinuousOperator(
            weight=np.empty((1, 1, 10)),
            kernel_weight_x=np.empty((1, 1, 10)),
            kernel_weight_y=np.empty((1, 1, 10)),
            kernel=np.empty((1, 1, 5)),
            dx=1.0
        )


def test_density(density, lin_dx):
    np.testing.assert_array_less(0, density, "density is negative")
    np.testing.assert_allclose(np.sum(density) * lin_dx ** np.ndim(density), num_nodes,
                               err_msg="density does not integrate to number of nodes")


def test_continuous_diffusion_operator(continuous_diffusion_operator, connectivity, density, lin_dx):
    # Evaluate the properties of the operator
    np.testing.assert_allclose(
        continuous_diffusion_operator.weight,
        - gd.evaluate_expected_degree(connectivity, density, lin_dx)[None, None],
        err_msg="weight must be equal to the negative expected degree"
    )
    np.testing.assert_allclose(continuous_diffusion_operator.kernel_weight_x, 1,
                               err_msg="kernel_weight_x must be equal to one")
    np.testing.assert_allclose(continuous_diffusion_operator.kernel_weight_y, density[None, None],
                               err_msg="kernel_weight_y must equal to the density")
    np.testing.assert_allclose(continuous_diffusion_operator.kernel, connectivity[None, None],
                               err_msg="kernel must equal to connectivity")


def test_continuous_averaging_operator(continuous_averaging_operator, connectivity, density, lin_dx):
    # Evaluate the properties of the operator
    np.testing.assert_allclose(continuous_averaging_operator.weight, -1, err_msg="weight must be equal to minus one")
    np.testing.assert_allclose(
        continuous_averaging_operator.kernel_weight_x,
        1 / gd.evaluate_expected_degree(connectivity, density, lin_dx)[None, None],
        err_msg="kernel_weight_x must be equal to the inverse expected degree"
        )
    np.testing.assert_allclose(continuous_averaging_operator.kernel_weight_y, density[None, None],
                               err_msg="kernel_weight_y must equal to the density")
    np.testing.assert_allclose(continuous_averaging_operator.kernel, connectivity[None, None],
                               err_msg="kernel must equal to connectivity")


def test_continuous_diffusion_gradient(continuous_diffusion_operator, density, initial_conditions):
    # Compute the gradient
    grad = continuous_diffusion_operator.evaluate_gradient(initial_conditions)
    # Check that the average gradient is close to zero (total is conserved in diffusion)
    assert np.abs(np.mean(grad * density)) < 1e-10, \
        "average gradient is not close to zero"


def test_continuous_diffusion_integration(continuous_diffusion_operator, density, initial_conditions):
    _, z_numeric, _ = _test_integration(continuous_diffusion_operator, initial_conditions)
    # Ensure the diffusion substance is conserved
    np.testing.assert_allclose(
        np.sum(initial_conditions * density),
        np.sum(z_numeric * density),
        err_msg="diffusion substance not conserved"
    )
    # TODO: compare with reference


def test_continuous_diffusion_integration_with_control(continuous_diffusion_operator,
                                                       density, initial_conditions):
    continuous_diffusion_operator.control = np.pi * np.ones_like(density)[None]
    _, z_numeric, _ = _test_integration(continuous_diffusion_operator, initial_conditions,
                                        time=np.sqrt(2))
    # Ensure the diffusion substance is conserved
    np.testing.assert_allclose(
        np.mean((initial_conditions + np.pi * np.sqrt(2)) * density),
        np.mean(z_numeric * density),
        err_msg="diffusion substance not conserved (subject to control)"
    )


def test_continuous_diffusion_integration_shape(continuous_diffusion_operator, integration_method, time):
    _test_integration_shape(continuous_diffusion_operator, integration_method, time)


def test_continous_oscillation_operator(continuous_oscillation_operator, density, coordinate_tensor,
                                        num_dims, periodic):
    center = np.random.uniform(0, 1, num_dims)
    ic = gd.evaluate_gaussian_kernel(coordinate_tensor, center, 1, 0.05 ** 2)
    _test_integration(continuous_oscillation_operator, [ic, np.zeros_like(ic)])


def test_continous_oscillation_operator_with_control(continuous_oscillation_operator, density,
                                                     coordinate_tensor, num_dims, periodic):
    center = np.random.uniform(0, 1, num_dims)
    ic = gd.evaluate_gaussian_kernel(coordinate_tensor, center, 1, 0.05 ** 2)
    ic = [ic, np.zeros_like(ic)]
    continuous_oscillation_operator.control = np.ones_like(ic)
    _test_integration(continuous_oscillation_operator, ic)


@pytest.mark.parametrize('control_weight', [0, 0.1])
def test_continuous_oscillation_optimal_control(continuous_oscillation_operator, density,
                                                coordinate_tensor, num_dims, periodic, control_weight):
    if not continuous_oscillation_operator.has_analytic_solution:
        pytest.skip()
    # Define the inputs
    t = np.sqrt(2)
    ic = gd.evaluate_gaussian_kernel(coordinate_tensor, .25, 1, 0.05 ** 2)
    ic = [ic, np.zeros_like(ic)]
    setpoint = gd.evaluate_gaussian_kernel(coordinate_tensor, .75, 1, 0.05 ** 2)
    setpoint = [setpoint, np.zeros_like(setpoint)]

    z_nc = continuous_oscillation_operator.integrate_analytic(ic, t)
    # Evaluate the control field
    control = continuous_oscillation_operator.evaluate_control(ic, setpoint, np.eye(2),
                                                               control_weight * np.eye(2), t)
    continuous_oscillation_operator.control = control
    z_wc = continuous_oscillation_operator.integrate_analytic(ic, t)

    # Check that the result with control is better than the one without
    corrcoef_nc = np.corrcoef(np.ravel(setpoint), z_nc.ravel())[0, 1]
    corrcoef_wc = np.corrcoef(np.ravel(setpoint), z_wc.ravel())[0, 1]
    assert corrcoef_wc > corrcoef_nc

    # Assert that we get the "right" result if the control weight is zero
    if control_weight == 0:
        gd.assert_correlated(setpoint, z_wc)
