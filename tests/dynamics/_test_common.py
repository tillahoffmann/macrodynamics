import numpy as np
import pytest

import graph_dynamics as gd


def _test_integration(operator, z0=None, time=1):
    if z0 is None:
        z0 = np.random.normal(0, 1, operator.shape)
    else:
        z0 = np.reshape(z0, operator.shape)

    # Run the integration
    z_numeric = operator.integrate_numeric(z0, time)
    z_naive = operator.integrate_naive(z0, np.linspace(0, time, 1000) if np.isscalar(time) else time)[-1]
    # Use slightly laxer tolerance on the offset because of the finite integration step
    gd.assert_correlated(z_naive, z_numeric)

    if not operator.has_analytic_solution:
        return z0, z_numeric, None

    # Make sure the values match
    z_analytic = operator.integrate_analytic(z0, time)
    gd.assert_correlated(z_numeric, z_analytic)
    return z0, z_numeric, z_analytic


def _test_integration_shape(operator, integration_method, time):
    z0 = np.random.normal(0, 1, operator.shape)
    if not operator.has_analytic_solution and integration_method == 'analytic':
        with pytest.raises(ValueError):
            operator.integrate(z0, time, integration_method)
    else:
        if np.isscalar(time) and integration_method == 'naive':
            pytest.skip("naive integration requires a vector of time points")
        z = operator.integrate(z0, time, integration_method)
        if np.isscalar(time):
            assert z.shape == z0.shape, "output shape does not match input shape"
        else:
            desired_shape = np.shape(time) + z0.shape
            assert z.shape == desired_shape, "expected shape `%s` but got `%s`" % \
                (desired_shape, z.shape)
