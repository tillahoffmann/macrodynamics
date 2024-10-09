import numpy as np
import pytest

import macrodynamics as md


def _test_integration(operator, z0=None, time=1, num=200, control=None):
    """
    Test that naive, numeric, and analytic integration are consistent. The comparison
    with analytic integration is omitted if the operator does not have a (known)
    analytic solution.
    """
    if z0 is None:
        z0 = np.random.normal(0, 1, operator.shape)
    else:
        z0 = np.reshape(z0, operator.shape)

    def callback(*args):
        callback.called = True

    # Run the integration
    z_numeric = operator.integrate_numeric(z0, time, callback=callback, control=control)
    z_naive = operator.integrate_naive(
        z0, np.linspace(0, time, num) if np.isscalar(time) else time, control=control
    )[-1]
    md.assert_correlated(z_naive, z_numeric)

    assert getattr(callback, "called", True), "callback not called"

    if not operator.has_analytic_solution:
        return z0, z_numeric, None

    # Make sure the values match
    z_analytic = operator.integrate_analytic(z0, time, control=control)
    md.assert_correlated(z_numeric, z_analytic)
    return z0, z_numeric, z_analytic


def _test_integration_shape(operator, integration_method, time, control=None):
    """
    Test that the output of an integration has the desired shape or raises a
    `ValueError` if the integration cannot be performed..
    """
    z0 = np.random.normal(0, 1, operator.shape)
    if (not operator.has_analytic_solution and integration_method == "analytic") or (
        np.isscalar(time) and integration_method == "naive"
    ):
        with pytest.raises(ValueError):
            operator.integrate(z0, time, integration_method, control=control)
    else:
        z = operator.integrate(z0, time, integration_method, control=control)
        if np.isscalar(time):
            assert z.shape == z0.shape, "output shape does not match input shape"
        else:
            desired_shape = np.shape(time) + z0.shape
            assert z.shape == desired_shape, "expected shape `%s` but got `%s`" % (
                desired_shape,
                z.shape,
            )
