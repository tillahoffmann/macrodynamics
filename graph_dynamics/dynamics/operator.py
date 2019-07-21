import functools as ft
import numpy as np
import scipy.integrate

from ..util import lazy_property


class Operator:
    """
    Base class for differential operators.

    Parameters
    ----------
    control : numpy.ndarray
        Static control to apply to the dynamics.
    """
    def __init__(self, control=None):
        self._control = None
        self.control = control

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, control):
        if control is not None:
            control = self._assert_valid_shape(control)
        self._control = control

    @lazy_property
    def shape(self):
        """tuple : Shape of the state array."""
        raise NotImplementedError

    def evaluate_gradient(self, z, t=None):
        """
        Evaluate the time derivative of `z`.

        Parameters
        ----------
        z : numpy.ndarray
            State for which to evaluate the gradient.
        t : float
            Time at which to evaluate the gradient.

        Returns
        -------
        grad : numpy.ndarray
            Gradient for the dynamics with the same shape as `z`.

        Notes
        -----
        If `evaluate_gradient` does not depend on `t`, the corresponding differential equation is
        homogeneous.
        """
        raise NotImplementedError

    def integrate_analytic(self, z, t):
        """
        Solve for `z` as a function of `t` analytically.

        Parameters
        ----------
        z : numpy.ndarray
            Initial state.
        t : numpy.ndarray
            Time at which to solve for `z`.

        Returns
        -------
        z : numpy.ndarray
            Solution of `z` for each `t`.
        """
        raise NotImplementedError

    def _evaluate_flat_gradient(self, t, z, callback=None):
        """
        Helper function to reshape `z` to the same shape as the state vector associated with this
        operator if necessary, compute the time derivative, and flatten the gradient.
        """
        z = np.reshape(z, self.shape)
        grad = self.evaluate_gradient(z, t)
        if callback:
            callback(t, z, grad)
        return grad.ravel()

    def _assert_valid_shape(self, z):
        """
        Helper function to assert that `z` has the same shape as the state vector associated with
        this operator.
        """
        z = np.atleast_2d(z)
        assert z.shape == self.shape, "expected state shape `%s` (state dim, *spatial dims) but " \
            "got `%s`" % (self.shape, z.shape)
        return z

    def integrate_numeric(self, z, t, callback=None, method='LSODA', **kwargs):
        """
        Solve for `z` as a function of `t` numerically.

        Parameters
        ----------
        z : numpy.ndarray
            Initial state.
        t : numpy.ndarray
            Time at which to solve for `z`.
        callback : callable or None
            Function for each evaluation of the RHS.
        method : str
            Integration method to use. See `scipy.integrate.ode.solve_ivp` for details.
        **kwargs : dict
            Keyword arguments passed to `scipy.integrate.ode.solve_ivp`.

        Returns
        -------
        z : numpy.ndarray
            Solution of `z` for each `t`.
        """
        # Validate the input tensor
        z = self._assert_valid_shape(z)
        times = np.asarray([0, t] if np.isscalar(t) else t)

        # Solve the initial value problem
        result = scipy.integrate.solve_ivp(
            ft.partial(self._evaluate_flat_gradient, callback=callback),
            (times[0], times[-1]),
            z.ravel(),
            t_eval=times,
            method=method,
            **kwargs
        )
        # Reshape the flattened array to the desired shape
        z = result.y.T.reshape((-1, * self.shape))
        return z[-1] if np.isscalar(t) else z

    def integrate_naive(self, z, t):
        """
        Solve for `z` as a function of `t` using naive finite difference integration.

        Parameters
        ----------
        z : numpy.ndarray
            Initial state.
        t : numpy.ndarray
            Time at which to solve for `z`.

        Returns
        -------
        z : numpy.ndarray
            Solution of `z` for each `t`.

        Notes
        -----
        This function is expected to perform worse in terms of performance and accuracy than
        `integrate_numeric`.
        """
        self._assert_valid_shape(z)
        zs = [z]

        if np.ndim(t) != 1:
            raise ValueError("time vector must be one-dimensional for naive integration")

        time = t[0]
        for next_time in t[1:]:
            grad = self.evaluate_gradient(z, time)
            dt = next_time - time
            z = z + dt * grad
            zs.append(z.copy())
            time = next_time

        return np.asarray(zs)

    def integrate(self, z, t, method, **kwargs):
        """
        Solve for `z` as a function of `t`.

        Parameters
        ----------
        z : numpy.ndarray
            Initial state.
        t : numpy.ndarray
            Time at which to solve for `z`.
        method : str
            Method used for simulation (one of 'analytic', 'numeric' or 'naive').
        **kwargs : dict
            Keyword arguments passed to the selected integration `method`.

        Returns
        -------
        z : numpy.ndarray
            Solution of `z` for each `t`.
        """
        self._assert_valid_shape(z)
        if method == 'analytic':
            return self.integrate_analytic(z, t, **kwargs)
        elif method == 'numeric':
            return self.integrate_numeric(z, t, **kwargs)
        elif method == 'naive':
            return self.integrate_naive(z, t, **kwargs)
        else:
            raise KeyError(method)

    @lazy_property
    def has_analytic_solution(self):
        """bool : whether the differential operator has an analytic solution"""
        raise NotImplementedError
