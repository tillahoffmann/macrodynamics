import functools as ft
import numpy as np
import scipy.integrate
from typing import Literal, overload, Any, Callable

from ..util import lazy_property


class Operator:
    """
    Base class for differential operators.
    """

    @lazy_property
    def shape(self) -> tuple[int, ...]:
        """tuple : Shape of the state array."""
        raise NotImplementedError

    def evaluate_gradient(
        self, z: np.ndarray, t: float | None = None, control: np.ndarray | None = None
    ) -> np.ndarray:
        """Evaluate the time derivative of `z`.

        Args:
            z: State for which to evaluate the gradient.
            t: Time at which to evaluate the gradient.
            control: Static control to apply to the dynamics.

        Returns:
            Gradient for the dynamics with the same shape as `z`.

        Note:
            If `evaluate_gradient` does not depend on `t`, the corresponding differential
            equation is homogeneous.
        """
        raise NotImplementedError

    def integrate_analytic(
        self, z: np.ndarray, t: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        """Solve for `z` as a function of `t` analytically.

        Args:
            z: Initial state.
            t: Time at which to solve for `z`.
            control: Static control to apply to the dynamics.

        Returns:
            Solution of `z` for each `t`.
        """
        raise NotImplementedError

    def _evaluate_flat_gradient(
        self,
        t: float,
        z: np.ndarray,
        control: np.ndarray | None,
        callback: Callable[[float, np.ndarray, np.ndarray], Any] | None = None,
    ) -> np.ndarray:
        """
        Helper function to reshape `z` to the same shape as the state vector associated
        with this operator if necessary, compute the time derivative, and flatten the
        gradient.
        """
        z = np.reshape(z, self.shape)
        grad = self.evaluate_gradient(z, t, control)
        if callback:
            callback(t, z, grad)
        return grad.ravel()

    @overload
    def _assert_valid_shape(
        self, z: np.ndarray, strict: Literal[True]
    ) -> np.ndarray: ...

    @overload
    def _assert_valid_shape(
        self, z: np.ndarray | None, strict: Literal[False]
    ) -> np.ndarray | None: ...

    @overload
    def _assert_valid_shape(self, z: np.ndarray) -> np.ndarray: ...

    def _assert_valid_shape(
        self, z: np.ndarray | None, strict: bool = True
    ) -> np.ndarray | None:
        """
        Helper function to assert that `z` has the same shape as the state vector
        associated with this operator.
        """
        if z is None:
            assert not strict, "Input must not be `None`."
            return
        z = np.atleast_2d(z)
        assert z.shape == self.shape, (
            f"Expected state shape `{self.shape}` (state dim, *spatial dims) but "
            f"got `{z.shape}`."
        )
        return z

    def integrate_numeric(
        self,
        z: np.ndarray,
        t: np.ndarray | float,
        control: np.ndarray | None = None,
        callback: Callable[[float, np.ndarray, np.ndarray], Any] | None = None,
        method: str = "LSODA",
        **kwargs,
    ) -> np.ndarray:
        """Solve for `z` as a function of `t` numerically.

        Args:
            z: Initial state.
            t: Time at which to solve for `z`.
            control: Static control to apply to the dynamics.
            callback: Function for each evaluation of the RHS.
            method: Integration method to use. See `scipy.integrate.ode.solve_ivp` for details.
            **kwargs: Keyword arguments passed to `scipy.integrate.ode.solve_ivp`.

        Returns:
            Solution of `z` for each `t`.
        """
        # Validate the input tensor
        z = self._assert_valid_shape(z)
        times = np.asarray([0, t] if np.isscalar(t) else t)

        # Solve the initial value problem
        result = scipy.integrate.solve_ivp(
            ft.partial(
                self._evaluate_flat_gradient, callback=callback, control=control
            ),
            (times[0], times[-1]),
            z.ravel(),
            t_eval=times,
            method=method,
            **kwargs,
        )
        # Reshape the flattened array to the desired shape
        z = result.y.T.reshape((-1, *self.shape))
        return z[-1] if np.isscalar(t) else z

    def integrate_naive(
        self, z: np.ndarray, t: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        """Solve for `z` as a function of `t` using naive finite difference integration.

        Args:
            z: Initial state.
            t: Time at which to solve for `z`.
            control: Static control to apply to the dynamics.

        Returns:
            Solution of `z` for each `t`.

        Note:
            This function is expected to perform worse in terms of performance and accuracy
            than `integrate_numeric`.
        """
        self._assert_valid_shape(z)
        zs = [z]

        if np.ndim(t) != 1:
            raise ValueError(
                "time vector must be one-dimensional for naive integration"
            )

        time = t[0]
        for next_time in t[1:]:
            grad = self.evaluate_gradient(z, time, control)
            dt = next_time - time
            z = z + dt * grad
            zs.append(z.copy())
            time = next_time

        return np.asarray(zs)

    def integrate(
        self,
        z: np.ndarray,
        t: np.ndarray,
        method: Literal["analytic", "numeric", "naive"],
        control: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Solve for `z` as a function of `t`.

        Args:
            z: Initial state.
            t: Time at which to solve for `z`.
            method: Method used for simulation (one of 'analytic', 'numeric' or 'naive').
            control: Static control to apply to the dynamics.
            **kwargs: Keyword arguments passed to the selected integration `method`.

        Returns:
            Solution of `z` for each `t`.
        """
        self._assert_valid_shape(z)
        if method == "analytic":
            return self.integrate_analytic(z, t, control, **kwargs)
        elif method == "numeric":
            return self.integrate_numeric(z, t, control, **kwargs)
        elif method == "naive":
            return self.integrate_naive(z, t, control, **kwargs)
        else:
            raise KeyError(method)

    @lazy_property
    def has_analytic_solution(self):
        """bool : Whether the differential operator has an analytic solution."""
        raise NotImplementedError
