import numpy as np

from ..util import lazy_property, origin_array, is_homogeneous, first_element, add_leading_dims
from .operator import Operator


class ContinuousOperator(Operator):
    r"""
    Differential operator for linear dynamics on continuous graphs embedded in Euclidean space.

    The operator evaluates the gradient

    \frac{\partial z(x, t)}{\partial t} = F(x) z(x) + G(x) \int dy \, H(x - y) L(y) z(y, t) + u(x),

    where all tensors `F` through `L` have shape `(k, k, *n)`, `k` is the number of states at each
    spatial point, and `n` is the shape of the space.

    * `F(x)` accounts for the independent dynamics of each point in the space.
    * `H(x - y)` accounts for the interactions between points at `x` and `y`. If
      `H(x - y) = H(|x - y|)`, the kernel is isotropic and homogeneous such that the Fourier modes
      evolve independently.
    * `G(x)` weights the effect of the spatial convolution in real space.
    * `L(y)` weights the effect of the field prior to applying the convolution.
    * `u(x)` is the static control field applied to the dynamics.

    Parameters
    ----------
    weight : np.ndarray
        pointwise multiplicative weight `F(x)` for the state `z(x, t)` with shape `(k, k, *n)`
    kernel : np.ndarray
        homogeneous interaction kernel `H(x - y)` with shape `(k, k, *n)`
    kernel_weight_x : np.ndarray
        pointwise multiplicative weight `G(x)` applied to the convolution with shape `(k, k, *n)`
    kernel_weight_y : np.ndarray
        pointwise multiplicative weight `L(y)` applied to the argument of the convolution `z(y, t)`
         with shape `(k, k, *n)` for periodic boundary conditions or `>= (k, k, *2 * (n - 1))` for
         aperiodic boundary conditions
    dx : np.ndarray or float
        spacing between sample points
    control : np.ndarray
        static control field to apply to the dynamics
    """
    def __init__(self, weight, kernel, kernel_weight_x, kernel_weight_y, dx, control=None):
        self.weight = np.asarray(weight)
        self.kernel = np.asarray(kernel)
        self.kernel_weight_x = np.asarray(kernel_weight_x)
        self.kernel_weight_y = np.asarray(kernel_weight_y)
        # Broadcast the volume elements
        self.dx = dx * np.ones(self.ndim)

        # Validate the shape of the multiplicative weight
        assert self.weight.ndim > 2, "expected at least three-dimensional shape for `weight` but " \
            "got `%s`" % (self.weight.shape,)
        assert is_homogeneous(self.weight.shape[:2]), "leading two dimensions for `weight` must " \
            "be square but got `%s`" % (self.weight.shape,)

        # Validate the shape of the kernel weights
        items = {
            'kernel_weight_x': self.kernel_weight_x,
            'kernel_weight_y': self.kernel_weight_y,
        }
        for key, value in items.items():
            assert value.shape == self.weight.shape, "expected shape of `%s` to be the same as " \
                "`weight.shape = %s` but got `%s`" % (key, self.weight.shape, value.shape)

        # Validate the shape of the kernel
        assert self.kernel.shape[:2] == self.weight.shape[:2], "expected the state shape of " \
            "`kernel` to be the same as `weight.shape[:2] = %s` but got `%s`" % \
            (self.weight.shape[:2], self.kernel.shape[:2])
        spatial_weight_shape = np.asarray(self.weight.shape[2:])
        spatial_kernel_shape = np.asarray(self.kernel.shape[2:])

        if np.array_equal(spatial_kernel_shape, spatial_weight_shape):
            self.periodic_boundary_conditions = True
        elif np.all(spatial_kernel_shape >= 2 * (spatial_weight_shape - 1)):
            self.periodic_boundary_conditions = False
        else:
            raise ValueError(
                "expected the spatial shape of `kernel` to be the same as `weight.shape[2:] = %s` "
                "(for periodic boundary conditions) or at least as large as "
                "`2 * (weight.shape[2:] - 1 = %s)` for aperiodic boundary conditions but got `%s`" %
                (spatial_weight_shape, 2 * spatial_weight_shape - 1, spatial_kernel_shape)
            )

        # Store the control field
        self._control = None
        self.control = control

    @classmethod
    def from_matrix(cls, weight, kernel, kernel_weight_x, kernel_weight_y, dx, **kwargs):
        args = [add_leading_dims(x, 2) for x in [weight, kernel, kernel_weight_x, kernel_weight_y]]
        return cls(*args, dx, **kwargs)

    @lazy_property
    def control(self):
        return self._control

    @control.setter
    def control(self, control):
        if control is not None:
            control = self._assert_valid_shape(control)
        self._control = control

    @lazy_property
    def shape(self):
        # We have to use the multiplicative weight because the kernel may be larger than the state
        # for non-periodic boundary conditions. Drop the first dimension because it does not represent
        # the shape of the state.
        return self.weight.shape[1:]

    @lazy_property
    def dV(self):
        """float : differential volume element"""
        return np.prod(self.dx)

    @lazy_property
    def ndim(self):
        """int : spatial dimensionality of the kernel"""
        # Take of two for the leading state dimension
        return self.kernel.ndim - 2

    @lazy_property
    def fft_kernel(self):
        """np.ndarray : Fourier transform of the kernel"""
        return self._evaluate_fft(self.kernel, True)

    def _evaluate_fft(self, x, forward):
        """
        Evaluate the FFT of `x` along the trailing dimensions depending on `forward`.
        """
        shape = self.kernel.shape[2:]
        axes = self._evaluate_spatial_axes(x)

        if forward:
            return np.fft.rfftn(x, shape, axes)
        else:
            return np.fft.irfftn(x, shape, axes)

    def _evaluate_spatial_axes(self, x):
        """
        Evaluate the spatial axes of `x` assuming that the spatial dimensions are fully determined
        by the connectivity kernel.
        """
        x = np.asarray(x)
        offset = x.ndim - self.ndim
        assert offset >= 0
        return np.arange(offset, x.ndim)

    @lazy_property
    def _inhomogeneous_attrs(self):
        attrs = {
            key: getattr(self, key) for key in ['weight', 'kernel_weight_x', 'kernel_weight_y']}

        return [key for key, value in attrs.items()
                if not np.all(is_homogeneous(value, self._evaluate_spatial_axes(value)))]

    @lazy_property
    def has_analytic_solution(self):
        return not self._inhomogeneous_attrs and self.kernel_weight_x.shape == self.kernel.shape

    def integrate_analytic(self, z, t):
        z = self._assert_valid_shape(z)
        # Check all the weight functions are homogeneous
        if not self.has_analytic_solution:
            raise ValueError("analytic solution is not available for inhomogeneous %s" % self._inhomogeneous_attrs)

        # Get the first dimensions
        args = [self.weight, self.kernel_weight_x, self.kernel_weight_y]
        for i, arg in enumerate(args):
            axes = self._evaluate_spatial_axes(arg)
            arg = first_element(arg, axes, axes)
            assert arg.ndim == 2 and is_homogeneous(arg.shape), \
                "expected square matrix but got shape `%s` for argument %d" % (arg.shape, i)
            args[i] = arg

        weight, kernel_weight_x, kernel_weight_y = args

        # Evaluate the operator and move the spatial part to the leading axes
        operator = weight + np.einsum('ij,jk...,kl->...il', kernel_weight_x, self.fft_kernel,
                                      kernel_weight_y) * self.dV
        # Now we need to diagonalise each element of the big matrix
        evals, evecs = np.linalg.eig(operator)
        ievecs = np.linalg.inv(evecs)

        # Take the fourier transform of the initial state (state_dim, *fourier_dims)
        ft_z = self._evaluate_fft(z, True)
        # Project into the diagonal space of the operator (*fourier_dims, state_dim)
        ft_z = np.einsum('...ij,j...->...i', ievecs, ft_z)
        # Evolve the state (manual multiplication to support broadcast (which *= doesn't))
        # (time_dim, *fourier_dims, state_dim)
        ft_z = np.exp(evals * np.reshape(t, (-1, *np.ones(ft_z.ndim, int)))) * ft_z
        # Apply the control field
        if self.control is not None:
            # Move to the Fourier space (state_dim, *fourier_dims)
            ft_control = self._evaluate_fft(self.control, True)
            # Move to the eigenspace of the operator (*fourier_dims, state_dim)
            ft_control = np.einsum('...ij,j...->...i', ievecs, ft_control)
            # Evolve in the eigenspace (time_dim, *fourier_dims, state_dim)
            # Note that we replace zero eigenvalues in the denominator with ones because the solution
            # is NaN otherwise. Zero eigenvalues do not contribute anyway because expm1(0) = 0.
            ft_control = np.expm1(evals * np.reshape(t, (-1, *np.ones(ft_control.ndim, int)))) * \
                ft_control / np.where(evals == 0, 1.0, evals)
            ft_z += ft_control
        # Project back into the original space
        ft_z = np.einsum('...ij,t...j->ti...', evecs, ft_z)
        # Compute the inverse transform
        z = self._evaluate_fft(ft_z, False)
        # Extract the region near the origin
        z = origin_array(z, self.shape[1:], axes=2 + np.arange(self.ndim))
        return z[-1] if np.isscalar(t) else z

    def _evaluate_dot(self, a, b):
        """
        Evaluate the dot product along the leading dimension broadcasting the remainder.
        """
        return np.einsum('ij...,j...->i...', a, b)

    def evaluate_gradient(self, z, t=None):
        z = self._assert_valid_shape(z)
        # Evaluate the kernel weighted field
        w = self._evaluate_dot(self.kernel_weight_y, z)
        # Compute the FFT of the kernel-weighted field
        ft_w = self._evaluate_fft(w, True)
        # Multiply by the kernel and invert
        ft_kernel_w = self._evaluate_dot(self.fft_kernel, ft_w)
        grad = self._evaluate_fft(ft_kernel_w, False) * self.dV
        # Extract the region near the origin (but only the spatial dims)
        grad = origin_array(grad, self.shape[1:], np.arange(1, grad.ndim))
        # Weight the gradient
        grad = self._evaluate_dot(self.kernel_weight_x, grad)
        # Add the elementwise multiplicative contribution
        grad += self._evaluate_dot(self.weight, z)
        # Add the control field if present
        if self.control is not None:
            grad += self.control
        return grad
