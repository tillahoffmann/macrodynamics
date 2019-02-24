import numpy as np

from ..util import lazy_property, origin_array, is_homogeneous, first_element, add_leading_dims
from .operator import Operator


class ContinuousOperator(Operator):
    r"""
    Differential operator for linear dynamics on continuous graphs embedded in Euclidean space.

    The operator evaluates the gradient

    \frac{\partial z(x, t)}{\partial t} = F(x) z(x) + G(x) \int dy \, H(x - y) L(y) z(y, t),

    where all tensors `F` through `L` have shape `(k, k, *n)`, `k` is the number of states at each
    spatial point, and `n` is the shape of the space.

    * `F(x)` accounts for the independent dynamics of each point in the space.
    * `H(x - y)` accounts for the interactions between points at `x` and `y`. If
      `H(x - y) = H(|x - y|)`, the kernel is isotropic and homogeneous such that the Fourier modes
      evolve independently.
    * `G(x)` weights the effect of the spatial convolution in real space.
    * `L(y)` weights the effect of the field prior to applying the convolution.

    Parameters
    ----------
    weight : np.ndarray or float
        pointwise multiplicative weight `F(x)` for the state `z(x, t)`
    kernel : np.ndarray
        homogeneous interaction kernel `H(x - y)`
    kernel_weight_x : np.ndarray or float
        pointwise multiplicative weight `G(x)` applied to the convolution
    kernel_weight_y : np.ndarray or float
        pointwise multiplicative weight `L(y)` applied to the argument of the convolution `z(y, t)`
    dx : np.ndarray or float
        spacing between sample points
    """
    def __init__(self, weight, kernel, kernel_weight_x, kernel_weight_y, dx):
        self.weight = np.asarray(weight)
        # The weight must have at least three dimensions
        assert self.weight.ndim > 2, "expected at least three dimensions for the multiplicative " \
            "weight but got %d" % self.weight.shape
        # The weight must have square leading dimensions
        assert is_homogeneous(self.weight.shape[:2]), "leading dimensions must be square"

        self.kernel = np.asarray(kernel)
        self.kernel_weight_x = np.asarray(kernel_weight_x)
        self.kernel_weight_y = np.asarray(kernel_weight_y)
        # Broadcast the volume elements
        self.dx = dx * np.ones(self.ndim)

        # Check that the shapes match up
        attrs = ['kernel_weight_x', 'kernel_weight_y']
        for attr in attrs:
            x = getattr(self, attr)
            assert x.shape == self.weight.shape, "expected `%s` to match the shape of the " \
                "multiplicative weight %s but got %s" % (attr, self.shape, x.shape)

        # The leading dimensions of the kernel must match
        np.testing.assert_equal(self.kernel.shape[:2], self.weight.shape[:2], "kernel must have "
                                "same leading dimensions as the multiplicative weight")

        # The shape will match for periodic boundary conditions but won't match for aperiodic boundary conditions
        assert np.all(np.asarray(self.shape) <= self.kernel.shape[1:]), "expected kernel shape to be " \
            "larger than or equal to the shape of the state %s but got %s" % \
            (self.shape, self.kernel.shape)

    @classmethod
    def from_matrix(cls, weight, kernel, kernel_weight_x, kernel_weight_y, dx):
        args = [add_leading_dims(x, 2) for x in [weight, kernel, kernel_weight_x, kernel_weight_y]]
        return cls(*args, dx)

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
        return self._evaluate_fft(self.kernel, True) * self.dV

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
        return not self._inhomogeneous_attrs

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
                                      kernel_weight_y)
        # Now we need to diagonalise each element of the big matrix
        evals, evecs = np.linalg.eig(operator)
        ievecs = np.linalg.inv(evecs)

        # Take the fourier transform of the initial state
        ft_z = self._evaluate_fft(z, True)
        # Project into the diagonal space of the operator
        ft_z = np.einsum('...ij,j...->...i', ievecs, ft_z)
        # Evolve the state (manual multiplication to support broadcast (which *= doesn't))
        ft_z = np.exp(evals * np.reshape(t, (-1, *np.ones(ft_z.ndim, int)))) * ft_z
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
        grad = self._evaluate_fft(ft_kernel_w, False)
        # Extract the region near the origin (but only the spatial dims)
        grad = origin_array(grad, self.shape[1:], np.arange(1, grad.ndim))
        # Weight the gradient
        grad = self._evaluate_dot(self.kernel_weight_x, grad)
        # Add the elementwise multiplicative contribution
        grad += self._evaluate_dot(self.weight, z)
        return grad
