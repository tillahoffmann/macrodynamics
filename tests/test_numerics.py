import itertools as it
import numpy as np
import pytest
import scipy.integrate
import scipy.signal


def test_eig():
    # Construct a matrix
    x = np.random.normal(0, 1, (50, 50))
    # Get the eigenvalues and eigenvectors
    w, v = np.linalg.eig(x)
    # Reconstruct the matrix
    np.testing.assert_allclose(
        v.dot(np.diag(w)).dot(np.linalg.inv(v)),
        x,
        err_msg="could not reconstruct matrix",
    )


@pytest.mark.parametrize("method", ["riemann", "trapz"])
def test_integrate(method):
    num_steps = 100

    if method == "riemann":
        y = np.ones(num_steps)
        z = np.sum(y) / num_steps
    elif method == "trapz":
        y = np.ones(num_steps)
        z = scipy.integrate.trapezoid(y, dx=1 / (num_steps - 1))
    else:
        raise KeyError(method)

    np.testing.assert_allclose(z, 1, err_msg="unexpected integral")


@pytest.mark.parametrize("n", [10, 11])
def test_fftconvolve_convolve2d_fill(n):
    # Check that fftconvolve and convolve2d produce the same result for zero-padded
    # boundary conditions.
    x, y = np.random.normal(0, 1, (2, n, n))
    a = scipy.signal.convolve2d(x, y, "same", "fill")
    b = scipy.signal.fftconvolve(x, y, "same")
    np.testing.assert_allclose(a, b)


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


@pytest.mark.parametrize(
    "boundary, n",
    it.product(
        ["fill", "wrap"],
        [10, 11],
    ),
)
def test_convolve2d(boundary, n):
    """
    This test ensures that we use the fft appropriately.

    * `scipy.signal.convolve2d` supports different boundary conditions 'fill', 'wrap',
      'symm'. The latter corresponds to reflective boundaries and is likely not
        relevant.
    * `scipy.signal.fftconvolve` only supports zero-padded boundaries. Internally, it
      computes the shape `s1 + s2 - 1`, where `s1` and `s2` are the input shapes, pads
      with zeros, determines the next fast fft size and proceeds with the standard fft
      -> product -> ifft method.

    The methods `fftconvolve` and `convolve2d` with 'fill' boundary conditions agree as
    demonstrated in `test_fftconvolve_convolve2d_fill` above. In this test, we thus only
    have to establish a relation between `convolve2d` and the manual approach using
    Fourier transforms.
    """
    # Create inputs
    lin = np.linspace(0, 1, n, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    x = np.exp(-7 * ((x - 0.2) ** 2 + (y - 0.7) ** 2))
    y = x.T
    # Convolve using scipy
    z_desired = scipy.signal.convolve2d(x, y, mode="same", boundary=boundary)

    # Roll the result if necessary
    if boundary == "wrap":
        z_desired = np.roll(
            z_desired, (np.asarray(z_desired.shape) - 1) // 2, axis=(0, 1)
        )

    # Pad if necessary
    if boundary == "fill":
        shape = np.asarray(x.shape) + np.asarray(y.shape) - 1
    else:
        shape = x.shape

    # Convolve manually
    fx = np.fft.rfftn(x, shape)
    fy = np.fft.rfftn(y, shape)
    z_actual = np.fft.irfftn(fx * fy, shape)

    # Crop (this is a noop if the image already has the right size)
    z_actual = _centered(z_actual, x.shape)

    # Check for agreement
    np.testing.assert_allclose(z_actual, z_desired, err_msg="convolutions do not agree")
