import functools as ft
import matplotlib as mpl
import numpy as np
import pytest
import macrodynamics as md

mpl.use("Agg")

SEED = np.random.randint(2**16)


@pytest.fixture(autouse=True)
def seed_rng(request):
    np.random.seed(SEED)


# Define general fixtures
kernel = md.list_fixture(
    [
        ft.partial(md.evaluate_uniform_kernel, norm=0.05),
        ft.partial(md.evaluate_gaussian_kernel, norm=1, cov=0.05**2),
        ft.partial(md.evaluate_tophat_kernel, norm=1, cov=0.05**2),
        ft.partial(md.evaluate_laplace_kernel, norm=1, cov=0.05**2),
    ],
    [
        "uniform_kernel",
        "gaussian_kernel",
        "tophat_kernel",
        "laplace_kernel",
    ],
)

num_dims = md.list_fixture([2, 1], ["2d", "1d"])

integration_method = md.list_fixture(["analytic", "numeric", "naive"])

time = md.list_fixture([1, [0, 0.25, 0.5, 0.75, 1]])

periodic = md.list_fixture([True, False], ["periodic", "non-periodic"])

num_lin = md.list_fixture([50, 51])
