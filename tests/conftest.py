import functools as ft
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pytest
import graph_dynamics as gd


SEED = np.random.randint(2**16)

@pytest.fixture(autouse=True)
def seed_rng(request):
    np.random.seed(SEED)


# Define general fixtures
kernel = gd.list_fixture([
    ft.partial(gd.evaluate_uniform_kernel, norm=0.05),
    ft.partial(gd.evaluate_gaussian_kernel, norm=1, cov=0.05 ** 2),
    ft.partial(gd.evaluate_tophat_kernel, norm=1, cov=0.05 ** 2),
    ft.partial(gd.evaluate_laplace_kernel, norm=1, cov=0.05 ** 2),
], [
    'uniform_kernel',
    'gaussian_kernel',
    'tophat_kernel',
    'laplace_kernel',
])

num_dims = gd.list_fixture([1, 2])

integration_method = gd.list_fixture(['analytic', 'numeric', 'naive'])

time = gd.list_fixture([1, [0, 0.25, 0.5, 0.75, 1]])

periodic = gd.list_fixture([True, False], ['periodic', 'non-periodic'])

num_lin = gd.list_fixture([50, 51])
