import glob
import os

from setuptools import setup, find_packages
from setuptools.extension import Extension

# Determine whether to use the trans-compiled *.c files or start from *.pyx
USE_CYTHON = os.environ.get('USE_CYTHON')
if USE_CYTHON:
    if USE_CYTHON.lower() == 'false':
        USE_CYTHON = False
    elif USE_CYTHON.lower() == 'true':
        USE_CYTHON = True
    else:
        raise ValueError(USE_CYTHON)
else:
    try:
        import Cython
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False

# Generate extensions
ext = ".pyx" if USE_CYTHON else ".c"
filenames = glob.glob("graph_dynamics/*" + ext)
ext_modules = [Extension(os.path.splitext(filename)[0].replace("/", "."), [filename])
               for filename in filenames]

if USE_CYTHON:
    from Cython.Build import cythonize
    ext_modules = cythonize(filenames, annotate=True)

tests_require = [
    "pytest",
    "pytest-cov",
    "sphinx",
]

setup(
    name='graph_dynamics',
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    tests_require=tests_require,
    extras_require={
        "tests": tests_require,
    },
)
