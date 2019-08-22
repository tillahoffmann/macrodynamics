from setuptools import setup, find_packages

tests_require = [
    "pytest",
    "pytest-cov",
    "sphinx",
    "numpydoc",
]

setup(
    name="macrodynamics",
    version="0.1",
    author="Till Hoffmann",
    url="https://github.com/tillahoffmann/macrodynamics",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    tests_require=tests_require,
    extras_require={
        "tests": tests_require,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics",
    ]
)
