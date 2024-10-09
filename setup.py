from setuptools import setup, find_packages


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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
