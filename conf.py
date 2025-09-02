import inspect
from numpydoc import docscrape_sphinx
import sphinx
import sphinx.util

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "macrodynamics"
copyright = "2019, Till Hoffmann"
author = "Till Hoffmann"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    ".venv",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Specify the baseurls for the projects I want to link to
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
}


master_doc = "README"


LOGGER = sphinx.util.logging.getLogger("validator")


def get_code_location(obj):
    obj = getattr(obj, "__wrapped__", obj)
    if not hasattr(obj, "__code__") and hasattr(obj, "__init__"):
        obj = obj.__init__
    return f"{obj.__code__.co_filename}:{obj.__code__.co_firstlineno}"


def assert_standard_sentence(value, type, name, obj):
    if isinstance(value, list):
        value = " ".join(value)
    try:
        assert value, "is missing"
        assert value[0].isupper(), "should start with uppercase letter"
        assert value.endswith("."), "should end with period"
    except AssertionError as ex:
        LOGGER.warn(
            f"documentation '{value}' for {type} `{name}` {ex} for {obj.__name__} at "
            f"{get_code_location(obj)}"
        )


def autodoc_process_docstring(app, what, name, obj, options, lines):
    # Don't try to validate the docstrings of modules
    if what == "module":
        return
    parsed = docscrape_sphinx.get_doc_object(obj, what)

    if what == "property":
        return

    # Validate the parameters
    documented_parameters = set()
    for parameter in parsed.get("Parameters", []):
        documented_parameters.add(parameter.name)
        assert_standard_sentence(parameter.desc, "parameter", parameter.name, obj)

    # Check that the parameters match the signature
    signature = inspect.signature(obj)
    available_parameters = set()
    for i, parameter in enumerate(signature.parameters.values()):
        if i == 0 and parameter.name == "self":
            continue
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            available_parameters.add("*" + parameter.name)
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            available_parameters.add("**" + parameter.name)
        else:
            available_parameters.add(parameter.name)

    # Find undocumented parameters
    undocumented = available_parameters - documented_parameters
    if undocumented:
        undocumented = ", ".join(f"`{param}`" for param in undocumented)
        LOGGER.warn(
            f"parameters {undocumented} should be documented for {obj.__name__} at "
            f"{get_code_location(obj)}"
        )

    # Find parameters that are documented but not available
    missing = documented_parameters - available_parameters
    if missing:
        missing = ", ".join(f"`{param}`" for param in missing)
        LOGGER.warn(
            f"parameters {missing} are documented but not defined for {obj.__name__} at "
            f"{get_code_location(obj)}"
        )

    # Check if the function could return something
    if what in ("function", "method"):
        source = [line.strip() for line in inspect.getsource(obj).splitlines()]
        could_return = any(
            line.startswith("return ") or line.startswith("raise NotImplemented")
            for line in source
        )
        returns = parsed.get("Returns")
        for parameter in parsed.get("Returns", []):
            assert_standard_sentence(
                parameter.desc, "return value", parameter.name, obj
            )

        if could_return and not returns:
            LOGGER.warn(
                f"return values should be documented for {obj.__name__} at "
                f"{get_code_location(obj)}"
            )
        if not could_return and returns:
            LOGGER.warn(
                f"return values are documented but not defined for {obj.__name__} at "
                f"{get_code_location(obj)}"
            )


def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
