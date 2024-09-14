# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import datetime
import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "TemporalScope"
author = "Philip Ndikum, Serge Ndikum, Kane Norman"
copyright = f"{datetime.datetime.now().year}, {author}"
version = release = importlib.metadata.version("temporalscope")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "autoapi.extension",
]

autoapi_dirs = ["../src"]
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]


master_doc = "index"
autoclass_content = "class"
autosummary_generate = True

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = project
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/philip-ndikum/TemporalScope",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ]
}

myst_enable_extensions = ["colon_fence"]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True
