"""TemporalScope/docs/conf.py

This module configures Sphinx for the TemporalScope project to automatically generate documentation.
It ensures the Python path is set correctly and dynamically updates the autosummary files to reflect
the current structure of the project.

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
from pathlib import Path

# Add your project source directory to sys.path
sys.path.insert(0, os.path.abspath('../../temporalscope'))

# -- Project information -----------------------------------------------------
project = "TemporalScope"
copyright = "2024, Philip Ndikum"
author = "Philip Ndikum"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

# Generate autosummary files automatically
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_end": ["search-field.html", "navbar-icon-links.html"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/philip-ndikum/TemporalScope/",
            "icon": "fab fa-github-square",
        },
    ],
}

html_static_path = ["_static"]

# -- Custom Setup Function ---------------------------------------------------
def setup(app):
    print("Sphinx documentation generation initiated...")
    # You can add additional setup logic if needed
