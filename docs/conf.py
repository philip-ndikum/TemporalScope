# -- Path setup --------------------------------------------------------------
import os
import sys

# Adjust the path to point to the correct source directory (temporalscope/temporalscope)
source_path = os.path.abspath("../../temporalscope/temporalscope")
sys.path.insert(0, source_path)

# Print out the path being used for the source code
print(f"Source code path added to sys.path: {source_path}")

# -- Project information -----------------------------------------------------
project = "TemporalScope"
copyright = "2024 Philip Ndikum. Apache 2.0 License"
author = "Philip Ndikum"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

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
    # Darker version of Ivy Green color for the theme
    # "primary_color": "#264d40",  # Dark Ivy Green
}

html_static_path = ["_static"]
