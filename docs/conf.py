"""TemporalScope/docs/conf.py

This module configures Sphinx for the TemporalScope project to automatically generate documentation.
It ensures the Python path is set correctly and dynamically updates the autosummary files to reflect
the current structure of the project.
"""

import sys
from pathlib import Path
import pkgutil
from sphinx.application import Sphinx
import sphinx.ext.autosummary.generate as autosummary_generate
import shutil

# Constants for directory paths
ROOT_DIR = "TemporalScope"
SRC_DIR = "temporalscope"
DOCUMENTATION_DIR = "_autosummary"

<<<<<<< HEAD

=======
>>>>>>> 6ecf0623f3d8d3c1f7607c1dd06e9c824d0dab98
def setup_documentation_path():
    """
    Configures the system path for Sphinx documentation generation.
    """
    current_directory = Path(__file__).resolve().parent
    project_root = current_directory.parent.parent
    source_path = project_root / ROOT_DIR / SRC_DIR

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    sys.path.insert(0, str(source_path))
    print(f"Source code path added to sys.path: {source_path}")

<<<<<<< HEAD

=======
>>>>>>> 6ecf0623f3d8d3c1f7607c1dd06e9c824d0dab98
def generate_autosummary(app: Sphinx):
    """
    Automatically generates autosummary files for all modules in the project.
    """
    document_base_path = Path(app.srcdir) / DOCUMENTATION_DIR
    if document_base_path.exists():
        shutil.rmtree(document_base_path)  # Clear existing files
    document_base_path.mkdir(parents=True, exist_ok=True)

    root_package_path = Path(app.srcdir).parent / ROOT_DIR / SRC_DIR
    if not root_package_path.exists():
        raise FileNotFoundError(f"Package path does not exist: {root_package_path}")

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=[str(root_package_path)], prefix=f"{ROOT_DIR}.{SRC_DIR}."
    ):
        output_file_path = document_base_path / f"{modname}.rst"
        with open(output_file_path, "w") as f:
            autosummary_generate.generate_autosummary_content(
                modname, None, f, app.config, template_dir="_templates/autosummary"
            )

<<<<<<< HEAD

def setup(app: Sphinx):
    app.connect("builder-inited", generate_autosummary)


=======
def setup(app: Sphinx):
    app.connect("builder-inited", generate_autosummary)

>>>>>>> 6ecf0623f3d8d3c1f7607c1dd06e9c824d0dab98
setup_documentation_path()  # Ensure this is called at the beginning

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
