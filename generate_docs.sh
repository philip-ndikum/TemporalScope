#!/bin/bash

# TemporalScope/generate_docs.sh
# This script automates the process of cleaning old documentation files and caches,
# generating new stubs, and building Sphinx documentation for the TemporalScope project.

# Function to clear Python and MyPy caches
clear_caches() {
    echo "Clearing Python and MyPy caches..."
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null
    echo "Caches cleared."
}

# Function to clear old Sphinx build and autosummary directories
clear_old_docs() {
    echo "Clearing old Sphinx build and autosummary directories..."
    rm -rf docs/_build docs/_autosummary docs/*.rst
    echo "Old documentation files cleared."
}

# Function to generate new API documentation stubs
generate_stubs() {
    echo "Generating new API documentation stubs with sphinx-apidoc..."
    sphinx-apidoc -o docs/ temporalscope/ 2>/dev/null
    echo "New stubs generated."
}

# Function to build the HTML documentation with Sphinx (Verbose mode and log the output)
build_docs() {
    echo "Building HTML documentation with Sphinx (verbose mode)..."
    cd docs
    sphinx-build -b html . _build/html -v | tee sphinx_build.log
    cd ..
    echo "HTML documentation built in docs/_build/html"
}

# Function to check if the virtual environment is active
check_virtualenv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Virtual environment already active."
    else
        echo "Please activate your virtual environment."
        exit 1
    fi
}

# Main function to run the entire documentation generation process
main() {
    check_virtualenv
    clear_caches
    clear_old_docs
    generate_stubs
    build_docs

    if [ -f "docs/_build/html/index.html" ]; then
        echo "To view the documentation locally, open the following file in your browser:"
        echo "$(pwd)/docs/_build/html/index.html"
        echo "✨    TemporalScope documentation successfully generated!"
    else
        echo "Failed to build HTML documentation with Sphinx. Check the sphinx_build.log file for details."
    fi
}

# Run the main function
main
