#!/bin/bash

echo "Generating documentation..."

# Navigate to the docs directory
cd docs

# Clear the _build directory to avoid conflicts with old files
echo "Clearing the _build directory..."
rm -rf _build/*

# Run Sphinx to generate the HTML documentation
sphinx-build -b html . _build/html

# Notify the user where the documentation has been generated
echo "Documentation generated in docs/_build/html"

# Print the command to view the documentation locally
echo "To view the documentation locally, open the following file in your browser:"
echo "$(pwd)/_build/html/index.html"

# Suggest using xdg-open to open the documentation automatically
echo "You can also open the documentation automatically using the following command:"
echo "xdg-open $(pwd)/_build/html/index.html"
