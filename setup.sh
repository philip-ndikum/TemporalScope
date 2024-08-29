#!/bin/bash

# Function to recursively remove all __pycache__ directories
remove_pycaches() {
    echo "Removing all __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "__pycache__ directories removed."
}

# Function to copy README.md to the /temporalscope directory and adjust image paths
copy_and_adjust_readme() {
    if [ -f "README.md" ]; then
        echo "Copying README.md to the temporalscope directory..."
        cp README.md temporalscope/

        # Adjust relative paths for images in the copied README.md
        echo "Adjusting image paths in the copied README.md..."
        sed -i 's/src="assets\//src="\.\.\/assets\//g' temporalscope/README.md
        echo "Image paths adjusted successfully."
    else
        echo "README.md not found. Skipping copy."
    fi
}

# Remove all __pycache__ directories
remove_pycaches

# Check for Python version
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=10
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
echo "Current Python version: $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -ne "$REQUIRED_PYTHON_MAJOR" || "$PYTHON_MINOR" -lt "$REQUIRED_PYTHON_MINOR" ]]; then
    echo "Error: Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi
echo "Python version is compatible."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "Poetry installed."
else
    echo "Poetry is already installed."
fi

# Ensure the virtual environment uses the current Python version
echo "Configuring the virtual environment to use Python $PYTHON_VERSION..."
poetry env use python3.$PYTHON_MINOR
echo "Virtual environment set to use Python $PYTHON_VERSION."

# Install project dependencies
echo "Installing project dependencies..."
poetry install
echo "Project dependencies installed."

# Set up Jupyter kernel
echo "Setting up Jupyter kernel named 'temporalscope'..."
poetry run python3 -m ipykernel install --user --name=temporalscope --display-name "Python (temporalscope)"
echo "Jupyter kernel 'temporalscope' set up successfully."

# Copy README.md to /temporalscope directory and adjust paths
copy_and_adjust_readme

echo "Setup completed successfully."
