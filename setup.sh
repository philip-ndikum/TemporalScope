#!/bin/bash

# Function to recursively remove all __pycache__ directories
remove_pycaches() {
    echo "Removing all __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "__pycache__ directories removed."
}

# Remove all __pycache__ directories
remove_pycaches

# Check for Python version
REQUIRED_PYTHON="3.10"
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Current Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "$REQUIRED_PYTHON"* ]]; then
    echo "Error: Python $REQUIRED_PYTHON.x is required. Current version: $PYTHON_VERSION"
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

# Ensure the virtual environment uses Python 3.10
echo "Configuring the virtual environment to use Python 3.10..."
poetry env use python3.10
echo "Virtual environment set to use Python 3.10."

# Install project dependencies
echo "Installing project dependencies..."
poetry install
echo "Project dependencies installed."

echo "Setup completed successfully."