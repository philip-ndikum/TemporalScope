#!/bin/bash

# Check for Python version
REQUIRED_PYTHON="3.10"
PYTHON_VERSION=$(python3 --version | awk '{print $2}')

if [[ "$PYTHON_VERSION" != "$REQUIRED_PYTHON"* ]]; then
    echo "Python $REQUIRED_PYTHON.x is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "Poetry is already installed."
fi

# Install project dependencies
poetry install

echo "Setup completed successfully."
