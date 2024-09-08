#!/bin/bash

# TemporalScope/setup.sh
# This script automates the environment setup for the TemporalScope project. 
# It supports both Poetry and Hatch for managing environments.

# Ensure the script runs from the root project directory (one level up from the script)
cd "$(dirname "$0")/.."

# Function to print a divider for clarity
print_divider() {
    echo -e "\n================================================================"
}

# Function to remove __pycache__ directories
remove_pycaches() {
    print_divider
    echo "Removing all __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "__pycache__ directories removed."
}

# Function to check if Python version is compatible
check_python_version() {
    print_divider
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
}

# ============= Poetry Setup Functions ============= #

# Function to install Poetry if not already installed
install_poetry() {
    print_divider
    if ! command -v poetry &> /dev/null; then
        echo "Poetry not found. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        echo "Poetry installed."
    else
        echo "Poetry is already installed."
    fi
}

# Function to configure the Poetry virtual environment
setup_poetry_virtual_environment() {
    print_divider
    PYTHON_MINOR=$(python3 --version | awk '{print $2}' | cut -d. -f2)
    echo "Configuring the Poetry virtual environment to use Python $PYTHON_VERSION..."
    poetry env use python3.$PYTHON_MINOR
    echo "Virtual environment set to use Python $PYTHON_VERSION."
}

# Function to install project dependencies via Poetry
install_poetry_dependencies() {
    print_divider
    echo "Installing project dependencies using Poetry..."
    poetry install
    echo "Poetry dependencies installed."
}

# Function to set up a Jupyter kernel for the project
setup_poetry_jupyter_kernel() {
    print_divider
    echo "Setting up Jupyter kernel named 'temporalscope_poetry'..."
    poetry run python3 -m ipykernel install --user --name=temporalscope_poetry --display-name "Python (temporalscope_poetry)"
    echo "Jupyter kernel 'temporalscope_poetry' set up successfully."
}

# ============= Hatch Setup Functions ============= #

# Function to install project dependencies via Hatch
install_hatch_dependencies() {
    print_divider
    echo "Installing project dependencies using Hatch..."
    hatch env create
    echo "Hatch dependencies installed."
}

# Function to set up a Jupyter kernel for the project using Hatch
setup_hatch_jupyter_kernel() {
    print_divider
    echo "Setting up Jupyter kernel named 'temporalscope_hatch'..."
    hatch run python3 -m ipykernel install --user --name=temporalscope_hatch --display-name "Python (temporalscope_hatch)"
    echo "Jupyter kernel 'temporalscope_hatch' set up successfully."
}

# Function to handle Hatch environment removal
remove_hatch_env() {
    print_divider
    echo "Removing existing Hatch environment..."
    hatch env remove default
    echo "Hatch environment removed."
}

# ============= Main Flow ============= #

# Function to copy and adjust the README.md file
copy_readme() {
    print_divider
    if [ -f "README.md" ]; then
        echo "Copying README.md to the temporalscope directory..."
        cp README.md temporalscope/
        echo "Adjusting image paths in the copied README.md..."
        sed -i 's/src="assets\//src="\.\.\/assets\//g' temporalscope/README.md
        echo "Image paths adjusted successfully."
    else
        echo "README.md not found. Skipping copy."
    fi
}

# Main script execution flow
main() {
    # Choose between Hatch and Poetry
    ENV_TOOL=$1  # Accept either 'poetry' or 'hatch' as argument, default to 'hatch'
    ENV_TOOL=${ENV_TOOL:-hatch}

    remove_pycaches
    check_python_version

    if [ "$ENV_TOOL" == "poetry" ]; then
        # Use Poetry for environment setup
        install_poetry
        setup_poetry_virtual_environment
        install_poetry_dependencies
        setup_poetry_jupyter_kernel
    else
        # Default to Hatch for environment setup
        remove_hatch_env  # Optional: remove existing Hatch environment
        install_hatch_dependencies
        setup_hatch_jupyter_kernel
    fi

    copy_readme

    # Set vim as the global editor for Git
    git config --global core.editor "vim"

    echo "✨   TemporalScope setup completed successfully using $ENV_TOOL!"
    echo "Run the package using:"
    if [ "$ENV_TOOL" == "poetry" ]; then
        echo "     $ poetry shell # to activate the Poetry shell"
    else
        echo "     $ hatch shell # to activate the Hatch environment"
    fi
    echo "Access Jupyter Notebooks via the 'temporalscope_${ENV_TOOL}' kernel installed."
}

# Run the main function with the provided argument (or default to Hatch)
main "$@"
