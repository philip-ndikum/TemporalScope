#!/bin/bash

print_divider() {
    echo -e "\n================================================================"
}

remove_pycaches() {
    print_divider
    echo "Removing all __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "__pycache__ directories removed."
}

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

install_poetry() {
    print_divider
    if ! command -v poetry &> /dev/null
    then
        echo "Poetry not found. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        echo "Poetry installed."
    else
        echo "Poetry is already installed."
    fi
}

setup_virtual_environment() {
    print_divider
    PYTHON_MINOR=$(python3 --version | awk '{print $2}' | cut -d. -f2)
    echo "Configuring the virtual environment to use Python $PYTHON_VERSION..."
    poetry env use python3.$PYTHON_MINOR
    echo "Virtual environment set to use Python $PYTHON_VERSION."
}

install_dependencies() {
    print_divider
    echo "Installing project dependencies..."
    poetry install
    echo "Project dependencies installed."
}

setup_jupyter_kernel() {
    print_divider
    echo "Setting up Jupyter kernel named 'temporalscope'..."
    poetry run python3 -m ipykernel install --user --name=temporalscope --display-name "Python (temporalscope)"
    echo "Jupyter kernel 'temporalscope' set up successfully."
}

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

# Main script execution
remove_pycaches
check_python_version
install_poetry
setup_virtual_environment
install_dependencies
setup_jupyter_kernel
copy_readme

echo "TemporalScope ✨ setup completed successfully!"
echo "Run the package using:"
echo "     $ poetry shell # to activate the poetry shell"
echo "Access Jupyter Notebooks via the 'temporalscope' kernel installed."
