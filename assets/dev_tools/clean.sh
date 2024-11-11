#!/bin/bash

echo "🧹 Cleaning TemporalScope development environment..."

# Check if we're in a hatch shell
if [[ -n "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Please exit the hatch shell first (use 'exit' command)"
    exit 1
fi

# List current kernels
echo "📋 Current Jupyter kernels:"
jupyter kernelspec list

# Remove all Jupyter kernels related to TemporalScope
echo "🗑️  Removing TemporalScope Jupyter kernels..."
jupyter kernelspec list | grep -E "temporalscope|gpu-temporalscope" | awk '{print $1}' | xargs -I {} jupyter kernelspec remove -f {}

# Clean Hatch environments
echo "🧼 Cleaning Hatch environments..."
hatch env prune
rm -rf ~/.local/share/hatch/env/virtual/temporalscope*

# Clean Conda environments (if conda is available)
if command -v conda &> /dev/null; then
    echo "🐍 Cleaning Conda environments..."
    # List all Conda environments containing 'temporalscope'
    CONDA_ENVS=$(conda env list | grep temporalscope | awk '{print $1}')
    if [ -n "$CONDA_ENVS" ]; then
        echo "Found Conda environments to remove:"
        echo "$CONDA_ENVS"
        for env in $CONDA_ENVS; do
            echo "Removing Conda environment: $env"
            conda env remove -n "$env" -y
        done
    else
        echo "No TemporalScope Conda environments found"
    fi
else
    echo "⚠️ Conda is not installed or available on this system."
fi

# Clean Poetry virtual environments (if Poetry is being used)
if command -v poetry &> /dev/null; then
    echo "🎵 Cleaning Poetry environments..."
    POETRY_ENVS=$(poetry env list --full-path | grep temporalscope | awk '{print $1}')
    if [ -n "$POETRY_ENVS" ]; then
        echo "Found Poetry environments to remove:"
        echo "$POETRY_ENVS"
        for env_path in $POETRY_ENVS; do
            echo "Removing Poetry environment at: $env_path"
            rm -rf "$env_path"
        done
    else
        echo "No TemporalScope Poetry environments found."
    fi
else
    echo "⚠️ Poetry is not installed or available on this system."
fi

# Remove stale pip cache related to TemporalScope
echo "🛠️  Cleaning pip cache..."
pip cache purge

# Clear Python cache
echo "🗑️  Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -name "*.pyc" -delete

# Clear build artifacts
echo "🔨 Removing build artifacts..."
find . -type d -name "build" -exec rm -r {} +
find . -type d -name "dist" -exec rm -r {} +
find . -type d -name "*.egg-info" -exec rm -r {} +

# Clear Hatch metadata cache (if applicable)
echo "🧼 Cleaning Hatch metadata cache..."
rm -rf ~/.cache/hatch

# Clear old log files (optional)
echo "📝 Cleaning old log files..."
find . -type f -name "*.log" -delete

echo "✨ Environment cleaned! Now you can:"
echo "   1. For CPU development:"
echo "      - Run 'hatch shell'"
echo ""
echo "   2. For GPU development:"
echo "      - Run 'conda env create -f environment-extras.yml'"
echo "      - Or 'conda create -n temporalscope-conda-extras -c conda-forge -c rapidsai -c nvidia python=3.11 vaex<5.0 cudf>=23.08.00 pytorch cudatoolkit=11.3 tensorflow'"
echo ""
echo "   3. For testing environments:"
echo "      - Run 'hatch shell test'"
