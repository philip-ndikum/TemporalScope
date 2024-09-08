#!/bin/bash

# TemporalScope/precommit_quality_assurance.sh

# Ensure the script runs from the root project directory (one level up from the script)
cd "$(dirname "$0")/.."

# Function to print centered text
print_centered() {
    termwidth=$(tput cols)
    padding=$(printf '%0.1s' "="{1..500})

    # Adjust the text length manually for emoji
    text="$1"
    emoji_adjust=2  # Adjust by 2 characters for the emoji
    printf '%*.*s %s %*.*s\n' 0 "$(((termwidth-2-${#text}-emoji_adjust)/2))" "$padding" "$text" 0 "$(((termwidth-1-${#text}-emoji_adjust)/2))" "$padding"
}


# Function to clear caches
clear_cache() {
    print_centered "Clearing caches and temporary files"
    # Remove all __pycache__ and .ipynb_checkpoints recursively from the temporalscope directory
    find temporalscope -name '__pycache__' -type d -exec rm -rf {} +
    find temporalscope -name '.ipynb_checkpoints' -type d -exec rm -rf {} +
    print_centered "Cache cleared"
}

# Print the centered title bar for each step
print_centered "✨  Running Pre-Commit Quality Assurance Checks"

# Clear caches before running checks
clear_cache



# Run Black only on the temporalscope directory
print_centered "Running Black"
black temporalscope/

# Run Flake8 only on the temporalscope directory
print_centered "Running Flake8"
flake8 temporalscope/

# Run Mypy only on the temporalscope directory
print_centered "Running Mypy"
mypy temporalscope/

# Run Bandit with the .bandit config file and exclude the test directory
print_centered "Running Bandit"
bandit -r temporalscope/ -c .bandit

# Run Pytest for unit tests
print_centered "Running Pytest"
pytest --maxfail=1 --disable-warnings  # Adjust options as needed

print_centered "✨  TemporalScope Pre-commit QA checks completed!"

