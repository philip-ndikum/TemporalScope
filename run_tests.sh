#!/bin/bash

# Function to run tests with optional verbose mode
run_tests() {
    local directory=$1
    local verbose_flag=""

    if [ "$2" == "verbose" ]; then
        verbose_flag="--verbose"
    fi

    if [ -d "$directory" ]; then
        echo "Running tests in $directory..."
        poetry run pytest "$directory" $verbose_flag
    else
        echo "Directory $directory does not exist. Skipping..."
    fi
}

# Run all tests in the 'tests' directory
if [ "$1" == "verbose" ]; then
    echo "Running all tests in verbose mode..."
    poetry run pytest temporalscope/tests --verbose
else
    echo "Running all tests in default mode..."
    poetry run pytest temporalscope/tests
fi

# Run unit tests
run_tests "temporalscope/tests/unit" "$1"

# Run integration tests
run_tests "temporalscope/tests/integration" "$1"

# Run benchmark tests
run_tests "temporalscope/tests/benchmark" "$1"
