#!/bin/bash

# TemporalScope/run_tests.sh
# This script automates running the test suites (unit, integration, and benchmark tests)
# using Poetry and Pytest for the TemporalScope project. It supports running tests
# in verbose mode and will check if the provided test directories exist before running
# tests. In case of failure in any test suite, the script will return an appropriate exit code.

# Function to run tests with optional verbose mode
run_tests() {
    local directory=$1
    local verbose_flag=""

    if [ "$2" == "verbose" ]; then
        verbose_flag="--verbose"
    fi

    if [ -d "$directory" ]; then
        echo "Running tests in $directory..."
        poetry run pytest "$directory" $verbose_flag || {
            echo "Tests failed in $directory."
            exit 1
        }
    else
        echo "Directory $directory does not exist. Skipping..."
    fi
}

# Run all tests in the 'tests' directory with optional verbose mode
run_all_tests() {
    if [ "$1" == "verbose" ]; then
        echo "Running all tests in verbose mode..."
        poetry run pytest temporalscope/tests --verbose || {
            echo "All tests failed."
            exit 1
        }
    else
        echo "Running all tests in default mode..."
        poetry run pytest temporalscope/tests || {
            echo "All tests failed."
            exit 1
        }
    fi
}

# Main function to run tests
main() {
    run_all_tests "$1"

    # Run unit tests
    run_tests "temporalscope/tests/unit" "$1"

    # Run integration tests
    run_tests "temporalscope/tests/integration" "$1"

    # Run benchmark tests
    run_tests "temporalscope/tests/benchmark" "$1"

    echo "✨    All TemporalScope tests completed successfully!"
}

# Run the main function
main "$1"
