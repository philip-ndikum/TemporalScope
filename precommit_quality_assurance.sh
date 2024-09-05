#!/bin/bash

# Pre-Commit Quality Assurance checks

print_centered() {
    termwidth=$(tput cols)
    padding=$(printf '%0.1s' "="{1..500})
    printf '%*.*s %s %*.*s\n' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Print the centered title bar for each step

print_centered "Running Pre-Commit Quality Assurance Checks"

print_centered "Running Black"
black .

print_centered "Running Flake8"
flake8 .

print_centered "Running Mypy"
mypy .

print_centered "Running Bandit"
bandit -r .

print_centered "TemporalScope ✨ Pre-commit QA checks completed!"
