#!/bin/bash

# Step 1: Generate documentation
echo "Generating documentation using mkdocstrings..."
poetry run mkdocs build

# Step 2: Serve documentation locally for preview
echo "Serving documentation at http://127.0.0.1:8000"
poetry run mkdocs serve
