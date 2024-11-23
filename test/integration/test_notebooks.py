"""Simple tests to verify tutorial notebooks execute successfully using Papermill."""
from pathlib import Path

import papermill as pm
import pytest


def get_notebooks():
    """Get all notebooks from tutorial directory."""
    notebook_dir = Path("tutorial_notebooks")
    return list(notebook_dir.rglob("*.ipynb"))

@pytest.mark.parametrize("notebook_path", get_notebooks())
def test_notebook_runs(notebook_path, tmp_path):
    """Test that notebook executes without errors using Papermill."""
    output_path = tmp_path / notebook_path.name
    try:
        # Execute notebook and save to temporary directory
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            kernel_name="python3",
            progress_bar=False,  # Disable progress bar for cleaner test output
            stdout_file=None,    # Don't capture stdout
            stderr_file=None     # Don't capture stderr
        )
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {str(e)}")
