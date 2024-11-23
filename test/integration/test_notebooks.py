"""Simple tests to verify tutorial notebooks execute successfully."""
import pytest
import papermill as pm
from pathlib import Path

def get_notebooks():
    """Get all notebooks from tutorial directory."""
    notebook_dir = Path("tutorial_notebooks")
    return list(notebook_dir.rglob("*.ipynb"))

@pytest.mark.parametrize("notebook_path", get_notebooks())
def test_notebook_runs(notebook_path):
    """Test that notebook executes without errors."""
    try:
        pm.execute_notebook(
            str(notebook_path),
            "/dev/null",  # Discard output
            kernel_name="python3"
        )
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {str(e)}")
