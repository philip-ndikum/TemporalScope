# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests to verify tutorial notebooks execute successfully using Papermill.

This module provides automated testing of Jupyter notebooks using Papermill,
ensuring notebooks can execute successfully in any environment. The @pytest.mark.notebook
marker is used to separate these execution tests from standard unit tests, as they serve
different purposes:

- Unit tests: Verify specific functionality and code correctness
- Notebook tests: Ensure notebooks can execute end-to-end without errors

This separation allows for more efficient CI/CD pipelines and clearer test organization.
"""

from pathlib import Path

import papermill as pm
import pytest


def get_notebooks():
    """Get all notebooks from tutorial directory."""
    notebook_dir = Path("tutorial_notebooks")
    return list(notebook_dir.rglob("*.ipynb"))


@pytest.mark.notebook
@pytest.mark.parametrize("notebook_path", get_notebooks())
def test_notebook_runs(notebook_path, tmp_path):
    """Test that notebook executes without errors using Papermill.

    Uses the notebook marker to separate execution tests from unit tests,
    focusing solely on verifying that notebooks can run successfully.
    """
    output_path = tmp_path / notebook_path.name
    try:
        # Execute notebook and save to temporary directory
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            kernel_name="python3",
            progress_bar=False,  # Disable progress bar for cleaner test output
            stdout_file=None,  # Don't capture stdout
            stderr_file=None,  # Don't capture stderr
        )
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed to execute: {str(e)}")
