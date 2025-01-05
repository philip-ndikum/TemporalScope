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

"""Tests to verify tutorial notebooks execute successfully."""

from pathlib import Path

import papermill as pm
import pytest


def collect_notebooks():
    """Helper function to collect all notebooks in the tutorial directory."""
    notebook_dir = Path("tutorial_notebooks")
    return list(notebook_dir.rglob("*.ipynb"))


@pytest.mark.notebook
@pytest.mark.parametrize("notebook_path", collect_notebooks())
def test_notebook_runs(notebook_path: Path, tmp_path: Path):
    """Test that tutorial notebooks execute without errors."""
    output_path = tmp_path / f"{notebook_path.stem}_executed.ipynb"

    pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        progress_bar=False,
        stdout_file=None,
        stderr_file=None,
    )
