""" temporalscope/tests/unit/test_core_config.py

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import polars as pl
import pandas as pd
import modin.pandas as mpd
from temporalscope.config import (
    get_default_backend_cfg,
    validate_backend,
    validate_input,
)


def test_get_default_backend_cfg():
    """Test that the default backend configuration is returned correctly."""
    expected_cfg = {
        "BACKENDS": {"pl": "polars", "pd": "pandas", "mpd": "modin"},
    }
    result = get_default_backend_cfg()
    assert result == expected_cfg


def test_validate_backend_supported():
    """Test that supported backends are validated successfully."""
    validate_backend("pl")
    validate_backend("pd")
    validate_backend("mpd")


@pytest.mark.parametrize("invalid_backend", ["tf", "spark", "unknown"])
def test_validate_backend_unsupported(invalid_backend):
    """Test that unsupported backends raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        validate_backend(invalid_backend)


@pytest.mark.parametrize(
    "backend, df",
    [
        ("pl", pl.DataFrame({"a": [1, 2, 3]})),  # Polars DataFrame
        ("pd", pd.DataFrame({"a": [1, 2, 3]})),  # Pandas DataFrame
        ("mpd", mpd.DataFrame({"a": [1, 2, 3]})),  # Modin DataFrame
    ],
)
def test_validate_input_valid(backend, df):
    """Test that valid DataFrame types are accepted based on the backend."""
    validate_input(df, backend)


@pytest.mark.parametrize(
    "backend, df",
    [
        ("pl", pd.DataFrame({"a": [1, 2, 3]})),  # Invalid Polars input
        ("pd", pl.DataFrame({"a": [1, 2, 3]})),  # Invalid Pandas input
        ("mpd", pd.DataFrame({"a": [1, 2, 3]})),  # Invalid Modin input
    ],
)
def test_validate_input_invalid(backend, df):
    """Test that invalid DataFrame types raise a TypeError based on the backend."""
    with pytest.raises(TypeError, match="Expected a .* DataFrame"):
        validate_input(df, backend)
