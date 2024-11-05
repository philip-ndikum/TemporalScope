# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import pytest
import warnings
import os
from dotenv import load_dotenv
from temporalscope.core.core_utils import (
    get_narwhals_backends,
    get_default_backend_cfg,
    get_temporalscope_backends,
    validate_backend,
    get_api_keys,
    print_divider,
    convert_to_backend,
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    TEMPORALSCOPE_OPTIONAL_BACKENDS,
    UnsupportedBackendError,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series
import pandas as pd
import modin.pandas as mpd
import pyarrow as pa
import polars as pl
import dask.dataframe as dd

# Constants
VALID_BACKENDS = ["pandas", "modin", "pyarrow", "polars", "dask"]
INVALID_BACKEND = "unsupported_backend"

# ========================= Tests for get_narwhals_backends =========================


def test_get_narwhals_backends():
    """Test the retrieval of Narwhals-supported backends."""
    backends = get_narwhals_backends()
    assert "pandas" in backends, "Expected 'pandas' backend in Narwhals backends list."
    assert "modin" in backends, "Expected 'modin' backend in Narwhals backends list."


# ========================= Tests for get_default_backend_cfg =========================


def test_get_default_backend_cfg():
    """Test retrieval of the default backend configuration for Narwhals."""
    cfg = get_default_backend_cfg()
    assert isinstance(cfg, dict), "Expected default backend configuration to be a dictionary."
    assert "BACKENDS" in cfg, "Expected 'BACKENDS' key in default configuration."
    assert all(
        backend in cfg["BACKENDS"] for backend in get_narwhals_backends()
    ), "Mismatch in default backend configuration."


# ========================= Tests for get_temporalscope_backends =========================


def test_get_temporalscope_backends():
    """Test that only TemporalScope-compatible backends are returned."""
    backends = get_temporalscope_backends()
    assert all(
        backend in VALID_BACKENDS for backend in backends
    ), "Non-compatible backend found in TemporalScope backends."


# ========================= Tests for validate_backend =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_validate_backend_supported(backend):
    """Test that validate_backend passes for supported backends."""
    try:
        validate_backend(backend)
    except UnsupportedBackendError:
        pytest.fail(f"validate_backend raised UnsupportedBackendError for valid backend '{backend}'.")


def test_validate_backend_unsupported():
    """Test that validate_backend raises error for unsupported backend."""
    with pytest.raises(UnsupportedBackendError):
        validate_backend(INVALID_BACKEND)


def test_validate_backend_optional_warning():
    """Test that validate_backend issues a warning for optional backends if available."""
    # Check if "cudf" is optional in TemporalScope
    if "cudf" in TEMPORALSCOPE_OPTIONAL_BACKENDS:
        # Expect a warning if "cudf" is not installed
        with pytest.warns(UserWarning, match="optional and requires additional setup"):
            validate_backend("cudf")
    else:
        pytest.skip("Skipping test as 'cudf' is not an optional backend in this configuration.")


# ========================= Tests for get_api_keys =========================


def test_get_api_keys_present(monkeypatch):
    """Test retrieval of API keys when they are set in the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_openai")
    monkeypatch.setenv("CLAUDE_API_KEY", "test_key_claude")
    api_keys = get_api_keys()
    assert api_keys["OPENAI_API_KEY"] == "test_key_openai", "Expected OPENAI_API_KEY to match environment variable."
    assert api_keys["CLAUDE_API_KEY"] == "test_key_claude", "Expected CLAUDE_API_KEY to match environment variable."


def test_get_api_keys_absent(monkeypatch, capsys):
    """Test warnings when API keys are missing from environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    get_api_keys()
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY is not set" in captured.out, "Expected warning for missing OPENAI_API_KEY."
    assert "CLAUDE_API_KEY is not set" in captured.out, "Expected warning for missing CLAUDE_API_KEY."


# ========================= Tests for print_divider =========================


def test_print_divider_default(capsys):
    """Test default divider output."""
    print_divider()
    captured = capsys.readouterr()
    assert captured.out == "=" * 70 + "\n", "Expected default divider output."


def test_print_divider_custom(capsys):
    """Test custom character and length in divider output."""
    print_divider(char="*", length=30)
    captured = capsys.readouterr()
    assert captured.out == "*" * 30 + "\n", "Expected custom divider output."


# ========================= Tests for convert_to_backend =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_convert_to_backend_valid(backend):
    """Test DataFrame conversion to each backend."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=2)
    converted_df = convert_to_backend(df, backend)
    expected_type = TEMPORALSCOPE_CORE_BACKEND_TYPES[backend]
    assert isinstance(converted_df, expected_type), f"Expected {expected_type} for backend '{backend}'."


def test_convert_to_backend_invalid():
    """Test that convert_to_backend raises error with unsupported backend."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=2)
    with pytest.raises(ValueError):
        convert_to_backend(df, INVALID_BACKEND)
