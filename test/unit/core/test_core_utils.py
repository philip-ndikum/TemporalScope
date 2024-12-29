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

"""TemporalScope/test/unit/core/test_core_utils.py

This module contains unit tests for core utility functions. Tests use synthetic data
generation at runtime to ensure backend-agnostic operations work correctly across
all supported DataFrame backends (pandas, polars, etc.).

The testing pattern follows these principles:
1. Use synthetic_data_generator to create test data for each backend
2. Test functions with actual DataFrame implementations at runtime
3. Verify backend-agnostic behavior using Narwhals operations
"""

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    TimeColumnError,
    check_dataframe_empty,
    check_dataframe_nulls_nans,
    convert_datetime_column_to_numeric,
    convert_time_column_to_datetime,
    convert_to_numeric,
    get_api_keys,
    get_default_backend_cfg,
    get_narwhals_backends,
    print_divider,
    sort_dataframe_time,
    validate_and_convert_time_column,
    validate_dataframe_column_types,
    validate_time_column_type,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = get_narwhals_backends()


@pytest.fixture(params=VALID_BACKENDS)
def synthetic_df(request):
    """Generate synthetic DataFrame for each backend."""
    return generate_synthetic_time_series(backend=request.param, num_samples=10, num_features=2, drop_time=False)


# ========================= Tests for get_api_keys =========================


def test_get_api_keys_present(monkeypatch):
    """Test retrieval of API keys when they are set in the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_openai")
    monkeypatch.setenv("CLAUDE_API_KEY", "test_key_claude")
    api_keys = get_api_keys()
    assert api_keys["OPENAI_API_KEY"] == "test_key_openai"
    assert api_keys["CLAUDE_API_KEY"] == "test_key_claude"


def test_get_api_keys_absent(monkeypatch, capsys):
    """Test warnings when API keys are missing from environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    get_api_keys()
    captured = capsys.readouterr()
    assert "Warning: OPENAI_API_KEY is not set" in captured.out
    assert "Warning: CLAUDE_API_KEY is not set" in captured.out


# ========================= Tests for print_divider =========================


def test_print_divider_default(capsys):
    """Test default divider output."""
    print_divider()
    captured = capsys.readouterr()
    assert captured.out == "=" * 70 + "\n"


def test_print_divider_custom(capsys):
    """Test custom character and length in divider output."""
    print_divider(char="*", length=30)
    captured = capsys.readouterr()
    assert captured.out == "*" * 30 + "\n"


# ========================= Tests for get_narwhals_backends =========================


def test_get_narwhals_backends():
    """Test the retrieval of Narwhals-supported backends."""
    backends = get_narwhals_backends()
    assert isinstance(backends, list)
    assert all(isinstance(backend, str) for backend in backends)
    assert "pandas" in backends


# ========================= Tests for get_default_backend_cfg =========================


def test_get_default_backend_cfg():
    """Test retrieval of the default backend configuration."""
    cfg = get_default_backend_cfg()
    assert isinstance(cfg, dict)
    assert "BACKENDS" in cfg
    assert isinstance(cfg["BACKENDS"], list)
    assert all(backend in cfg["BACKENDS"] for backend in get_narwhals_backends())


# ========================= Tests for check_dataframe_empty =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_empty_df(backend):
    """Test check_dataframe_empty returns True for empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    assert check_dataframe_empty(df) is True


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_data(backend):
    """Test check_dataframe_empty returns False for non-empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=10, num_features=1, drop_time=True)
    assert check_dataframe_empty(df) is False


def test_check_dataframe_empty_none():
    """Test check_dataframe_empty error handling for None input."""
    with pytest.raises(ValueError, match="DataFrame cannot be None"):
        check_dataframe_empty(None)


# ========================= Tests for check_dataframe_nulls_nans =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_no_nulls(backend):
    """Test check_dataframe_nulls_nans with DataFrame containing no null values."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=2, with_nulls=False, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert result == {"feature_1": 0}


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_with_nulls(backend):
    """Test check_dataframe_nulls_nans with DataFrame containing null values."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=2, with_nulls=True, null_percentage=0.5, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1", "feature_2"])
    # With 50% null percentage, expect around half the values to be null
    assert 4 <= result["feature_1"] <= 6
    assert 4 <= result["feature_2"] <= 6


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_empty_dataframe(backend):
    """Test check_dataframe_nulls_nans with an empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        check_dataframe_nulls_nans(df, ["feature_1"])


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_nonexistent_column(backend):
    """Test check_dataframe_nulls_nans with nonexistent column."""
    df = generate_synthetic_time_series(backend=backend, num_samples=10, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        check_dataframe_nulls_nans(df, ["nonexistent"])


# ========================= Tests for convert_to_numeric =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_convert_to_numeric(backend):
    """Test convert_to_numeric with datetime column."""
    # Generate synthetic data with datetime time column
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=1, time_col_numeric=False, drop_time=False
    )
    # Convert to Narwhals format
    df_nw = nw.from_native(df)
    # Convert to numeric
    result = convert_to_numeric(df_nw, "time")
    # Verify result is float (handle different backend type names)
    dtype_str = str(result["time"].dtype).lower()
    assert any(float_type in dtype_str for float_type in ["float", "f64"])


def test_convert_to_numeric_invalid():
    """Test convert_to_numeric with non-datetime column."""
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError):
        convert_to_numeric(df, "time")


# ========================= Tests for convert_datetime_column_to_numeric =========================


def test_convert_datetime_column_to_numeric(synthetic_df):
    """Test convert_datetime_column_to_numeric with valid input."""
    result = convert_datetime_column_to_numeric(nw.from_native(synthetic_df), "time")
    # Check if dtype indicates float (handle different backends)
    dtype_str = str(result["time"].dtype).lower()
    assert any(float_type in dtype_str for float_type in ["float", "f64"])


def test_convert_datetime_column_to_numeric_already_numeric():
    """Test convert_datetime_column_to_numeric with already numeric column."""
    df = pd.DataFrame({"time": [1, 2, 3]})
    df = nw.from_native(df)
    result = convert_datetime_column_to_numeric(df, "time")
    # Convert both to pandas DataFrames for comparison
    result_pd = result if isinstance(result, pd.DataFrame) else result.to_native()
    df_pd = df if isinstance(df, pd.DataFrame) else df.to_native()
    pd.testing.assert_frame_equal(result_pd, df_pd)


def test_convert_datetime_column_to_numeric_invalid():
    """Test convert_datetime_column_to_numeric with invalid column."""
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError):
        convert_datetime_column_to_numeric(df, "time")


# ========================= Tests for convert_time_column_to_datetime =========================


def test_convert_time_column_to_datetime_from_string():
    """Test convert_time_column_to_datetime with string input."""
    df = pd.DataFrame({"time": ["2023-01-01", "2023-01-02"]})
    df = nw.from_native(df)
    result = convert_time_column_to_datetime(df, "time")
    assert pd.api.types.is_datetime64_dtype(result["time"].dtype)


def test_convert_time_column_to_datetime_from_numeric():
    """Test convert_time_column_to_datetime with numeric input."""
    df = pd.DataFrame({"time": [1672531200, 1672617600]})  # Unix timestamps
    df = nw.from_native(df)
    result = convert_time_column_to_datetime(df, "time")
    assert pd.api.types.is_datetime64_dtype(result["time"].dtype)


def test_convert_time_column_to_datetime_invalid():
    """Test convert_time_column_to_datetime with invalid input."""
    df = pd.DataFrame({"time": [True, False]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError):
        convert_time_column_to_datetime(df, "time")


# ========================= Tests for validate_time_column_type =========================


def test_validate_time_column_type_numeric():
    """Test validate_time_column_type with numeric column."""
    df = pd.DataFrame({"time": [1, 2, 3]})
    df = nw.from_native(df)
    validate_time_column_type(df, "time")


def test_validate_time_column_type_datetime():
    """Test validate_time_column_type with datetime column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    df = nw.from_native(df)
    validate_time_column_type(df, "time")


def test_validate_time_column_type_invalid():
    """Test validate_time_column_type with invalid column."""
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        validate_time_column_type(df, "time")


# ========================= Tests for validate_and_convert_time_column =========================


def test_validate_and_convert_time_column_numeric(synthetic_df):
    """Test validation and numeric conversion of time column."""
    result = validate_and_convert_time_column(nw.from_native(synthetic_df), "time", conversion_type="numeric")
    # Check if dtype indicates float (handle different backends)
    dtype_str = str(result["time"].dtype).lower()
    assert any(float_type in dtype_str for float_type in ["float", "f64"])


def test_validate_and_convert_time_column_datetime():
    """Test validation and datetime conversion of time column."""
    df = pd.DataFrame({"time": [1672531200, 1672617600]})
    df = nw.from_native(df)
    result = validate_and_convert_time_column(df, "time", conversion_type="datetime")
    assert pd.api.types.is_datetime64_dtype(result["time"].dtype)


def test_validate_and_convert_time_column_missing():
    """Test for missing time column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError):
        validate_and_convert_time_column(df, "time")


def test_validate_and_convert_time_column_invalid_type():
    """Test for invalid conversion type."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        validate_and_convert_time_column(df, "time", conversion_type="invalid")


# ========================= Tests for validate_dataframe_column_types =========================


def test_validate_dataframe_column_types_valid(synthetic_df):
    """Test validate_dataframe_column_types with valid columns."""
    validate_dataframe_column_types(synthetic_df, "time")


def test_validate_dataframe_column_types_invalid_time():
    """Test validate_dataframe_column_types with invalid time column."""
    df = pd.DataFrame({"time": ["a", "b", "c"], "value": [1, 2, 3]})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        validate_dataframe_column_types(df, "time")


def test_validate_dataframe_column_types_invalid_feature():
    """Test validate_dataframe_column_types with invalid feature column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "value": ["a", "b", "c"]})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        validate_dataframe_column_types(df, "time")


# ========================= Tests for sort_dataframe_time =========================


def test_sort_dataframe_time_numeric():
    """Test sort_dataframe_time with numeric time column."""
    df = pd.DataFrame({"time": [3, 1, 2], "value": [30, 10, 20]})
    df = nw.from_native(df)
    result = sort_dataframe_time(df, "time")
    expected = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
    # Convert result to pandas DataFrame if needed
    result_pd = result if isinstance(result, pd.DataFrame) else result.to_native()
    pd.testing.assert_frame_equal(result_pd.reset_index(drop=True), expected)


def test_sort_dataframe_time_datetime(synthetic_df):
    """Test sort_dataframe_time with datetime time column."""
    df_nw = nw.from_native(synthetic_df)
    # Create a copy by selecting all columns
    shuffled = df_nw.select([nw.col(col) for col in df_nw.columns])
    # Shuffle the DataFrame - handle different backends
    native_df = shuffled.to_native()
    # Check for specific DataFrame types
    if hasattr(native_df, "iloc"):  # pandas-like DataFrame
        shuffled = nw.from_native(native_df.sample(frac=1, random_state=42))
    else:  # polars DataFrame
        shuffled = nw.from_native(native_df.sample(fraction=1.0, seed=42))
    result = sort_dataframe_time(shuffled, "time")
    # Convert both to pandas Series for comparison
    result_time = pd.Series(result["time"].to_native() if hasattr(result["time"], "to_native") else result["time"])
    df_time = pd.Series(df_nw["time"].to_native() if hasattr(df_nw["time"], "to_native") else df_nw["time"])
    pd.testing.assert_series_equal(result_time.reset_index(drop=True), df_time.reset_index(drop=True))


def test_sort_dataframe_time_missing():
    """Test sort_dataframe_time with missing time column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        sort_dataframe_time(df, "time")
