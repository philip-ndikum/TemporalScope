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

"""TemporalScope/test/unit/core/test_core_utils.py

This module contains unit tests for core utility functions.
"""

from typing import Any, Callable, Dict
from unittest import mock

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    TEMPORALSCOPE_BACKEND_CONVERTERS,
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    TEMPORALSCOPE_OPTIONAL_BACKENDS,
    TimeColumnError,
    UnsupportedBackendError,
    check_dataframe_empty,
    check_dataframe_nulls_nans,
    check_strict_temporal_ordering,
    check_temporal_order_and_uniqueness,
    convert_to_backend,
    convert_to_datetime,
    convert_to_numeric,
    get_api_keys,
    get_dataframe_backend,
    get_default_backend_cfg,
    get_narwhals_backends,
    get_temporalscope_backends,
    is_lazy_evaluation,
    is_valid_temporal_backend,
    is_valid_temporal_dataframe,
    print_divider,
    sort_dataframe_time,
    validate_and_convert_time_column,
    validate_dataframe_column_types,
    validate_time_column_type,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = ["pandas", "modin", "pyarrow", "polars", "dask"]
INVALID_BACKEND = "unsupported_backend"

# ========================= Fixtures =========================


@pytest.fixture(params=VALID_BACKENDS)
def synthetic_df(request):
    """Fixture providing synthetic DataFrames for each backend.

    :param request: pytest request object containing the backend parameter
    :return: DataFrame in the specified backend format
    """
    return generate_synthetic_time_series(backend=request.param, num_samples=10, num_features=2)


@pytest.fixture
def narwhalified_df():
    """Fixture providing a narwhalified DataFrame."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=2)
    return nw.from_native(df)


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


# ========================= Tests for is_valid_temporal_backend =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_is_valid_temporal_backend_supported(backend):
    """Test that is_valid_temporal_backend passes for supported backends."""
    try:
        is_valid_temporal_backend(backend)
    except UnsupportedBackendError:
        pytest.fail(f"is_valid_temporal_backend raised UnsupportedBackendError for valid backend '{backend}'.")


def test_is_valid_temporal_backend_unsupported():
    """Test that is_valid_temporal_backend raises error for unsupported backend."""
    with pytest.raises(UnsupportedBackendError):
        is_valid_temporal_backend(INVALID_BACKEND)


def test_is_valid_temporal_backend_optional_warning():
    """Test that is_valid_temporal_backend issues a warning for optional backends if available."""
    # Check if "cudf" is optional in TemporalScope
    if "cudf" in TEMPORALSCOPE_OPTIONAL_BACKENDS:
        # Expect a warning if "cudf" is not installed
        with pytest.warns(UserWarning, match="optional and requires additional setup"):
            is_valid_temporal_backend("cudf")
    else:
        pytest.skip("Skipping test as 'cudf' is not an optional backend in this configuration.")


# ========================= Tests for is_valid_temporal_dataframe =========================


def test_is_valid_temporal_dataframe_supported(synthetic_df):
    """Test that is_valid_temporal_dataframe returns True for supported DataFrame types."""
    # Check if the DataFrame is valid
    is_valid, df_type = is_valid_temporal_dataframe(synthetic_df)
    assert is_valid, "Expected DataFrame to be valid."
    assert df_type == "native", "Expected DataFrame type to be 'native'."


def test_is_valid_temporal_dataframe_narwhalified(narwhalified_df):
    """Test that is_valid_temporal_dataframe handles narwhalified DataFrames."""
    # Check if the narwhalified DataFrame is valid
    is_valid, df_type = is_valid_temporal_dataframe(narwhalified_df)
    assert is_valid, "Expected narwhalified DataFrame to be valid."
    assert df_type == "narwhals", "Expected DataFrame type to be 'narwhals' for narwhalified DataFrame."


def test_is_valid_temporal_dataframe_unsupported():
    """Test that is_valid_temporal_dataframe returns False for unsupported DataFrame types."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()

    is_valid, df_type = is_valid_temporal_dataframe(df)
    assert not is_valid, "Expected DataFrame to be invalid for unsupported type."
    assert df_type is None, "Expected DataFrame type to be None for unsupported type."


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


# ========================= Tests for is_lazy_evaluation =========================


def test_is_lazy_evaluation_invalid_dataframe():
    """Test is_lazy_evaluation raises error for unsupported DataFrame."""

    class InvalidDataFrame:
        pass

    df = InvalidDataFrame()
    with pytest.raises(UnsupportedBackendError, match="The input DataFrame is not supported by TemporalScope."):
        is_lazy_evaluation(df)


def test_is_lazy_evaluation_dask(synthetic_df, request):
    """Test lazy evaluation detection for dask backend."""
    if request.node.callspec.params["synthetic_df"] == "dask":
        assert is_lazy_evaluation(synthetic_df), "Expected dask DataFrame to use lazy evaluation"


def test_is_lazy_evaluation_polars_lazy(synthetic_df, request):
    """Test lazy evaluation detection for polars lazy backend."""
    if request.node.callspec.params["synthetic_df"] == "polars-lazy":
        assert is_lazy_evaluation(synthetic_df), "Expected polars lazy DataFrame to use lazy evaluation"


def test_is_lazy_evaluation_eager(synthetic_df, request):
    """Test lazy evaluation detection for eager backends."""
    backend = request.node.callspec.params["synthetic_df"]
    if backend not in ["dask", "polars-lazy"]:
        assert not is_lazy_evaluation(synthetic_df), f"Expected {backend} DataFrame to use eager evaluation"


def test_is_lazy_evaluation_narwhalified(narwhalified_df):
    """Test lazy evaluation detection for narwhalified DataFrame."""
    assert not is_lazy_evaluation(narwhalified_df), "Expected narwhalified DataFrame to use eager evaluation"


@pytest.fixture(params=get_temporalscope_backends())
def backend(request) -> str:
    """Fixture providing all supported backends for testing."""
    return request.param


@pytest.fixture
def data_config(backend: str) -> Callable[..., Dict[str, Any]]:
    """Base fixture for data generation configuration."""

    def _config(**kwargs) -> Dict[str, Any]:
        default_config = {
            "num_samples": 3,
            "num_features": 2,
            "with_nulls": False,
            "with_nans": False,
            "backend": backend,
            "drop_time": True,
            "random_seed": 42,
        }
        default_config.update(kwargs)
        return default_config

    return _config


# ========================= check_dataframe_empty Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_empty_df(backend: str) -> None:
    """Test check_dataframe_empty returns True for empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    df = nw.from_native(df)
    assert check_dataframe_empty(df) is True


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_data(backend: str) -> None:
    """Test check_dataframe_empty returns False for non-empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    df = nw.from_native(df)
    assert check_dataframe_empty(df) is False


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_lazy_evaluation(backend: str) -> None:
    """Test check_dataframe_empty works with lazy evaluation."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    df = nw.from_native(df)
    if hasattr(df.to_native(), "lazy"):
        df = df.to_native().lazy()
        df = nw.from_native(df)
    assert check_dataframe_empty(df) is True


@pytest.mark.parametrize("backend", ["dask"])
def test_check_dataframe_empty_lazy_compute(backend: str) -> None:
    """Test check_dataframe_empty handles lazy evaluation with compute."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    df = nw.from_native(df)

    if hasattr(df, "compute"):
        df.compute = lambda: nw.from_native({"col": []})
        assert check_dataframe_empty(df) is True


# Tests for error handling
def test_check_dataframe_empty_error_handling() -> None:
    """Test check_dataframe_empty error handling for None input."""
    with pytest.raises(ValueError, match="DataFrame cannot be None"):
        check_dataframe_empty(None)


def test_check_dataframe_empty_unsupported_type() -> None:
    """Test check_dataframe_empty raises ValueError for unsupported DataFrame type."""

    class UnsupportedDataFrame:
        """Custom DataFrame type not supported by Narwhals API."""

        pass

    df = UnsupportedDataFrame()
    with pytest.raises(ValueError, match="Unsupported DataFrame type"):
        check_dataframe_empty(df)


# ========================= check_dataframe_nulls_nans Tests =========================


@pytest.fixture
def sample_df():
    """Generate sample DataFrame for testing check_dataframe_nulls_nans function.

    Uses synthetic_data_generator to create consistent test data across backends.
    """
    return generate_synthetic_time_series(
        backend="pandas", num_samples=3, num_features=2, with_nulls=True, null_percentage=0.3, drop_time=True
    )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_no_nulls(backend: str) -> None:
    """Test check_dataframe_nulls_nans with DataFrame containing no null values.

    Uses synthetic data generator with no nulls to verify check_dataframe_nulls_nans correctly
    identifies when there are no null values in specified columns.
    """
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, with_nulls=False, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert result == {"feature_1": 0}


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_with_nulls(backend: str) -> None:
    """Test check_dataframe_nulls_nans with DataFrame containing null values.

    Uses synthetic data generator with nulls to verify check_dataframe_nulls_nans correctly
    counts null values in specified columns.
    """
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=1, with_nulls=True, null_percentage=0.5, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert 4 <= result["feature_1"] <= 6  # ~50% nulls


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_empty_dataframe(backend: str) -> None:
    """Test check_dataframe_nulls_nans with an empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Empty"):  # Generic pattern
        check_dataframe_nulls_nans(df, ["feature_1"])


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_nonexistent_column(backend: str) -> None:
    """Test check_dataframe_nulls_nans with nonexistent column."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found."):  # Match the specific message
        check_dataframe_nulls_nans(df, ["nonexistent"])


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_empty_column_list(backend: str) -> None:
    """Test check_dataframe_nulls_nans with empty list of columns.

    Verifies check_dataframe_nulls_nans returns empty dict for empty column list.
    """
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    result = check_dataframe_nulls_nans(df, [])
    assert result == {}


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_all_nulls(backend: str) -> None:
    """Test check_dataframe_nulls_nans with columns containing all null values.

    Uses synthetic data generator with 100% null percentage to verify
    check_dataframe_nulls_nans correctly identifies when all values are null.
    """
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, with_nulls=True, null_percentage=1.0, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert result == {"feature_1": 3}


@pytest.mark.parametrize("backend", ["dask"])
def test_check_dataframe_nulls_nans_lazy_compute(backend: str) -> None:
    """Test check_dataframe_nulls_nans handles lazy evaluation with compute."""
    # Generate a real synthetic Dask DataFrame with nulls
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=1, with_nulls=True, null_percentage=0.5, drop_time=True
    )

    # Perform the test directly without mocking
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    print(f"Result from check_dataframe_nulls_nans:\n{result}")  # Debug: Verify result

    # Verify null counts (depends on generator behavior)
    assert "feature_1" in result
    assert 0 <= result["feature_1"] <= 5  # Adjust based on null_percentage


# ========================= validate_and_convert_time_column Tests =========================


@pytest.mark.parametrize("backend", ["pandas", "modin", "polars", "pyarrow", "dask"])
def test_validate_and_convert_time_column_numeric_single(backend):
    """Test validation and numeric conversion of time column for a single backend."""
    # Generate synthetic data with the specified backend
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, drop_time=False, time_col_numeric=False
    )

    # Convert the time column to numeric
    result = validate_and_convert_time_column(df, "time", conversion_type="numeric")

    # Ensure the column was converted properly to numeric
    if backend in ["pyarrow"]:
        # PyArrow: Check if `time` exists in the schema (field names)
        assert "time" in [
            field.name for field in result.schema
        ], f"'time' column not found in result for backend: {backend}"
    else:
        assert "time" in result.columns, f"'time' column not found in result for backend: {backend}"

    # Backend-specific dtype validation
    if backend in ["pandas", "modin"]:
        # Pandas/Modin: Use DataFrame dtypes directly
        resolved_dtype = result["time"].dtype
        assert resolved_dtype in ["float64", "float32"], f"Expected numeric dtype for 'time', got {resolved_dtype}"

    elif backend == "polars":
        # Polars: Use `schema` for dtype resolution
        resolved_dtype = result.schema.get("time")
        assert str(resolved_dtype).lower() in [
            "float64",
            "float32",
        ], f"Expected numeric dtype for Polars, got {resolved_dtype}"

    elif backend == "pyarrow":
        # PyArrow: Validate `ChunkedArray` dtype
        resolved_dtype = result.schema.field("time").type
        assert str(resolved_dtype).lower() in [
            "double",
            "float64",
        ], f"Expected numeric dtype for PyArrow, got {resolved_dtype}"

    elif backend == "dask":
        # Dask: Resolve dtype from `_meta`
        resolved_dtype = result["time"]._meta.dtype
        assert resolved_dtype in ["float64", "float32"], f"Expected numeric dtype for Dask, got {resolved_dtype}"


# ========================= Edge case tests =========================


def test_convert_to_numeric_unsupported_dataframe() -> None:
    """Test convert_to_numeric raises ValueError for unsupported DataFrame type."""

    class UnsupportedDataFrame:
        """Mock class to simulate an unsupported DataFrame."""

        pass

    # Create an instance of the unsupported DataFrame
    unsupported_df = UnsupportedDataFrame()

    # Attempt to call convert_to_numeric and expect a ValueError
    with pytest.raises(ValueError, match=f"Unsupported DataFrame type: {type(unsupported_df).__name__}"):
        convert_to_numeric(unsupported_df, time_col="time", col_expr=None, col_dtype="datetime64[ns]")


def test_convert_to_numeric_with_timezones():
    """Test convert_to_numeric with timezone-aware and naive datetime columns.

    This test ensures that:
    - Timezone-aware datetime columns are correctly converted to numeric timestamps.
    - Naive datetime columns are handled correctly.
    - Columns with invalid types raise appropriate errors.
    """
    # Create a DataFrame with different datetime columns
    df = pd.DataFrame(
        {
            "naive_datetime": pd.date_range("2023-01-01", periods=3),
            "aware_datetime": pd.date_range("2023-01-01", periods=3, tz="UTC"),
            "invalid_column": ["not_a_datetime", "still_not", "nope"],
        }
    )

    # Update col_dtype to match actual Pandas dtypes
    naive_dtype = df["naive_datetime"].dtype
    aware_dtype = df["aware_datetime"].dtype

    # Test naive datetime column
    result = convert_to_numeric(df, "naive_datetime", nw.col("naive_datetime"), naive_dtype)
    assert "naive_datetime" in result.columns
    assert pd.api.types.is_float_dtype(result["naive_datetime"]), "Expected numeric dtype for naive datetime."

    # Test timezone-aware datetime column
    result = convert_to_numeric(df, "aware_datetime", nw.col("aware_datetime"), aware_dtype)
    assert "aware_datetime" in result.columns
    assert pd.api.types.is_float_dtype(result["aware_datetime"]), "Expected numeric dtype for timezone-aware datetime."

    # Test invalid column type
    with pytest.raises(ValueError, match="not a datetime column"):
        convert_to_numeric(df, "invalid_column", nw.col("invalid_column"), df["invalid_column"].dtype)


def test_convert_to_numeric_error_handling():
    """Test convert_to_numeric error handling for invalid column types."""
    # Create a DataFrame with an invalid column type
    df = pd.DataFrame({"invalid_col": ["not_a_datetime", "nope", "still_not"]})

    # Test invalid column type
    with pytest.raises(ValueError, match="is not a datetime column"):
        convert_to_numeric(df, "invalid_col", nw.col("invalid_col"), df["invalid_col"].dtype)


def test_validate_time_column_type():
    """Test validate_time_column_type for various scenarios."""
    # Test valid numeric column
    validate_time_column_type("numeric_col", "float64")  # Should not raise an error

    # Test valid datetime column
    validate_time_column_type("datetime_col", "datetime64[ns]")  # Should not raise an error

    # Test invalid column type
    with pytest.raises(ValueError, match="neither numeric nor datetime"):
        validate_time_column_type("invalid_col", "string")

    # Test mixed-type column (invalid)
    with pytest.raises(ValueError, match="neither numeric nor datetime"):
        validate_time_column_type("mixed_col", "object")

    # Test custom/user-defined types (invalid)
    with pytest.raises(ValueError, match="neither numeric nor datetime"):
        validate_time_column_type("custom_col", "custom_type")


def test_convert_to_datetime_error_handling():
    """Test convert_to_datetime error handling for invalid column types."""
    # Create a DataFrame with an invalid column type
    df = pd.DataFrame({"invalid_col": ["not_a_datetime", "nope", "still_not"]})

    # Test invalid column type
    with pytest.raises(ValueError, match="neither string nor numeric"):
        convert_to_datetime(df, "invalid_col", nw.col("invalid_col"), df["invalid_col"].dtype)


def test_validate_time_column_type_edge_cases():
    """Test validate_time_column_type for edge cases."""
    # Test very long column name
    long_col_name = "a" * 300
    validate_time_column_type(long_col_name, "datetime64[ns]")  # Should not raise an error

    # Test numeric column with unusual dtype
    validate_time_column_type("unusual_numeric", "float128")  # Should not raise an error

    # Test datetime column with timezone
    validate_time_column_type("tz_datetime", "datetime64[ns, UTC]")  # Should not raise an error


def test_convert_to_datetime_with_string_column():
    """Test convert_to_datetime with string datetime column."""
    # Create a DataFrame with a string datetime column
    df = pd.DataFrame(
        {
            "string_datetime": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }
    )

    # Call convert_to_datetime for the string column
    result = convert_to_datetime(df, "string_datetime", nw.col("string_datetime"), "string")

    # Check that the column was converted correctly
    assert "string_datetime" in result.columns, "Column 'string_datetime' not found in result."
    assert pd.api.types.is_datetime64_any_dtype(
        result["string_datetime"]
    ), "Expected column 'string_datetime' to be converted to datetime."


def test_convert_to_datetime_with_numeric_column():
    """Test convert_to_datetime with numeric timestamp column."""
    # Create a DataFrame with a numeric timestamp column
    df = pd.DataFrame(
        {
            "numeric_timestamp": [1672531200, 1672617600, 1672704000],  # Unix timestamps for Jan 1-3, 2023
        }
    )

    # Call convert_to_datetime for the numeric column
    result = convert_to_datetime(df, "numeric_timestamp", nw.col("numeric_timestamp"), "float")

    # Check that the column was converted correctly
    assert "numeric_timestamp" in result.columns, "Column 'numeric_timestamp' not found in result."
    assert pd.api.types.is_datetime64_any_dtype(
        result["numeric_timestamp"]
    ), "Expected column 'numeric_timestamp' to be converted to datetime."


def test_validate_and_convert_time_column_missing_time_column():
    """Test for missing time column in validate_and_convert_time_column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(TimeColumnError, match="Column 'time' does not exist in the DataFrame."):
        validate_and_convert_time_column(df, "time", conversion_type="numeric")


def test_validate_and_convert_time_column_invalid_conversion_type():
    """Test for invalid conversion type in validate_and_convert_time_column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    with pytest.raises(
        ValueError, match="Invalid conversion_type 'invalid_type'. Must be one of 'numeric', 'datetime', or None."
    ):
        validate_and_convert_time_column(df, "time", conversion_type="invalid_type")


def test_validate_and_convert_time_column_to_numeric():
    """Test validate_and_convert_time_column with numeric conversion."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    result = validate_and_convert_time_column(df, "time", conversion_type="numeric")
    assert "time" in result.columns
    assert pd.api.types.is_float_dtype(result["time"]), "Expected numeric dtype for 'time' column."


def test_validate_and_convert_time_column_to_datetime():
    """Test validate_and_convert_time_column with datetime conversion."""
    df = pd.DataFrame({"time": [1672531200, 1672617600, 1672704000]})
    result = validate_and_convert_time_column(df, "time", conversion_type="datetime")
    assert "time" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["time"]), "Expected datetime dtype for 'time' column."


def test_validate_and_convert_time_column_validation_only():
    """Test validate_and_convert_time_column with validation-only path."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    result = validate_and_convert_time_column(df, "time", conversion_type=None)
    assert "time" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["time"]), "Expected datetime dtype for 'time' column."


# ========================= validate_dataframe_column_types Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_validate_dataframe_column_types_basic(backend: str) -> None:
    """Test validate_dataframe_column_types with valid numeric and datetime columns."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=3,
        num_features=2,
        time_col_numeric=False,  # This gives us a datetime time column
        drop_time=False,
    )
    df = nw.from_native(df)
    validate_dataframe_column_types(df, "time")  # Should not raise error


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_validate_dataframe_column_types_with_lazy_evaluation(backend: str) -> None:
    """Test validate_dataframe_column_types handles lazy evaluation correctly."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=2, drop_time=False)
    df = nw.from_native(df)

    # Make DataFrame lazy if possible
    if hasattr(df.to_native(), "lazy"):
        df = df.to_native().lazy()
        df = nw.from_native(df)

    validate_dataframe_column_types(df, "time")  # Should not raise error


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_validate_dataframe_column_types_missing_column(backend: str) -> None:
    """Test validate_dataframe_column_types raises error for missing column."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=3,
        num_features=2,
        drop_time=True,  # This removes the time column
    )
    df = nw.from_native(df)

    with pytest.raises(ValueError, match="Column 'nonexistent' does not exist"):
        validate_dataframe_column_types(df, "nonexistent")


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_validate_dataframe_column_types_invalid_type(backend: str) -> None:
    """Test validate_dataframe_column_types raises error for invalid column type."""
    # Create DataFrame with string column
    if backend == "pandas":
        df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "string_col": ["a", "b", "c"]})
    else:
        df = convert_to_backend(
            pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "string_col": ["a", "b", "c"]}), backend
        )

    with pytest.raises(ValueError, match="Column 'string_col' must be numeric but found type 'String'"):
        validate_dataframe_column_types(df, "time")


@pytest.mark.parametrize("backend", ["dask"])
def test_validate_dataframe_column_types_lazy_compute(backend: str) -> None:
    """Test validate_dataframe_column_types handles lazy computation correctly."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=2, drop_time=False)
    df = nw.from_native(df)

    # The function should handle compute() internally
    validate_dataframe_column_types(df, "time")  # Should not raise error


# ========================= Tests for sort_dataframe_time =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_sort_dataframe_time_valid(backend: str) -> None:
    """Test sort_dataframe_time with valid time column for all backends."""
    # Generate backend-native synthetic data
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=10,
        num_features=3,
        time_col_numeric=True,
        drop_time=False,
    )

    # Materialize the original DataFrame if lazy
    if hasattr(df, "collect"):
        df = df.collect()
    elif hasattr(df, "compute"):
        df = df.compute()

    # Sort using Narwhalified function
    sorted_df = sort_dataframe_time(df, time_col="time", ascending=True)

    # Materialize the sorted DataFrame if lazy
    if hasattr(sorted_df, "collect"):
        sorted_df = sorted_df.collect()
    elif hasattr(sorted_df, "compute"):
        sorted_df = sorted_df.compute()

    # Convert both sorted and original to Pandas for validation
    actual = nw.from_native(sorted_df).to_pandas()
    expected = nw.from_native(df).to_pandas().sort_values("time", ascending=True)

    # Assert that the time column is sorted
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,  # Ignore dtype differences caused by backend conversions
    )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_sort_dataframe_time_missing_time_column(backend: str) -> None:
    """Test sort_dataframe_time raises an error for missing time column."""
    # Generate synthetic data without a time column
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=10,
        num_features=3,
        drop_time=True,  # Drop the time column
    )

    # Expect ValueError for missing 'time' column
    with pytest.raises(ValueError, match="Column 'time' does not exist in the DataFrame"):
        sort_dataframe_time(df, time_col="time", ascending=True)


# ========================= convert_to_backend Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_convert_to_backend_narwhalified(backend: str) -> None:
    """Test converting narwhalified DataFrame to target backend."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=3)
    df_narwhals = nw.from_native(df)

    # Mock `to_native` to verify it's called
    with mock.patch.object(df_narwhals, "to_native", wraps=df_narwhals.to_native) as mock_to_native:
        converted_df = convert_to_backend(df_narwhals, backend)
        expected_type = TEMPORALSCOPE_CORE_BACKEND_TYPES[backend]
        assert isinstance(converted_df, expected_type), f"Expected {expected_type}, got {type(converted_df)}."
        mock_to_native.assert_called_once()


def test_convert_to_backend_lazy_dataframe() -> None:
    """Test handling of lazy DataFrame materialization."""
    df = generate_synthetic_time_series(backend="dask", num_samples=10, num_features=3)

    # Mock `is_lazy_evaluation` to verify it's called
    with mock.patch("temporalscope.core.core_utils.is_lazy_evaluation", wraps=is_lazy_evaluation) as mock_is_lazy:
        converted_df = convert_to_backend(df, "pandas")
        assert isinstance(converted_df, TEMPORALSCOPE_CORE_BACKEND_TYPES["pandas"])
        mock_is_lazy.assert_called_once_with(df)


def test_convert_to_backend_lazy_compute() -> None:
    """Test lazy evaluation materialization using compute."""
    df = generate_synthetic_time_series(backend="dask", num_samples=10, num_features=3)
    converted_df = convert_to_backend(df, "pandas")
    assert isinstance(converted_df, pd.DataFrame)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_convert_to_backend_valid(backend: str) -> None:
    """Test DataFrame conversion to each valid backend."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=3)
    converted_df = convert_to_backend(df, backend)
    expected_type = TEMPORALSCOPE_CORE_BACKEND_TYPES[backend]
    assert isinstance(converted_df, expected_type), f"Expected {expected_type}, got {type(converted_df)}."


@pytest.mark.parametrize("backend", ["unsupported_backend"])
def test_convert_to_backend_invalid_backend(backend: str) -> None:
    """Test that convert_to_backend raises error for unsupported backend."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=3)
    with pytest.raises(
        UnsupportedBackendError, match=f"Backend '{backend}' is not supported by TemporalScope. Supported backends are:"
    ):
        convert_to_backend(df, backend)


def test_convert_to_backend_error_handling() -> None:
    """Test handling of conversion errors."""

    class InvalidDataFrame:
        def __init__(self):
            self._df = "Invalid"

        def compute(self):
            raise Exception("Computation failed")

    df = InvalidDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Input DataFrame type 'InvalidDataFrame' is not supported"):
        convert_to_backend(df, "pandas")


# ========================= check_dataframe_nulls_nans Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_valid_columns(backend: str) -> None:
    """Test check_dataframe_nulls_nans for valid columns."""
    data = {"col1": [1, None, 2], "col2": [None, 2, 3]}

    # Convert to the specific backend
    if backend == "dask":
        df = TEMPORALSCOPE_BACKEND_CONVERTERS[backend](pd.DataFrame(data), npartitions=1)
    else:
        df = TEMPORALSCOPE_BACKEND_CONVERTERS[backend](pd.DataFrame(data), 1)

    # Call check_dataframe_nulls_nans and validate the result
    result = check_dataframe_nulls_nans(df, column_names=["col1", "col2"])
    assert result == {"col1": 1, "col2": 1}, f"Unexpected result for backend {backend}: {result}"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_empty_df(backend: str) -> None:
    """Test check_dataframe_nulls_nans raises error for empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=3)
    with pytest.raises(ValueError, match="Empty DataFrame provided."):
        check_dataframe_nulls_nans(df, column_names=["col1", "col2"])


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_missing_columns(backend: str) -> None:
    """Test check_dataframe_nulls_nans raises error for missing columns."""
    df = generate_synthetic_time_series(backend=backend, num_samples=10, num_features=2)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found."):
        check_dataframe_nulls_nans(df, column_names=["nonexistent"])


def test_check_dataframe_nulls_nans_error_handling() -> None:
    """Test check_dataframe_nulls_nans raises UnsupportedBackendError for unsupported DataFrame."""

    class InvalidDataFrame:
        """A mock class for an unsupported DataFrame type."""

        pass

    # Instantiate the invalid DataFrame
    invalid_df = InvalidDataFrame()

    # Ensure UnsupportedBackendError is raised during validation
    with pytest.raises(UnsupportedBackendError, match="The input DataFrame is not supported by TemporalScope."):
        check_dataframe_nulls_nans(invalid_df, column_names=["col1"])


# ========================= Tests for get_dataframe_backend =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_get_dataframe_backend_valid(backend: str) -> None:
    """Test get_dataframe_backend returns correct backend for valid DataFrame types."""
    df = generate_synthetic_time_series(backend=backend, num_samples=10, num_features=2)
    result = get_dataframe_backend(df)
    assert result == backend, f"Expected backend '{backend}', but got '{result}'."


def test_get_dataframe_backend_narwhalified(narwhalified_df) -> None:
    """Test get_dataframe_backend handles narwhalified DataFrames."""
    # Expected to return the native backend (e.g., 'pandas')
    backend = get_dataframe_backend(narwhalified_df)
    assert backend == "pandas", f"Expected backend 'pandas', but got '{backend}'."


def test_get_dataframe_backend_unsupported_type() -> None:
    """Test get_dataframe_backend raises UnsupportedBackendError for unsupported DataFrame types."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Unknown DataFrame type"):
        get_dataframe_backend(df)


def test_get_dataframe_backend_unknown_type() -> None:
    """Test get_dataframe_backend raises UnsupportedBackendError for unknown valid types."""

    class CustomDataFrame:
        def __init__(self):
            self.data = pd.DataFrame({"col": [1, 2, 3]})

        def to_pandas(self):
            return self.data

    df = CustomDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Unknown DataFrame type"):
        get_dataframe_backend(df)


# ========================= Unsupported DataFrame Tests =========================


def test_convert_to_datetime_unsupported_dataframe() -> None:
    """Test convert_to_datetime raises UnsupportedBackendError for unsupported DataFrame."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()
    with pytest.raises(UnsupportedBackendError, match=f"Unsupported DataFrame type: {type(df).__name__}"):
        convert_to_datetime(df, time_col="time", col_expr=None, col_dtype="string")


def test_validate_and_convert_time_column_unsupported_dataframe() -> None:
    """Test validate_and_convert_time_column raises UnsupportedBackendError for unsupported DataFrame."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()
    with pytest.raises(UnsupportedBackendError, match=f"Unsupported DataFrame type: {type(df).__name__}"):
        validate_and_convert_time_column(df, time_col="time", conversion_type="numeric")


def test_validate_dataframe_column_types_unsupported_dataframe() -> None:
    """Test validate_dataframe_column_types raises UnsupportedBackendError for unsupported DataFrame."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()
    with pytest.raises(UnsupportedBackendError, match=f"Unsupported DataFrame type: {type(df).__name__}"):
        validate_dataframe_column_types(df, time_col="time")


def test_sort_dataframe_time_unsupported_dataframe() -> None:
    """Test sort_dataframe_time raises UnsupportedBackendError for unsupported DataFrame."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()
    with pytest.raises(UnsupportedBackendError, match=f"Unsupported DataFrame type: {type(df).__name__}"):
        sort_dataframe_time(df, time_col="time", ascending=True)


def test_sort_dataframe_time_with_numeric_column():
    """Test sort_dataframe_time function with a numeric time column."""
    df = pd.DataFrame({"time": [3, 1, 2], "value": [30, 10, 20]})

    # Sort the DataFrame by the numeric time column
    sorted_df = sort_dataframe_time(df, time_col="time", ascending=True)

    # Assert the time column is sorted
    expected = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
    pd.testing.assert_frame_equal(sorted_df.reset_index(drop=True), expected)


def test_sort_dataframe_time_with_datetime_column():
    """Test sort_dataframe_time function with a datetime time column."""
    df = pd.DataFrame({"time": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]), "value": [30, 10, 20]})

    # Sort the DataFrame by the datetime time column
    sorted_df = sort_dataframe_time(df, time_col="time", ascending=True)

    # Assert the time column is sorted
    expected = pd.DataFrame({"time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]), "value": [10, 20, 30]})
    pd.testing.assert_frame_equal(sorted_df.reset_index(drop=True), expected)


def test_sort_dataframe_time_with_missing_time_column():
    """Test sort_dataframe_time raises an error for missing time column."""
    df = pd.DataFrame({"value": [30, 10, 20]})

    with pytest.raises(ValueError, match="Column 'time' does not exist in the DataFrame."):
        sort_dataframe_time(df, time_col="time", ascending=True)


# ======================= check_temporal_order_and_uniqueness Tests =======================


def test_check_temporal_order_and_uniqueness_no_duplicates():
    """Test check_temporal_order_and_uniqueness with no duplicates or violations."""
    df = pd.DataFrame({"time": [1, 2, 3, 4, 5]})
    check_temporal_order_and_uniqueness(df, time_col="time")


def test_check_temporal_order_and_uniqueness_duplicates():
    """Test check_temporal_order_and_uniqueness raises error for duplicate timestamps."""
    df = pd.DataFrame({"time": [1, 1, 2, 3]})
    # Convert to Narwhals backend and check
    df_backend = nw.from_native(df)  # Ensure proper Narwhalification
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        check_temporal_order_and_uniqueness(df_backend, time_col="time")


def test_check_temporal_order_and_uniqueness_non_monotonic():
    """Test check_temporal_order_and_uniqueness raises error for non-monotonic timestamps."""
    df = pd.DataFrame({"time": [1, 3, 2, 4]})
    # Let @nw.narwhalify handle conversion
    with pytest.raises(ValueError, match=r".*strictly increasing.*"):  # Use regex pattern
        check_temporal_order_and_uniqueness(df, time_col="time")


def test_check_temporal_order_and_uniqueness_warn_on_failure():
    """Test check_temporal_order_and_uniqueness warns instead of erroring."""
    df = pd.DataFrame({"time": [1, 1, 2, 3]})
    with pytest.warns(UserWarning, match="Duplicate timestamps"):
        check_temporal_order_and_uniqueness(df, time_col="time", raise_error=False)


def test_check_temporal_order_and_uniqueness_with_context():
    """Test check_temporal_order_and_uniqueness includes context in error message."""
    df = pd.DataFrame({"time": [1, 1, 2, 3]})
    with pytest.raises(ValueError, match="group 'A'"):
        check_temporal_order_and_uniqueness(df, time_col="time", context="group 'A'")


# Generalized Backend-Agnostic Test
@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_temporal_order_and_uniqueness_all_backends(backend):
    """Test check_temporal_order_and_uniqueness across all supported backends."""
    test_cases = {
        "no_duplicates": pd.DataFrame({"time": [1, 2, 3, 4, 5]}),
        "duplicates": pd.DataFrame({"time": [1, 1, 2, 3]}),
        "non_monotonic": pd.DataFrame({"time": [1, 3, 2, 4]}),
    }

    for case_name, df in test_cases.items():
        if case_name == "no_duplicates":
            check_temporal_order_and_uniqueness(df, time_col="time")
        elif case_name == "duplicates":
            with pytest.raises(ValueError, match=r"(?i).*duplicate.*"):
                check_temporal_order_and_uniqueness(df, time_col="time")
        else:  # non_monotonic
            with pytest.raises(ValueError, match=r".*strictly increasing.*"):
                check_temporal_order_and_uniqueness(df, time_col="time")


def test_check_temporal_order_and_uniqueness_invalid_time_column():
    """Test check_temporal_order_and_uniqueness with invalid time column.

    Tests the first validation step:
    try:
        df = nw.from_native(validate_and_convert_time_column(df, time_col))
    except Exception as e:
        raise TimeColumnError(f"Invalid time column: {str(e)}")
    """
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    with pytest.raises(TimeColumnError, match=r"Invalid time column:.*"):
        check_temporal_order_and_uniqueness(df, time_col="time")


# ======================== check_strict_temporal_ordering Tests =========================


def test_check_strict_temporal_ordering_invalid_time_column():
    """Test check_strict_temporal_ordering with invalid time column.

    Tests the validation in sort_dataframe_time():
    validate_time_column_type(time_col, df.schema.get(time_col, None))
    """
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    with pytest.raises(ValueError, match=r".*neither numeric nor datetime.*"):
        check_strict_temporal_ordering(df, time_col="time")


def test_check_strict_temporal_ordering_missing_group_col():
    """Test check_strict_temporal_ordering raises error for missing group column."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
    with pytest.raises(ValueError, match="Column 'group_col' does not exist"):
        check_strict_temporal_ordering(df, time_col="time", group_col="group_col")


def test_check_strict_temporal_ordering_empty_group():
    """Test check_strict_temporal_ordering with empty group."""
    df = pd.DataFrame({"group_col": [], "time": [], "value": []})
    with pytest.raises(ValueError, match="Invalid or empty DataFrame"):
        check_strict_temporal_ordering(df, time_col="time", group_col="group_col")


def test_check_strict_temporal_ordering_complex_sorting():
    """Test check_strict_temporal_ordering handles sorting with id_col."""
    df = pd.DataFrame(
        {
            "id_col": [2, 1, 2, 1],
            "time": [3, 1, 4, 2],
            "value": [10, 20, 30, 40],
        }
    )
    check_strict_temporal_ordering(df, time_col="time", id_col="id_col")


def test_check_strict_temporal_ordering_missing_columns():
    """Test check_strict_temporal_ordering with missing columns.

    Tests step 3 and step 5:
    # Step 3: Column validation
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Step 5: Additional sort by id_col if provided
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' does not exist in the DataFrame.")
    """
    # Create DataFrame with some data
    df = pd.DataFrame(
        {
            "other": [1, 2, 3],  # Some other column
        }
    )

    # Test missing time column
    with pytest.raises(ValueError, match=r"Column 'time' does not exist in the DataFrame"):
        check_strict_temporal_ordering(df, time_col="time")

    # Test missing id column
    df["time"] = [1, 2, 3]  # Add valid time column
    with pytest.raises(ValueError, match=r"Column 'id' does not exist in the DataFrame"):
        check_strict_temporal_ordering(df, time_col="time", id_col="id")


def test_check_strict_temporal_ordering_group_validation():
    """Test check_strict_temporal_ordering with group validation."""
    # Create DataFrame with valid groups
    df = pd.DataFrame(
        {
            "time": [1, 2, 1, 2],
            "group": ["A", "A", "B", "B"],
            "feature_1": [10, 20, 30, 40],
            "feature_2": [50, 60, 70, 80],
            "target": [90, 100, 110, 120],
        }
    )

    # Should pass - each group has monotonic time
    check_strict_temporal_ordering(df, time_col="time", group_col="group")
