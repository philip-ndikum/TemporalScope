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

# Constants
# Use TEST_BACKENDS for test environment
from temporalscope.core.core_utils import (
    TEST_BACKENDS,
    TimeColumnError,
    convert_column_to_datetime_type,
    convert_datetime_column_to_microseconds,
    convert_datetime_column_to_timestamp,
    count_dataframe_column_nulls,
    get_api_keys,
    get_default_backend_cfg,
    get_narwhals_backends,
    is_dataframe_empty,
    print_divider,
    sort_dataframe_time,
    validate_and_convert_time_column,
    validate_column_numeric_or_datetime,
    validate_feature_columns_numeric,
    validate_temporal_ordering,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series


@pytest.fixture(params=TEST_BACKENDS)
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


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_is_dataframe_empty_with_empty_df(backend):
    """Test is_dataframe_empty returns True for empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    assert is_dataframe_empty(df) is True


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_is_dataframe_empty_with_data(backend):
    """Test is_dataframe_empty returns False for non-empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=10, num_features=1, drop_time=True)
    assert is_dataframe_empty(df) is False


def test_is_dataframe_empty_none():
    """Test is_dataframe_empty error handling for None input."""
    with pytest.raises(ValueError, match="DataFrame cannot be None"):
        is_dataframe_empty(None)


# ========================= Tests for count_dataframe_column_nulls =========================


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_count_dataframe_column_nulls_no_nulls(backend):
    """Test count_dataframe_column_nulls with DataFrame containing no null values."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=2, with_nulls=False, drop_time=True
    )
    result = count_dataframe_column_nulls(df, ["feature_1"])
    assert result == {"feature_1": 0}


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_count_dataframe_column_nulls_with_nulls(backend):
    """Test count_dataframe_column_nulls with DataFrame containing null values."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=2, with_nulls=True, null_percentage=0.5, drop_time=True
    )
    result = count_dataframe_column_nulls(df, ["feature_1", "feature_2"])
    # With 50% null percentage, expect around half the values to be null
    assert 4 <= result["feature_1"] <= 6
    assert 4 <= result["feature_2"] <= 6


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_count_dataframe_column_nulls_empty_dataframe(backend):
    """Test count_dataframe_column_nulls with an empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        count_dataframe_column_nulls(df, ["feature_1"])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_count_dataframe_column_nulls_nonexistent_column(backend):
    """Test count_dataframe_column_nulls with nonexistent column."""
    df = generate_synthetic_time_series(backend=backend, num_samples=10, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        count_dataframe_column_nulls(df, ["nonexistent"])


# ========================= Tests for convert_datetime_column_to_microseconds =========================


def test_convert_datetime_column_to_microseconds(synthetic_df):
    """Test convert_datetime_column_to_microseconds with valid input."""
    result = convert_datetime_column_to_microseconds(nw.from_native(synthetic_df), "time")
    # Check if dtype indicates float (handle different backends)
    dtype_str = str(result["time"].dtype).lower()
    assert any(float_type in dtype_str for float_type in ["float", "f64"])


def test_convert_datetime_column_to_microseconds_already_numeric():
    """Test convert_datetime_column_to_microseconds with already numeric column."""
    df = pd.DataFrame({"time": [1, 2, 3]})
    df = nw.from_native(df)
    result = convert_datetime_column_to_microseconds(df, "time")
    # Convert both to pandas DataFrames for comparison
    result_pd = result if isinstance(result, pd.DataFrame) else result.to_native()
    df_pd = df if isinstance(df, pd.DataFrame) else df.to_native()
    pd.testing.assert_frame_equal(result_pd, df_pd)


def test_convert_datetime_column_to_microseconds_invalid():
    """Test convert_datetime_column_to_microseconds with invalid column."""
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError):
        convert_datetime_column_to_microseconds(df, "time")


# ========================= Tests for convert_column_to_datetime_type =========================


def test_convert_column_to_datetime_type_missing_column():
    """Test convert_column_to_datetime_type with missing column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError, match="Column 'time' does not exist in DataFrame"):
        convert_column_to_datetime_type(df, "time")


def test_convert_column_to_datetime_type_already_datetime():
    """Test convert_column_to_datetime_type with already datetime column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    df = nw.from_native(df)
    result = convert_column_to_datetime_type(df, "time")
    # Verify data is unchanged and type is still datetime
    assert pd.api.types.is_datetime64_dtype(result["time"].dtype)
    # Convert both to pandas for comparison
    result_pd = result if isinstance(result, pd.DataFrame) else result.to_native()
    df_pd = df if isinstance(df, pd.DataFrame) else df.to_native()
    pd.testing.assert_frame_equal(result_pd, df_pd)


def test_convert_column_to_datetime_type_from_string():
    """Test convert_column_to_datetime_type with string input."""
    df = pd.DataFrame({"time": ["2023-01-01", "2023-01-02"]})
    df = nw.from_native(df)
    result = convert_column_to_datetime_type(df, "time")
    assert pd.api.types.is_datetime64_dtype(result["time"].dtype)


def test_convert_column_to_datetime_type_from_numeric():
    """Test convert_column_to_datetime_type with numeric input."""
    df = pd.DataFrame({"time": [1672531200, 1672617600]})  # Unix timestamps
    df = nw.from_native(df)
    result = convert_column_to_datetime_type(df, "time")
    assert pd.api.types.is_datetime64_dtype(result["time"].dtype)


def test_convert_column_to_datetime_type_conversion_failure():
    """Test convert_column_to_datetime_type when all conversions fail."""
    df = pd.DataFrame({"time": ["invalid", "dates", "here"]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError, match="Column 'time' must be string or numeric to convert to datetime"):
        convert_column_to_datetime_type(df, "time")


def test_convert_column_to_datetime_type_invalid():
    """Test convert_column_to_datetime_type with invalid input."""
    df = pd.DataFrame({"time": [True, False]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError):
        convert_column_to_datetime_type(df, "time")


# ========================= Tests for validate_column_numeric_or_datetime =========================


def test_validate_column_numeric_or_datetime_numeric():
    """Test validate_column_numeric_or_datetime with numeric column."""
    df = pd.DataFrame({"time": [1, 2, 3]})
    df = nw.from_native(df)
    validate_column_numeric_or_datetime(df, "time")


def test_validate_column_numeric_or_datetime_datetime():
    """Test validate_column_numeric_or_datetime with datetime column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    df = nw.from_native(df)
    validate_column_numeric_or_datetime(df, "time")


def test_validate_column_numeric_or_datetime_invalid():
    """Test validate_column_numeric_or_datetime with invalid column."""
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        validate_column_numeric_or_datetime(df, "time")


# ========================= Tests for validate_and_convert_time_column =========================


def test_validate_and_convert_time_column_validation_only():
    """Test validate_and_convert_time_column with validation only (no conversion)."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    df = nw.from_native(df)
    result = validate_and_convert_time_column(df, "time", conversion_type=None)
    # Verify data is unchanged
    result_pd = result if isinstance(result, pd.DataFrame) else result.to_native()
    df_pd = df if isinstance(df, pd.DataFrame) else df.to_native()
    pd.testing.assert_frame_equal(result_pd, df_pd)


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


# ========================= Tests for validate_feature_columns_numeric =========================


def test_validate_feature_columns_numeric_all_columns(synthetic_df):
    """Test validate_feature_columns_numeric with all columns when time_col=None."""
    # Create DataFrame with only numeric columns
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
        }
    )
    df = nw.from_native(df)
    validate_feature_columns_numeric(df)  # Should pass


def test_validate_feature_columns_numeric_exclude_time(synthetic_df):
    """Test validate_feature_columns_numeric with time column excluded."""
    validate_feature_columns_numeric(synthetic_df, time_col="time")  # Should pass


def test_validate_feature_columns_numeric_missing_time():
    """Test validate_feature_columns_numeric with missing time column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    df = nw.from_native(df)
    with pytest.raises(TimeColumnError, match="Column 'time' does not exist"):
        validate_feature_columns_numeric(df, time_col="time")


def test_validate_feature_columns_numeric_invalid_feature():
    """Test validate_feature_columns_numeric with non-numeric feature column."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=3),
            "value": ["a", "b", "c"],  # String type (invalid)
        }
    )
    df = nw.from_native(df)
    with pytest.raises(ValueError, match="Column 'value' must be numeric"):
        validate_feature_columns_numeric(df, time_col="time")


def test_validate_feature_columns_numeric_all_invalid():
    """Test validate_feature_columns_numeric with all non-numeric columns."""
    df = pd.DataFrame(
        {
            "col1": ["a", "b", "c"],
            "col2": ["d", "e", "f"],
        }
    )
    df = nw.from_native(df)
    with pytest.raises(ValueError, match="Column 'col1' must be numeric"):
        validate_feature_columns_numeric(df)


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


# ========================= Tests for validate_temporal_ordering =========================


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_validate_temporal_ordering_basic_backend(backend):
    """Test basic temporal validation across backends."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=4, num_features=1, time_col_numeric=True, drop_time=False
    )
    validate_temporal_ordering(df, "time")


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_validate_temporal_ordering_equidistant_backend(backend):
    """Test equidistant sampling across backends."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=4, num_features=1, time_col_numeric=False, drop_time=False
    )
    validate_temporal_ordering(df, "time", enforce_equidistant_sampling=True)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_validate_temporal_ordering_multi_entity_backend(backend):
    """Test multi-entity validation across backends."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=4, num_features=1, time_col_numeric=True, drop_time=False
    )
    # Test basic validation
    validate_temporal_ordering(df, "time")

    # Test missing id_col error
    with pytest.raises(ValueError, match="Column 'missing_id' does not exist"):
        validate_temporal_ordering(df, "time", id_col="missing_id")


def test_validate_temporal_ordering_basic():
    """Test validate_temporal_ordering with default ML/DL settings.

    Validates that:
    1. No information leakage in train/test splits
    2. Proper sequence ordering for time series models
    3. Valid feature/target relationships

    Note: In ML context, we typically want strict ordering
    to prevent data leakage and ensure proper temporal splits.
    """
    # Simple test data with strictly increasing timestamps
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df = nw.from_native(df)  # Convert to Narwhals DataFrame
    validate_temporal_ordering(df, "time")  # Should pass


def test_validate_temporal_ordering_duplicates():
    """Test validate_temporal_ordering with duplicate timestamps.

    Should raise TimeColumnError when duplicates are found.
    """
    # Test data with duplicate timestamps
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 2.0, 3.0],  # Duplicate at t=2
            "value": [10.0, 20.0, 25.0, 30.0],
        }
    )
    df = nw.from_native(df)

    with pytest.raises(TimeColumnError, match="Duplicate timestamps found"):
        validate_temporal_ordering(df, "time")


def test_validate_temporal_ordering_multi_entity():
    """Test validate_temporal_ordering with multiple entities.

    Validates proper handling of grouped time series data.
    """
    # Test data with multiple entities, each with unique timestamps
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1.0, 2.0, 1.0, 2.0],  # Same times ok for different ids
            "value": [10.0, 20.0, 15.0, 25.0],
        }
    )
    df = nw.from_native(df)
    validate_temporal_ordering(df, "time", id_col="id")  # Should pass

    # Test data with duplicate timestamps within an entity
    df_bad = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1.0, 1.0, 2.0, 3.0],  # Duplicate for id=1
            "value": [10.0, 15.0, 20.0, 30.0],
        }
    )
    df_bad = nw.from_native(df_bad)

    with pytest.raises(TimeColumnError, match="Duplicate timestamps found within groups"):
        validate_temporal_ordering(df_bad, "time", id_col="id")


def test_validate_temporal_ordering_equidistant():
    """Test validate_temporal_ordering with equidistant sampling requirement.

    Validates proper handling of regular time intervals required by
    classical time series models like ARIMA.
    """
    # Test data with regular intervals
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],  # Equal spacing of 1.0
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df = nw.from_native(df)
    validate_temporal_ordering(df, "time", enforce_equidistant_sampling=True)  # Should pass

    # Test data with irregular intervals
    df_irregular = pd.DataFrame(
        {
            "time": [1.0, 2.0, 4.0, 7.0],  # Irregular spacing
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df_irregular = nw.from_native(df_irregular)

    with pytest.raises(TimeColumnError, match="Irregular time sampling found"):
        validate_temporal_ordering(df_irregular, "time", enforce_equidistant_sampling=True)


def test_validate_temporal_ordering_datetime():
    """Test validate_temporal_ordering with datetime values.

    Validates proper handling of datetime timestamps.
    """
    # Test data with datetime timestamps
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=4, freq="D"),
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df = nw.from_native(df)
    validate_temporal_ordering(df, "time")  # Should pass
    validate_temporal_ordering(df, "time", enforce_equidistant_sampling=True)  # Should pass with daily frequency

    # Test data with irregular datetime intervals
    df_irregular = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-04"),  # Gap
                pd.Timestamp("2023-01-07"),  # Gap
            ],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df_irregular = nw.from_native(df_irregular)

    validate_temporal_ordering(df_irregular, "time")  # Should pass without equidistant requirement
    with pytest.raises(TimeColumnError, match="Irregular time sampling found"):
        validate_temporal_ordering(df_irregular, "time", enforce_equidistant_sampling=True)


def test_validate_temporal_ordering_hierarchical():
    """Test validate_temporal_ordering with hierarchical time series."""
    # Create test data with different sampling rates
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 1, 2, 3, 4],
            "id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "value": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    df = nw.from_native(df)
    validate_temporal_ordering(df, "time", id_col="id")  # Should pass
    validate_temporal_ordering(df, "time", id_col="id", enforce_equidistant_sampling=True)  # Should pass


def test_validate_temporal_ordering_multi_sensor():
    """Test validate_temporal_ordering with multi-sensor data.

    Special case: Multi-sensor data with different sampling rates
    - Each sensor (group) has its own timeline
    - No overlap required between groups
    - Common in IoT and industrial monitoring
    """
    # Test data with multiple sensors, each with unique timestamps
    df = pd.DataFrame(
        {
            "id": ["A", "A", "B", "B"],  # Two sensors
            "time": [1.0, 2.0, 1.0, 2.0],  # Same times ok for different sensors
            "value": [10.0, 20.0, 15.0, 25.0],
        }
    )
    df = nw.from_native(df)
    validate_temporal_ordering(df, "time", id_col="id")  # Should pass

    # Test data with duplicate timestamps within a sensor
    df_bad = pd.DataFrame(
        {
            "id": ["A", "A", "B", "B"],
            "time": [1.0, 1.0, 2.0, 3.0],  # Duplicate for sensor A
            "value": [10.0, 15.0, 20.0, 30.0],
        }
    )
    df_bad = nw.from_native(df_bad)

    with pytest.raises(TimeColumnError, match="Duplicate timestamps found within groups"):
        validate_temporal_ordering(df_bad, "time", id_col="id")


def test_validate_temporal_ordering_mixed_frequency():
    """Test validate_temporal_ordering with mixed frequency data.

    Tests handling of:
    1. Regular sampling within groups
    2. Different frequencies across groups
    3. Equidistant sampling validation
    """
    # Test data with different but regular frequencies per group
    df = pd.DataFrame(
        {
            "id": ["A", "A", "A", "B", "B", "B"],
            "time": [1, 2, 3, 2, 4, 6],  # A: interval=1, B: interval=2
            "value": [10, 20, 30, 40, 50, 60],
        }
    )
    df = nw.from_native(df)

    # Should pass without equidistant requirement
    validate_temporal_ordering(df, "time", id_col="id")

    # Should pass with equidistant requirement (each group has regular intervals)
    validate_temporal_ordering(df, "time", id_col="id", enforce_equidistant_sampling=True)

    # Test irregular intervals within a group
    df_irregular = pd.DataFrame(
        {
            "id": ["A", "A", "A"],
            "time": [1, 2, 4],  # Irregular spacing
            "value": [10, 20, 30],
        }
    )
    df_irregular = nw.from_native(df_irregular)

    with pytest.raises(TimeColumnError, match="Irregular time sampling found"):
        validate_temporal_ordering(df_irregular, "time", enforce_equidistant_sampling=True)


def test_sort_dataframe_time_missing():
    """Test sort_dataframe_time with missing time column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    df = nw.from_native(df)
    with pytest.raises(ValueError):
        sort_dataframe_time(df, "time")


def test_sort_dataframe_time_invalid_type():
    """Test sort_dataframe_time with invalid time column type."""
    df = pd.DataFrame({"time": ["a", "b", "c"]})  # String type (invalid)
    df = nw.from_native(df)
    with pytest.raises(ValueError, match="Column 'time' is neither numeric nor datetime"):
        sort_dataframe_time(df, "time")


# ========================= Tests for convert_datetime_column_to_timestamp =========================


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_convert_datetime_column_to_timestamp_microseconds(backend):
    """Test convert_datetime_column_to_timestamp with microsecond precision."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, time_col_numeric=False, drop_time=False
    )
    result = convert_datetime_column_to_timestamp(df, "time", "us")
    # Check if dtype indicates float for microseconds
    dtype_str = str(result["time"].dtype).lower()
    assert any(float_type in dtype_str for float_type in ["float", "f64"])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_convert_datetime_column_to_timestamp_nanoseconds(backend):
    """Test convert_datetime_column_to_timestamp with nanosecond precision."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, time_col_numeric=False, drop_time=False
    )
    result = convert_datetime_column_to_timestamp(df, "time", "ns")
    # Check if dtype indicates int for nanoseconds
    dtype_str = str(result["time"].dtype).lower()
    assert "int" in dtype_str


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_convert_datetime_column_to_timestamp_already_numeric(backend):
    """Test convert_datetime_column_to_timestamp with already numeric column."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, time_col_numeric=True, drop_time=False
    )
    result = convert_datetime_column_to_timestamp(df, "time")
    # Both result and df should be identical since numeric columns are returned as-is
    assert (result["time"] == df["time"]).all()


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_convert_datetime_column_to_timestamp_missing_column(backend):
    """Test convert_datetime_column_to_timestamp with missing column."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Column 'time' does not exist in the DataFrame."):
        convert_datetime_column_to_timestamp(df, "time")


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_convert_datetime_column_to_timestamp_empty_df(backend):
    """Test convert_datetime_column_to_timestamp with empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=False)
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        convert_datetime_column_to_timestamp(df, "time")


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_convert_datetime_column_to_timestamp_invalid_type(backend):
    """Test convert_datetime_column_to_timestamp with invalid column type."""
    # Create DataFrame with invalid column type
    df = pd.DataFrame({"time": ["a", "b", "c"]})  # String type (invalid)
    df = nw.from_native(df)  # Convert to Narwhals format
    with pytest.raises(TimeColumnError, match="Column 'time' must be datetime type to convert"):
        convert_datetime_column_to_timestamp(df, "time")
