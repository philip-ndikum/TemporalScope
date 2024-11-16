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

"""Unit Test Design for TemporalScope's TimeFrame Class.

This module implements a systematic approach to testing the TimeFrame class across multiple DataFrame backends
while maintaining consistency and reliability in test execution.

Testing Philosophy
-----------------
The testing strategy follows three core principles:

1. Backend-Agnostic Operations:
   - All DataFrame manipulations use the Narwhals API (@nw.narwhalify) to ensure consistent behavior
   - Operations are written once and work across all supported backends (Pandas, Polars, Modin)
   - Backend-specific code is avoided to maintain test uniformity

2. Fine-Grained Data Generation:
   - PyTest fixtures provide flexible, parameterized test data generation
   - Base configuration fixture allows easy overrides for specific test cases
   - Each test case can specify exact data characteristics needed (nulls, NaNs, dtypes)
   - Data generation simulates real end-user scenarios and edge cases

3. Consistent Validation Pattern:
   - All validation steps convert to Pandas via .to_pandas() for reliable comparisons
   - This ensures consistent behavior across backends during assertion checks
   - Complex validations are encapsulated in reusable helper functions
   - Assertions focus on business logic rather than implementation details

See Also
--------
- TemporalScope documentation: https://temporalscope.readthedocs.io/
- Narwhals API documentation: https://narwhals.readthedocs.io/
- PyTest documentation: https://docs.pytest.org/

"""

from datetime import datetime
from typing import Any, Callable, Dict, Generator

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    MODE_SINGLE_STEP,
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    SupportedTemporalDataFrame,
    get_dataframe_backend,
)
from temporalscope.core.exceptions import ModeValidationError, TimeColumnError, UnsupportedBackendError
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = list(TEMPORALSCOPE_CORE_BACKEND_TYPES.keys())  # Include all backends
TEST_MODES = [MODE_SINGLE_STEP]  # Add MODE_MULTI_STEP when implemented

# Custom types
DataConfigType = Callable[..., Dict[str, Any]]


# Test Configurations
@pytest.fixture
def data_config() -> DataConfigType:
    """Base fixture for data generation configuration.

    Provides a callable that returns a dictionary of data generation parameters.
    This allows tests to override specific parameters while maintaining defaults.
    """

    def _config(**kwargs) -> Dict[str, Any]:
        default_config = {
            "num_samples": 5,
            "num_features": 3,
            "with_nulls": False,
            "with_nans": False,
            "mode": "single_step",
            "time_col_numeric": True,
        }
        default_config.update(kwargs)
        return default_config

    return _config


# DataFrame Fixtures
@pytest.fixture(params=VALID_BACKENDS)
def simple_df(request, data_config: DataConfigType) -> Generator[SupportedTemporalDataFrame, None, None]:
    """Generate a simple, clean DataFrame for each backend."""
    backend = request.param
    config = data_config()
    df = generate_synthetic_time_series(backend=backend, **config)
    yield df


@pytest.fixture(params=VALID_BACKENDS)
def df_with_nulls(request, data_config: DataConfigType) -> Generator[SupportedTemporalDataFrame, None, None]:
    """Generate a DataFrame with null values for each backend."""
    backend = request.param
    config = data_config(with_nulls=True)
    df = generate_synthetic_time_series(backend=backend, **config)
    yield df


@pytest.fixture(params=VALID_BACKENDS)
def df_with_nans(request, data_config: DataConfigType) -> Generator[SupportedTemporalDataFrame, None, None]:
    """Generate a DataFrame with NaN values for each backend."""
    backend = request.param
    config = data_config(with_nans=True)
    df = generate_synthetic_time_series(backend=backend, **config)
    yield df


@pytest.fixture(params=VALID_BACKENDS)
def df_with_datetime(request, data_config: DataConfigType) -> Generator[SupportedTemporalDataFrame, None, None]:
    """Generate a DataFrame with datetime time column for each backend."""
    backend = request.param
    config = data_config(time_col_numeric=False)
    df = generate_synthetic_time_series(backend=backend, **config)
    yield df


# Assertion Helpers
@nw.narwhalify
def assert_df_properties(df: SupportedTemporalDataFrame, expected_rows: int, expected_cols: int) -> None:
    """Validate DataFrame properties using Narwhals operations."""
    # Get row count using Narwhals and convert to pandas for consistent behavior
    row_count = df.select(nw.len()).to_pandas().iloc[0, 0]
    assert row_count == expected_rows, f"Expected {expected_rows} rows, got {row_count}"

    # Get column count
    assert len(df.columns) == expected_cols, f"Expected {expected_cols} columns, got {len(df.columns)}"

    # Verify required columns exist
    assert "time" in df.columns, "time column not found"
    assert "target" in df.columns, "target column not found"


@nw.narwhalify
def assert_no_nulls(df: SupportedTemporalDataFrame) -> None:
    """Verify there are no null values in the DataFrame using Narwhals."""
    for col in df.columns:
        null_count = df.select(nw.col(col).is_null().sum()).to_pandas().iloc[0, 0]
        assert null_count == 0, f"Found {null_count} null values in column {col}"


@nw.narwhalify
def assert_sorted_by_time(df: SupportedTemporalDataFrame, ascending: bool = True) -> None:
    """Verify DataFrame is sorted by time column."""
    time_values = df.select(nw.col("time")).to_pandas()["time"].tolist()
    sorted_values = sorted(time_values, reverse=not ascending)
    assert time_values == sorted_values, "DataFrame is not properly sorted by time"


# Tests
@pytest.mark.parametrize("mode", TEST_MODES)
def test_timeframe_basic_initialization(simple_df: SupportedTemporalDataFrame, mode: str) -> None:
    """Test basic TimeFrame initialization with clean data."""
    # Initialize TimeFrame
    tf = TimeFrame(df=simple_df, time_col="time", target_col="target", mode=mode)

    # Validate properties using Narwhals operations
    assert_df_properties(tf.df, expected_rows=5, expected_cols=5)
    assert_no_nulls(tf.df)
    assert_sorted_by_time(tf.df)

    # Verify TimeFrame properties
    assert tf.mode == mode
    assert tf.backend == get_dataframe_backend(simple_df)  # Use get_dataframe_backend directly
    assert tf.ascending is True


def test_timeframe_backend_inference(simple_df: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame correctly infers backend when none is specified."""
    tf = TimeFrame(df=simple_df, time_col="time", target_col="target")
    expected_backend = get_dataframe_backend(simple_df)
    assert tf.backend == expected_backend, f"Expected backend {expected_backend}, got {tf.backend}"
    assert_df_properties(tf.df, expected_rows=5, expected_cols=5)


@pytest.mark.parametrize("target_backend", ["pandas", "polars"])
def test_timeframe_explicit_backend_conversion(simple_df: SupportedTemporalDataFrame, target_backend: str) -> None:
    """Test TimeFrame initialization with backend conversion."""
    current_backend = get_dataframe_backend(simple_df)

    # Skip problematic backend combinations
    if current_backend == target_backend:
        pytest.skip(f"Current backend is already {target_backend}")
    if current_backend == "dask":
        pytest.skip("Skipping conversion test for dask source due to conversion limitations")

    tf = TimeFrame(df=simple_df, time_col="time", target_col="target", dataframe_backend=target_backend)
    assert tf.backend == target_backend, f"Backend conversion failed: expected {target_backend}, got {tf.backend}"
    assert_df_properties(tf.df, expected_rows=5, expected_cols=5)


def test_timeframe_explicit_backend_same(simple_df: SupportedTemporalDataFrame) -> None:
    """Test TimeFrame initialization with explicitly specified same backend."""
    current_backend = get_dataframe_backend(simple_df)

    # Skip for dask backend due to known conversion limitations
    if current_backend == "dask":
        pytest.skip("Skipping same-backend test for dask due to conversion limitations")

    tf = TimeFrame(df=simple_df, time_col="time", target_col="target", dataframe_backend=current_backend)
    assert tf.backend == current_backend
    assert_df_properties(tf.df, expected_rows=5, expected_cols=5)


def test_timeframe_invalid_backend_raises(simple_df: SupportedTemporalDataFrame) -> None:
    """Test TimeFrame initialization with invalid backend specification."""
    with pytest.raises(UnsupportedBackendError, match=".*not supported by TemporalScope.*"):
        TimeFrame(df=simple_df, time_col="time", target_col="target", dataframe_backend="invalid_backend")


def test_timeframe_rejects_nulls(df_with_nulls: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame properly validates and rejects data with nulls."""
    with pytest.raises(ValueError, match="Missing values detected"):
        TimeFrame(df=df_with_nulls, time_col="time", target_col="target", mode=MODE_SINGLE_STEP)


def test_timeframe_rejects_nans(df_with_nans: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame properly validates and rejects data with NaNs."""
    with pytest.raises(ValueError, match="Missing values detected"):
        TimeFrame(df=df_with_nans, time_col="time", target_col="target", mode=MODE_SINGLE_STEP)


def test_timeframe_accepts_datetime(df_with_datetime: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame properly handles datetime time columns."""
    tf = TimeFrame(df=df_with_datetime, time_col="time", target_col="target", mode=MODE_SINGLE_STEP)

    # Get time column directly using df indexing instead of select
    time_col = tf.df[tf._time_col].to_pandas() if hasattr(tf.df, "to_pandas") else tf.df[tf._time_col]

    # Verify all values are datetime objects
    assert all(isinstance(t, (datetime, pd.Timestamp)) for t in time_col), "Time column should contain datetime values"

    # Additional validation for correct sorting
    assert_sorted_by_time(tf.df)


def test_timeframe_invalid_mode(simple_df: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame properly rejects invalid modes."""
    with pytest.raises(ModeValidationError):
        TimeFrame(df=simple_df, time_col="time", target_col="target", mode="invalid_mode")


def test_timeframe_missing_columns(simple_df: SupportedTemporalDataFrame) -> None:
    """Test error handling for missing columns."""
    with pytest.raises(TimeColumnError, match=r".*must exist.*"):
        TimeFrame(simple_df, time_col="invalid_time", target_col="target")

    with pytest.raises(TimeColumnError, match=r".*must exist.*"):
        TimeFrame(simple_df, time_col="time", target_col="invalid_target")


def test_timeframe_sort_order(simple_df: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame respects sort order parameter."""
    tf_ascending = TimeFrame(df=simple_df, time_col="time", target_col="target", ascending=True)
    assert_sorted_by_time(tf_ascending.df, ascending=True)

    tf_descending = TimeFrame(df=simple_df, time_col="time", target_col="target", ascending=False)
    assert_sorted_by_time(tf_descending.df, ascending=False)


@pytest.fixture(params=["pandas"])  # Only test pandas backend for now
def df_with_non_numeric(request, data_config: DataConfigType) -> Generator[SupportedTemporalDataFrame, None, None]:
    """Generate a DataFrame with non-numeric values in feature columns."""
    pd_df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "target": [0.5, 0.6, 0.7, 0.8, 0.9],
            "feature_1": ["a", "b", "c", "d", "e"],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature_3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    yield pd_df


def test_timeframe_rejects_non_numeric(df_with_non_numeric: SupportedTemporalDataFrame) -> None:
    """Test that TimeFrame properly rejects data with non-numeric values in feature columns."""
    with pytest.raises(ValueError, match=r"could not convert string to float: 'a'"):
        TimeFrame(df=df_with_non_numeric, time_col="time", target_col="target", mode=MODE_SINGLE_STEP)


@nw.narwhalify
def test_validate_data_with_invalid_datetime_string(simple_df: SupportedTemporalDataFrame) -> None:
    """Test validate_data with an invalid datetime string in the time column."""
    # Create new DataFrame with invalid date using Narwhals operations
    invalid_df = simple_df.select([
        nw.lit("invalid_date").alias("time"),
        *[nw.col(col) for col in simple_df.columns if col != "time"]
    ])
    
    with pytest.raises(TimeColumnError, match=r"time_col must be numeric, datetime, or a valid datetime string"):
        TimeFrame(df=invalid_df, time_col="time", target_col="target")


@nw.narwhalify
def test_setup_timeframe_calls_validate_data(simple_df: SupportedTemporalDataFrame) -> None:
    """Test that _setup_timeframe calls validate_data and handles exceptions."""
    # Create new DataFrame with invalid date using Narwhals operations
    invalid_df = simple_df.select([
        nw.lit("invalid_date").alias("time"),
        *[nw.col(col) for col in simple_df.columns if col != "time"]
    ])
    
    with pytest.raises(TimeColumnError):
        TimeFrame(df=invalid_df, time_col="time", target_col="target")


def test_sort_data_method(simple_df: SupportedTemporalDataFrame) -> None:
    """Test the sort_data method to ensure it sorts correctly."""
    tf = TimeFrame(df=simple_df, time_col="time", target_col="target")
    
    # Get initial time values
    initial_df = tf.df.to_pandas() if hasattr(tf.df, "to_pandas") else tf.df
    initial_values = initial_df["time"].tolist()
    
    # Sort descending
    sorted_df = tf.sort_data(tf.df, ascending=False)
    sorted_df_pd = sorted_df.to_pandas() if hasattr(sorted_df, "to_pandas") else sorted_df
    sorted_values = sorted_df_pd["time"].tolist()
    
    # Verify sorting
    assert sorted_values == sorted(initial_values, reverse=True), "DataFrame not sorted correctly in descending order"


def test_update_data_method(simple_df: SupportedTemporalDataFrame) -> None:
    """Test the update_data method to ensure it updates the DataFrame and column configurations correctly."""
    tf = TimeFrame(df=simple_df, time_col="time", target_col="target")
    
    # Create new DataFrame with modified target using Narwhals operations
    new_df = tf.df.to_pandas() if hasattr(tf.df, "to_pandas") else tf.df
    new_df["new_target"] = new_df["target"] * 2
    
    tf.update_data(new_df, new_target_col="new_target")
    df_pd = tf.df.to_pandas() if hasattr(tf.df, "to_pandas") else tf.df
    assert "new_target" in df_pd.columns, "new_target column not found in updated DataFrame"
    assert tf._target_col == "target"  # Original target column name should not change
