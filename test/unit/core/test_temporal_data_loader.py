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

"""Unit tests for TimeFrame class.

Tests the TimeFrame class which acts as a delegator to core_utils functions
for temporal data loading and validation. Tests use synthetic data generation
to ensure backend-agnostic operations work correctly.
"""

import pandas as pd
import pytest

from temporalscope.core.core_utils import TEST_BACKENDS, TimeColumnError
from temporalscope.core.temporal_data_loader import MODE_MULTI_TARGET, MODE_SINGLE_TARGET, TimeFrame
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series


@pytest.fixture(params=TEST_BACKENDS)
def synthetic_df(request):
    """Generate synthetic DataFrame for each backend."""
    return generate_synthetic_time_series(backend=request.param, num_samples=10, num_features=2, drop_time=False)


# ========================= Tests for TimeFrame initialization =========================


def test_timeframe_init_basic(synthetic_df):
    """Test basic TimeFrame initialization."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    assert tf.mode == MODE_SINGLE_TARGET
    assert tf.ascending is True
    # No need to check DataFrame type since we use native pandas


def test_timeframe_init_invalid_time_col():
    """Test TimeFrame initialization with invalid time column."""
    df = pd.DataFrame({"wrong_col": [1, 2, 3], "target": [10, 20, 30]})
    with pytest.raises(ValueError, match="Column 'time' does not exist"):
        TimeFrame(df, time_col="time", target_col="target")


def test_timeframe_init_invalid_target_col(synthetic_df):
    """Test TimeFrame initialization with invalid target column."""
    with pytest.raises(ValueError, match="Column 'wrong_target' does not exist"):
        TimeFrame(synthetic_df, time_col="time", target_col="wrong_target")


def test_timeframe_init_empty_df():
    """Test TimeFrame initialization with empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        TimeFrame(df, time_col="time", target_col="target")


def test_timeframe_init_invalid_mode(synthetic_df):
    """Test TimeFrame initialization with invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        TimeFrame(synthetic_df, time_col="time", target_col="feature_1", mode="invalid_mode")


def test_timeframe_init_multi_target(synthetic_df):
    """Test TimeFrame initialization in multi-target mode."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1", mode=MODE_MULTI_TARGET)
    assert tf.mode == MODE_MULTI_TARGET


def test_timeframe_init_descending_sort(synthetic_df):
    """Test TimeFrame initialization with descending sort."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1", ascending=False)
    assert tf.ascending is False


def test_timeframe_init_no_sort(synthetic_df):
    """Test TimeFrame initialization without sorting."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1", sort=False)
    # Original order should be preserved
    # Convert both to pandas for comparison if needed
    df1 = tf.df.to_pandas() if hasattr(tf.df, "to_pandas") else tf.df
    df2 = synthetic_df.to_pandas() if hasattr(synthetic_df, "to_pandas") else synthetic_df
    pd.testing.assert_frame_equal(df1, df2)


def test_timeframe_init_invalid_parameter_types(synthetic_df):
    """Test TimeFrame initialization with invalid parameter types."""
    # Test invalid time_col type
    with pytest.raises(TypeError, match="`time_col` must be a string"):
        TimeFrame(synthetic_df, time_col=123, target_col="feature_1")

    # Test invalid target_col type
    with pytest.raises(TypeError, match="`target_col` must be a string"):
        TimeFrame(synthetic_df, time_col="time", target_col=123)

    # Test invalid sort type
    with pytest.raises(TypeError, match="`sort` must be a boolean"):
        TimeFrame(synthetic_df, time_col="time", target_col="feature_1", sort="yes")

    # Test invalid ascending type
    with pytest.raises(TypeError, match="`ascending` must be a boolean"):
        TimeFrame(synthetic_df, time_col="time", target_col="feature_1", ascending="yes")

    # Test invalid verbose type
    with pytest.raises(TypeError, match="`verbose` must be a boolean"):
        TimeFrame(synthetic_df, time_col="time", target_col="feature_1", verbose="yes")

    # Test invalid id_col type
    with pytest.raises(TypeError, match="`id_col` must be a string or None"):
        TimeFrame(synthetic_df, time_col="time", target_col="feature_1", id_col=123)


# ========================= Tests for TimeFrame validation =========================


def test_timeframe_validate_numeric_time(synthetic_df):
    """Test TimeFrame validation with numeric time column."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    tf.validate_dataframe(synthetic_df)  # Should pass


def test_timeframe_validate_datetime_time():
    """Test TimeFrame validation with datetime time column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "target": [1, 2, 3]})
    tf = TimeFrame(df, time_col="time", target_col="target")
    tf.validate_dataframe(df)  # Should pass


def test_timeframe_validate_non_numeric_features():
    """Test TimeFrame validation with non-numeric feature columns."""
    df = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "target": [1, 2, 3],
            "category": ["A", "B", "C"],  # Non-numeric column
        }
    )
    with pytest.raises(ValueError, match="must be numeric"):
        TimeFrame(df, time_col="time", target_col="target")


# ========================= Tests for TimeFrame sorting =========================


def test_timeframe_sort_numeric_time():
    """Test TimeFrame sorting with numeric time column."""
    df = pd.DataFrame({"time": [3, 1, 2], "target": [30, 10, 20]})
    tf = TimeFrame(df, time_col="time", target_col="target")
    assert list(tf.df["time"]) == [1, 2, 3]
    assert list(tf.df["target"]) == [10, 20, 30]


def test_timeframe_sort_datetime_time():
    """Test TimeFrame sorting with datetime time column."""
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"time": [dates[2], dates[0], dates[1]], "target": [30, 10, 20]})
    tf = TimeFrame(df, time_col="time", target_col="target")
    # Convert to pandas if needed and reset index names for comparison
    time_series = tf.df["time"].to_pandas() if hasattr(tf.df["time"], "to_pandas") else tf.df["time"]
    sorted_dates = pd.DatetimeIndex(time_series)
    sorted_dates.name = None  # Reset name for comparison
    pd.testing.assert_index_equal(sorted_dates, dates)
    assert list(tf.df["target"]) == [10, 20, 30]


def test_timeframe_sort_descending():
    """Test TimeFrame descending sort."""
    df = pd.DataFrame({"time": [1, 2, 3], "target": [10, 20, 30]})
    tf = TimeFrame(df, time_col="time", target_col="target", ascending=False)
    assert list(tf.df["time"]) == [3, 2, 1]
    assert list(tf.df["target"]) == [30, 20, 10]


# ========================= Tests for TimeFrame conversion =========================


def test_timeframe_convert_to_numeric():
    """Test TimeFrame time column conversion to numeric."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "target": [1, 2, 3]})
    tf = TimeFrame(df, time_col="time", target_col="target", time_col_conversion="numeric")
    assert pd.api.types.is_float_dtype(tf.df["time"].dtype)


def test_timeframe_convert_to_datetime():
    """Test TimeFrame time column conversion to datetime."""
    df = pd.DataFrame({"time": [1672531200, 1672617600, 1672704000], "target": [1, 2, 3]})
    tf = TimeFrame(df, time_col="time", target_col="target", time_col_conversion="datetime")
    assert pd.api.types.is_datetime64_dtype(tf.df["time"].dtype)


def test_timeframe_convert_invalid_type(synthetic_df):
    """Test TimeFrame conversion with invalid type."""
    with pytest.raises(ValueError, match="Invalid `time_col_conversion`"):
        TimeFrame(synthetic_df, time_col="time", target_col="feature_1", time_col_conversion="invalid")


# ========================= Tests for TimeFrame temporal validation =========================


def test_timeframe_temporal_uniqueness():
    """Test TimeFrame temporal uniqueness validation."""
    df = pd.DataFrame({"time": [1, 2, 2, 3], "target": [10, 20, 30, 40]})  # Duplicate at t=2
    with pytest.raises(TimeColumnError, match="Duplicate timestamps found"):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True)


def test_timeframe_temporal_uniqueness_by_group():
    """Test TimeFrame temporal uniqueness validation within groups."""
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],  # Same times ok for different ids
            "target": [10, 20, 30, 40],
        }
    )
    tf = TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True, id_col="id")
    assert len(tf.df) == 4  # All rows preserved


def test_timeframe_temporal_uniqueness_violation_in_group():
    """Test TimeFrame temporal uniqueness violation within a group."""
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 1, 2, 3],  # Duplicate for id=1
            "target": [10, 20, 30, 40],
        }
    )
    with pytest.raises(TimeColumnError, match="Duplicate timestamps found within groups"):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True, id_col="id")


# ========================= Tests for TimeFrame update =========================


def test_timeframe_update_dataframe(synthetic_df):
    """Test TimeFrame DataFrame update."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    new_df = generate_synthetic_time_series(backend="pandas", num_samples=5, num_features=2, drop_time=False)
    tf.update_dataframe(new_df)
    pd.testing.assert_frame_equal(tf.df, new_df)


def test_timeframe_update_invalid_df(synthetic_df):
    """Test TimeFrame update with invalid DataFrame."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    invalid_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="Column 'time' does not exist"):
        tf.update_dataframe(invalid_df)


def test_timeframe_update_empty_df(synthetic_df):
    """Test TimeFrame update with empty DataFrame."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        tf.update_dataframe(empty_df)


def test_timeframe_update_with_verbose(synthetic_df):
    """Test TimeFrame update with verbose mode."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1", verbose=True)
    new_df = generate_synthetic_time_series(backend="pandas", num_samples=5, num_features=2, drop_time=False)
    tf.update_dataframe(new_df)  # Should print success message


# ========================= Tests for TimeFrame metadata =========================


def test_timeframe_metadata(synthetic_df):
    """Test TimeFrame metadata storage."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    tf.metadata["description"] = "Test dataset"
    tf.metadata["model_details"] = {"type": "LSTM"}
    assert tf.metadata["description"] == "Test dataset"
    assert tf.metadata["model_details"]["type"] == "LSTM"


def test_timeframe_metadata_empty(synthetic_df):
    """Test TimeFrame metadata starts empty."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1")
    assert tf.metadata == {}


def test_timeframe_setup_empty_df():
    """Test TimeFrame setup with empty DataFrame."""
    df = pd.DataFrame()
    tf = TimeFrame(pd.DataFrame({"time": [1], "target": [1]}), time_col="time", target_col="target")
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        tf.setup(df)


def test_timeframe_setup_missing_columns():
    """Test TimeFrame setup with missing required columns."""
    df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    tf = TimeFrame(pd.DataFrame({"time": [1], "target": [1]}), time_col="time", target_col="target")
    with pytest.raises(ValueError, match="Column 'time' does not exist"):
        tf.setup(df)


def test_timeframe_metadata_access():
    """Test TimeFrame metadata access and modification."""
    df = pd.DataFrame({"time": [1, 2, 3], "target": [10, 20, 30]})
    tf = TimeFrame(df, time_col="time", target_col="target")

    # Test metadata access
    assert isinstance(tf.metadata, dict)

    # Test metadata modification
    tf.metadata["test_key"] = "test_value"
    assert tf.metadata["test_key"] == "test_value"

    # Test nested metadata
    tf.metadata["nested"] = {"key": "value"}
    assert tf.metadata["nested"]["key"] == "value"


def test_timeframe_validate_dataframe_empty():
    """Test validate_dataframe with empty DataFrame."""
    df = pd.DataFrame()
    tf = TimeFrame(pd.DataFrame({"time": [1], "target": [1]}), time_col="time", target_col="target")
    with pytest.raises(ValueError, match="Empty DataFrame provided"):
        tf.validate_dataframe(df)


def test_timeframe_validate_dataframe_missing_time_col():
    """Test validate_dataframe with missing time column."""
    df = pd.DataFrame({"wrong_col": [1, 2, 3], "target": [10, 20, 30]})
    tf = TimeFrame(pd.DataFrame({"time": [1], "target": [1]}), time_col="time", target_col="target")
    with pytest.raises(ValueError, match="Column 'time' does not exist"):
        tf.validate_dataframe(df)


def test_timeframe_validate_dataframe_missing_target_col():
    """Test validate_dataframe with missing target column."""
    df = pd.DataFrame({"time": [1, 2, 3], "wrong_col": [10, 20, 30]})
    tf = TimeFrame(pd.DataFrame({"time": [1], "target": [1]}), time_col="time", target_col="target")
    with pytest.raises(ValueError, match="Column 'target' does not exist"):
        tf.validate_dataframe(df)


def test_timeframe_validate_dataframe_with_verbose():
    """Test validate_dataframe with verbose mode."""
    df = pd.DataFrame({"time": [1, 2, 3], "target": [10, 20, 30]})
    tf = TimeFrame(df, time_col="time", target_col="target", verbose=True)
    tf.validate_dataframe(df)  # Should print validation message


def test_timeframe_setup_with_verbose(synthetic_df):
    """Test TimeFrame setup with verbose mode."""
    tf = TimeFrame(synthetic_df, time_col="time", target_col="feature_1", verbose=True)
    # Test both conversion and temporal validation messages
    tf.setup(synthetic_df, time_col_conversion="numeric", enforce_temporal_uniqueness=True)
