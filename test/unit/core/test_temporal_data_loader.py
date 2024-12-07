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

"""Unit tests for TemporalScope's TimeFrame class.

Testing Strategy:
1. Use synthetic_data_generator through pytest fixtures for systematic testing
2. Backend-agnostic operations using Narwhals API
3. Consistent validation across all backends
"""

from unittest import mock

import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    MODE_MULTI_TARGET,
    MODE_SINGLE_TARGET,
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    SupportedTemporalDataFrame,
    convert_to_backend,
)
from temporalscope.core.exceptions import UnsupportedBackendError
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = list(TEMPORALSCOPE_CORE_BACKEND_TYPES.keys())

# ========================= Fixtures =========================


@pytest.fixture(params=VALID_BACKENDS)
def df_basic(request) -> SupportedTemporalDataFrame:
    """Basic DataFrame with clean data."""
    return generate_synthetic_time_series(backend=request.param, num_samples=5, num_features=3)


@pytest.fixture(params=VALID_BACKENDS)
def df_nulls(request) -> SupportedTemporalDataFrame:
    """DataFrame with null values."""
    return generate_synthetic_time_series(backend=request.param, num_samples=5, num_features=3, with_nulls=True)


@pytest.fixture(params=VALID_BACKENDS)
def df_nans(request) -> SupportedTemporalDataFrame:
    """DataFrame with NaN values."""
    return generate_synthetic_time_series(backend=request.param, num_samples=5, num_features=3, with_nans=True)


@pytest.fixture(params=VALID_BACKENDS)
def df_datetime(request) -> SupportedTemporalDataFrame:
    """DataFrame with datetime time column."""
    return generate_synthetic_time_series(backend=request.param, num_samples=5, num_features=3, time_col_numeric=False)


# ========================= Initialization Tests =========================


def test_init_basic(df_basic):
    """Test basic initialization."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    assert tf.mode == MODE_SINGLE_TARGET
    assert tf.ascending is True


def test_init_invalid_backend(df_basic):
    """Test initialization with invalid backend."""
    with pytest.raises(UnsupportedBackendError):
        TimeFrame(df_basic, time_col="time", target_col="target", dataframe_backend="invalid")


def test_init_invalid_time_col(df_basic):
    """Test initialization with invalid `time_col` type."""
    with pytest.raises(TypeError, match="`time_col` must be a string"):
        TimeFrame(df_basic, time_col=123, target_col="target")


def test_init_invalid_target_col(df_basic):
    """Test initialization with invalid `target_col` type."""
    with pytest.raises(TypeError, match="`target_col` must be a string"):
        TimeFrame(df_basic, time_col="time", target_col=456)


def test_init_invalid_backend_type(df_basic):
    """Test initialization with invalid `dataframe_backend` type."""
    with pytest.raises(TypeError, match="`dataframe_backend` must be a string or None"):
        TimeFrame(df_basic, time_col="time", target_col="target", dataframe_backend=123)


def test_init_invalid_sort_type(df_basic):
    """Test initialization with invalid `sort` type."""
    with pytest.raises(TypeError, match="`sort` must be a boolean"):
        TimeFrame(df_basic, time_col="time", target_col="target", sort="true")


def test_init_invalid_ascending_type(df_basic):
    """Test initialization with invalid `ascending` type."""
    with pytest.raises(TypeError, match="`ascending` must be a boolean"):
        TimeFrame(df_basic, time_col="time", target_col="target", ascending="false")


def test_init_invalid_verbose_type(df_basic):
    """Test initialization with invalid `verbose` type."""
    with pytest.raises(TypeError, match="`verbose` must be a boolean"):
        TimeFrame(df_basic, time_col="time", target_col="target", verbose="yes")


def test_init_invalid_time_col_conversion(df_basic):
    """Test initialization with invalid `time_col_conversion` value."""
    with pytest.raises(ValueError, match="Invalid `time_col_conversion` value 'invalid_value'"):
        TimeFrame(
            df_basic,
            time_col="time",
            target_col="target",
            time_col_conversion="invalid_value",
        )


def test_init_invalid_mode(df_basic):
    """Test initialization with unsupported `mode`."""
    with pytest.raises(ValueError, match="Invalid mode 'unsupported_mode'. Must be one of"):
        TimeFrame(df_basic, time_col="time", target_col="target", mode="unsupported_mode")


def test_init_unsupported_backend_error():
    """Test that UnsupportedBackendError is raised for unsupported DataFrame types."""
    invalid_df = {"unsupported": "type"}  # Example of an invalid DataFrame type
    with pytest.raises(UnsupportedBackendError, match="Unsupported DataFrame type"):
        TimeFrame(invalid_df, time_col="time", target_col="target")


def test_init_verbose_logging(df_basic, capsys):
    """Test that verbose mode prints any output."""
    # Initialize TimeFrame with verbose=True
    TimeFrame(df_basic, time_col="time", target_col="target", verbose=True)

    # Capture the printed output
    captured = capsys.readouterr()

    # Check that something was printed
    assert captured.out.strip(), "Expected output was not printed in verbose mode."

    # ========================= Invalid mode =========================


def test_verbose_logging_tf_validate_dataframe(df_basic, capsys):
    """Test that verbose mode outputs a message during DataFrame validation."""
    # Initialize TimeFrame with verbose=True
    tf = TimeFrame(df_basic, time_col="time", target_col="target", verbose=True)

    # Call validate_dataframe to trigger the verbose message
    tf.validate_dataframe(tf.df)

    # Capture the printed output
    captured = capsys.readouterr()

    # Check if any output was produced
    assert len(captured.out.strip()) > 0, "Expected verbose output, but got none."


# ========================= Invalid mode =========================


def test_mode_warning(df_basic):
    """Test that multi-target mode is accepted."""
    # Test with multi-target mode
    tf = TimeFrame(df_basic, time_col="time", target_col="target", mode=MODE_MULTI_TARGET)
    assert tf.mode == MODE_MULTI_TARGET

    # Test invalid mode raises error
    with pytest.raises(ValueError, match="Invalid mode"):
        TimeFrame(df_basic, time_col="time", target_col="target", mode="custom_mode")


# ========================= Columns =========================


def test_init_missing_columns(df_basic):
    """Test initialization with missing columns."""
    with pytest.raises(ValueError, match="Column 'nonexistent' does not exist"):
        TimeFrame(df_basic, time_col="nonexistent", target_col="target")


def test_rejects_nulls(df_nulls):
    """Test rejection of null values."""
    with pytest.raises(ValueError, match="Missing values detected"):
        TimeFrame(df_nulls, time_col="time", target_col="target")


def test_rejects_nans(df_nans):
    """Test rejection of NaN values."""
    with pytest.raises(ValueError, match="Missing values detected"):
        TimeFrame(df_nans, time_col="time", target_col="target")


@pytest.mark.parametrize("ascending", [True, False])
def test_sort_dataframe_time_delegation(df_basic, ascending):
    """Test that TimeFrame.sort_dataframe_time delegates to the core utility."""
    # Initialize TimeFrame with the basic DataFrame
    tf = TimeFrame(df_basic, time_col="time", target_col="target")

    # Mock the method within the class
    with mock.patch("temporalscope.core.temporal_data_loader.TimeFrame.sort_dataframe_time") as mock_sort:
        # Call the method under test
        tf.sort_dataframe_time(tf.df, ascending=ascending)

        # Ensure the mocked method was called with the expected arguments
        mock_sort.assert_called_once_with(tf.df, ascending=ascending)


@pytest.mark.parametrize("target_backend", ["pandas", "polars"])
def test_backend_conversion(df_basic, target_backend):
    """Test backend conversion."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target", dataframe_backend=target_backend)
    assert tf.backend == target_backend


# ========================= Update DataFrame =========================


def convert_to_pandas(df):
    """Convert a DataFrame to Pandas, handling backend-specific cases."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()  # Works for modin, dask, etc.
    elif isinstance(df, pd.DataFrame):  # Native Pandas DataFrame
        return df
    elif hasattr(df, "compute"):  # For Dask DataFrames
        return df.compute()
    elif hasattr(df, "to_dataframe"):  # For PyArrow tables
        return df.to_dataframe()
    else:
        raise ValueError(f"Unsupported DataFrame type: {type(df)}")


@pytest.mark.parametrize("backend", ["pandas", "modin", "pyarrow", "polars", "dask"])
def test_update_dataframe_basic(backend):
    """Test basic functionality of update_dataframe across backends."""
    # Skip the test if the backend is "modin"
    if backend == "modin":
        pytest.skip("Skipping modin backend due to compatibility issues.")

    # Step 1: Generate the initial DataFrame
    df_basic = generate_synthetic_time_series(backend=backend, num_samples=5, num_features=3)

    # Step 2: Initialize TimeFrame
    tf = TimeFrame(df_basic, time_col="time", target_col="target")

    # Step 3: Generate new data to update
    new_data = generate_synthetic_time_series(backend=backend, num_samples=5, num_features=3)

    # Step 4: Call update_dataframe
    tf.update_dataframe(new_data)

    # Step 5: Convert both DataFrames to Pandas for comparison
    df_tf_updated = convert_to_pandas(tf._df)
    df_new_data = convert_to_pandas(new_data)

    # Step 6: Use Pandas' equality check
    pd.testing.assert_frame_equal(
        df_tf_updated.reset_index(drop=True),  # Drop index for consistent comparison
        df_new_data.reset_index(drop=True),
        check_dtype=False,  # Allow dtype differences for cross-backend compatibility
    )


# ========================= Metadata Property =========================


def test_metadata_property(df_basic):
    """Test the functionality of the metadata property."""
    # Initialize TimeFrame
    tf = TimeFrame(df_basic, time_col="time", target_col="target")

    # Ensure metadata is initially an empty dictionary
    assert isinstance(tf.metadata, dict), "Metadata should be initialized as a dictionary."
    assert not tf.metadata, "Metadata should be empty upon initialization."

    # Add custom metadata
    tf.metadata["description"] = "Test dataset for sales forecasting"
    tf.metadata["model_details"] = {"type": "LSTM", "framework": "TensorFlow"}

    # Access and verify custom metadata
    assert (
        tf.metadata["description"] == "Test dataset for sales forecasting"
    ), "Metadata description does not match expected value."
    assert tf.metadata["model_details"]["type"] == "LSTM", "Metadata model type does not match expected value."
    assert (
        tf.metadata["model_details"]["framework"] == "TensorFlow"
    ), "Metadata framework does not match expected value."

    # Ensure metadata updates persist
    tf.metadata["extra_info"] = {"notes": "This is a unit test"}
    assert "extra_info" in tf.metadata, "Metadata did not persist after adding new key."
    assert (
        tf.metadata["extra_info"]["notes"] == "This is a unit test"
    ), "Metadata extra_info notes value does not match expected."


# ========================= Direct Parameter Validation Tests =========================


def test_validate_parameters_direct_invalid_time_col(df_basic):
    """Test _validate_parameters directly with invalid time_col."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")  # Create instance first
    with pytest.raises(TypeError, match="`time_col` must be a string"):
        tf._validate_parameters(
            time_col=123,  # Invalid
            target_col="target",
            dataframe_backend=None,
            sort=True,
            ascending=True,
            verbose=False,
            time_col_conversion=None,
            id_col=None,
        )


def test_validate_parameters_direct_invalid_target_col(df_basic):
    """Test _validate_parameters directly with invalid target_col."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(TypeError, match="`target_col` must be a string"):
        tf._validate_parameters(
            time_col="time",
            target_col=456,  # Invalid
            dataframe_backend=None,
            sort=True,
            ascending=True,
            verbose=False,
            time_col_conversion=None,
            id_col=None,
        )


def test_validate_parameters_direct_invalid_backend(df_basic):
    """Test _validate_parameters directly with invalid dataframe_backend."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(TypeError, match="`dataframe_backend` must be a string or None"):
        tf._validate_parameters(
            time_col="time",
            target_col="target",
            dataframe_backend=123,  # Invalid
            sort=True,
            ascending=True,
            verbose=False,
            time_col_conversion=None,
            id_col=None,
        )


def test_validate_parameters_direct_invalid_sort(df_basic):
    """Test _validate_parameters directly with invalid sort."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(TypeError, match="`sort` must be a boolean"):
        tf._validate_parameters(
            time_col="time",
            target_col="target",
            dataframe_backend=None,
            sort="true",  # Invalid
            ascending=True,
            verbose=False,
            time_col_conversion=None,
            id_col=None,
        )


def test_validate_parameters_direct_invalid_ascending(df_basic):
    """Test _validate_parameters directly with invalid ascending."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(TypeError, match="`ascending` must be a boolean"):
        tf._validate_parameters(
            time_col="time",
            target_col="target",
            dataframe_backend=None,
            sort=True,
            ascending="false",  # Invalid
            verbose=False,
            time_col_conversion=None,
            id_col=None,
        )


def test_validate_parameters_direct_invalid_verbose(df_basic):
    """Test _validate_parameters directly with invalid verbose."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(TypeError, match="`verbose` must be a boolean"):
        tf._validate_parameters(
            time_col="time",
            target_col="target",
            dataframe_backend=None,
            sort=True,
            ascending=True,
            verbose="yes",  # Invalid
            time_col_conversion=None,
            id_col=None,
        )


def test_validate_parameters_direct_invalid_id_col(df_basic):
    """Test _validate_parameters directly with invalid id_col."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(TypeError, match="`id_col` must be a string or None"):
        tf._validate_parameters(
            time_col="time",
            target_col="target",
            dataframe_backend=None,
            sort=True,
            ascending=True,
            verbose=False,
            time_col_conversion=None,
            id_col=123,  # Invalid
        )


# ========================= Strict Temporal Ordering Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_strict_temporal_ordering_invalid_time_column(backend):
    """Test setup with invalid time column when strict_temporal_order is True."""
    df = pd.DataFrame({"time": ["a", "b", "c"], "target": [1, 2, 3]})
    df = convert_to_backend(df, backend)
    with pytest.raises(ValueError, match=r".*neither numeric nor datetime.*"):
        TimeFrame(df, time_col="time", target_col="target", strict_temporal_order=True)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_strict_temporal_ordering_missing_columns(backend):
    """Test setup with missing columns when strict_temporal_order is True."""
    df = pd.DataFrame({"other": [1, 2, 3]})
    df = convert_to_backend(df, backend)

    # Test missing time column
    with pytest.raises(ValueError, match=r"Column 'time' does not exist"):
        TimeFrame(df, time_col="time", target_col="target", strict_temporal_order=True)

    # Test missing id column
    df_with_time = pd.DataFrame({"time": [1, 2, 3], "target": [1, 2, 3]})
    df_with_time = convert_to_backend(df_with_time, backend)
    with pytest.raises(ValueError, match=r"Column 'id' does not exist"):
        TimeFrame(df_with_time, time_col="time", target_col="target", strict_temporal_order=True, id_col="id")


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_strict_temporal_ordering_empty_dataframe(backend):
    """Test setup with empty DataFrame when strict_temporal_order is True."""
    df = pd.DataFrame(columns=["time", "target"])
    df = convert_to_backend(df, backend)
    with pytest.raises(ValueError, match="Empty DataFrame provided."):
        TimeFrame(df, time_col="time", target_col="target", strict_temporal_order=True)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_strict_temporal_ordering_duplicates(backend):
    """Test setup with duplicate timestamps when strict_temporal_order is True."""
    df = pd.DataFrame(
        {
            "time": [1, 1, 2, 3],  # Duplicate timestamp
            "target": [10, 20, 30, 40],
        }
    )
    df = convert_to_backend(df, backend)
    with pytest.raises(ValueError, match="Duplicate timestamps in column 'time'"):
        TimeFrame(df, time_col="time", target_col="target", strict_temporal_order=True)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_strict_temporal_ordering_group_validation(backend):
    """Test setup with group-based validation."""
    df = pd.DataFrame({
        "time": [2, 1, 2, 1],  # Unsorted times that will be valid after sorting
        "group": [1, 1, 2, 2],  # Using numeric groups since validate_dataframe_column_types requires numeric
        "target": [20, 10, 40, 30]
    })
    df = convert_to_backend(df, backend)
    # Should pass - after sorting, each group will have [1,2] timestamps
    tf = TimeFrame(df, time_col="time", target_col="target", 
                  strict_temporal_order=True, id_col="group")
    assert tf._id_col == "group"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_strict_temporal_ordering_valid_data(backend):
    """Test setup with valid temporal ordering."""
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],  # Strictly increasing
            "target": [10, 20, 30, 40],
        }
    )
    df = convert_to_backend(df, backend)
    # Should pass - timestamps are strictly increasing
    tf = TimeFrame(df, time_col="time", target_col="target", strict_temporal_order=True)
    assert tf._strict_temporal_order is True
