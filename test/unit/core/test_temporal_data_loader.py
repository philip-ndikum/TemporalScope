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
    """Test initialization with an invalid backend."""
    with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
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
    with pytest.raises(UnsupportedBackendError, match="Unsupported backend: Backend '123' is not supported"):
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
    # Attempt to create an invalid TimeFrame instance with non-string time_col
    with pytest.raises(TypeError, match="`time_col` must be a string"):
        TimeFrame(  # No need to assign to `tf`
            df_basic,
            time_col=123,  # Invalid
            target_col="target",
            dataframe_backend=None,
            sort=True,
            ascending=True,
            time_col_conversion=None,
            enforce_temporal_uniqueness=False,
            id_col=None,
            verbose=False,
        )


def test_validate_parameters_direct_invalid_target_col(df_basic):
    """Test invalid target_col during TimeFrame initialization."""
    with pytest.raises(TypeError, match="`target_col` must be a string"):
        TimeFrame(
            df_basic,
            time_col="time",
            target_col=456,  # Invalid target_col
        )


def test_validate_parameters_direct_invalid_backend(df_basic):
    """Test _validate_parameters directly with invalid dataframe_backend."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")  # Create instance first
    tf._backend = 123  # Directly set invalid backend
    with pytest.raises(TypeError, match="`dataframe_backend` must be a string or None"):
        tf._validate_parameters()


def test_validate_parameters_direct_invalid_sort(df_basic):
    """Test _validate_parameters directly with invalid sort."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")  # Create instance first
    tf._sort = "true"  # Assign invalid value to the instance variable
    with pytest.raises(TypeError, match="`sort` must be a boolean"):
        tf._validate_parameters()


def test_validate_parameters_direct_invalid_ascending(df_basic):
    """Test _validate_parameters directly with invalid ascending."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")  # Create instance first
    tf._ascending = "false"  # Assign invalid value to the instance variable
    with pytest.raises(TypeError, match="`ascending` must be a boolean"):
        tf._validate_parameters()


def test_validate_parameters_direct_invalid_verbose(df_basic):
    """Test _validate_parameters directly with invalid verbose."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")  # Create instance first
    tf._verbose = "yes"  # Assign invalid value to the instance variable
    with pytest.raises(TypeError, match="`verbose` must be a boolean"):
        tf._validate_parameters()


def test_validate_parameters_direct_invalid_id_col(df_basic):
    """Test _validate_parameters directly with invalid id_col."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")  # Create instance first
    tf._id_col = 123  # Assign invalid value to the instance variable
    with pytest.raises(TypeError, match="`id_col` must be a string or None"):
        tf._validate_parameters()


# ========================= Setup Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_enforce_temporal_uniquenessing_invalid_time_column(backend):
    """Test setup with invalid time column when enforce_temporal_uniqueness is True."""
    df = pd.DataFrame({"time": ["a", "b", "c"], "target": [1, 2, 3]})
    df = convert_to_backend(df, backend)
    with pytest.raises(ValueError, match=r".*neither numeric nor datetime.*"):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_validate_columns_missing_columns(backend):
    """Test setup with missing columns required for validation."""
    df = pd.DataFrame({"other": [1, 2, 3]})
    df = convert_to_backend(df, backend)

    # Test missing time column
    with pytest.raises(ValueError, match=r"Column .* does not exist"):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True)

    # Test missing id column when enforcing temporal uniqueness
    df_with_time = pd.DataFrame({"time": [1, 2, 3], "target": [1, 2, 3]})
    df_with_time = convert_to_backend(df_with_time, backend)
    with pytest.raises(ValueError, match=r"Column .* does not exist"):
        TimeFrame(
            df_with_time,
            time_col="time",
            target_col="target",
            enforce_temporal_uniqueness=True,
            id_col="id",  # Missing id_col
        )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_enforce_temporal_uniquenessing_empty_dataframe(backend):
    """Test setup with empty DataFrame when enforce_temporal_uniqueness is True."""
    df = pd.DataFrame(columns=["time", "target"])
    df = convert_to_backend(df, backend)
    with pytest.raises(ValueError, match="Empty DataFrame provided."):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_enforce_temporal_uniquenessing_duplicates(backend):
    """Test setup with duplicate timestamps when enforce_temporal_uniqueness is True."""
    df = pd.DataFrame(
        {
            "time": [1, 1, 2, 3],  # Duplicate timestamp
            "target": [10, 20, 30, 40],
        }
    )
    df = convert_to_backend(df, backend)
    # Update regex to match 'None' or empty string
    with pytest.raises(ValueError, match=r"Duplicate timestamps in id_col '.*' column 'time'."):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_enforce_temporal_uniqueness_group_validation(backend):
    """Test setup with group-based validation ensuring duplicate timestamps raise ValueError."""
    # Create a DataFrame with duplicates in each group
    # Group 1 has time=[1,1], Group 2 has time=[2,2]
    df = pd.DataFrame(
        {
            "time": [1, 1, 2, 2],
            "group": [1, 1, 2, 2],
            "target": [20, 10, 40, 30],
        }
    )
    df = convert_to_backend(df, backend)

    # Expecting a ValueError due to duplicate timestamps in 'group' column 'time'
    with pytest.raises(ValueError, match=r"Duplicate timestamps in id_col 'group' column 'time'."):
        TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True, id_col="group")


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_enforce_temporal_uniquenessing_valid_data(backend):
    """Test setup with valid temporal ordering."""
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],  # Strictly increasing
            "target": [10, 20, 30, 40],
        }
    )
    df = convert_to_backend(df, backend)
    # Should pass - timestamps are strictly increasing
    tf = TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=True)
    assert tf._enforce_temporal_uniqueness is True


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_print_statements_numeric_conversion(backend, capsys):
    """Test that the numeric conversion print statement is triggered."""
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2023-01-01", periods=3, freq="D"),  # Valid datetime
            "target": [10, 20, 30],
        }
    )
    df = convert_to_backend(df, backend)

    # Initialize TimeFrame with verbose=True to enable printing
    tf = TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=False, verbose=True)
    tf.setup(df, time_col_conversion="numeric")  # Trigger numeric conversion

    # Capture printed output
    captured = capsys.readouterr()
    expected_message = "Converted column 'time' to numeric (Unix timestamp)."
    assert expected_message in captured.out


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_setup_print_statements_datetime_conversion(backend, capsys):
    """Test that the datetime conversion print statement is triggered."""
    df = pd.DataFrame(
        {
            "time": [1672531200, 1672617600, 1672704000],  # Unix timestamps
            "target": [10, 20, 30],
        }
    )
    df = convert_to_backend(df, backend)

    # Initialize TimeFrame with verbose=True to enable printing
    tf = TimeFrame(df, time_col="time", target_col="target", enforce_temporal_uniqueness=False, verbose=True)
    tf.setup(df, time_col_conversion="datetime")  # Trigger datetime conversion

    # Capture printed output
    captured = capsys.readouterr()
    expected_message = "Converted column 'time' to datetime."
    assert expected_message in captured.out


# ========================= Valid Numeric Conversion =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_time_col_conversion_to_numeric_valid_datetime(backend):
    """Test numeric conversion of a valid datetime column."""
    # Step 1: Create the initial DataFrame in Pandas
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2023-01-01", periods=3, freq="D"),
            "target": [10, 20, 30],
        }
    )

    # Step 2: Convert to the specified backend
    df = convert_to_backend(df, backend)

    # Step 3: Initialize the TimeFrame
    tf = TimeFrame(df, time_col="time", target_col="target")

    # Step 4: Perform the setup with conversion to numeric
    converted_df = tf.setup(df, time_col_conversion="numeric")

    # Step 5: Convert back to Pandas for validation
    pandas_df = convert_to_backend(converted_df, backend="pandas")

    # Step 6: Validate that the time column is numeric
    assert pd.api.types.is_numeric_dtype(pandas_df["time"]), "Time column was not converted to numeric."


# ========================= Invalid Numeric Conversion =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_time_col_conversion_to_numeric_invalid_non_datetime(backend):
    """Test numeric conversion skips already numeric columns gracefully."""
    # Step 1: Create a DataFrame with numeric columns
    df = pd.DataFrame(
        {
            "time": [1672531200, 1672617600, 1672704000],  # Already numeric
            "target": [10, 20, 30],
        }
    )

    # Step 2: Convert the DataFrame to the specified backend
    df = convert_to_backend(df, backend)

    # Step 3: Initialize TimeFrame and run the setup
    tf = TimeFrame(df, time_col="time", target_col="target")
    converted_df = tf.df

    # Step 4: Convert back to Pandas for comparison using `convert_to_backend`
    converted_df_native = convert_to_backend(converted_df, "pandas")
    original_df_native = convert_to_backend(df, "pandas")

    # Step 5: Explicitly handle Modin backend if necessary
    if backend == "modin":
        converted_df_native = converted_df_native._to_pandas()
        original_df_native = original_df_native._to_pandas()

    # Step 6: Assert that the numeric column remains unchanged
    pd.testing.assert_frame_equal(converted_df_native, original_df_native)


# ========================= Already Numeric Input =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_time_col_conversion_to_numeric_already_numeric(backend):
    """Test behavior with a column that is already numeric."""
    # Step 1: Create the DataFrame in Pandas
    df = pd.DataFrame(
        {
            "time": [1672531200, 1672617600, 1672704000],  # Unix timestamps
            "target": [10, 20, 30],
        }
    )
    # Step 2: Convert to backend
    df = convert_to_backend(df, backend)
    # Step 3: Validate that no change is made
    tf = TimeFrame(df, time_col="time", target_col="target")
    converted_df = tf.setup(df, time_col_conversion="numeric")
    # Step 4: Convert back to Pandas for validation
    converted_df = convert_to_backend(converted_df, backend="pandas")
    # Step 5: Validate that the time column is still numeric
    assert pd.api.types.is_numeric_dtype(converted_df["time"]), "Time column was not numeric."


# ========================= Mixed Valid and Invalid Values =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_time_col_conversion_to_numeric_mixed_invalid(backend):
    """Test numeric conversion with a column containing mixed valid and invalid datetime values."""
    # Step 1: Create a DataFrame with mixed valid and invalid datetime values
    df = pd.DataFrame(
        {
            "time": ["2023-01-01", "2023-01-02", "invalid_date"],  # Mixed valid and invalid
            "target": [10, 20, 30],
        }
    )

    # Step 2: Convert the DataFrame to the specified backend
    df = convert_to_backend(df, backend)

    # Step 3: Assert that initializing TimeFrame raises a ValueError
    with pytest.raises(ValueError, match="Column 'time' is neither numeric nor datetime."):
        TimeFrame(df, time_col="time", target_col="target")
