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

# TemporalScope/test/unit/test_core_utils.py

import warnings
import pytest
from unittest.mock import patch
from typing import Optional, Tuple, Union

import modin.pandas as mpd
import pandas as pd
import polars as pl
import numpy as np

# Import core utility functions
from temporalscope.core.core_utils import (
    check_nans,
    check_nulls,
    get_api_keys,
    get_default_backend_cfg,
    validate_and_convert_input,
    validate_backend,
    print_divider,
    infer_backend_from_dataframe,
    is_timestamp_like,
    is_numeric,
    has_mixed_frequencies,
    sort_dataframe,
    check_empty_columns
)

# Import exceptions
from temporalscope.core.exceptions import UnsupportedBackendError, MixedFrequencyWarning, MixedTimezonesWarning

# Import the sample data generation and fixture from test_data_utils
from temporalscope.datasets.synthetic_data_generator import create_sample_data, sample_df_with_conditions

# # Constants
# BACKEND_PANDAS = "pd"
# BACKEND_MODIN = "mpd"
# BACKEND_POLARS = "pl"
# SUPPORTED_BACKENDS = [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS]

# # Mock API key constants
# MOCK_OPENAI_API_KEY = "mock_openai_key"
# MOCK_CLAUDE_API_KEY = "mock_claude_key"

# # --- Tests with Parametrization ---

# @pytest.mark.parametrize(
#     "check_func, with_nulls, with_nans",
#     [
#         (check_nulls, True, False),  # Test with nulls, no NaNs
#         (check_nulls, False, False),  # Test without nulls
#         (check_nans, False, True),  # Test with NaNs
#         (check_nans, False, False),  # Test without NaNs
#     ]
# )
# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_check_funcs(backend, sample_df_with_conditions, check_func, with_nulls, with_nans):
#     """Test check_nulls and check_nans for both nulls and NaNs across backends."""
#     df, _ = sample_df_with_conditions(backend=backend, with_nulls=with_nulls, with_nans=with_nans)
#     result = validate_and_convert_input(df, backend)

#     if check_func == check_nulls:
#         # Calculate nulls for each backend
#         if backend == BACKEND_POLARS:
#             # Polars: Check if null count is greater than 0
#             result_check = result.null_count().select(pl.col("*").sum()).to_numpy().sum() > 0
#         else:
#             # Pandas and Modin
#             result_check = result.isnull().any().any()
#         expected = with_nulls  # True if nulls were introduced, else False
#     else:
#         # Calculate NaNs for each backend
#         if backend == BACKEND_POLARS:
#             # Polars: Use .is_nan() on each column and sum up NaN values
#             result_check = result.select(pl.col("*").is_nan().sum()).to_numpy().sum() > 0
#         else:
#             # Pandas and Modin
#             result_check = result.isna().any().any()
#         expected = with_nans  # True if NaNs were introduced, else False

#     assert result_check == expected, f"Expected {expected} but got {result_check} for backend {backend} using {check_func.__name__}"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_check_nulls(backend, sample_df_with_conditions):
#     """Test check_nulls for detecting null values across backends."""
#     # Case 1: DataFrame with nulls
#     df_with_nulls, _ = sample_df_with_conditions(backend=backend, with_nulls=True)
#     result_with_nulls = check_nulls(df_with_nulls, backend)
#     assert result_with_nulls is True, f"Expected True but got {result_with_nulls} for backend {backend} with nulls"

#     # Case 2: DataFrame without nulls
#     df_without_nulls, _ = sample_df_with_conditions(backend=backend, with_nulls=False)
#     result_without_nulls = check_nulls(df_without_nulls, backend)
#     assert result_without_nulls is False, f"Expected False but got {result_without_nulls} for backend {backend} without nulls"


# @pytest.mark.parametrize("unsupported_backend", ["unsupported_backend", "invalid_backend", "spark"])
# def test_check_nulls_unsupported_backend(unsupported_backend):
#     """Test that check_nulls raises UnsupportedBackendError for unsupported backends."""
#     df = pd.DataFrame({"col1": [1, 2, 3]})  # Sample DataFrame
#     with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
#         check_nulls(df, unsupported_backend)


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_check_nans(backend, sample_df_with_conditions):
#     """Test check_nans for detecting NaN values across backends."""
#     # Case 1: DataFrame with NaNs
#     df_with_nans, _ = sample_df_with_conditions(backend=backend, with_nans=True)
#     result_with_nans = check_nans(df_with_nans, backend)
#     assert result_with_nans is True, f"Expected True but got {result_with_nans} for backend {backend} with NaNs"

#     # Case 2: DataFrame without NaNs
#     df_without_nans, _ = sample_df_with_conditions(backend=backend, with_nans=False)
#     result_without_nans = check_nans(df_without_nans, backend)
#     assert result_without_nans is False, f"Expected False but got {result_without_nans} for backend {backend} without NaNs"


# @pytest.mark.parametrize("unsupported_backend", ["unsupported_backend", "invalid_backend", "spark"])
# def test_check_nans_unsupported_backend(unsupported_backend):
#     """Test that check_nans raises UnsupportedBackendError for unsupported backends."""
#     df = pd.DataFrame({"col1": [1, 2, 3]})  # Sample DataFrame
#     with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
#         check_nans(df, unsupported_backend)


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_validate_backend_supported(backend):
#     """Test that supported backends are validated successfully."""
#     validate_backend(backend)


# @pytest.mark.parametrize("invalid_backend", ["tf", "spark", "unknown"])
# def test_validate_backend_unsupported(invalid_backend):
#     """Test that unsupported backends raise an UnsupportedBackendError."""
#     with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
#         validate_backend(invalid_backend)


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# @pytest.mark.parametrize("target_backend", SUPPORTED_BACKENDS)
# def test_validate_and_convert_input(sample_df_with_conditions, backend, target_backend):
#     """Test that DataFrame conversion between backends works correctly."""
#     df, _ = sample_df_with_conditions(backend=backend, with_nulls=False)
#     result = validate_and_convert_input(df, target_backend)

#     if target_backend == BACKEND_PANDAS:
#         assert isinstance(result, pd.DataFrame), f"Expected Pandas DataFrame but got {type(result)}"
#     elif target_backend == BACKEND_POLARS:
#         assert isinstance(result, pl.DataFrame), f"Expected Polars DataFrame but got {type(result)}"
#     elif target_backend == BACKEND_MODIN:
#         assert isinstance(result, mpd.DataFrame), f"Expected Modin DataFrame but got {type(result)}"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_validate_and_convert_input_invalid_type(backend):
#     """Test that validate_and_convert_input raises TypeError when given an invalid DataFrame type."""
#     invalid_df = "This is not a DataFrame"
#     with pytest.raises(TypeError, match="Input DataFrame type"):
#         validate_and_convert_input(invalid_df, backend)


# @pytest.mark.parametrize("invalid_backend", ["unsupported_backend", "excel", "json", None])
# def test_validate_and_convert_input_invalid_backend(sample_df_with_conditions, invalid_backend):
#     """Test that validate_and_convert_input raises UnsupportedBackendError for invalid or None backend."""
#     df, _ = sample_df_with_conditions(backend=BACKEND_PANDAS)
#     with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
#         validate_and_convert_input(df, invalid_backend)


# def test_print_divider(capsys):
#     """Test the print_divider function outputs the correct string."""
#     print_divider("-", 50)
#     captured = capsys.readouterr()
#     assert captured.out == "-" * 50 + "\n"


# def test_get_api_keys():
#     """Test that get_api_keys retrieves environment variables correctly."""
#     with patch.dict("os.environ", {"OPENAI_API_KEY": MOCK_OPENAI_API_KEY, "CLAUDE_API_KEY": MOCK_CLAUDE_API_KEY}):
#         api_keys = get_api_keys()
#         assert api_keys["OPENAI_API_KEY"] == MOCK_OPENAI_API_KEY
#         assert api_keys["CLAUDE_API_KEY"] == MOCK_CLAUDE_API_KEY

#     with patch.dict("os.environ", {}, clear=True):
#         api_keys = get_api_keys()
#         assert api_keys["OPENAI_API_KEY"] is None
#         assert api_keys["CLAUDE_API_KEY"] is None


# def test_get_default_backend_cfg():
#     """Test that the default backend configuration is returned correctly."""
#     expected_cfg = {
#         "BACKENDS": {
#             BACKEND_POLARS: "polars",
#             BACKEND_PANDAS: "pandas",
#             BACKEND_MODIN: "modin",
#         }
#     }
#     result = get_default_backend_cfg()
#     assert result == expected_cfg, f"Expected {expected_cfg} but got {result}"


# def test_validate_and_convert_input_modin_to_polars(sample_df_with_conditions):
#     """Test Modin DataFrame conversion to Polars."""
#     # Create a sample Modin DataFrame
#     df_modin, _ = sample_df_with_conditions(backend=BACKEND_MODIN)

#     # Mock the _to_pandas method to ensure it's called
#     with patch.object(mpd.DataFrame, "_to_pandas", return_value=df_modin._to_pandas()) as mock_to_pandas:
#         # Convert from Modin to Polars
#         result = validate_and_convert_input(df_modin, BACKEND_POLARS)
#         assert isinstance(result, pl.DataFrame), f"Expected Polars DataFrame but got {type(result)}"
#         mock_to_pandas.assert_called_once()  # Ensure _to_pandas is called



# @pytest.mark.parametrize(
#     "input_df, expected_backend",
#     [
#         (pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), BACKEND_PANDAS),  # Pandas DataFrame
#         (pl.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), BACKEND_POLARS),  # Polars DataFrame
#         (mpd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), BACKEND_MODIN),  # Modin DataFrame
#     ]
# )
# def test_infer_backend_from_dataframe(input_df, expected_backend):
#     """Test the infer_backend_from_dataframe function for supported backends."""
#     assert infer_backend_from_dataframe(input_df) == expected_backend

# def test_infer_backend_from_dataframe_unsupported():
#     """Test that infer_backend_from_dataframe raises an UnsupportedBackendError for unsupported backends."""
#     invalid_df = "This is not a DataFrame"
#     with pytest.raises(UnsupportedBackendError, match="Unsupported DataFrame type"):
#         infer_backend_from_dataframe(invalid_df)


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_is_timestamp_like(backend, sample_df_with_conditions):
#     """Test is_timestamp_like for timestamp-like columns across backends."""
#     df, _ = sample_df_with_conditions(backend=backend, timestamp_like=True)
#     result = is_timestamp_like(df, "time")
#     assert result is True, f"Expected True for timestamp-like column but got {result}"

#     df, _ = sample_df_with_conditions(backend=backend, numeric=True)  # Non-timestamp column
#     result = is_timestamp_like(df, "time")
#     assert result is False, f"Expected False for non-timestamp column but got {result}"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_is_numeric(backend, sample_df_with_conditions):
#     """Test is_numeric for numeric columns across backends."""
#     df, _ = sample_df_with_conditions(backend=backend, numeric=True)
#     result = is_numeric(df, "time")
#     assert result is True, f"Expected True for numeric column but got {result}"

#     df, _ = sample_df_with_conditions(backend=backend, timestamp_like=True)
#     result = is_numeric(df, "time")
#     assert result is False, f"Expected False for non-numeric column but got {result}"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_has_mixed_frequencies(backend, sample_df_with_conditions):
#     """Test has_mixed_frequencies for mixed frequency time columns across backends."""
#     df, _ = sample_df_with_conditions(backend=backend, mixed_frequencies=True)
#     result = has_mixed_frequencies(df, "time")
#     assert result is True, f"Expected True for mixed frequencies but got {result}"

#     df, _ = sample_df_with_conditions(backend=backend, timestamp_like=True)
#     result = has_mixed_frequencies(df, "time")
#     assert result is False, f"Expected False for consistent frequencies but got {result}"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_time_column_not_found(backend, sample_df_with_conditions):
#     """Test that ValueError is raised when the time column does not exist."""
#     df, _ = sample_df_with_conditions(backend=backend)  # Create a sample DataFrame without the 'non_existing_time_col'
#     with pytest.raises(ValueError, match="Column 'non_existing_time_col' not found"):
#         is_timestamp_like(df, "non_existing_time_col")

#     with pytest.raises(ValueError, match="Column 'non_existing_time_col' not found"):
#         is_numeric(df, "non_existing_time_col")

#     with pytest.raises(ValueError, match="Column 'non_existing_time_col' not found"):
#         has_mixed_frequencies(df, "non_existing_time_col")


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_empty_dataframe(backend):
#     """Test handling of empty DataFrames across backends."""
#     # Create an empty DataFrame based on the backend
#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame({"time": []})  # Empty but with a time column
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame({"time": []})
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame({"time": []})

#     # Ensure the functions return False or handle empty DataFrames gracefully
#     assert not is_timestamp_like(df, "time"), "Expected False for is_timestamp_like on empty DataFrame"
#     assert not is_numeric(df, "time"), "Expected False for is_numeric on empty DataFrame"
#     assert not has_mixed_frequencies(df, "time"), "Expected False for has_mixed_frequencies on empty DataFrame"


# @pytest.mark.parametrize(
#     "backend, wrong_backend, expected_exception, expected_message",
#     [
#         # Generalized the regex to match a rough pattern for class types, without being too specific
#         (BACKEND_POLARS, BACKEND_PANDAS, TypeError, r"Expected Pandas DataFrame but got .*polars.*"),
#         (BACKEND_PANDAS, BACKEND_POLARS, TypeError, r"Expected Polars DataFrame but got .*pandas.*"),
#         (BACKEND_MODIN, BACKEND_PANDAS, TypeError, r"Expected Pandas DataFrame but got .*modin.*"),
#         (BACKEND_MODIN, BACKEND_POLARS, TypeError, r"Expected Polars DataFrame but got .*modin.*"),
#         (BACKEND_PANDAS, "unsupported_backend", UnsupportedBackendError, r"Unsupported backend: unsupported_backend\. Supported backends are 'pd', 'mpd', 'pl'\."),
#     ]
# )
# def test_sort_dataframe_exceptions(sample_df_with_conditions, backend, wrong_backend, expected_exception, expected_message):
#     """Test that sort_dataframe raises the correct exceptions for invalid DataFrame types and unsupported backends."""
#     # Create a sample DataFrame for the correct backend
#     df, _ = sample_df_with_conditions(backend=backend, numeric=True)

#     # Try sorting the DataFrame using the wrong backend and expect exceptions
#     with pytest.raises(expected_exception, match=expected_message):
#         sort_dataframe(df, "time", wrong_backend)


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# @pytest.mark.parametrize("ascending", [True, False])
# def test_sort_dataframe(sample_df_with_conditions, backend, ascending):
#     """Test that sort_dataframe correctly sorts the DataFrame by time column."""
#     # Create a sample DataFrame with a numeric time column
#     df, _ = sample_df_with_conditions(backend=backend, numeric=True)

#     # Sort the DataFrame using the utility function
#     sorted_df = sort_dataframe(df, "time", backend, ascending)

#     # Extract the time column from the sorted DataFrame
#     sorted_time_column = sorted_df["time"].to_numpy() if backend != BACKEND_POLARS else sorted_df["time"].to_numpy().flatten()

#     # Calculate the expected sorted time column
#     expected_sorted_time_column = sorted(df["time"].to_numpy(), reverse=not ascending)

#     # Ensure the time column is correctly sorted
#     assert all(sorted_time_column == expected_sorted_time_column), f"Expected sorted time column {expected_sorted_time_column} but got {sorted_time_column} for backend {backend} with ascending={ascending}"

# # --- IKdividual tests for Modin backend ---

# @pytest.mark.parametrize(
#     "wrong_backend, expected_exception, expected_substring",
#     [
#         (BACKEND_PANDAS, TypeError, "Expected Pandas DataFrame"),  # Substring matching for Pandas
#         (BACKEND_POLARS, TypeError, "Expected Polars DataFrame"),  # Substring matching for Polars
#         ("unsupported_backend", UnsupportedBackendError, "Unsupported backend"),  # Catch unsupported backend
#     ]
# )
# def test_sort_dataframe_modin_exceptions(sample_df_with_conditions, wrong_backend, expected_exception, expected_substring):
#     """Test that sort_dataframe raises the correct exceptions for Modin DataFrames and wrong backends."""
#     # Create a sample DataFrame for Modin backend
#     df, _ = sample_df_with_conditions(backend=BACKEND_MODIN, numeric=True)

#     # Try sorting the Modin DataFrame using the wrong backend and expect exceptions
#     with pytest.raises(expected_exception) as exc_info:
#         sort_dataframe(df, "time", wrong_backend)

#     # Ensure that the expected substring is in the exception message
#     assert expected_substring in str(exc_info.value), f"Expected substring '{expected_substring}' in exception message but got: {str(exc_info.value)}"

# @pytest.mark.parametrize(
#     "wrong_df, backend, expected_exception, expected_substring",
#     [
#         (pd.DataFrame({"time": [1, 2, 3]}), BACKEND_MODIN, TypeError, "Expected Modin DataFrame"),  # Force the specific TypeError for Modin
#     ]
# )
# def test_sort_dataframe_modin_type_error(sample_df_with_conditions, wrong_df, backend, expected_exception, expected_substring):
#     """Test that sort_dataframe raises TypeError for non-Modin DataFrames when the backend is Modin."""
#     # Try sorting a non-Modin DataFrame using the Modin backend and expect exceptions
#     with pytest.raises(expected_exception) as exc_info:
#         sort_dataframe(wrong_df, "time", backend)

#     # Ensure that the expected substring is in the exception message
#     assert expected_substring in str(exc_info.value), f"Expected substring '{expected_substring}' in exception message but got: {str(exc_info.value)}"


# @pytest.mark.parametrize(
#     "backend, with_empty_columns, expected_result",
#     [
#         (BACKEND_PANDAS, True, True),  # Test for empty columns in Pandas
#         (BACKEND_PANDAS, False, False),  # Test for no empty columns in Pandas
#         (BACKEND_POLARS, True, True),  # Test for empty columns in Polars
#         (BACKEND_POLARS, False, False),  # Test for no empty columns in Polars
#         (BACKEND_MODIN, True, True),  # Test for empty columns in Modin
#         (BACKEND_MODIN, False, False),  # Test for no empty columns in Modin
#     ]
# )
# def test_check_empty_columns(backend, sample_df_with_conditions, with_empty_columns, expected_result):
#     """Test check_empty_columns for detecting empty columns across backends."""

#     # Case 1: Create a sample DataFrame with or without empty columns
#     if with_empty_columns:
#         data = create_sample_data(num_samples=100)
#         # Fill empty column with None (consistent length with other columns)
#         data["empty_col"] = [None] * 100
#     else:
#         data = create_sample_data(num_samples=100)

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     # Check for empty columns
#     result = check_empty_columns(df, backend)

#     # Ensure the result matches the expected outcome
#     assert result == expected_result, f"Expected {expected_result} but got {result} for backend {backend} with empty columns={with_empty_columns}"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_check_empty_columns_empty_dataframe(backend):
#     """Test that check_empty_columns raises ValueError for empty DataFrames across all backends."""

#     # Case: Empty DataFrame (no columns)
#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame()  # Create an empty DataFrame for Pandas
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame()  # Create an empty DataFrame for Polars
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame()  # Create an empty DataFrame for Modin

#     # Expect a ValueError due to the lack of columns
#     with pytest.raises(ValueError, match="The DataFrame contains no columns to check."):
#         check_empty_columns(df, backend)


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_check_no_empty_columns(backend):
#     """Test that check_empty_columns returns False when all columns have non-empty data."""

#     # Case: DataFrame with non-empty columns (no NaN/None values)
#     data = create_sample_data(num_samples=100)

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     # Check for empty columns, should return False since no columns are empty
#     result = check_empty_columns(df, backend)

#     # Assert that the function returns False (indicating no empty columns)
#     assert result is False, "Expected False when no columns are empty, but got True"


# @pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
# def test_check_empty_columns_no_empty(backend):
#     """Test that check_empty_columns returns False when no columns are empty."""

#     # Case: DataFrame with no empty columns (all columns have valid data)
#     data = create_sample_data(num_samples=100)

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     # Check for empty columns, should return False since no columns are empty
#     result = check_empty_columns(df, backend)

#     # Ensure that the function correctly returns False (indicating no empty columns)
#     assert result is False, "Expected False when no columns are empty, but got True"
