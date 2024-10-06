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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations under the License.

# TemporalScope/test/unit/test_core_temporal_data_loader.py

import pytest
from typing import Dict, Union, Optional, List
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import polars as pl
import modin.pandas as mpd

from temporalscope.core.temporal_data_loader import TimeFrame

from temporalscope.core.exceptions import (
    TimeColumnError,
    MixedTypesWarning,
    MixedFrequencyWarning,
    UnsupportedBackendError,
)

BACKEND_POLARS = "pl"
BACKEND_PANDAS = "pd"
BACKEND_MODIN = "mpd"


# Utility to create sample data for various edge cases
# def create_sample_data(
#     num_samples: int = 100, num_features: int = 3, missing_values: bool = False,
#     mixed_frequencies: bool = False, non_numeric_time: bool = False, mixed_timezones: bool = False
# ) -> Dict[str, Union[List[datetime], List[float]]]:
#     """Create sample data to test edge cases with numeric features and a time column."""

#     start_date = datetime(2021, 1, 1)
#     if non_numeric_time:
#         data = {"time": ["non_numeric" for _ in range(num_samples)]}
#     elif mixed_timezones:
#         data = {"time": [(start_date + timedelta(days=i)).replace(tzinfo=timezone.utc if i % 2 == 0 else None)
#                          for i in range(num_samples)]}
#     else:
#         data = {"time": [start_date + timedelta(days=i) for i in range(num_samples)]}

#     for i in range(1, num_features + 1):
#         data[f"feature_{i}"] = np.random.rand(num_samples).tolist()

#     if mixed_frequencies:
#         data["time"] = pd.date_range(start='2021-01-01', periods=num_samples // 2, freq='D').tolist()
#         data["time"] += pd.date_range(start='2021-02-01', periods=num_samples // 2, freq='M').tolist()

#     if missing_values:
#         for i in range(num_samples):
#             if i % 10 == 0:
#                 data[f"feature_1"][i] = None

#     data["target"] = np.random.rand(num_samples).tolist()
#     return data


# @pytest.mark.parametrize(
#     "backend, case_type, expected_error, match_message",
#     [
#         (BACKEND_POLARS, "non_numeric_time_col", TimeColumnError, r"`time_col` must be numeric or timestamp-like"),
#         (BACKEND_PANDAS, "non_numeric_time_col", TimeColumnError, r"`time_col` must be numeric or timestamp-like"),
#         (BACKEND_MODIN, "non_numeric_time_col", TimeColumnError, r"`time_col` must be numeric or timestamp-like"),
#         (BACKEND_PANDAS, "mixed_frequencies", None, r"Mixed timestamp frequencies detected in the time column."),  # Update match message
#     ]
# )
# def test_validation_edge_cases(backend, case_type, expected_error, match_message):
#     """Test validation logic for edge cases like non-numeric time columns and mixed frequencies."""

#     if case_type == "non_numeric_time_col":
#         data = create_sample_data(non_numeric_time=True)
#     elif case_type == "mixed_frequencies":
#         data = create_sample_data(mixed_frequencies=True)

#     if backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     if expected_error:
#         with pytest.raises(expected_error, match=match_message):
#             TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)
#     else:
#         if case_type == "mixed_frequencies":
#             with pytest.warns(MixedFrequencyWarning, match=match_message):  # Expect MixedFrequencyWarning
#                 TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)


# @pytest.mark.parametrize(
#     "backend, time_col, target_col, expected_error, match_message, infer_backend",
#     [
#         # Valid cases with explicit backend
#         (BACKEND_PANDAS, "time", "target", None, None, False),
#         (BACKEND_POLARS, "time", "target", None, None, False),
#         (BACKEND_MODIN, "time", "target", None, None, False),

#         # Valid cases with inferred backend
#         (BACKEND_PANDAS, "time", "target", None, None, True),
#         (BACKEND_POLARS, "time", "target", None, None, True),
#         (BACKEND_MODIN, "time", "target", None, None, True),

#         # Invalid `time_col` cases
#         (BACKEND_PANDAS, "", "target", ValueError, "`time_col` must be a non-empty string.", False),
#         (BACKEND_POLARS, "", "target", ValueError, "`time_col` must be a non-empty string.", False),
#         (BACKEND_MODIN, "", "target", ValueError, "`time_col` must be a non-empty string.", False),

#         # Invalid `target_col` cases
#         (BACKEND_PANDAS, "time", "", ValueError, "`target_col` must be a non-empty string.", False),
#         (BACKEND_POLARS, "time", "", ValueError, "`target_col` must be a non-empty string.", False),
#         (BACKEND_MODIN, "time", "", ValueError, "`target_col` must be a non-empty string.", False),

#         # Invalid backend cases
#         ("invalid_backend", "time", "target", UnsupportedBackendError, "Unsupported backend", False),
#     ]
# )
# def test_timeframe_init_and_get_data(backend, time_col, target_col, expected_error, match_message, infer_backend):
#     """Test initialization of TimeFrame class and `get_data` method across backends, including invalid cases."""

#     data = create_sample_data()

#     if backend in [BACKEND_PANDAS, BACKEND_POLARS, BACKEND_MODIN]:
#         if backend == BACKEND_PANDAS:
#             df = pd.DataFrame(data)
#         elif backend == BACKEND_POLARS:
#             df = pl.DataFrame(data)
#         elif backend == BACKEND_MODIN:
#             df = mpd.DataFrame(data)
#     else:
#         # Use a dummy Pandas DataFrame for unsupported backend
#         df = pd.DataFrame(data)

#     if expected_error:
#         with pytest.raises(expected_error, match=match_message):
#             if infer_backend:
#                 # Don't pass the backend to trigger inference
#                 TimeFrame(df, time_col=time_col, target_col=target_col)
#             else:
#                 # Pass backend explicitly (invalid backend case covered here)
#                 TimeFrame(df, time_col=time_col, target_col=target_col, dataframe_backend=backend)
#     else:
#         # Initialize the TimeFrame
#         if infer_backend:
#             tf = TimeFrame(df, time_col=time_col, target_col=target_col)
#         else:
#             tf = TimeFrame(df, time_col=time_col, target_col=target_col, dataframe_backend=backend)

#         # Ensure `get_data` returns the correct DataFrame
#         result_df = tf.get_data()
#         assert result_df.shape[0] == 100  # Ensure the DataFrame has the expected number of rows
#         assert tf.time_col == time_col  # Time column should match the expected value
#         assert tf.target_col == target_col  # Target column should match the expected value

#         if infer_backend:
#             # Check that the backend was correctly inferred
#             assert tf.dataframe_backend == backend, f"Expected inferred backend {backend}, but got {tf.dataframe_backend}"

# @pytest.mark.parametrize(
#     "backend, time_col, target_col, expected_error, match_message",
#     [
#         # Missing `time_col` in DataFrame (should raise ValueError)
#         (BACKEND_PANDAS, "invalid_time", "target", ValueError, "`time_col` 'invalid_time' not found"),
#         (BACKEND_POLARS, "invalid_time", "target", ValueError, "`time_col` 'invalid_time' not found"),
#         (BACKEND_MODIN, "invalid_time", "target", ValueError, "`time_col` 'invalid_time' not found"),

#         # Missing `target_col` in DataFrame (should raise ValueError)
#         (BACKEND_PANDAS, "time", "invalid_target", ValueError, "`target_col` 'invalid_target' not found"),
#         (BACKEND_POLARS, "time", "invalid_target", ValueError, "`target_col` 'invalid_target' not found"),
#         (BACKEND_MODIN, "time", "invalid_target", ValueError, "`target_col` 'invalid_target' not found"),
#     ]
# )
# def test_timeframe_missing_columns(backend, time_col, target_col, expected_error, match_message):
#     """Test that missing columns raise ValueError with the correct message."""
#     data = create_sample_data()

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     with pytest.raises(expected_error, match=match_message):
#         TimeFrame(df, time_col=time_col, target_col=target_col, dataframe_backend=backend)


# @pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_POLARS, BACKEND_MODIN])
# def test_sort_data(backend):
#     """Test sorting of the DataFrame by time column using `sort_data` method."""

#     data = create_sample_data(num_samples=10)

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)

#     # Sort the DataFrame in descending order
#     tf.sort_data(ascending=False)

#     sorted_df = tf.get_data()

#     # For Polars use .to_numpy() or .row() to access the rows
#     if backend == BACKEND_POLARS:
#         time_col_np = sorted_df["time"].to_numpy()
#         assert time_col_np[0] > time_col_np[-1]
#     else:
#         assert sorted_df["time"].iloc[0] > sorted_df["time"].iloc[-1]



# @pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_POLARS, BACKEND_MODIN])
# def test_update_target_col(backend):
#     """Test `update_target_col` method across backends by updating the target column."""

#     data = create_sample_data()

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)

#     # New target column
#     new_target_col = np.random.rand(100)

#     if backend == BACKEND_PANDAS:
#         new_target_series = pd.Series(new_target_col, name="target")
#     elif backend == BACKEND_POLARS:
#         new_target_series = pl.Series("target", new_target_col)
#     elif backend == BACKEND_MODIN:
#         new_target_series = mpd.Series(new_target_col, name="target")

#     # Update the target column
#     tf.update_target_col(new_target_series)

#     # Ensure the target column has been updated correctly
#     updated_df = tf.get_data()
#     assert np.allclose(updated_df["target"], new_target_col), "Target column update failed."


# @pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_POLARS, BACKEND_MODIN])
# def test_validate_and_update_data(backend):
#     """Test `update_data` method by updating the entire DataFrame."""

#     data = create_sample_data(num_samples=100)

#     if backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)

#     # Create a new DataFrame with 50 samples
#     new_data = create_sample_data(num_samples=50)

#     if backend == BACKEND_PANDAS:
#         new_df = pd.DataFrame(new_data)
#     elif backend == BACKEND_POLARS:
#         new_df = pl.DataFrame(new_data)
#     elif backend == BACKEND_MODIN:
#         new_df = mpd.DataFrame(new_data)

#     # Update the DataFrame in the TimeFrame instance
#     tf.update_data(new_df)

#     # Ensure the new data has been updated correctly
#     updated_df = tf.get_data()
#     assert updated_df.shape[0] == 50, "DataFrame update failed. Expected 50 rows."
