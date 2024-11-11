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

import pytest
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.core.exceptions import TimeColumnError, UnsupportedBackendError
from temporalscope.core.core_utils import get_temporalscope_backends
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series


@pytest.fixture(params=get_temporalscope_backends())
def backend(request):
    """Fixture to dynamically iterate through all supported backends."""
    return request.param


@pytest.fixture
def sample_data(backend):
    """Fixture to generate synthetic time series data for a specific backend."""
    return generate_synthetic_time_series(backend=backend, num_samples=100, num_features=3)


def test_timeframe_initialization(backend, sample_data):
    """Test TimeFrame initialization for each backend."""
    tf = TimeFrame(sample_data, time_col="time", target_col="target", dataframe_backend=backend)

    # Assert correct initialization
    assert tf.df.shape[0] == 100, "Row count mismatch in the DataFrame."
    assert tf._time_col == "time", "Time column mismatch."
    assert tf._target_col == "target", "Target column mismatch."
    assert tf.backend == backend, f"Backend mismatch for {backend}."

    # Check backend inference
    tf_inferred = TimeFrame(sample_data, time_col="time", target_col="target")
    assert tf_inferred.backend == backend, f"Failed to infer backend for {backend}."


def test_missing_columns(backend, sample_data):
    """Test error handling for missing columns."""
    # Missing time_col
    with pytest.raises(ValueError, match=r"`time_col` 'invalid_time' not found"):
        TimeFrame(sample_data, time_col="invalid_time", target_col="target", dataframe_backend=backend)

    # Missing target_col
    with pytest.raises(ValueError, match=r"`target_col` 'invalid_target' not found"):
        TimeFrame(sample_data, time_col="time", target_col="invalid_target", dataframe_backend=backend)


def test_sort_data(backend, sample_data):
    """Test DataFrame sorting by time column for each backend."""
    tf = TimeFrame(sample_data, time_col="time", target_col="target", dataframe_backend=backend)

    # Sort descending
    sorted_df = tf.sort_data(tf.df, ascending=False)

    # Verify sorted order
    time_col_values = sorted_df["time"].to_numpy() if backend == "polars" else sorted_df["time"].values
    assert time_col_values[0] > time_col_values[-1], f"Sorting failed for backend {backend}."


def test_update_data(backend, sample_data):
    """Test updating TimeFrame data for each backend."""
    tf = TimeFrame(sample_data, time_col="time", target_col="target", dataframe_backend=backend)

    # New synthetic data
    new_data = generate_synthetic_time_series(backend=backend, num_samples=50, num_features=3)

    # Update and validate
    tf.update_data(new_data)
    assert tf.df.shape[0] == 50, f"Data update failed for backend {backend}."


def test_validation_edge_cases(backend):
    """Test edge cases like non-numeric time columns or missing data."""
    # Invalid time column type
    data = generate_synthetic_time_series(backend=backend, num_samples=100, num_features=3)
    data["time"] = ["not_a_date"] * 100  # Invalid time column

    with pytest.raises(TimeColumnError, match=r"`time_col` must be numeric or timestamp-like"):
        TimeFrame(data, time_col="time", target_col="target", dataframe_backend=backend)

    # Missing values in critical columns
    data.loc[0, "time"] = None  # Missing value in time column
    with pytest.raises(ValueError, match=r"Missing values detected in `time` or `target`."):
        TimeFrame(data, time_col="time", target_col="target", dataframe_backend=backend)
