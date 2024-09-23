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

# TemporalScope/test/unit/test_core_temporal_data_loader.py


from datetime import date, timedelta
from typing import Dict, List, Union

import modin.pandas as mpd
import numpy as np
import pandas as pd
import polars as pl
import pytest

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
)
from temporalscope.core.temporal_data_loader import TimeFrame


def create_sample_data(num_samples: int = 100, num_features: int = 3) -> Dict[str, Union[List[date], List[float]]]:
    """Create a sample data dictionary for testing.

    :param num_samples: Number of samples to generate, defaults to 100
    :type num_samples: int, optional
    :param num_features: Number of feature columns to generate, defaults to 3
    :type num_features: int, optional
    :return: A dictionary containing generated data with keys 'time', 'feature_1', ..., 'feature_n', and 'target'
    :rtype: Dict[str, Union[List[date], List[float]]]
    """
    start_date = date(2021, 1, 1)
    data = {
        "time": [start_date + timedelta(days=i) for i in range(num_samples)],
    }

    # Generate feature columns
    for i in range(1, num_features + 1):
        data[f"feature_{i}"] = np.random.rand(num_samples).tolist()

    # Generate target column (e.g., sum of features plus noise)
    data["target"] = [
        sum(data[f"feature_{j}"][i] for j in range(1, num_features + 1)) + np.random.normal(0, 0.1)
        for i in range(num_samples)
    ]

    return data


@pytest.fixture(params=[BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def sample_dataframe(request):
    """Fixture to create sample DataFrames for each backend.

    :param request: Pytest fixture request object containing the backend parameter.
    :type request: _pytest.fixtures.SubRequest
    :return: A tuple of the DataFrame and the backend identifier.
    :rtype: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    data = create_sample_data()
    backend = request.param

    if backend == BACKEND_POLARS:
        # Ensure 'time' column is properly typed
        data["time"] = pl.Series(data["time"])
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return df, backend


def test_timeframe_initialization(sample_dataframe):
    """Test the initialization of TimeFrame with various backends.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    assert tf.backend == backend
    assert tf.time_col == "time"
    assert tf.target_col == "target"
    assert len(tf.get_data()) == len(df)


def test_sort_data(sample_dataframe):
    """Test the sort_data method.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend, sort=False)
    # Shuffle the data
    if backend == BACKEND_POLARS:
        shuffled_df = tf.get_data().sample(fraction=1.0)
    else:
        shuffled_df = tf.get_data().sample(frac=1).reset_index(drop=True)
    tf.update_data(shuffled_df)
    tf.sort_data(ascending=True)
    sorted_df = tf.get_data()
    # Verify that data is sorted
    times = sorted_df[tf.time_col].to_list() if backend == BACKEND_POLARS else sorted_df[tf.time_col].tolist()
    assert times == sorted(times)


def test_update_data(sample_dataframe):
    """Test the update_data method.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    new_data = create_sample_data(num_samples=50)
    if backend == BACKEND_POLARS:
        new_data["time"] = pl.Series(new_data["time"])
        new_df = pl.DataFrame(new_data)
    elif backend == BACKEND_PANDAS:
        new_df = pd.DataFrame(new_data)
    elif backend == BACKEND_MODIN:
        new_df = mpd.DataFrame(new_data)
    tf.update_data(new_df)
    assert len(tf.get_data()) == 50


def test_update_target_col(sample_dataframe):
    """Test the update_target_col method.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    new_target = np.random.rand(len(df))
    if backend == BACKEND_POLARS:
        new_target_col = pl.Series(new_target)
    elif backend == BACKEND_PANDAS:
        new_target_col = pd.Series(new_target)
    elif backend == BACKEND_MODIN:
        new_target_col = mpd.Series(new_target)
    tf.update_target_col(new_target_col)
    updated_target = (
        tf.get_data()[tf.target_col].to_numpy() if backend == BACKEND_POLARS else tf.get_data()[tf.target_col].values
    )
    np.testing.assert_array_almost_equal(updated_target, new_target)


def test_missing_columns(sample_dataframe):
    """Test initialization with missing required columns.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    # Remove the target column
    if backend == BACKEND_POLARS:
        df = df.drop(["target"])
    else:
        df = df.drop(columns=["target"])
    with pytest.raises(ValueError) as excinfo:
        TimeFrame(df, time_col="time", target_col="target", backend=backend)
    assert "Missing required columns" in str(excinfo.value)


def test_invalid_backend(sample_dataframe):
    """Test initialization with an invalid backend.

    :param sample_dataframe: Fixture providing the DataFrame.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, _ = sample_dataframe
    invalid_backend = "invalid_backend"
    with pytest.raises(ValueError) as excinfo:
        TimeFrame(df, time_col="time", target_col="target", backend=invalid_backend)
    assert f"Unsupported backend '{invalid_backend}'" in str(excinfo.value)


def test_invalid_time_col_type(sample_dataframe):
    """Test initialization with invalid time_col type.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    with pytest.raises(ValueError) as excinfo:
        TimeFrame(df, time_col=123, target_col="target", backend=backend)
    assert "time_col must be a non-empty string." in str(excinfo.value)


def test_invalid_target_col_type(sample_dataframe):
    """Test initialization with invalid target_col type.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    with pytest.raises(ValueError) as excinfo:
        TimeFrame(df, time_col="time", target_col=None, backend=backend)
    assert "target_col must be a non-empty string." in str(excinfo.value)


def test_invalid_dataframe_type():
    """Test initialization with an invalid DataFrame type."""
    invalid_df = "This is not a DataFrame"
    with pytest.raises(TypeError):
        TimeFrame(invalid_df, time_col="time", target_col="target", backend=BACKEND_POLARS)


def test_sort_data_invalid_backend():
    """Test initialization with an unsupported backend."""
    data = create_sample_data()
    df = pd.DataFrame(data)
    with pytest.raises(ValueError) as excinfo:
        TimeFrame(df, time_col="time", target_col="target", backend="unsupported_backend")
    assert "Unsupported backend" in str(excinfo.value)


def test_update_target_col_invalid_length(sample_dataframe):
    """Test update_target_col with mismatched length."""
    df, backend = sample_dataframe
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    new_target = np.random.rand(len(df) - 1)  # Mismatch length by 1
    if backend == BACKEND_POLARS:
        new_target_col = pl.Series(new_target)
    elif backend == BACKEND_PANDAS:
        new_target_col = pd.Series(new_target)
    elif backend == BACKEND_MODIN:
        new_target_col = mpd.Series(new_target)

    with pytest.raises(ValueError) as excinfo:
        tf.update_target_col(new_target_col)

    assert "The new target column must have the same number of rows as the DataFrame." in str(excinfo.value)



def test_update_target_col_invalid_type(sample_dataframe):
    """Test update_target_col with invalid Series type.

    :param sample_dataframe: Fixture providing the DataFrame and backend.
    :type sample_dataframe: Tuple[Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], str]
    """
    df, backend = sample_dataframe
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    invalid_series = "This is not a Series"
    with pytest.raises(TypeError) as excinfo:
        tf.update_target_col(invalid_series)
    assert "Expected a" in str(excinfo.value)


@pytest.mark.parametrize(
    "df_backend,expected_backend",
    [(BACKEND_POLARS, BACKEND_POLARS), (BACKEND_PANDAS, BACKEND_PANDAS), (BACKEND_MODIN, BACKEND_MODIN)],
)
def test_infer_backend(sample_dataframe, df_backend, expected_backend):
    """Test that the backend is correctly inferred for Polars, Pandas, and Modin DataFrames."""
    df, backend = sample_dataframe
    if backend == df_backend:
        tf = TimeFrame(df, time_col="time", target_col="target")
        inferred_backend = tf._infer_backend(df)
        assert inferred_backend == expected_backend


def test_infer_backend_invalid():
    """Test that a ValueError is raised for unsupported DataFrame types."""
    invalid_df = "This is not a DataFrame"

    # Creating a valid TimeFrame object first to avoid column validation
    valid_df = pd.DataFrame({"time": [1, 2, 3], "target": [1, 2, 3]})
    tf = TimeFrame(valid_df, time_col="time", target_col="target")  # Placeholder

    # Now test the _infer_backend method directly on the invalid data
    with pytest.raises(ValueError) as excinfo:
        tf._infer_backend(invalid_df)
    assert "Unsupported DataFrame type" in str(excinfo.value)
