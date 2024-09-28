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

# TemporalScope/test/unit/test_core_temporal_target_shifter.py

import modin.pandas as mpd
import numpy as np
import pandas as pd
import polars as pl
import pytest

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
    MODE_MACHINE_LEARNING,
    MODE_DEEP_LEARNING,
)
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.core.temporal_target_shifter import TemporalTargetShifter


# Fixture to generate sample dataframes for different data_formats
@pytest.fixture(params=[BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def sample_dataframe(request):
    """Fixture to generate sample dataframes for different data_formats."""
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }
    data_format = request.param
    if data_format == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif data_format == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif data_format == BACKEND_MODIN:
        df = mpd.DataFrame(data)
    return df, data_format, "target"


# Parametrized Test for data_format Inference, n_lags, and Modes
@pytest.mark.parametrize(
    "n_lags, mode, sequence_length",
    [
        (1, MODE_MACHINE_LEARNING, None),
        (3, MODE_MACHINE_LEARNING, None),
        (1, MODE_DEEP_LEARNING, 5),
    ],
)
@pytest.mark.parametrize("data_format", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])  # Parametrizing data_formats as well
def test_data_format_inference(data_format, n_lags, mode, sequence_length):
    """Test data_format inference and shifting functionality across all data_formats."""
    # Generate data for the current data_format
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if data_format == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif data_format == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif data_format == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    # Initialize shifter
    shifter = TemporalTargetShifter(n_lags=n_lags, mode=mode, sequence_length=sequence_length, target_col="target")

    # Test fitting the dataframe and checking the inferred data_format
    shifter.fit(df)
    assert shifter.data_format == data_format

    # Test transformation (ensure no crashes)
    transformed = shifter.transform(df)
    assert transformed is not None


# Parametrized test for invalid data and expected errors across data_formats
@pytest.mark.parametrize(
    "invalid_data",
    [
        None,  # Null input should raise an error
        pd.DataFrame(),  # Empty DataFrame should raise an error
    ],
)
@pytest.mark.parametrize("data_format", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_invalid_data_handling(data_format, invalid_data):
    """Test invalid data handling for empty or None DataFrames across data_formats."""
    shifter = TemporalTargetShifter(n_lags=1, target_col="target")

    with pytest.raises(ValueError):
        shifter.fit(invalid_data)


# Parametrized test for TimeFrame inputs and transformation across all data_formats
@pytest.mark.parametrize("n_lags", [1, 2])
@pytest.mark.parametrize("data_format", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_time_frame_input(data_format, n_lags):
    """Test TimeFrame input handling and transformation across all data_formats."""
    # Generate data for the current data_format
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if data_format == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif data_format == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif data_format == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    # Ensure TimeFrame uses dataframe_backend
    tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend=data_format)
    shifter = TemporalTargetShifter(n_lags=n_lags, target_col="target")

    # Test fitting and transforming TimeFrame
    shifter.fit(tf)
    transformed = shifter.transform(tf)
    assert transformed is not None


# Parametrized test for deep learning mode with different sequence lengths across all data_formats
@pytest.mark.parametrize("sequence_length", [3, 5])
@pytest.mark.parametrize("data_format", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_deep_learning_mode(data_format, sequence_length):
    """Test deep learning mode sequence generation across all data_formats."""
    # Generate data for the current data_format
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if data_format == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif data_format == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif data_format == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    shifter = TemporalTargetShifter(
        n_lags=1, mode=MODE_DEEP_LEARNING, sequence_length=sequence_length, target_col="target"
    )

    shifter.fit(df)
    transformed = shifter.transform(df)
    assert transformed is not None


# Test verbose mode with stdout capture
@pytest.mark.parametrize("data_format", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_verbose_mode(data_format, capfd):
    """Test verbose mode output and row dropping information."""
    # Generate data for the current data_format
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if data_format == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif data_format == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif data_format == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    shifter = TemporalTargetShifter(n_lags=1, target_col="target", verbose=True)

    shifter.fit(df)
    shifter.transform(df)

    # Capture stdout and check for printed verbose information
    captured = capfd.readouterr()
    assert "Rows before shift" in captured.out


# Parametrized test for fit_transform method for all data_formats
@pytest.mark.parametrize("n_lags", [1, 2])
@pytest.mark.parametrize("data_format", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_fit_transform(data_format, n_lags):
    """Test fit_transform() method for all data_formats."""
    # Generate data for the current data_format
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if data_format == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif data_format == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif data_format == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    shifter = TemporalTargetShifter(n_lags=n_lags, target_col="target")

    transformed = shifter.fit_transform(df)
    assert transformed is not None
