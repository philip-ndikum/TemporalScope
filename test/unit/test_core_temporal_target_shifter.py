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

"""TemporalScope/test/unit/test_core_temporal_target_shifter.py

This file contains unit tests for the TemporalTargetShifter class to ensure it behaves correctly across different
backends (pandas, modin, polars), modes of operation (machine_learning, deep_learning), and various configurations.

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import modin.pandas as mpd
import numpy as np
import pandas as pd
import polars as pl
import pytest

from temporalscope.core.core_utils import BACKEND_MODIN, BACKEND_PANDAS, BACKEND_POLARS
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.core.temporal_target_shifter import TemporalTargetShifter


# Fixture to generate sample dataframes for different backends
@pytest.fixture(params=[BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def sample_dataframe(request):
    """Fixture to generate sample dataframes for different backends."""
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }
    backend = request.param
    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)
    return df, backend, "target"


# Parametrized Test for Backend Inference, n_lags, and Modes
@pytest.mark.parametrize(
    "n_lags, mode, sequence_length",
    [
        (1, TemporalTargetShifter.MODE_MACHINE_LEARNING, None),
        (3, TemporalTargetShifter.MODE_MACHINE_LEARNING, None),
        (1, TemporalTargetShifter.MODE_DEEP_LEARNING, 5),
    ],
)
@pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])  # Parametrizing backends as well
def test_backend_inference(backend, n_lags, mode, sequence_length):
    """Test backend inference and shifting functionality across all backends."""
    # Generate data for the current backend
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    # Initialize shifter
    shifter = TemporalTargetShifter(n_lags=n_lags, mode=mode, sequence_length=sequence_length, target_col="target")

    # Test fitting the dataframe and checking the inferred backend
    shifter.fit(df)
    assert shifter.backend == backend

    # Test transformation (ensure no crashes)
    transformed = shifter.transform(df)
    assert transformed is not None


# Parametrized test for invalid data and expected errors across backends
@pytest.mark.parametrize(
    "invalid_data",
    [
        None,  # Null input should raise an error
        pd.DataFrame(),  # Empty DataFrame should raise an error
    ],
)
@pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_invalid_data_handling(backend, invalid_data):
    """Test invalid data handling for empty or None DataFrames across backends."""
    shifter = TemporalTargetShifter(n_lags=1, target_col="target")

    with pytest.raises(ValueError):
        shifter.fit(invalid_data)


# Parametrized test for TimeFrame inputs and transformation across all backends
@pytest.mark.parametrize("n_lags", [1, 2])
@pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_time_frame_input(backend, n_lags):
    """Test TimeFrame input handling and transformation across all backends."""
    # Generate data for the current backend
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    shifter = TemporalTargetShifter(n_lags=n_lags, target_col="target")

    # Test fitting and transforming TimeFrame
    shifter.fit(tf)
    transformed = shifter.transform(tf)
    assert transformed is not None


# Parametrized test for deep learning mode with different sequence lengths across all backends
@pytest.mark.parametrize("sequence_length", [3, 5])
@pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_deep_learning_mode(backend, sequence_length):
    """Test deep learning mode sequence generation across all backends."""
    # Generate data for the current backend
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    shifter = TemporalTargetShifter(
        n_lags=1, mode=TemporalTargetShifter.MODE_DEEP_LEARNING, sequence_length=sequence_length, target_col="target"
    )

    shifter.fit(df)
    transformed = shifter.transform(df)
    assert transformed is not None


# Test verbose mode with stdout capture
@pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_verbose_mode(backend, capfd):
    """Test verbose mode output and row dropping information."""
    # Generate data for the current backend
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    shifter = TemporalTargetShifter(n_lags=1, target_col="target", verbose=True)

    shifter.fit(df)
    shifter.transform(df)

    # Capture stdout and check for printed verbose information
    captured = capfd.readouterr()
    assert "Rows before shift" in captured.out


# Parametrized test for fit_transform method for all backends
@pytest.mark.parametrize("n_lags", [1, 2])
@pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
def test_fit_transform(backend, n_lags):
    """Test fit_transform() method for all backends."""
    # Generate data for the current backend
    data = {
        "time": pd.date_range(start="2022-01-01", periods=100),
        "target": np.random.rand(100),
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }

    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data)
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    shifter = TemporalTargetShifter(n_lags=n_lags, target_col="target")

    transformed = shifter.fit_transform(df)
    assert transformed is not None
