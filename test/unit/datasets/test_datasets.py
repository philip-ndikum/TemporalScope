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

import pandas as pd
import pytest

from temporalscope.core.core_utils import BACKEND_MODIN, BACKEND_PANDAS, BACKEND_POLARS
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.datasets.datasets import DatasetLoader


@pytest.fixture
def dataset_loader():
    """Fixture to create a DatasetLoader instance for the macrodata dataset."""
    return DatasetLoader(dataset_name="macrodata")


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_init_timeframes_for_backends_parametrized(dataset_loader, backend):
    """Test initializing TimeFrame objects for different backends."""
    df, target_col = dataset_loader._load_dataset_and_target()

    timeframes = dataset_loader.init_timeframes_for_backends(df, target_col, backends=(backend,))

    assert isinstance(timeframes[backend], TimeFrame)

    # Check that the backend is correct
    assert timeframes[backend].dataframe_backend == backend


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_load_and_init_timeframes_parametrized(dataset_loader, backend):
    """Test loading dataset and initializing TimeFrames for each backend."""
    timeframes = dataset_loader.load_and_init_timeframes(backends=(backend,))

    # Check if the returned TimeFrame object is valid for the backend
    assert isinstance(timeframes[backend], TimeFrame)
    assert timeframes[backend].dataframe_backend == backend


def test_invalid_backend_raises_error(dataset_loader):
    """Test that initializing with an invalid backend raises a ValueError."""
    df, target_col = dataset_loader._load_dataset_and_target()

    with pytest.raises(ValueError, match="Unsupported backend"):
        dataset_loader.init_timeframes_for_backends(df, target_col, backends=("invalid_backend",))


def test_invalid_dataset_name():
    """Test that initializing DatasetLoader with an invalid dataset name raises a ValueError."""
    with pytest.raises(ValueError, match="Dataset 'invalid' is not supported"):
        DatasetLoader(dataset_name="invalid")


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_init_timeframes_with_custom_backend(dataset_loader, backend):
    """Test initializing TimeFrames with a custom backend selection."""
    df, target_col = dataset_loader._load_dataset_and_target()
    timeframes = dataset_loader.init_timeframes_for_backends(df, target_col, backends=(backend,))

    # Ensure only the requested backend is initialized
    assert backend in timeframes
    assert isinstance(timeframes[backend], TimeFrame)


def test_load_dataset_internal_call(mocker):
    """Test the internal call to _load_dataset_and_target and check the dataset loader function."""
    mocker.patch("temporalscope.datasets.datasets._load_macrodata", return_value=(pd.DataFrame(), "realgdp"))
    dataset_loader = DatasetLoader(dataset_name="macrodata")

    df, target_col = dataset_loader._load_dataset_and_target()

    assert target_col == "realgdp"
    assert isinstance(df, pd.DataFrame)


def test_load_dataset_and_verify_time_column(dataset_loader):
    """Test to ensure that the 'ds' column is created and of type datetime."""
    df, target_col = dataset_loader._load_dataset_and_target()

    # Ensure 'ds' column exists and is of datetime type
    assert "ds" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])


@pytest.mark.parametrize(
    "backends",
    [(BACKEND_PANDAS,), (BACKEND_MODIN,), (BACKEND_POLARS,), (BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS)],
)
def test_load_and_init_timeframes_return(dataset_loader, backends):
    """Test that the returned timeframes object is a dictionary and contains the expected backends."""
    timeframes = dataset_loader.load_and_init_timeframes(backends=backends)

    # Ensure the return value is a dictionary
    assert isinstance(timeframes, dict)

    # Check that the returned dictionary contains the expected backends
    for backend in backends:
        assert backend in timeframes
        assert isinstance(timeframes[backend], TimeFrame)
