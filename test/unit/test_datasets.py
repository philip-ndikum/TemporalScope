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

import pytest
from temporalscope.datasets.datasets import DatasetLoader
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.core.core_utils import BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS
import pandas as pd
import modin.pandas as mpd
import polars as pl

@pytest.fixture
def dataset_loader():
    """Fixture to create a DatasetLoader instance for the macrodata dataset."""
    return DatasetLoader(dataset_name="macrodata")


def test_load_dataset_and_target(dataset_loader):
    """Test loading the dataset and its target column."""
    df, target_col = dataset_loader._load_dataset_and_target()
    assert isinstance(df, pd.DataFrame)
    assert target_col == "realgdp"
    assert "ds" in df.columns
    assert len(df) > 0  # Ensure the dataset is not empty


def test_init_timeframes_for_backends(dataset_loader):
    """Test initializing TimeFrame objects for multiple backends."""
    df, target_col = dataset_loader._load_dataset_and_target()

    timeframes = dataset_loader.init_timeframes_for_backends(df, target_col)
    
    # Check if the returned TimeFrame objects for each backend are valid
    assert isinstance(timeframes[BACKEND_PANDAS], TimeFrame)
    assert isinstance(timeframes[BACKEND_MODIN], TimeFrame)
    assert isinstance(timeframes[BACKEND_POLARS], TimeFrame)

    # Ensure correct data in each backend
    assert timeframes[BACKEND_PANDAS].dataframe_backend == BACKEND_PANDAS
    assert timeframes[BACKEND_MODIN].dataframe_backend == BACKEND_MODIN
    assert timeframes[BACKEND_POLARS].dataframe_backend == BACKEND_POLARS


def test_load_and_init_timeframes(dataset_loader):
    """Test loading dataset and initializing TimeFrames for all backends."""
    timeframes = dataset_loader.load_and_init_timeframes()

    # Check if the returned TimeFrame objects for each backend are valid
    assert isinstance(timeframes[BACKEND_PANDAS], TimeFrame)
    assert isinstance(timeframes[BACKEND_MODIN], TimeFrame)
    assert isinstance(timeframes[BACKEND_POLARS], TimeFrame)


def test_invalid_backend_raises_error(dataset_loader):
    """Test that initializing with an invalid backend raises a ValueError."""
    df, target_col = dataset_loader._load_dataset_and_target()

    with pytest.raises(ValueError, match="Unsupported backend"):
        dataset_loader.init_timeframes_for_backends(df, target_col, backends=("invalid_backend",))


def test_invalid_dataset_name():
    """Test that initializing DatasetLoader with an invalid dataset name raises a ValueError."""
    with pytest.raises(ValueError, match="Dataset 'invalid' is not supported"):
        DatasetLoader(dataset_name="invalid")


def test_init_timeframes_with_custom_backend(dataset_loader):
    """Test initializing TimeFrames with a custom selection of backends."""
    df, target_col = dataset_loader._load_dataset_and_target()
    timeframes = dataset_loader.init_timeframes_for_backends(df, target_col, backends=(BACKEND_PANDAS,))

    # Ensure only the requested backend is initialized
    assert BACKEND_PANDAS in timeframes
    assert BACKEND_MODIN not in timeframes
    assert BACKEND_POLARS not in timeframes


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
