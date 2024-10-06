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

# TemporalScope/test/unit/datasets/test_synthetic_data_generator.py

import numpy as np
import pandas as pd
import polars as pl
import pytest

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
    MODE_MULTI_STEP,
    MODE_SINGLE_STEP,
)
from temporalscope.datasets.synthetic_data_generator import create_sample_data


# Skip unsupported backends for multi-step mode and Pandas-to-Polars conversion
@pytest.mark.parametrize(
    "num_samples, num_features, mode",
    [
        (100, 3, MODE_SINGLE_STEP),  # Single-step mode
        pytest.param(
            100, 3, MODE_MULTI_STEP, marks=pytest.mark.xfail(reason="Unsupported multi-step mode for Modin and Polars")
        ),
        (0, 0, MODE_SINGLE_STEP),  # Zero samples and features
        (1000, 10, MODE_SINGLE_STEP),  # Large data
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        BACKEND_PANDAS,
        BACKEND_MODIN,
        pytest.param(BACKEND_POLARS, marks=pytest.mark.xfail(reason="Pandas to Polars conversion not supported")),
    ],
)
def test_create_sample_data_basic(num_samples, num_features, mode, backend):
    """Test that data generation works for both single-step and multi-step modes."""
    # Generate synthetic data
    df = create_sample_data(backend=backend, num_samples=num_samples, num_features=num_features, mode=mode)

    # Check if DataFrame is empty before accessing data
    if num_samples == 0:
        if backend == BACKEND_POLARS:
            assert df.is_empty(), "DataFrame should be empty when num_samples is 0 for Polars."
        else:
            assert df.empty, "DataFrame should be empty when num_samples is 0."
    else:
        assert len(df) == num_samples, f"Mismatch in expected number of samples: {num_samples}"

        # Check if target is scalar for single-step mode
        if mode == MODE_SINGLE_STEP:
            if backend == BACKEND_POLARS:
                assert isinstance(df["target"][0], float), "Single-step mode should generate scalar target values."
            else:
                assert np.isscalar(df["target"].iloc[0]), "Single-step mode should generate scalar target values."

        # Check if target is vector for multi-step mode
        if mode == MODE_MULTI_STEP:
            assert isinstance(
                df["target"][0], (list, np.ndarray)
            ), "Multi-step mode should generate vectorized target values."


@pytest.mark.parametrize(
    "timestamp_like, numeric, mixed_frequencies, mixed_timezones",
    [
        (True, False, False, False),  # Timestamp-like time column
        (False, True, False, False),  # Numeric time column
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        BACKEND_PANDAS,
        BACKEND_MODIN,
        pytest.param(BACKEND_POLARS, marks=pytest.mark.xfail(reason="Pandas to Polars conversion not supported")),
    ],
)
def test_time_column_generation(timestamp_like, numeric, mixed_frequencies, mixed_timezones, backend):
    """Test that time columns are generated with the correct type and properties."""
    num_samples, num_features = 100, 3
    df = create_sample_data(
        backend=backend,
        num_samples=num_samples,
        num_features=num_features,
        timestamp_like=timestamp_like,
        numeric=numeric,
        mixed_frequencies=mixed_frequencies,
        mixed_timezones=mixed_timezones,
    )

    # Validate the type of the time column based on configuration
    if timestamp_like:
        if backend == BACKEND_POLARS:
            assert isinstance(df["time"][0], pl.datatypes.Datetime), "Expected a timestamp-like time column"
        else:
            assert isinstance(df["time"].iloc[0], pd.Timestamp), "Expected a timestamp-like time column"

    if numeric:
        if backend == BACKEND_POLARS:
            assert isinstance(df["time"][0], float), "Expected a numeric time column"
        else:
            assert isinstance(df["time"].iloc[0], np.float64), "Expected a numeric time column"
