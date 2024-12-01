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

import narwhals as nw
import pytest

from temporalscope.core.core_utils import (
    MODE_MULTI_TARGET,
    MODE_SINGLE_TARGET,
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    SupportedTemporalDataFrame,
)
from temporalscope.core.exceptions import TimeColumnError, UnsupportedBackendError
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


# ========================= Tests =========================


def test_init_basic(df_basic):
    """Test basic initialization."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    assert tf.mode == MODE_SINGLE_TARGET
    assert tf.ascending is True


def test_init_invalid_backend(df_basic):
    """Test initialization with invalid backend."""
    with pytest.raises(UnsupportedBackendError):
        TimeFrame(df_basic, time_col="time", target_col="target", dataframe_backend="invalid")


# ========================= Invalid mode =========================


def test_mode_warning(df_basic):
    """Test that non-single-target modes emit a warning but are stored as metadata."""
    # Test with multi-target mode
    with pytest.warns(UserWarning):
        tf = TimeFrame(df_basic, time_col="time", target_col="target", mode=MODE_MULTI_TARGET)
        assert tf.mode == MODE_MULTI_TARGET

    # Test with custom mode
    with pytest.warns(UserWarning):
        tf = TimeFrame(df_basic, time_col="time", target_col="target", mode="custom_mode")
        assert tf.mode == "custom_mode"


# ========================= Columns =========================


def test_init_missing_columns(df_basic):
    """Test initialization with missing columns."""
    with pytest.raises(TimeColumnError):
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
def test_sorting(df_basic, ascending):
    """Test sorting functionality."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target", ascending=ascending)
    # Convert to narwhals DataFrame before using select
    nw_df = tf._to_narwhals(tf.df)
    time_values = nw_df.select(nw.col("time")).to_pandas()["time"].tolist()
    assert time_values == sorted(time_values, reverse=not ascending)


def test_update_invalid_target(df_basic):
    """Test updating with invalid target."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target")
    with pytest.raises(ValueError):
        tf.update_dataframe(df_basic, new_target_col="nonexistent")


@pytest.mark.parametrize("target_backend", ["pandas", "polars"])
def test_backend_conversion(df_basic, target_backend):
    """Test backend conversion."""
    tf = TimeFrame(df_basic, time_col="time", target_col="target", dataframe_backend=target_backend)
    assert tf.backend == target_backend
