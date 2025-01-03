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

"""Unit Test Design for TemporalScope's Padding Functions.

This module implements systematic testing of the padding functions in functional.py
across multiple DataFrame backends while maintaining consistency and reliability.

Testing Philosophy
-----------------
1. Backend-Agnostic Operations:
   - All DataFrame manipulations use the Narwhals API (@nw.narwhalify)
   - Operations are written once and work across all supported backends
   - Backend-specific code is avoided to maintain test uniformity

2. Fine-Grained Data Generation:
   - PyTest fixtures provide flexible, parameterized test data
   - Base configuration fixture allows easy overrides
   - Each test case specifies exact data characteristics needed

3. Consistent Validation Pattern:
   - All validation steps convert to Pandas for reliable comparisons
   - Complex validations use reusable helper functions
   - Assertions focus on business logic rather than implementation
"""

from typing import Any, Callable, Dict

import narwhals as nw
import pytest
from narwhals.typing import FrameT

from temporalscope.core.core_utils import (
    get_narwhals_backends as get_temporalscope_backends,
)
from temporalscope.core.core_utils import (
    is_dataframe_empty as is_lazy_evaluation,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series
from temporalscope.partition.single_target.padding.functional import mean_fill_pad


@pytest.fixture(params=get_temporalscope_backends())
def backend(request) -> str:
    """Fixture providing all supported backends for testing."""
    return request.param


@pytest.fixture
def data_config(backend: str) -> Callable[..., Dict[str, Any]]:
    """Base fixture for data generation configuration."""

    def _config(**kwargs) -> Dict[str, Any]:
        default_config = {
            "num_samples": 3,
            "num_features": 2,
            "with_nulls": False,
            "with_nans": False,
            "backend": backend,
            "drop_time": True,  # Use our new parameter to ensure only numerical data
            "random_seed": 42,  # Add for reproducibility
        }
        default_config.update(kwargs)
        return default_config

    return _config


@pytest.fixture
def sample_df(data_config: Callable[..., Dict[str, Any]]) -> FrameT:
    """Generate sample DataFrame for testing."""
    config = data_config()
    return generate_synthetic_time_series(**config)


@nw.narwhalify
def assert_padding_length(df: FrameT, target_len: int) -> None:
    """Verify padding length using Narwhals operations."""
    count_result = df.select([nw.col(df.columns[0]).count().cast(nw.Int64).alias("count")])

    # Handle both lazy and eager evaluation
    if is_lazy_evaluation(count_result):
        count_val = count_result.collect()["count"][0]
    else:
        count_val = count_result.to_pandas()["count"][0]

    # Handle PyArrow scalar if needed
    if hasattr(count_val, "as_py"):
        count_val = count_val.as_py()

    assert count_val == target_len


@nw.narwhalify
def assert_mean_values(df: FrameT, original_df: FrameT) -> None:
    """Verify mean values in padding rows match original column means."""
    for col in df.columns:
        mean_expr = nw.col(col).mean().cast(nw.Float64).alias("mean")

        # Get original mean, handling both lazy and eager evaluation
        original_result = original_df.select([mean_expr])
        if is_lazy_evaluation(original_result):
            original_mean = original_result.collect().to_pandas()["mean"].iloc[0]
        else:
            original_mean = original_result.to_pandas()["mean"].iloc[0]

        # Get padded mean, handling both lazy and eager evaluation
        padded_result = df.select([mean_expr])
        if is_lazy_evaluation(padded_result):
            padded_mean = padded_result.collect().to_pandas()["mean"].iloc[0]
        else:
            padded_mean = padded_result.to_pandas()["mean"].iloc[0]

        assert abs(original_mean - padded_mean) < 1e-6, f"Mean mismatch for column {col}"


def test_mean_fill_pad_basic(sample_df: FrameT) -> None:
    """Test basic mean-fill padding functionality."""
    df = nw.from_native(sample_df)
    target_len = 5

    # Test post-padding
    result = mean_fill_pad(df, target_len=target_len, padding="post")
    assert_padding_length(result, target_len)
    assert_mean_values(result, df)

    # Test pre-padding
    result = mean_fill_pad(df, target_len=target_len, padding="pre")
    assert_padding_length(result, target_len)
    assert_mean_values(result, df)


def test_mean_fill_pad_null_data(data_config: Callable[..., Dict[str, Any]]) -> None:
    """Test padding with null values in data."""
    config = data_config(with_nulls=True)
    df = generate_synthetic_time_series(**config)

    # Check if nulls are present before attempting padding
    with pytest.raises(ValueError, match="Cannot process data containing null values"):
        mean_fill_pad(df, target_len=5)


def test_mean_fill_pad_nan_data(data_config: Callable[..., Dict[str, Any]]) -> None:
    """Test padding with NaN values in data."""
    config = data_config(with_nans=True)
    df = generate_synthetic_time_series(**config)

    # Check if NaNs are handled properly (they are caught as nulls)
    with pytest.raises(ValueError, match="Cannot process data containing null values"):
        mean_fill_pad(df, target_len=5)


def test_mean_fill_pad_invalid_target_len(sample_df: FrameT) -> None:
    """Test invalid target length handling."""
    df = nw.from_native(sample_df)

    with pytest.raises(ValueError, match="target_len .* must be greater than current length"):
        mean_fill_pad(df, target_len=2)


def test_mean_fill_pad_invalid_padding(sample_df: FrameT) -> None:
    """Test invalid padding direction handling."""
    df = nw.from_native(sample_df)

    with pytest.raises(ValueError, match="padding must be 'pre' or 'post'"):
        mean_fill_pad(df, target_len=5, padding="invalid")
