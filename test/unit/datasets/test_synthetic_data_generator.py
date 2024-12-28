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

"""TemporalScope/test/unit/datasets/test_synthetic_data_generator.py

This module contains unit tests for the synthetic data generator, ensuring it works
correctly with Narwhals and different DataFrame backends.
"""

from datetime import datetime
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from narwhals.typing import FrameT
from narwhals.utils import Implementation

from temporalscope.core.core_utils import MODE_SINGLE_TARGET
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = [impl.name.lower() for impl in Implementation]
INVALID_BACKEND = "unsupported_backend"


@nw.narwhalify(eager_only=True)
def get_shape(df: FrameT) -> tuple[int, int]:
    """Get shape of DataFrame using Narwhals operations."""
    # For empty DataFrames, use len() directly
    if len(df.columns) == 0:
        return (0, 0)

    # For non-empty DataFrames, use len() for row count
    row_count = len(df.select([nw.col(df.columns[0])]))
    col_count = len(df.columns)

    return (row_count, col_count)


@nw.narwhalify(eager_only=True)
def get_row_value(df: FrameT, col_name: str, row_idx: int = 0) -> Any:
    """Get a value from a DataFrame using Narwhals operations."""
    # Get the first value from the specified column at the specified row
    value = df.select([nw.col(col_name).alias("value")]).item(row=row_idx, column=0)

    return value


@nw.narwhalify(eager_only=True)
def get_column_names(df: FrameT) -> set[str]:
    """Get column names from DataFrame using Narwhals operations."""
    return set(df.columns)


# ========================= Basic Data Generation Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("num_samples, num_features", [(100, 3), (0, 0), (1000, 10)])
@pytest.mark.parametrize("with_nulls, with_nans", [(True, False), (False, True), (True, True)])
def test_generate_synthetic_time_series_basic(
    backend: str, num_samples: int, num_features: int, with_nulls: bool, with_nans: bool
) -> None:
    """Test basic functionality of generate_synthetic_time_series across backends."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=num_samples,
        num_features=num_features,
        with_nulls=with_nulls,
        with_nans=with_nans,
        mode=MODE_SINGLE_TARGET,
    )

    # Get shape
    shape = get_shape(df)

    # Validate number of samples and features
    if num_samples == 0:
        assert shape[0] == 0, f"Expected empty DataFrame, got {shape[0]} rows"
    else:
        assert shape[0] == num_samples, f"Expected {num_samples} rows, got {shape[0]}"
        feature_columns = [col for col in df.columns if "feature_" in col]
        assert len(feature_columns) == num_features, f"Expected {num_features} feature columns"

    # Test target column type for single-step mode
    if num_samples > 0:
        target_val = get_row_value(df, "target")
        assert isinstance(target_val, (float, np.floating)), "Expected numeric target value"


# ========================= Time Column Generation Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("time_col_numeric", [True, False])
def test_time_column_generation(backend: str, time_col_numeric: bool) -> None:
    """Test time column generation as numeric or timestamp."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=100,
        num_features=3,
        time_col_numeric=time_col_numeric,
        mode=MODE_SINGLE_TARGET,
    )

    # Get first time value
    time_val = get_row_value(df, "time")

    # Check time column type
    if time_col_numeric:
        assert isinstance(time_val, (float, np.floating)), "Expected numeric time column"
    else:
        assert isinstance(time_val, (pd.Timestamp, datetime)), "Expected datetime time column"


# ========================= Error Handling Tests =========================


def test_invalid_backend() -> None:
    """Test error handling for unsupported backends."""
    with pytest.raises(ValueError, match="is not supported by Narwhals"):
        generate_synthetic_time_series(backend=INVALID_BACKEND)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_unsupported_mode(backend: str) -> None:
    """Test that an unsupported mode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported mode: multi_target"):
        generate_synthetic_time_series(backend=backend, mode="multi_target")


# ========================= Narwhals-Specific Tests =========================


def test_narwhals_backend_conversion() -> None:
    """Test that generated data can be converted between backends."""
    # Test pandas backend (default path)
    df_pandas = generate_synthetic_time_series(backend="pandas")
    assert isinstance(df_pandas, pd.DataFrame)

    # Test non-pandas backend (covers line 149)
    df_polars = generate_synthetic_time_series(backend="polars")
    df_polars_native = nw.from_native(df_polars).to_native()  # Convert back to native polars
    assert "polars" in str(type(df_polars_native)).lower()

    # Verify data consistency across backends
    assert set(df_polars.columns) == set(df_pandas.columns)
    assert len(df_polars.columns) == len(df_pandas.columns)


def test_narwhals_lazy_evaluation() -> None:
    """Test lazy evaluation handling."""
    # Generate data with dask backend (which is inherently lazy)
    df = generate_synthetic_time_series(backend="dask")

    # Verify we can chain operations without immediate execution
    result = nw.from_native(df).select([nw.col("target").mean().alias("mean"), nw.col("target").sum().alias("sum")])

    # The result should be a DataFrame but not yet computed
    assert "mean" in result.columns
    assert "sum" in result.columns


# ========================= Additional Edge Case Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_empty_data(backend: str) -> None:
    """Test generating empty datasets."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0)
    shape = get_shape(df)
    assert shape[0] == 0, f"Expected empty DataFrame, got {shape[0]} rows"


@nw.narwhalify(eager_only=True)
def check_nulls_nans(df: FrameT, col: str) -> int:
    """Check for nulls/nans in a column using Narwhals operations."""
    return df.select([nw.col(col).is_null().sum().cast(nw.Int64()).alias("null_count")]).item()


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_with_nulls_and_nans(backend: str) -> None:
    """Test null and NaN handling."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=100, num_features=5, with_nulls=True, with_nans=True
    )

    # Check for nulls/nans in feature columns
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    for col in feature_cols:
        null_count = check_nulls_nans(df, col)
        assert null_count > 0, f"Expected nulls/nans in {col}"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("drop_time", [True, False])
def test_generate_synthetic_time_series_drop_time(backend: str, drop_time: bool) -> None:
    """Test time column inclusion/exclusion."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=3,
        num_features=2,
        drop_time=drop_time,
        time_col_numeric=True,
    )

    # Check time column presence
    if drop_time:
        assert "time" not in df.columns
    else:
        assert "time" in df.columns
        time_val = get_row_value(df, "time")
        assert isinstance(time_val, (float, np.floating))


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_percentage_validation(backend: str) -> None:
    """Test null/nan percentage validation."""
    with pytest.raises(ValueError, match="null_percentage must be between 0.0 and 1.0"):
        generate_synthetic_time_series(backend=backend, null_percentage=1.5)

    with pytest.raises(ValueError, match="nan_percentage must be between 0.0 and 1.0"):
        generate_synthetic_time_series(backend=backend, nan_percentage=-0.1)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_single_row(backend: str) -> None:
    """Test single row dataset handling."""
    # Test with nulls (should take precedence)
    df = generate_synthetic_time_series(backend=backend, num_samples=1, num_features=2, with_nulls=True, with_nans=True)
    feature_val = get_row_value(df, "feature_1")
    assert pd.isna(feature_val), "Expected null/nan value"

    # Test with nans only (covers line 51)
    df_nans = generate_synthetic_time_series(backend=backend, num_samples=1, num_features=2, with_nulls=False, with_nans=True)
    feature_val_nan = get_row_value(df_nans, "feature_1")
    assert pd.isna(feature_val_nan), "Expected nan value"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("num_samples, num_features", [(-1, 3), (100, -2)])
def test_generate_synthetic_time_series_negative_values(backend: str, num_samples: int, num_features: int) -> None:
    """Test that negative values for samples or features raise ValueError."""
    with pytest.raises(ValueError, match="`num_samples` and `num_features` must be non-negative"):
        generate_synthetic_time_series(
            backend=backend,
            num_samples=num_samples,
            num_features=num_features
        )
