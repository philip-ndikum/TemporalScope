# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from datetime import datetime
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from temporalscope.core.core_utils import MODE_SINGLE_STEP, TEMPORALSCOPE_CORE_BACKEND_TYPES
from temporalscope.core.exceptions import UnsupportedBackendError
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = [backend for backend in TEMPORALSCOPE_CORE_BACKEND_TYPES.keys() if backend != "dask"]
INVALID_BACKEND = "unsupported_backend"

# Type variables for better type handling
T = TypeVar("T")


def is_dask_df(obj: Any) -> bool:
    """Check if object is a Dask DataFrame."""
    return type(obj).__module__.startswith("dask.")


def get_shape(df: Any) -> tuple[int, int]:
    """Get shape of DataFrame, handling different backends including Dask."""
    if is_dask_df(df):
        # Compute the shape components separately for Dask
        nrows = int(df.shape[0].compute())  # type: ignore
        ncols = df.shape[1]  # This is already an int
        return (nrows, ncols)
    return df.shape


def get_row_value(df: Any, col_name: str, row_idx: int = 0) -> Any:
    """Safely get a value from a DataFrame, handling different backends."""
    if is_dask_df(df):
        df = df.compute()  # type: ignore

    if isinstance(df, pl.DataFrame):
        return df.select(pl.col(col_name)).row(row_idx)[0]
    elif isinstance(df, pa.Table):
        return df.column(col_name)[row_idx].as_py()  # type: ignore
    else:
        return df.iloc[row_idx][col_name]


# ========================= Basic Data Generation Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("num_samples, num_features", [(100, 3), (0, 0), (1000, 10)])
@pytest.mark.parametrize("with_nulls, with_nans", [(True, False), (False, True), (True, True)])
def test_generate_synthetic_time_series_basic(
    backend: str, num_samples: int, num_features: int, with_nulls: bool, with_nans: bool
) -> None:
    """Test basic functionality of `generate_synthetic_time_series` across backends and configurations."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=num_samples,
        num_features=num_features,
        with_nulls=with_nulls,
        with_nans=with_nans,
        mode=MODE_SINGLE_STEP,
    )

    # Get shape and compute for Dask
    shape = get_shape(df)

    # Validate number of samples and features
    if num_samples == 0:
        assert shape[0] == 0, f"Expected empty DataFrame, got {shape[0]} rows"
    else:
        assert shape[0] == num_samples, f"Expected {num_samples} rows, got {shape[0]}"

        # For Dask, compute to finalize DataFrame creation
        if is_dask_df(df):
            df = df.compute()  # type: ignore

        # Check feature columns based on backend
        feature_columns: list[str] = []
        if isinstance(df, pa.Table):
            feature_columns = [str(name) for name in df.schema.names]  # type: ignore
            feature_columns = [name for name in feature_columns if "feature_" in name]
        else:
            feature_columns = [str(col) for col in df.columns]
            feature_columns = [col for col in feature_columns if "feature_" in col]

        assert (
            len(feature_columns) == num_features
        ), f"Expected {num_features} feature columns, got {len(feature_columns)}"

    # Verify DataFrame backend type
    expected_type: Any = TEMPORALSCOPE_CORE_BACKEND_TYPES[backend]
    if is_dask_df(df):
        assert type(df).__module__.startswith("dask."), "Expected Dask DataFrame"
    else:
        assert isinstance(df, expected_type), f"Expected {expected_type} for backend '{backend}'"  # type: ignore[arg-type]

    # Test target column type for single-step mode
    if num_samples > 0:
        target_val = get_row_value(df, "target")
        if isinstance(df, pa.Table):
            target_type = df.schema.field("target").type  # type: ignore
            assert pa.types.is_floating(target_type), "Expected scalar target value in PyArrow"
        elif isinstance(df, pl.DataFrame):
            assert isinstance(target_val, (float, int)), "Expected scalar target value in Polars"
        else:
            assert np.isscalar(target_val), "Expected scalar target value in Pandas-based backend"


# ========================= Time Column Generation Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("time_col_numeric", [True, False])
def test_time_column_generation(backend: str, time_col_numeric: bool) -> None:
    """Test the `time` column generation as numeric or timestamp based on `time_col_numeric`."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=100,
        num_features=3,
        time_col_numeric=time_col_numeric,
        mode=MODE_SINGLE_STEP,
    )

    # Get first time value
    time_val = get_row_value(df, "time")

    # Check time column based on type
    if time_col_numeric:
        if isinstance(df, pl.DataFrame):
            assert isinstance(time_val, (float, int)), "Expected Polars numeric column"
        elif isinstance(df, pa.Table):
            time_type = df.schema.field("time").type  # type: ignore
            assert pa.types.is_floating(time_type), "Expected PyArrow float column"
        else:
            assert isinstance(time_val, (np.float64, float)), "Expected numeric column"
    else:
        if isinstance(df, pl.DataFrame):
            assert isinstance(time_val, datetime), "Expected Polars datetime column"
        elif isinstance(df, pa.Table):
            time_type = df.schema.field("time").type  # type: ignore
            assert isinstance(time_type, pa.TimestampType), "Expected PyArrow timestamp column"
        else:
            assert isinstance(time_val, (pd.Timestamp, datetime)), "Expected timestamp column"


# ========================= Error Handling Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_invalid_backend(backend: str) -> None:
    """Test error handling for unsupported backends."""
    with pytest.raises(UnsupportedBackendError, match="is not supported"):
        generate_synthetic_time_series(backend=INVALID_BACKEND, num_samples=100, num_features=5)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_unsupported_mode(backend: str) -> None:
    """Test that an unsupported mode raises the appropriate error."""
    with pytest.raises(ValueError, match="Unsupported mode: multi_step. Only 'single_step' mode is supported."):
        generate_synthetic_time_series(
            backend=backend,
            num_samples=100,
            num_features=5,
            mode="multi_step",
        )


# ========================= Additional Edge Case Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_empty_data(backend: str) -> None:
    """Test generating synthetic data with zero samples to verify empty dataset handling."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=3)
    shape = get_shape(df)
    assert shape[0] == 0, f"Expected empty DataFrame, got {shape[0]} rows"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_with_nulls_and_nans(backend: str) -> None:
    """Test generating synthetic data with both nulls and NaNs introduced to feature columns."""
    df = generate_synthetic_time_series(
        backend=backend, num_samples=100, num_features=5, with_nulls=True, with_nans=True
    )

    # Get first feature value
    feature_val = get_row_value(df, "feature_1")

    # Validate that both None and NaN values are present in the feature columns
    if isinstance(df, pa.Table):
        feature_field = df.schema.field("feature_1")  # type: ignore
        assert feature_field is not None, "Expected feature_1 column in PyArrow backend"
    elif isinstance(df, pl.DataFrame):
        # Get null count and ensure it's greater than 0
        has_nulls = df.select(pl.col("feature_1").is_null()).sum().item() > 0  # type: ignore
        assert has_nulls, "Expected None or NaN in Polars backend"
    else:
        assert pd.isna(feature_val), "Expected None or NaN in Pandas-based backend"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_defaults_only_backend(backend: str) -> None:
    """Test function call with only the backend parameter to confirm default value handling."""
    df = generate_synthetic_time_series(backend=backend)
    shape = get_shape(df)

    # Check if the DataFrame matches the default values (100 samples, 3 features)
    assert shape[0] == 100, f"Expected 100 rows, got {shape[0]}"

    # For Dask, compute to finalize DataFrame creation
    if is_dask_df(df):
        df = df.compute()  # type: ignore

    feature_columns: list[str] = []
    if isinstance(df, pa.Table):
        feature_columns = [str(name) for name in df.schema.names]  # type: ignore
        feature_columns = [name for name in feature_columns if "feature_" in name]
    else:
        feature_columns = [str(col) for col in df.columns]
        feature_columns = [col for col in feature_columns if "feature_" in col]

    assert len(feature_columns) == 3, "Expected 3 feature columns"


def test_generate_synthetic_time_series_negative_values() -> None:
    """Test that generating synthetic data with negative `num_samples` or `num_features` raises a ValueError."""
    with pytest.raises(ValueError, match="`num_samples` and `num_features` must be non-negative."):
        generate_synthetic_time_series(
            backend="pandas",
            num_samples=-10,
            num_features=5,
        )
    with pytest.raises(ValueError, match="`num_samples` and `num_features` must be non-negative."):
        generate_synthetic_time_series(backend="pandas", num_samples=100, num_features=-3)
