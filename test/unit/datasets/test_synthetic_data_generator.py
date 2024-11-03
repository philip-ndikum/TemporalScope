# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
import dask.dataframe as dd
import modin.pandas as mpd
import polars as pl
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series
from temporalscope.core.core_utils import TEMPORALSCOPE_CORE_BACKEND_TYPES, MODE_SINGLE_STEP
from temporalscope.core.exceptions import UnsupportedBackendError
from datetime import datetime

# Constants
VALID_BACKENDS = list(TEMPORALSCOPE_CORE_BACKEND_TYPES.keys())
INVALID_BACKEND = "unsupported_backend"

# ========================= Basic Data Generation Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("num_samples, num_features", [(100, 3), (0, 0), (1000, 10)])
@pytest.mark.parametrize("with_nulls, with_nans", [(True, False), (False, True), (True, True)])
def test_generate_synthetic_time_series_basic(backend, num_samples, num_features, with_nulls, with_nans):
    """Test basic functionality of `generate_synthetic_time_series` across backends and configurations."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=num_samples,
        num_features=num_features,
        with_nulls=with_nulls,
        with_nans=with_nans,
        mode=MODE_SINGLE_STEP,
    )

    # For Dask, compute to finalize DataFrame creation
    if backend == "dask":
        df = df.compute()

    # Validate number of samples and features
    if num_samples == 0:
        assert df.shape[0] == 0, f"Expected empty DataFrame, got {df.shape[0]} rows"
    else:
        assert df.shape[0] == num_samples, f"Expected {num_samples} rows, got {df.shape[0]}"
        
        # Check feature columns based on backend
        if backend == "pyarrow":
            feature_columns = [col for col in df.schema.names if "feature_" in col]
        else:
            feature_columns = [col for col in df.columns if "feature_" in col]
        
        assert len(feature_columns) == num_features, f"Expected {num_features} feature columns, got {len(feature_columns)}"

    # Verify DataFrame backend type
    expected_type = TEMPORALSCOPE_CORE_BACKEND_TYPES[backend]
    if backend == "dask":
        # Generalize check for Dask DataFrame
        assert dd.utils.is_dataframe_like(df), "Expected Dask DataFrame-like object"
    else:
        assert isinstance(df, expected_type), f"Expected {expected_type} for backend '{backend}'"

    # Test target column type for single-step mode
    if num_samples > 0:
        if backend == "pyarrow":
            # PyArrow requires special handling to access values in columns
            assert pa.types.is_floating(df.schema.field("target").type), "Expected scalar target value in PyArrow"
        elif backend == "polars":
            assert isinstance(df["target"][0], (float, int)), "Expected scalar target value in Polars"
        else:
            assert np.isscalar(df["target"].iloc[0]), "Expected scalar target value in Pandas-based backend"



# ========================= Time Column Generation Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("time_col_numeric", [True, False])
def test_time_column_generation(backend, time_col_numeric):
    """Test the `time` column generation as numeric or timestamp based on `time_col_numeric`."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=100,
        num_features=3,
        time_col_numeric=time_col_numeric,
        mode=MODE_SINGLE_STEP,
    )

    # For Dask, compute to finalize DataFrame creation
    if backend == "dask":
        df = df.compute()

    # Check time column based on type
    if time_col_numeric:
        # Verify that `time` is numeric
        if backend == "polars":
            assert isinstance(df["time"][0], (float, int)), "Expected Polars numeric column"
        elif backend == "pyarrow":
            assert pa.types.is_floating(df.schema.field("time").type), "Expected PyArrow float column"
        elif backend == "dask":
            assert dd.utils.is_series_like(df["time"]), "Expected Dask numeric-like column"
        else:
            assert isinstance(df["time"].iloc[0], np.float64), "Expected Pandas numeric column"
    else:
        # Verify that `time` is timestamp-like
        if backend == "polars":
            # Check for datetime.datetime in Polars
            assert isinstance(df["time"].to_list()[0], datetime), "Expected Polars datetime column"
        elif backend == "pyarrow":
            assert isinstance(df.schema.field("time").type, pa.lib.TimestampType), "Expected PyArrow timestamp column"
        elif backend == "dask":
            assert dd.utils.is_series_like(df["time"]), "Expected Dask timestamp-like column"
        else:
            assert isinstance(df["time"].iloc[0], pd.Timestamp), "Expected Pandas timestamp-like column"


# ========================= Error Handling Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_invalid_backend(backend):
    """Test error handling for unsupported backends."""
    with pytest.raises(UnsupportedBackendError, match="is not supported"):
        generate_synthetic_time_series(
            backend=INVALID_BACKEND,
            num_samples=100,     
            num_features=5       
        )

@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_unsupported_mode(backend):
    """Test that an unsupported mode raises the appropriate error."""
    with pytest.raises(ValueError, match="Unsupported mode: multi_step. Only 'single_step' mode is supported."):
        generate_synthetic_time_series(
            backend=backend,
            num_samples=100,       # Add num_samples
            num_features=5,        # Add num_features
            mode="multi_step"      # Use an unsupported mode
        )


# ========================= Additional Edge Case Tests =========================

@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_empty_data(backend):
    """Test generating synthetic data with zero samples to verify empty dataset handling."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=0,
        num_features=3
    )

    # For Dask, compute to finalize DataFrame creation
    if backend == "dask":
        df = df.compute()

    # Check that the DataFrame or equivalent backend structure is empty
    assert df.shape[0] == 0, f"Expected empty DataFrame, got {df.shape[0]} rows"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_with_nulls_and_nans(backend):
    """Test generating synthetic data with both nulls and NaNs introduced to feature columns."""
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=100,
        num_features=5,
        with_nulls=True,
        with_nans=True
    )

    # For Dask, compute to finalize DataFrame creation
    if backend == "dask":
        df = df.compute()

    # Validate that both None and NaN values are present in the feature columns
    if backend == "pyarrow":
        assert df["feature_1"][0].as_py() is None or pd.isna(df["feature_1"][0].as_py()), "Expected None or NaN in PyArrow backend"
    elif backend == "polars":
        assert df["feature_1"].is_null().sum() > 0, "Expected None or NaN in Polars backend"
    else:
        assert df.iloc[0, 2] is None or pd.isna(df.iloc[0, 2]), "Expected None or NaN in Pandas-based backend"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_defaults_only_backend(backend):
    """Test function call with only the backend parameter to confirm default value handling."""
    df = generate_synthetic_time_series(backend=backend)

    # For Dask, compute to finalize DataFrame creation
    if backend == "dask":
        df = df.compute()

    # Check if the DataFrame matches the default values (100 samples, 3 features)
    assert df.shape[0] == 100, f"Expected 100 rows, got {df.shape[0]}"
    assert len([col for col in (df.schema.names if backend == "pyarrow" else df.columns) if "feature_" in col]) == 3, "Expected 3 feature columns"


def test_generate_synthetic_time_series_negative_values():
    """Test that generating synthetic data with negative `num_samples` or `num_features` raises a ValueError."""
    with pytest.raises(ValueError, match="`num_samples` and `num_features` must be non-negative."):
        generate_synthetic_time_series(
            backend="pandas",  # Using a simple backend for targeted error checking
            num_samples=-10,
            num_features=5
        )
    with pytest.raises(ValueError, match="`num_samples` and `num_features` must be non-negative."):
        generate_synthetic_time_series(
            backend="pandas",
            num_samples=100,
            num_features=-3
        )
