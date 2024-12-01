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

from temporalscope.core.core_utils import MODE_SINGLE_TARGET, TEMPORALSCOPE_CORE_BACKEND_TYPES
from temporalscope.core.exceptions import UnsupportedBackendError
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = [backend for backend in TEMPORALSCOPE_CORE_BACKEND_TYPES if backend != "dask"]
INVALID_BACKEND = "unsupported_backend"

# Type variables for better type handling
T = TypeVar("T")


def is_dask_df(obj: Any) -> bool:
    """Check if object is a Dask DataFrame."""
    # Check both the module name and string representation
    return type(obj).__module__.startswith("dask.") or isinstance(obj, object) and "Dask DataFrame" in str(obj)


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


def get_column_names(df: Any) -> set[str]:
    """Get column names from DataFrame, handling different backends."""
    if isinstance(df, pa.Table):
        return set(df.schema.names)  # type: ignore
    elif is_dask_df(df):
        return set(df.columns)
    else:
        return set(df.columns)


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
        mode=MODE_SINGLE_TARGET,
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
        mode=MODE_SINGLE_TARGET,
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
    with pytest.raises(ValueError, match="Unsupported mode: multi_target"):
        generate_synthetic_time_series(
            backend=backend,
            num_samples=100,
            num_features=5,
            mode="multi_target",
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

    # Get all feature values
    if isinstance(df, pa.Table):
        feature_field = df.schema.field("feature_1")  # type: ignore
        assert feature_field is not None, "Expected feature_1 column in PyArrow backend"
    elif isinstance(df, pl.DataFrame):
        # Get null count and ensure it's greater than 0
        has_nulls = df.select(pl.col("feature_1").is_null()).sum().item() > 0  # type: ignore
        assert has_nulls, "Expected None or NaN in Polars backend"
    else:
        # Check any row has null/nan
        if hasattr(df["feature_1"], "compute"):  # For dask
            feature_vals = pd.Series(df["feature_1"].compute())
        else:  # For pandas and modin
            feature_vals = pd.Series(df["feature_1"])
        assert feature_vals.isna().any(), "Expected at least one None or NaN value"


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


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("drop_time", [True, False])
def test_generate_synthetic_time_series_drop_time(backend: str, drop_time: bool) -> None:
    """Test that drop_time parameter correctly handles time column inclusion/exclusion."""
    num_samples = 3
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=num_samples,
        num_features=2,
        drop_time=drop_time,
        time_col_numeric=True,  # Use numeric time for easier verification
    )

    # Get column names
    columns = get_column_names(df)

    # Test the dictionary unpacking operation directly
    time_dict = {"time": np.arange(num_samples, dtype=np.float64)} if not drop_time else {}
    expected_cols = {
        **time_dict,  # Test the same unpacking operation as in the implementation
        "target": "value",  # Placeholder value
        "feature_1": "value",
        "feature_2": "value",
    }.keys()

    # Verify column names match expected set from dictionary unpacking
    assert columns == set(expected_cols), "Column names don't match expected set from dictionary unpacking"

    # Verify time column values when present
    if not drop_time:
        time_val = get_row_value(df, "time")
        assert isinstance(time_val, (np.float64, float)), "time column should be numeric"
        assert time_val == 0.0, "time column should start at 0"

    # Verify target column values
    target_val = get_row_value(df, "target")
    assert isinstance(target_val, (np.float64, float)), "target column should be numeric"
    assert 0.0 <= target_val <= 1.0, "target values should be between 0 and 1"


def test_generate_synthetic_time_series_all_paths():
    """Test all code paths in synthetic data generation."""
    # Test with time column included
    df1 = generate_synthetic_time_series(
        backend="pandas",
        num_samples=3,
        num_features=2,
        time_col_numeric=True,
        drop_time=False,  # Include time column
    )
    assert "time" in df1.columns
    assert df1["time"].dtype == np.float64
    assert len(df1.columns) == 4  # time, target, feature_1, feature_2

    # Test with time column dropped
    df2 = generate_synthetic_time_series(
        backend="pandas",
        num_samples=3,
        num_features=2,
        drop_time=True,  # Drop time column
    )
    assert "time" not in df2.columns
    assert len(df2.columns) == 3  # target, feature_1, feature_2

    # Verify feature columns in both cases
    for df in [df1, df2]:
        for i in range(2):
            col = f"feature_{i+1}"
            assert col in df.columns
            assert all(0 <= x <= 1 for x in df[col])


def test_generate_synthetic_time_series_feature_loop():
    """Test feature column generation loop."""
    # Test with exactly one feature to force loop execution
    df = generate_synthetic_time_series(backend="pandas", num_samples=1, num_features=1, drop_time=True)

    # Verify feature column exists and has correct values
    assert "feature_1" in df.columns
    assert len(df["feature_1"]) == 1
    assert 0 <= df["feature_1"].iloc[0] <= 1

    # Test with zero features to cover loop initialization
    df_empty = generate_synthetic_time_series(backend="pandas", num_samples=1, num_features=0, drop_time=True)
    assert not any(col.startswith("feature_") for col in df_empty.columns)


def test_generate_synthetic_time_series_feature_value():
    """Test feature value generation."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=1, num_features=1, drop_time=True)

    # Verify feature value is generated correctly
    feature_val = df["feature_1"].values[0]
    assert isinstance(feature_val, float)
    assert 0 <= feature_val <= 1


def test_generate_synthetic_time_series_dask():
    """Test that Dask DataFrames are properly handled."""
    # Skip if dask is not in TEMPORALSCOPE_CORE_BACKEND_TYPES
    if "dask" not in TEMPORALSCOPE_CORE_BACKEND_TYPES:
        pytest.skip("Dask backend not available")

    df = generate_synthetic_time_series(backend="dask", num_samples=1, num_features=1, drop_time=True)

    # Verify it's a Dask DataFrame
    assert is_dask_df(df), "Expected Dask DataFrame"

    # Get a value to trigger computation
    value = get_row_value(df, "feature_1")
    assert isinstance(value, float)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_percentage_validation(backend: str) -> None:
    """Test validation of null and nan percentage parameters."""
    with pytest.raises(ValueError, match="null_percentage must be between 0.0 and 1.0"):
        generate_synthetic_time_series(
            backend=backend, num_samples=100, num_features=5, with_nulls=True, null_percentage=1.5
        )

    with pytest.raises(ValueError, match="nan_percentage must be between 0.0 and 1.0"):
        generate_synthetic_time_series(
            backend=backend, num_samples=100, num_features=5, with_nans=True, nan_percentage=-0.1
        )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_generate_synthetic_time_series_single_row(backend: str) -> None:
    """Test handling of nulls and nans with single row datasets."""
    # Test nulls take precedence
    df = generate_synthetic_time_series(backend=backend, num_samples=1, num_features=2, with_nulls=True, with_nans=True)
    feature_val = get_row_value(df, "feature_1")
    assert pd.isna(feature_val), "Expected null/nan value for single row with both nulls and nans"

    # Test nans only
    df = generate_synthetic_time_series(
        backend=backend, num_samples=1, num_features=2, with_nulls=False, with_nans=True
    )
    feature_val = get_row_value(df, "feature_1")
    assert pd.isna(feature_val), "Expected null/nan value for single row with nans only"
