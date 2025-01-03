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

"""TemporalScope/src/temporalscope/datasets/synthetic_data_generator.py

This module provides utilities for generating synthetic time series data specifically for testing
and validation purposes. While TemporalScope uses Narwhals for backend-agnostic operations, this
generator serves as a defensive programming tool to ensure:

1. Runtime Testing: Generate test data across different DataFrame backends to verify behavior
2. Edge Case Coverage: Create data with nulls, NaNs, and various data types
3. Backend Validation: Test Narwhals operations with different DataFrame implementations

The generator creates consistent test data that matches the TimeFrame API's expected structure
(see core_utils.py for data structure details). This helps maintain code stability by providing
reliable test data that works across all supported backends.

Note: This module is primarily intended for testing purposes, not for production data generation.

"""

import narwhals as nw
import numpy as np
import pandas as pd
from narwhals.typing import FrameT
from narwhals.utils import Implementation

# Set random seed for reproducibility for unit tests
RANDOM_SEED = 100


def _apply_nulls_nans_single_row(df: pd.DataFrame, feature_cols: list[str], with_nulls: bool, with_nans: bool) -> None:
    """Apply nulls/nans to a single row DataFrame using pandas operations.

    Parameters
    ----------
    df : pd.DataFrame
        Single row DataFrame to modify
    feature_cols : list[str]
        List of feature column names to apply nulls/nans to
    with_nulls : bool
        If True, apply None values to feature columns
    with_nans : bool
        If True and with_nulls is False, apply NaN values to feature columns

    Notes
    -----
    - Modifies DataFrame in-place
    - If both with_nulls and with_nans are True, nulls take precedence
    """
    if with_nulls:
        df.iloc[0, df.columns.get_indexer(feature_cols)] = None
    elif with_nans:
        df.iloc[0, df.columns.get_indexer(feature_cols)] = np.nan


def _apply_nulls_nans_multi_row(
    df: pd.DataFrame,
    feature_cols: list[str],
    with_nulls: bool,
    with_nans: bool,
    null_percentage: float,
    nan_percentage: float,
    num_samples: int,
) -> None:
    """Apply nulls/nans to multiple rows in a DataFrame using pandas operations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify
    feature_cols : list[str]
        List of feature column names to apply nulls/nans to
    with_nulls : bool
        If True, apply None values to feature columns
    with_nans : bool
        If True, apply NaN values to feature columns
    null_percentage : float
        Percentage of rows to contain null values (0.0 to 1.0)
    nan_percentage : float
        Percentage of rows to contain NaN values (0.0 to 1.0)
    num_samples : int
        Total number of rows in DataFrame

    Notes
    -----
    - Modifies DataFrame in-place
    - Ensures at least one row has nulls/nans if enabled
    - NaNs are only applied to rows that don't already have nulls
    """
    null_indices = []
    if with_nulls:
        num_null_rows = min(
            num_samples,  # Don't exceed total rows
            max(1, int(num_samples * null_percentage)),  # At least 1 row if enabled
        )
        null_indices = np.random.choice(num_samples, size=num_null_rows, replace=False)
        df.iloc[null_indices, df.columns.get_indexer(feature_cols)] = None

    if with_nans:
        # Use remaining rows after null application
        available_indices = np.setdiff1d(np.arange(num_samples), null_indices)
        if len(available_indices) > 0:  # Only if we have rows left
            num_nan_rows = min(
                len(available_indices),  # Don't exceed available rows
                max(1, int(num_samples * nan_percentage)),  # At least 1 row if enabled
            )
            nan_indices = np.random.choice(available_indices, size=num_nan_rows, replace=False)
            df.iloc[nan_indices, df.columns.get_indexer(feature_cols)] = np.nan


def _validate_synthetic_data_params(
    backend: str, num_samples: int, num_features: int, mode: str, null_percentage: float, nan_percentage: float
) -> None:
    """Validate parameters for synthetic data generation.

    Parameters
    ----------
    backend : str
        Backend to use for generated data
    num_samples : int
        Number of samples (rows) to generate
    num_features : int
        Number of feature columns to generate
    mode : str
        Mode for data generation
    null_percentage : float
        Percentage of rows to contain null values
    nan_percentage : float
        Percentage of rows to contain NaN values

    Raises
    ------
    ValueError
        If any parameters are invalid
    """
    if backend.lower() not in [impl.name.lower() for impl in Implementation]:
        raise ValueError(f"Backend '{backend}' is not supported by Narwhals.")

    if num_samples < 0 or num_features < 0:
        raise ValueError("`num_samples` and `num_features` must be non-negative.")
    if mode != "single_target":
        raise ValueError(f"Unsupported mode: {mode}. Only 'single_target' mode is supported.")
    if not 0.0 <= null_percentage <= 1.0:
        raise ValueError("null_percentage must be between 0.0 and 1.0")
    if not 0.0 <= nan_percentage <= 1.0:
        raise ValueError("nan_percentage must be between 0.0 and 1.0")


@nw.narwhalify
def generate_synthetic_time_series(
    backend: str,
    *,  # Force keyword arguments for better readability
    num_samples: int = 100,
    num_features: int = 3,
    with_nulls: bool = False,
    with_nans: bool = False,
    null_percentage: float = 0.05,
    nan_percentage: float = 0.05,
    mode: str = "single_target",
    time_col_numeric: bool = False,
    drop_time: bool = False,
    random_seed: int = RANDOM_SEED,
) -> FrameT:
    """Generate synthetic time series data with specified backend support and configurations.

    Parameters
    ----------
    backend : str
        Backend to use for generated data (must be supported by Narwhals)
    num_samples : int, optional
        Number of samples (rows) to generate, by default 100
    num_features : int, optional
        Number of feature columns to generate, by default 3
    with_nulls : bool, optional
        Whether to introduce None values in feature columns, by default False
    with_nans : bool, optional
        Whether to introduce NaN values in feature columns, by default False
    null_percentage : float, optional
        Percentage of rows to contain null values (0.0 to 1.0), by default 0.05
    nan_percentage : float, optional
        Percentage of rows to contain NaN values (0.0 to 1.0), by default 0.05
    mode : str, optional
        Mode for data generation, by default "single_target"
    time_col_numeric : bool, optional
        If True, time column is numeric instead of datetime, by default False
    drop_time : bool, optional
        If True, time column is omitted from output, by default False
    random_seed : int, optional
        Seed for random number generation, by default RANDOM_SEED

    Returns
    -------
    FrameT
        Narwhals DataFrame containing generated synthetic data

    Raises
    ------
    ValueError
        If backend not supported by Narwhals
        If invalid mode specified (only "single_target" supported)
        If invalid parameters (negative samples/features, invalid percentages)

    Notes
    -----
    - For datasets with few rows, ensures at least one row has nulls/NaNs if enabled
    - For single-row datasets, nulls take precedence over NaNs if both enabled
    - Time column can be numeric (timestamps) or datetime based on time_col_numeric
    """
    _validate_synthetic_data_params(
        backend=backend,
        num_samples=num_samples,
        num_features=num_features,
        mode=mode,
        null_percentage=null_percentage,
        nan_percentage=nan_percentage,
    )

    np.random.seed(random_seed)

    # Generate DataFrame
    time_column = (
        np.arange(num_samples, dtype=np.float64)
        if time_col_numeric
        else pd.date_range("2023-01-01", periods=num_samples)
    )

    columns = {}
    if not drop_time:
        columns["time"] = time_column
    columns["target"] = np.random.rand(num_samples)
    for i in range(num_features):
        columns[f"feature_{i+1}"] = np.random.rand(num_samples)

    df = pd.DataFrame(columns)

    # Apply nulls/nans if needed
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    if feature_cols and (with_nulls or with_nans):
        if num_samples == 1:
            _apply_nulls_nans_single_row(df, feature_cols, with_nulls, with_nans)
        else:
            _apply_nulls_nans_multi_row(
                df, feature_cols, with_nulls, with_nans, null_percentage, nan_percentage, num_samples
            )

    # Convert to Narwhals DataFrame and transform
    df_nw = nw.from_native(df)

    # Following Pattern 1 from notebook: proper column selection with nw.col() and alias()
    result = df_nw.select(
        [
            # Time column if present
            *([nw.col("time").alias("time")] if not drop_time else []),
            # Target column (always present)
            nw.col("target").alias("target"),
            # Feature columns
            *[nw.col(f"feature_{i+1}").alias(f"feature_{i+1}") for i in range(num_features)],
        ]
    )

    # Convert to requested backend
    if backend.lower() != "pandas":
        # First wrap in Narwhals to use its conversion capabilities
        df_nw = nw.from_native(result)
        # Then convert to the target backend using the backend's native module
        if backend.lower() == "polars":
            import polars as pl

            result = pl.from_pandas(df_nw.to_native())
        else:
            # For other backends, let Narwhals handle the conversion
            result = df_nw.to_native()

    return result
