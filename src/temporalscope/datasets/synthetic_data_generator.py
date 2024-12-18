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

This module provides utility functions for generating synthetic time series data, specifically designed to facilitate
unit testing and benchmarking of various components across the TemporalScope ecosystem. The generated data simulates
real-world time series with configurable features, ensuring comprehensive coverage of test cases for various modes, such
as single-step and multi-step target data handling. This module is intended for use within automated test pipelines, and
it plays a critical role in maintaining code stability and robustness across different test suites.

Core Purpose:
-------------
The `synthetic_data_generator` is a centralized utility for creating synthetic time series data that enables robust testing
of TemporalScope's core modules. It supports the generation of data for various modes, ensuring that TemporalScope modules
can handle a wide range of time series data, including edge cases and anomalies.

Supported Use Cases:
---------------------
- Single-step mode: Generates scalar target values for tasks where each row represents a single time step.
- Multi-step mode: Produces input-output sequence data for sequence forecasting, where input sequences (`X`) and output
  sequences (`Y`) are handled as part of a unified dataset but with vectorized targets.

Notes
-----
- **Batch size**: This package assumes no default batch size; batch size is typically managed by the data loader (e.g.,
 TensorFlow `DataLoader`, PyTorch `DataLoader`). The synthetic data generator provides the raw data structure, which is
 then partitioned and batched as needed in downstream pipelines (e.g., after target shifting or partitioning).

- **TimeFrame and Target Shape**: The TemporalScope framework checks if the target is scalar or vector (sequence). The
 generated data in multi-step mode follows a unified structure, with the target represented as a sequence in the same
 DataFrame. This ensures compatibility with popular machine learning libraries that are compatible with SHAP, LIME, and
 other explainability methods.


See Also
--------
For further details on the single-step and multi-step modes, refer to the core TemporalScope documentation on data handling.

Example Visualization:
----------------------
Here is a visual demonstration of the datasets generated for single-step and multi-step modes, including the shape
of input (`X`) and target (`Y`) data compatible with most popular ML frameworks like TensorFlow, PyTorch, and SHAP.

**Single-step mode**:

| Time       | Feature 1 | Feature 2 | Feature 3 | Target  |
|------------|-----------|-----------|-----------|---------|
| 2023-01-01 | 0.15      | 0.67      | 0.89      | 0.33    |
| 2023-01-02 | 0.24      | 0.41      | 0.92      | 0.28    |


Shape:

- `X`: (num_samples, num_features)
- `Y`: (num_samples, 1)


**Multi-step mode (with vectorized targets)**:

| Time       | Feature 1 | Feature 2 | Feature 3 | Target      |
|------------|-----------|-----------|-----------|-------------|
| 2023-01-01 | 0.15      | 0.67      | 0.89      | [0.3, 0.4]  |
| 2023-01-02 | 0.24      | 0.41      | 0.92      | [0.5, 0.6]  |


Shape:

- `X`: (num_samples, num_features)
- `Y`: (num_samples, sequence_length)

Examples
--------
```python
from temporalscope.core.core_utils import MODE_SINGLE_TARGET, MODE_MULTI_TARGET
from temporalscope.datasets.synthetic_data_generator import create_sample_data

# Generating data for single-step mode
df = create_sample_data(num_samples=100, num_features=3, mode=MODE_SINGLE_TARGET)
print(df.head())  # Shows the generated data with features and a scalar target.

# Generating data for multi-step mode
df = create_sample_data(num_samples=100, num_features=3, mode=MODE_MULTI_TARGET)
print(df.head())  # Shows the generated input sequence (`X`) and target sequence (`Y`).
```

"""

import dask.dataframe as dd
import numpy as np
import pandas as pd

from temporalscope.core.core_utils import SupportedTemporalDataFrame, convert_to_backend, is_valid_temporal_backend

# Set random seed for reproducibility for unit tests
RANDOM_SEED = 100


def _apply_nulls_nans_single_row(df: pd.DataFrame, feature_cols: list[str], with_nulls: bool, with_nans: bool) -> None:
    """Apply nulls/nans to a single row DataFrame using pandas operations.

    This is an internal utility function that operates on pandas DataFrames directly for efficiency
    and simplicity in null/nan application. The main function handles conversion to other backends.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to modify (modified in-place)
    feature_cols : list[str]
        List of feature column names to apply nulls/nans to
    with_nulls : bool
        Whether to apply null values
    with_nans : bool
        Whether to apply NaN values (only if with_nulls is False)
    df: pd.DataFrame :

    feature_cols: list[str] :

    with_nulls: bool :

    with_nans: bool :


    Returns
    -------
    None

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

    This is an internal utility function that operates on pandas DataFrames directly for efficiency
    and simplicity in null/nan application. The main function handles conversion to other backends.

    For nulls and nans:
    - Ensures at least 1 row is affected if enabled
    - Respects maximum available rows
    - Prevents overlap between null and nan rows
    - Uses random selection for realistic data generation

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to modify (modified in-place)
    feature_cols : list[str]
        List of feature column names to apply nulls/nans to
    with_nulls : bool
        Whether to apply null values
    with_nans : bool
        Whether to apply NaN values
    null_percentage : float
        Percentage of rows to contain null values (0.0 to 1.0)
    nan_percentage : float
        Percentage of rows to contain NaN values (0.0 to 1.0)
    num_samples : int
        Total number of rows in the DataFrame
    df: pd.DataFrame :

    feature_cols: list[str] :

    with_nulls: bool :

    with_nans: bool :

    null_percentage: float :

    nan_percentage: float :

    num_samples: int :


    Returns
    -------
    None

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


def generate_synthetic_time_series(
    backend: str,
    num_samples: int = 100,
    num_features: int = 3,
    with_nulls: bool = False,
    with_nans: bool = False,
    null_percentage: float = 0.05,  # Default 5% nulls
    nan_percentage: float = 0.05,  # Default 5% nans
    mode: str = "single_target",
    time_col_numeric: bool = False,
    drop_time: bool = False,
    random_seed: int = RANDOM_SEED,
) -> SupportedTemporalDataFrame:
    """
    Generate synthetic time series data with specified backend support and configurations.

    Parameters
    ----------
    backend : str
        The backend to use for the generated data.
    num_samples : int, optional
        Number of samples (rows) to generate in the time series data. Default is 100.
    num_features : int, optional
        Number of feature columns to generate in addition to 'time' and 'target' columns. Default is 3.
    with_nulls : bool, optional
        Whether to introduce None values in feature columns. Default is False.
    with_nans : bool, optional
        Whether to introduce NaN values in feature columns. Default is False.
    null_percentage : float, optional
        Percentage of rows to contain null values (0.0 to 1.0). Only used if `with_nulls` is True.
        - For datasets with few rows, ensures at least one row is affected if nulls are enabled.
        - For single-row datasets, nulls take precedence over NaNs if both are enabled.
        Default is 0.05 (5%).
    nan_percentage : float, optional
        Percentage of rows to contain NaN values (0.0 to 1.0). Only used if `with_nans` is True.
        - For datasets with few rows, ensures at least one row is affected if NaNs are enabled.
        - For single-row datasets, nulls take precedence over NaNs if both are enabled.
        Default is 0.05 (5%).
    mode : str, optional
        Mode for data generation. Currently, only 'single_target' is supported. Default is 'single_target'.
    time_col_numeric : bool, optional
        If True, the 'time' column is numeric instead of a datetime object. Default is False.
    drop_time : bool, optional
        If True, the time column is omitted from the output DataFrame. Default is False.
    random_seed : int, optional
        Seed for random number generation to ensure reproducible results. Default is `RANDOM_SEED`.

    Returns
    -------
    SupportedTemporalDataFrame
        DataFrame or table in the specified backend containing the generated synthetic data.

    Raises
    ------
    ValueError
        If an unsupported backend, mode, or invalid parameters are specified.

    """
    is_valid_temporal_backend(backend)

    if num_samples < 0 or num_features < 0:
        raise ValueError("`num_samples` and `num_features` must be non-negative.")
    if mode != "single_target":
        raise ValueError(f"Unsupported mode: {mode}. Only 'single_target' mode is supported.")
    if not 0.0 <= null_percentage <= 1.0:
        raise ValueError("null_percentage must be between 0.0 and 1.0")
    if not 0.0 <= nan_percentage <= 1.0:
        raise ValueError("nan_percentage must be between 0.0 and 1.0")

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

    result = convert_to_backend(df, backend)
    if isinstance(result, dd.DataFrame):
        result = result.persist()

    return result
