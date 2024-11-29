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

.. note::
   - **Batch size**: This package assumes no default batch size; batch size is typically managed by the data loader (e.g.,
     TensorFlow `DataLoader`, PyTorch `DataLoader`). The synthetic data generator provides the raw data structure, which is
     then partitioned and batched as needed in downstream pipelines (e.g., after target shifting or partitioning).

   - **TimeFrame and Target Shape**: The TemporalScope framework checks if the target is scalar or vector (sequence). The
     generated data in multi-step mode follows a unified structure, with the target represented as a sequence in the same
     DataFrame. This ensures compatibility with popular machine learning libraries that are compatible with SHAP, LIME, and
     other explainability methods.

.. seealso::
   For further details on the single-step and multi-step modes, refer to the core TemporalScope documentation on data handling.

Example Visualization:
----------------------
Here is a visual demonstration of the datasets generated for single-step and multi-step modes, including the shape
of input (`X`) and target (`Y`) data compatible with most popular ML frameworks like TensorFlow, PyTorch, and SHAP.

Single-step mode:
    +------------+------------+------------+------------+-----------+
    |   time     | feature_1  | feature_2  | feature_3  |  target   |
    +============+============+============+============+===========+
    | 2023-01-01 |   0.15     |   0.67     |   0.89     |   0.33    |
    +------------+------------+------------+------------+-----------+
    | 2023-01-02 |   0.24     |   0.41     |   0.92     |   0.28    |
    +------------+------------+------------+------------+-----------+

    Shape:
    - `X`: (num_samples, num_features)
    - `Y`: (num_samples, 1)  # Scalar target for each time step

Multi-step mode (with vectorized targets):

    +------------+------------+------------+------------+-------------+
    |   time     | feature_1  | feature_2  | feature_3  |    target   |
    +============+============+============+============+=============+
    | 2023-01-01 |   0.15     |   0.67     |   0.89     |  [0.3, 0.4] |
    +------------+------------+------------+------------+-------------+
    | 2023-01-02 |   0.24     |   0.41     |   0.92     |  [0.5, 0.6] |
    +------------+------------+------------+------------+-------------+

    Shape:
    - `X`: (num_samples, num_features)
    - `Y`: (num_samples, sequence_length)  # Vectorized target for each input sequence

Example Usage:
--------------
.. code-block:: python

    from temporalscope.core.core_utils import MODE_SINGLE_STEP, MODE_MULTI_STEP
    from temporalscope.datasets.synthetic_data_generator import create_sample_data

    # Generating data for single-step mode
    df = create_sample_data(num_samples=100, num_features=3, mode=MODE_SINGLE_STEP)
    print(df.head())  # Shows the generated data with features and a scalar target.

    # Generating data for multi-step mode
    df = create_sample_data(num_samples=100, num_features=3, mode=MODE_MULTI_STEP)
    print(df.head())  # Shows the generated input sequence (`X`) and target sequence (`Y`).
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

    :param df: Pandas DataFrame to modify (modified in-place)
    :type df: pd.DataFrame
    :param feature_cols: List of feature column names to apply nulls/nans to
    :type feature_cols: list[str]
    :param with_nulls: Whether to apply null values
    :type with_nulls: bool
    :param with_nans: Whether to apply NaN values (only if with_nulls is False)
    :type with_nans: bool
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

    :param df: Pandas DataFrame to modify (modified in-place)
    :type df: pd.DataFrame
    :param feature_cols: List of feature column names to apply nulls/nans to
    :type feature_cols: list[str]
    :param with_nulls: Whether to apply null values
    :type with_nulls: bool
    :param with_nans: Whether to apply NaN values
    :type with_nans: bool
    :param null_percentage: Percentage of rows to contain null values (0.0 to 1.0)
    :type null_percentage: float
    :param nan_percentage: Percentage of rows to contain NaN values (0.0 to 1.0)
    :type nan_percentage: float
    :param num_samples: Total number of rows in the DataFrame
    :type num_samples: int
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
    mode: str = "single_step",
    time_col_numeric: bool = False,
    drop_time: bool = False,
    random_seed: int = RANDOM_SEED,
) -> SupportedTemporalDataFrame:
    """Generate synthetic time series data with specified backend support and configurations.

    :param backend: The backend to use for the generated data.
    :type backend: str
    :param num_samples: Number of samples (rows) to generate in the time series data.
    :type num_samples: int, optional
    :param num_features: Number of feature columns to generate in addition to 'time' and 'target' columns.
    :type num_features: int, optional
    :param with_nulls: Introduces None values in feature columns if True.
    :type with_nulls: bool, optional
    :param with_nans: Introduces NaN values in feature columns if True.
    :type with_nans: bool, optional
    :param null_percentage: Percentage of rows to contain null values (0.0 to 1.0). Only used if with_nulls is True.
                          For datasets with few rows, ensures at least one row is affected if nulls are enabled.
                          For single-row datasets, nulls take precedence over NaNs if both are enabled.
    :type null_percentage: float, optional
    :param nan_percentage: Percentage of rows to contain NaN values (0.0 to 1.0). Only used if with_nans is True.
                         For datasets with few rows, ensures at least one row is affected if NaNs are enabled.
                         For single-row datasets, nulls take precedence over NaNs if both are enabled.
    :type nan_percentage: float, optional
    :param mode: Mode for data generation; currently only supports 'single_step'.
    :type mode: str, optional
    :param time_col_numeric: If True, 'time' column is numeric instead of datetime.
    :type time_col_numeric: bool, optional
    :param drop_time: If True, omits the time column from output DataFrame.
    :type drop_time: bool, optional
    :param random_seed: Seed for random number generation to ensure reproducible results.
    :type random_seed: int, optional

    :return: DataFrame or Table in the specified backend containing synthetic data.
    :rtype: SupportedTemporalDataFrame

    :raises ValueError: If unsupported backend, mode, or invalid parameters.
    """
    is_valid_temporal_backend(backend)

    if num_samples < 0 or num_features < 0:
        raise ValueError("`num_samples` and `num_features` must be non-negative.")
    if mode != "single_step":
        raise ValueError(f"Unsupported mode: {mode}. Only 'single_step' mode is supported.")
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
