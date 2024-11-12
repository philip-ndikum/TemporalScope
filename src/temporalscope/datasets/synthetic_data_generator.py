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

import pandas as pd
import numpy as np
import dask.dataframe as dd
from temporalscope.core.core_utils import validate_backend, convert_to_backend, SupportedTemporalDataFrame


def generate_synthetic_time_series(
    backend: str,
    num_samples: int = 100,
    num_features: int = 3,
    with_nulls: bool = False,
    with_nans: bool = False,
    mode: str = "single_step",
    time_col_numeric: bool = False,
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
    :param mode: Mode for data generation; currently only supports 'single_step'.
    :type mode: str, optional
    :param time_col_numeric: If True, 'time' column is numeric instead of datetime.
    :type time_col_numeric: bool, optional

    :return: DataFrame or Table in the specified backend containing synthetic data.
    :rtype: SupportedTemporalDataFrame

    :raises ValueError: If unsupported backend, mode, or invalid parameters.
    """
    validate_backend(backend)

    if num_samples < 0 or num_features < 0:
        raise ValueError("`num_samples` and `num_features` must be non-negative.")
    if mode != "single_step":
        raise ValueError(f"Unsupported mode: {mode}. Only 'single_step' mode is supported.")

    # Generate initial DataFrame with Pandas
    time_column = (
        np.arange(num_samples, dtype=np.float64)
        if time_col_numeric
        else pd.date_range("2023-01-01", periods=num_samples)
    )
    df = pd.DataFrame(
        {
            "time": time_column,
            "target": np.random.rand(num_samples),
            **{f"feature_{i+1}": np.random.rand(num_samples) for i in range(num_features)},
        }
    )

    # Apply nulls or NaNs if specified
    if with_nulls:
        df.iloc[0:5, 2:] = None
    if with_nans:
        df.iloc[0:5, 2:] = np.nan

    # Convert to specified backend
    result = convert_to_backend(df, backend)

    # Ensure Dask DataFrames are computed before returning
    if isinstance(result, dd.DataFrame):
        result = result.persist()

    return result
