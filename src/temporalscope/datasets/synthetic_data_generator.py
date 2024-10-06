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
Here’s a visual demonstration of the datasets generated for single-step and multi-step modes, including the shape
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

    # Generating data for single-step mode
    df = create_sample_data(num_samples=100, num_features=3, mode="single_step")
    print(df.head())  # Shows the generated data with features and a scalar target.

    # Generating data for multi-step mode
    df = create_sample_data(num_samples=100, num_features=3, mode="multi_step")
    print(df.head())  # Shows the generated input sequence (`X`) and target sequence (`Y`).
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    BACKEND_PANDAS,
    BACKEND_MODIN,
    BACKEND_POLARS,
    MODE_MULTI_STEP,
    MODE_SINGLE_STEP,
    SUPPORTED_MULTI_STEP_BACKENDS,
    SupportedBackendDataFrame,
    validate_and_convert_input,
    validate_backend,
    validate_mode,
)

# Constants defined locally in this file
DEFAULT_NUM_SAMPLES = 100
DEFAULT_NUM_FEATURES = 3
SEED = 42
DEFAULT_NAN_INTERVAL = 10  # Default interval for inserting NaNs
DEFAULT_NULL_INTERVAL = 15  # Default interval for inserting nulls


def create_sample_data(  # noqa: PLR0912
    backend: str,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_features: int = DEFAULT_NUM_FEATURES,
    with_nulls: bool = False,
    with_nans: bool = False,
    timestamp_like: bool = False,
    numeric: bool = False,
    mixed_frequencies: bool = False,
    mixed_timezones: bool = False,
    mode: str = MODE_SINGLE_STEP,
    seed: Optional[int] = SEED,
    nan_interval: int = DEFAULT_NAN_INTERVAL,
    null_interval: int = DEFAULT_NULL_INTERVAL,
) -> pd.DataFrame:
    """Generate synthetic time series data for testing XAI workflows across the TemporalScope ecosystem.

    This function generates synthetic time series data with configurable features, ensuring comprehensive test
    coverage for various machine learning, deep learning, and survival model applications. The generated data
    supports key XAI (Explainable AI) techniques that are model-agnostic, making this utility essential for
    testing model interpretability, such as SHAP, LIME, and other time series XAI workflows.

    For interoperability reasons, the data is first generated using Pandas and then converted to the preferred backend.

    :param backend:
        The backend to use ('pd' for Pandas, 'mpd' for Modin, 'pl' for Polars).
    :type backend: str
    :param num_samples:
        Number of rows (samples) to generate. (default: 100)
    :type num_samples: int
    :param num_features:
        Number of feature columns to generate. (default: 3)
    :type num_features: int
    :param with_nulls:
        If True, introduces null values. (default: False)
    :type with_nulls: bool
    :param with_nans:
        If True, introduces NaN values. (default: False)
    :type with_nans: bool
    :param timestamp_like:
        If True, includes a timestamp-like column. (default: False)
    :type timestamp_like: bool
    :param numeric:
        If True, includes a numeric 'time' column instead of a timestamp. (default: False)
    :type numeric: bool
    :param mixed_frequencies:
        If True, simulates mixed time intervals in the 'time' column. (default: False)
    :type mixed_frequencies: bool
    :param mixed_timezones:
        If True, generates both timezone-aware and naive time data. (default: False)
    :type mixed_timezones: bool
    :param mode:
        Mode for generating the target column. Supported modes: 'single_step', 'multi_step'. (default: 'single_step')
    :type mode: str
    :param seed:
        Random seed for reproducibility. (default: 42)
    :type seed: Optional[int]
    :param nan_interval:
        Interval at which NaN values are inserted in the second feature. (default: 10)
    :type nan_interval: int
    :param null_interval:
        Interval at which null values are inserted in the third feature. (default: 15)
    :type null_interval: int

    :return:
        A Pandas DataFrame. The DataFrame is converted to the specified backend after generation.
    :rtype: pd.DataFrame

    :raises ValueError:
        If an unsupported mode or incompatible configuration is provided, or if multi-step mode is used with unsupported backends.

    Example Visualization:
    ----------------------
    Single-step mode:
        +------------+------------+------------+------------+-----------+
        |   time     | feature_1  | feature_2  | feature_3  |  target   |
        +============+============+============+============+===========+
        | 2023-01-01 |   0.15     |   0.67     |   0.89     |   0.33    |
        +------------+------------+------------+------------+-----------+
        | 2023-01-02 |   0.24     |   0.41     |   0.92     |   0.28    |
        +------------+------------+------------+------------+-----------+

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
    """
    # Validate the backend and mode
    validate_backend(backend)
    validate_mode(backend, mode)

    # Check if multi-step mode is supported for the backend
    if mode == MODE_MULTI_STEP and backend not in SUPPORTED_MULTI_STEP_BACKENDS:
        raise ValueError(f"Multi-step mode is not supported for the '{backend}' backend.")

    if seed is not None:
        np.random.seed(seed)

    # Generate feature columns
    data = {f"feature_{i+1}": np.random.rand(num_samples) for i in range(num_features)}

    # Insert NaNs and nulls if required
    if with_nans:
        for i in range(0, num_samples, nan_interval):
            data["feature_2"][i] = np.nan
    if with_nulls:
        for i in range(0, num_samples, null_interval):
            data["feature_3"][i] = None

    # Handle timestamp-like or numeric columns
    if timestamp_like and numeric:
        raise ValueError("Cannot have both 'timestamp_like' and 'numeric' time columns.")

    if timestamp_like:
        data["time"] = pd.date_range("2023-01-01", periods=num_samples, freq="D")
    elif numeric:
        data["time"] = np.arange(num_samples, dtype=np.float64)

    # Handle mixed frequencies or timezones
    if mixed_frequencies:
        data["time"] = (
            pd.date_range("2023-01-01", periods=num_samples // 2, freq="D").tolist()
            + pd.date_range("2023-02-01", periods=num_samples // 2, freq="M").tolist()
        )

    if mixed_timezones:
        time_with_timezones = (
            pd.date_range("2023-01-01", periods=num_samples // 2, freq="D").tz_localize("UTC").tolist()
        )
        time_without_timezones = pd.date_range("2023-01-01", periods=num_samples // 2, freq="D").tolist()
        data["time"] = time_with_timezones + time_without_timezones

    # Generate target based on the mode
    if mode == MODE_SINGLE_STEP:
        data["target"] = np.random.rand(num_samples)
    elif mode == MODE_MULTI_STEP:
        data["target"] = np.array([np.random.rand(10) for _ in range(num_samples)])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Create the DataFrame using Pandas
    df = pd.DataFrame(data)

    # Convert to the specified backend if required
    if backend != BACKEND_PANDAS:
        df = validate_and_convert_input(df, backend)

    return df


@pytest.fixture
def sample_df_with_conditions() -> Callable[..., Tuple[SupportedBackendDataFrame, str]]:
    """Pytest fixture for creating DataFrames for each backend (Pandas, Modin, Polars) with customizable conditions.

    This function generates synthetic data using Pandas and leaves the conversion to the backend
    to be handled by the centralized `validate_and_convert_input` function.

    :return:
        A function that generates a DataFrame and the backend type based on user-specified conditions.
    :rtype: Callable[..., Tuple[Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], str]]

    .. example::

        .. code-block:: python

            df, backend = sample_df_with_conditions(backend="pd", with_nulls=True)
            assert df.isnull().sum().sum() > 0  # Ensure nulls are present

            df, backend = sample_df_with_conditions(backend="mpd", mode="multi_step")
            assert isinstance(df["target"][0], list)  # Multi-step mode returns sequences

    """

    def _create_sample_df(backend: Optional[str] = None, **kwargs: Any) -> Tuple[SupportedBackendDataFrame, str]:
        """Internal helper function to create a sample DataFrame based on the specified backend and options.

        This function generates the data with Pandas and leaves backend conversion to `validate_and_convert_input`.

        :param backend:
            The desired backend ('pd', 'mpd', or 'pl').
        :type backend: Optional[str]
        :param kwargs:
            Additional options for creating the sample data (e.g., with_nans, timestamp_like).
        :type kwargs: dict
        :return:
            A tuple containing the generated DataFrame and the backend type.
        :rtype: Tuple[Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], str]
        """
        # Assign a default backend if none is provided
        backend = backend or BACKEND_PANDAS

        # Generate the sample data using Pandas
        df = create_sample_data(backend=BACKEND_PANDAS, **kwargs)

        # Return the DataFrame and the provided backend
        return df, backend

    return _create_sample_df
