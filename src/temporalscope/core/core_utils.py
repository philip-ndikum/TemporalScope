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

"""TemporalScope/src/temporalscope/core/core_utils.py.

This module provides essential utility functions for the TemporalScope package,
including support for:
- Backend validation (Pandas, Modin, Polars).
- Checking for nulls, NaNs, and handling mixed frequency issues in time series
  data.
- Managing different modes (Single-step vs. Multi-step) for machine learning and
  deep learning workflows.

Engineering Design Assumptions:
-------------------------------
TemporalScope is designed around two core modes for time series workflows, based
on the assumption that users handle their own preprocessing (e.g., managing NaNs,
encoding categorical variables, scaling, etc.).

These modes represent generalized structures that support both machine learning
and deep learning tasks, giving users the flexibility to manage their own
model-building workflows.

1. Single-step mode:
   - In this mode, each row of the data corresponds to a single time step with a
     scalar target.
   - Compatible with traditional machine learning frameworks (e.g., Scikit-learn,
     XGBoost) as well as deep learning libraries like TensorFlow and PyTorch.
   - The data is structured as a single DataFrame, where each row is an
     observation, and the target is scalar.
   - Example workflows include regression, classification, and survival models.
   - TemporalScope allows simple shifting/lagging of variables within this mode.
   - After partitioning (e.g., using a sliding window), users can convert the
     data into the required format for their model.

2. Multi-step mode:
   - This mode supports tasks like sequence forecasting, where the target is a
     sequence of values (multi-step).
   - Data is split into two DataFrames: one for the input sequences (X)
     and one for the target sequences (Y).
   - This mode is most commonly used in deep learning frameworks such as
     TensorFlow and PyTorch, where the task involves predicting sequences of
     time steps (e.g., seq2seq models).
   - TemporalScope's partitioning algorithms (e.g., sliding window) can
     partition this data for time series forecasting, making it ready for
     sequential models.

Core Purpose:
-------------
TemporalScope provides utility support for popular APIs such as TensorFlow,
PyTorch, Keras, and model-agnostic explainability tools (SHAP, Boruta-SHAP, LIME).
These utilities allow TemporalScope to fit seamlessly into machine learning and
deep learning workflows, while providing model-agnostic insights.

Supported Modes:
----------------
The following table illustrates the two core modes supported by TemporalScope.
These are generalized super-structures for time series tasks. Users are expected
to customize their workflows for specific model-building tasks (e.g., tree-based
models, neural networks, etc.):

+----------------+-------------------------------------------------------------------+
| Mode           | Description                                                       |
|                | Data Structure                                                    |
+----------------+-------------------------------------------------------------------+
| single_step    | General machine learning tasks with scalar targets. Each row is   |
|                | a single time step, and the target is scalar.                     |
|                | Single DataFrame: each row is an observation.                     |
|                | Frameworks: Scikit-learn, XGBoost, TensorFlow, PyTorch, etc.      |
+----------------+-------------------------------------------------------------------+
| multi_step     | Sequential time series tasks (e.g., seq2seq) for deep learning.   |
|                | The data is split into sequences (input X, target Y).             |
|                | Two DataFrames: X for input sequences, Y for targets.             |
|                | Frameworks: TensorFlow, PyTorch, Keras.                           |
+----------------+-------------------------------------------------------------------+

.. note::

   The table above is illustrative of common time series workflows that are
   supported by machine learning and deep learning frameworks. Users will need
   to manage their own data preprocessing (e.g., handling NaNs, scaling features,
   encoding categorical variables) to ensure compatibility with these frameworks.

   TemporalScope provides tools for integrating popular model-agnostic
   explainability techniques such as SHAP, Boruta-SHAP, and LIME, allowing users
   to extract insights from any type of model.
"""

import os
from typing import Dict, Optional, Union, cast, Callable, Type
from datetime import datetime, timedelta, date
import warnings

import modin.pandas as mpd
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from temporalscope.core.exceptions import UnsupportedBackendError, MixedFrequencyWarning

# Load environment variables from the .env file
load_dotenv()

# Backend abbreviations
BACKEND_POLARS = "pl"
BACKEND_PANDAS = "pd"
BACKEND_MODIN = "mpd"

# Modes for TemporalScope
MODE_SINGLE_STEP = "single_step"
MODE_MULTI_STEP = "multi_step"

# Mapping of backend keys to their full names or module references
BACKENDS = {
    BACKEND_POLARS: "polars",
    BACKEND_PANDAS: "pandas",
    BACKEND_MODIN: "modin",
}

TF_DEFAULT_CFG = {
    "BACKENDS": BACKENDS,
}

SUPPORTED_MULTI_STEP_BACKENDS = [BACKEND_PANDAS]

# Define a type alias for DataFrames that support Pandas, Modin, and Polars backends
SupportedBackendDataFrame = Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]


def get_default_backend_cfg() -> Dict[str, Dict[str, str]]:
    """Retrieve the application configuration settings.

    :return: A dictionary of configuration settings.
    :rtype: Dict[str, Dict[str, str]]
    """
    return TF_DEFAULT_CFG.copy()


def validate_backend(backend: str) -> None:
    """Validate the backend against the supported backends in the configuration.

    :param backend: The backend to validate ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :raises UnsupportedBackendError: If the backend is not supported.
    """
    if backend not in TF_DEFAULT_CFG["BACKENDS"]:
        raise UnsupportedBackendError(backend)


def infer_backend_from_dataframe(df: SupportedBackendDataFrame) -> str:
    """Infer the backend from the DataFrame type.

    :param df: The input DataFrame.
    :type df: SupportedBackendDataFrame
    :return: The inferred backend ('pl', 'pd', or 'mpd').
    :rtype: str
    :raises UnsupportedBackendError: If the DataFrame type is unsupported.
    """
    if isinstance(df, pl.DataFrame):
        return BACKEND_POLARS
    elif isinstance(df, pd.DataFrame):
        return BACKEND_PANDAS
    elif isinstance(df, mpd.DataFrame):
        return BACKEND_MODIN
    else:
        raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df)}")


def validate_mode(backend: str, mode: str) -> None:
    """Validate if the backend supports the given mode.

    :param backend: The backend type ('pl', 'pd', or 'mpd').
    :param mode: The mode type ('single_step' or 'multi_step').
    :raises NotImplementedError: If the backend does not support the requested mode.
    """
    if mode == MODE_MULTI_STEP and backend not in SUPPORTED_MULTI_STEP_BACKENDS:
        raise NotImplementedError(f"The '{backend}' backend does not support multi-step mode.")


def validate_and_convert_input(
    df: SupportedBackendDataFrame,
    backend: str,
    time_col: Optional[str] = None,
    mode: str = MODE_SINGLE_STEP
) -> SupportedBackendDataFrame:
    """Validates and converts the input DataFrame to the specified backend type, with optional time column casting.

    :param df: The input DataFrame to validate and convert.
    :param backend: The desired backend type ('pl', 'pd', or 'mpd').
    :param time_col: Optional; the name of the time column for casting.
    :param mode: The processing mode ('single_step' or 'multi_step').
    :raises TypeError: If input DataFrame type doesn't match the specified backend or conversion fails.
    :raises NotImplementedError: If multi-step mode is requested for unsupported backends or unsupported conversion to Polars.
    :return: The DataFrame converted to the specified backend type.

    Example
    -------
    Here's how you would use this function to convert a Pandas DataFrame to Polars:

        .. code-block:: python

            import pandas as pd
            import polars as pl

            data = {'col1': [1, 2], 'col2': [3, 4], 'time': pd.date_range(start='1/1/2023', periods=2)}
            df = pd.DataFrame(data)

            # Convert the DataFrame from Pandas to Polars, with an optional time column for casting
            converted_df = validate_and_convert_input(df, 'pl', time_col='time')
            print(type(converted_df))  # Output: <class 'polars.DataFrame'>

            # If you don't need to cast the time column, just omit the time_col argument
            converted_df = validate_and_convert_input(df, 'pl')
            print(type(converted_df))  # Output: <class 'polars.DataFrame'>

    .. note::
        - This function first converts the input DataFrame into the appropriate backend.
        - If `time_col` is specified and the backend is Polars, it casts the time column to `pl.Datetime`.
        - Pandas to Polars conversion is currently unsupported and raises a `NotImplementedError`. This needs to be implemented later.
    """
    # Validate the backend and mode combination
    validate_backend(backend)
    validate_mode(backend, mode)

    # Backend conversion map
    backend_conversion_map: Dict[
        str, Dict[Type[SupportedBackendDataFrame], Callable[[SupportedBackendDataFrame], SupportedBackendDataFrame]]
    ] = {
        BACKEND_POLARS: {
            # Polars to Polars
            pl.DataFrame: lambda x: x,
            # Pandas to Polars - currently not supported
            pd.DataFrame: lambda x: (_ for _ in ()).throw(NotImplementedError("Pandas to Polars conversion is not currently supported.")),
            # Modin to Polars
            mpd.DataFrame: lambda x: pl.from_pandas(x._to_pandas()),
        },
        BACKEND_PANDAS: {
            pd.DataFrame: lambda x: x,  # Pandas to Pandas
            pl.DataFrame: lambda x: x.to_pandas(),  # Polars to Pandas
            mpd.DataFrame: lambda x: x._to_pandas() if hasattr(x, "_to_pandas") else x,  # Modin to Pandas
        },
        BACKEND_MODIN: {
            mpd.DataFrame: lambda x: x,  # Modin to Modin
            pd.DataFrame: lambda x: mpd.DataFrame(x),  # Pandas to Modin
            pl.DataFrame: lambda x: mpd.DataFrame(x.to_pandas()),  # Polars to Modin via Pandas
        },
    }

    # Step 1: Convert the DataFrame to the desired backend
    converted_df = None
    for dataframe_type, conversion_func in backend_conversion_map[backend].items():
        if isinstance(df, dataframe_type):
            converted_df = conversion_func(df)
            break

    if converted_df is None:
        raise TypeError(f"Input DataFrame type {type(df)} does not match the specified backend '{backend}'")

    # Step 2: Explicitly cast the time column to pl.Datetime if backend is Polars and the column exists
    if backend == BACKEND_POLARS and time_col and time_col in converted_df.columns:
        # Force cast time_col to pl.Datetime
        converted_df = converted_df.with_columns(pl.col(time_col).cast(pl.Datetime))

        # Check the type of the column and assert it is correct
        assert isinstance(converted_df[time_col][0], pl.Datetime), f"Expected a timestamp-like time column, but got {type(converted_df[time_col][0])}"

    return converted_df



def get_api_keys() -> Dict[str, Optional[str]]:
    """Retrieve API keys from environment variables.

    :return: A dictionary containing the API keys, or None if not found.
    :rtype: Dict[str, Optional[str]]
    """
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY"),
    }

    # Print warnings if keys are missing
    for key, value in api_keys.items():
        if value is None:
            print(f"Warning: {key} is not set in the environment variables.")

    return api_keys


def print_divider(char: str = "=", length: int = 70) -> None:
    """Prints a divider line made of a specified character and length.

    :param char: The character to use for the divider, defaults to '='
    :type char: str, optional
    :param length: The length of the divider, defaults to 70
    :type length: int, optional
    """
    print(char * length)


def check_nulls(df: SupportedBackendDataFrame, backend: str) -> bool:
    """Check for null values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for null values.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for the DataFrame ('pl', 'pd', 'mpd').
    :type backend: str
    :return: True if there are null values, False otherwise.
    :rtype: bool
    :raises UnsupportedBackendError: If the backend is not supported.
    """
    validate_backend(backend)

    if backend == BACKEND_PANDAS:
        return bool(cast(pd.DataFrame, df).isnull().values.any())
    elif backend == BACKEND_POLARS:
        null_count = cast(pl.DataFrame, df).null_count().select(pl.col("*").sum()).to_numpy().sum()
        return bool(null_count > 0)
    elif backend == BACKEND_MODIN:
        return bool(cast(mpd.DataFrame, df).isnull().values.any())

    # Suppress the warning since this path is unreachable due to `validate_backend`
    # mypy: ignore


def check_nans(df: SupportedBackendDataFrame, backend: str) -> bool:
    """Check for NaN values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for NaN values.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for the DataFrame ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin').
    :type backend: str
    :return: True if there are NaN values, False otherwise.
    :rtype: bool
    :raises UnsupportedBackendError: If the backend is not supported.
    """
    validate_backend(backend)

    if backend == BACKEND_PANDAS:
        return bool(cast(pd.DataFrame, df).isna().values.any())
    elif backend == BACKEND_POLARS:
        nan_count = cast(pl.DataFrame, df).select(pl.col("*").is_nan().sum()).to_numpy().sum()
        return bool(nan_count > 0)
    elif backend == BACKEND_MODIN:
        return bool(cast(mpd.DataFrame, df).isna().values.any())

    # Suppress the warning since this path is unreachable due to `validate_backend`
    # mypy: ignore


def is_timestamp_like(df: SupportedBackendDataFrame, time_col: str) -> bool:
    """Check if the specified column in the DataFrame is timestamp-like.

    This function can be used in the context of time series modeling to
    validate that the time column is in an appropriate format for further
    temporal operations such as sorting or windowing.

    This function assumes that the DataFrame has been pre-validated to ensure
    it is using a supported backend.

    :param df: The DataFrame containing the time column.
    :type df: SupportedBackendDataFrame
    :param time_col: The name of the column representing time data.
    :type time_col: str
    :return: True if the column is timestamp-like, otherwise False.
    :rtype: bool
    :raises ValueError: If the time_col does not exist in the DataFrame.

    .. note::
        This function is primarily used for warning users if the time column is not
        timestamp-like, but the final decision on how to handle this rests with the user.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in the DataFrame.")

    time_column = df[time_col]

    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        return pd.api.types.is_datetime64_any_dtype(time_column)
    elif isinstance(df, pl.DataFrame):
        return time_column.dtype == pl.Datetime


def is_numeric(df: SupportedBackendDataFrame, time_col: str) -> bool:
    """Check if the specified column in the DataFrame is numeric.

    :param df: The DataFrame containing the time column.
    :type df: SupportedBackendDataFrame
    :param time_col: The name of the column representing time data.
    :type time_col: str
    :return: True if the column is numeric, otherwise False.
    :rtype: bool
    :raises ValueError: If the time_col does not exist in the DataFrame.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in the DataFrame.")

    time_column = df[time_col]

    # Handle empty columns for different backends
    if isinstance(df, pl.DataFrame):
        # Polars: Check if the DataFrame has zero rows or if the column is empty
        if df.height == 0 or time_column.is_empty():
            return False
    elif isinstance(df, mpd.DataFrame):
        # Modin: Check if the column is empty by using length
        if len(time_column) == 0:
            return False
    elif isinstance(df, pd.DataFrame):
        # Pandas: Check if the column is empty
        if isinstance(time_column, pd.Series) and time_column.empty:
            return False

    # Check if the column is numeric based on the backend
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        return pd.api.types.is_numeric_dtype(time_column)
    elif isinstance(df, pl.DataFrame):
        return time_column.dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]


def has_mixed_frequencies(df: SupportedBackendDataFrame, time_col: str, min_non_null_values: int = 3) -> bool:
    """Check if the given time column in the DataFrame contains mixed frequencies.

    This function is essential in time series data, as mixed frequencies (e.g., a mix of daily
    and monthly data) can lead to inconsistent modeling outcomes. While some models may handle
    mixed frequencies, others might struggle with this data structure.

    The function ensures that a minimum number of non-null values are present
    before inferring the frequency to avoid issues with small datasets.

    :param df: The DataFrame containing the time column.
    :type df: SupportedBackendDataFrame
    :param time_col: The name of the column representing time data.
    :type time_col: str
    :param min_non_null_values: The minimum number of non-null values required to infer a frequency.
                                Default is 3, which ensures enough data points for frequency inference.
    :type min_non_null_values: int
    :return: True if mixed frequencies are detected, otherwise False.
    :rtype: bool
    :raises ValueError: If the time_col does not exist in the DataFrame.
    :raises UnsupportedBackendError: If the DataFrame is from an unsupported backend.
    :raises MixedFrequencyWarning: If mixed timestamp frequencies are detected.

    .. warning::
        If mixed frequencies are detected, the user should be aware of potential issues in modeling. This function
        will raise a warning but not prevent further operations, leaving it up to the user to handle.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in the DataFrame.")

    # Drop null values in the time column
    if isinstance(df, pl.DataFrame):
        time_column = df[time_col].drop_nulls()
    else:
        time_column = df[time_col].dropna()

    # Ensure there are at least min_non_null_values non-null values to infer frequency
    if len(time_column) < min_non_null_values:
        return False

    # Infer frequency depending on backend
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        inferred_freq = pd.infer_freq(time_column)
    elif isinstance(df, pl.DataFrame):
        inferred_freq = pd.infer_freq(time_column.to_pandas())

    if inferred_freq is None:
        warnings.warn("Mixed timestamp frequencies detected in the time column.", MixedFrequencyWarning)
        return True
    return False


def sort_dataframe(
    df: SupportedBackendDataFrame, time_col: str, backend: str, ascending: bool = True
) -> SupportedBackendDataFrame:
    """Sorts a DataFrame by the specified time column based on the backend.

    :param df: The DataFrame to be sorted.
    :type df: SupportedBackendDataFrame
    :param time_col: The name of the column to sort by.
    :type time_col: str
    :param backend: The backend used for the DataFrame ('pl', 'pd', or 'mpd').
    :type backend: str
    :param ascending: If True, sort in ascending order; if False, sort in descending order. Default is True.
    :type ascending: bool
    :return: The sorted DataFrame.
    :rtype: SupportedBackendDataFrame
    :raises TypeError: If the DataFrame type does not match the backend.
    :raises UnsupportedBackendError: If the backend is unsupported or validation fails.
    """
    # Validate backend
    validate_backend(backend)

    # Select backend-specific sorting logic
    if backend == BACKEND_POLARS:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame but got {type(df)}")
        return df.sort(by=time_col, descending=not ascending)

    elif backend == BACKEND_PANDAS:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected Pandas DataFrame but got {type(df)}")
        df.sort_values(by=time_col, ascending=ascending, inplace=True)
        return df

    elif backend == BACKEND_MODIN:
        if not isinstance(df, mpd.DataFrame):
            raise TypeError(f"Expected Modin DataFrame but got {type(df)}")
        df.sort_values(by=time_col, ascending=ascending, inplace=True)
        return df


def check_empty_columns(df: SupportedBackendDataFrame, backend: str) -> bool:
    """Check for empty columns in the DataFrame using the specified backend.

    This function ensures that none of the columns in the DataFrame are effectively empty
    (i.e., they contain only NaN or None values or are entirely empty).
    It returns True if any column is found to be effectively empty, and False otherwise.

    :param df: The DataFrame to check for empty columns.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for the DataFrame ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :return: True if there are any effectively empty columns, False otherwise.
    :rtype: bool
    :raises UnsupportedBackendError: If the backend is not supported.
    :raises ValueError: If the DataFrame does not contain columns.
    """
    # Validate the backend
    validate_backend(backend)

    # Check for columns in the DataFrame
    if df.shape[1] == 0:
        raise ValueError("The DataFrame contains no columns to check.")

    # Define backend-specific logic for checking empty columns
    if backend == BACKEND_PANDAS:
        if any(cast(pd.DataFrame, df)[col].isnull().all() for col in df.columns):
            return True
    elif backend == BACKEND_POLARS:
        if any(cast(pl.DataFrame, df)[col].null_count() == len(df) for col in df.columns):
            return True
    elif backend == BACKEND_MODIN:
        if any(cast(mpd.DataFrame, df)[col].isnull().all() for col in df.columns):
            return True

    # If no empty columns were found, return False
    return False
