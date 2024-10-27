# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance
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
- Backend validation (Narwhals).
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
import warnings
from typing import Dict, Optional

import narwhals as nw
from dotenv import load_dotenv

from temporalscope.core.exceptions import MixedFrequencyWarning, UnsupportedBackendError

# Load environment variables from the .env file
load_dotenv()

MODE_SINGLE_STEP = "single_step"
MODE_MULTI_STEP = "multi_step"

# Define a type alias for Narwhals-compatible DataFrame backends
SupportedBackendDataFrame = nw.NarwhalDataFrame

def get_default_backend_cfg() -> Dict[str, Dict[str, str]]:
    """Retrieve the application configuration settings.

    :return: A dictionary of configuration settings.
    :rtype: Dict[str, Dict[str, str]]
    """
    return {"BACKENDS": nw.available_backends()}


def validate_mode(backend: str, mode: str) -> None:
    """Validate if the backend supports the given mode.

    :param backend: The backend type.
    :type backend: str
    :param mode: The mode type ('single_step' or 'multi_step').
    :raises NotImplementedError: If the backend does not support the requested mode.
    """
    supported_modes = nw.supported_modes(backend)
    if mode not in supported_modes:
        raise NotImplementedError(f"The '{backend}' backend does not support '{mode}' mode.")


def get_api_keys() -> Dict[str, Optional[str]]:
    """Retrieve API keys from environment variables.

    :return: A dictionary containing the API keys, or None if not found.
    :rtype: Dict[str, Optional[str]]
    """
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY"),
    }
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
    :param backend: The backend used for the DataFrame.
    :type backend: str
    :return: True if there are null values, False otherwise.
    :rtype: bool
    :raises UnsupportedBackendError: If the backend is not supported.
    """
    validate_backend(backend)
    return bool(nw.check_nulls(df))


def check_nans(df: SupportedBackendDataFrame, backend: str) -> bool:
    """Check for NaN values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for NaN values.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for the DataFrame.
    :type backend: str
    :return: True if there are NaN values, False otherwise.
    :rtype: bool
    :raises UnsupportedBackendError: If the backend is not supported.
    """
    validate_backend(backend)
    return bool(nw.check_nans(df))


def is_timestamp_like(df: SupportedBackendDataFrame, time_col: str) -> bool:
    """Check if the specified column in the DataFrame is timestamp-like.

    :param df: The DataFrame containing the time column.
    :type df: SupportedBackendDataFrame
    :param time_col: The name of the column representing time data.
    :type time_col: str
    :return: True if the column is timestamp-like, otherwise False.
    :rtype: bool
    :raises ValueError: If the time_col does not exist in the DataFrame.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in the DataFrame.")
    return nw.is_timestamp(df[time_col])


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
    return nw.is_numeric(df[time_col])


def has_mixed_frequencies(df: SupportedBackendDataFrame, time_col: str, min_non_null_values: int = 3) -> bool:
    """Check if the given time column in the DataFrame contains mixed frequencies.

    :param df: The DataFrame containing the time column.
    :type df: SupportedBackendDataFrame
    :param time_col: The name of the column representing time data.
    :type time_col: str
    :param min_non_null_values: Minimum number of non-null values for frequency inference.
    :type min_non_null_values: int
    :return: True if mixed frequencies are detected, otherwise False.
    :rtype: bool
    :raises ValueError: If the time_col does not exist in the DataFrame.
    :raises UnsupportedBackendError: If the backend is unsupported.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in the DataFrame.")
    mixed_freq = nw.has_mixed_frequencies(df[time_col], min_non_null_values)
    if mixed_freq:
        warnings.warn("Mixed timestamp frequencies detected in the time column.", MixedFrequencyWarning)
    return mixed_freq

