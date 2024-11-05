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
import importlib
from typing import Dict, List, Union, Type, Set, Optional, Any
from dotenv import load_dotenv

import narwhals as nw
from narwhals.utils import Implementation
from narwhals.typing import FrameT  # Import FrameT from Narwhals for unified type hinting
from temporalscope.core.exceptions import UnsupportedBackendError, MixedFrequencyWarning

import pandas as pd
import modin.pandas as mpd
import pyarrow as pa
import polars as pl
import dask.dataframe as dd

# Load environment variables from the .env file
load_dotenv()

# Define constants for TemporalScope-supported modes
MODE_SINGLE_STEP = "single_step"
MODE_MULTI_STEP = "multi_step"
VALID_MODES = [MODE_SINGLE_STEP, MODE_MULTI_STEP]

# Backend constants for TemporalScope
TEMPORALSCOPE_CORE_BACKENDS = {"pandas", "modin", "pyarrow", "polars", "dask"}
# TODO: Add optional backend "cudf" when Conda setup is confirmed
TEMPORALSCOPE_OPTIONAL_BACKENDS = {"cudf"}

# Define a type alias combining Narwhals' FrameT with the supported TemporalScope dataframes
SupportedTemporalDataFrame = Union[FrameT, pd.DataFrame, mpd.DataFrame, pa.Table, pl.DataFrame, dd.DataFrame]

# Backend type classes for TemporalScope backends
TEMPORALSCOPE_CORE_BACKEND_TYPES: Dict[str, Type] = {
    "pandas": pd.DataFrame,
    "modin": mpd.DataFrame,
    "pyarrow": pa.Table,
    "polars": pl.DataFrame,
    "dask": dd.DataFrame,
}


def get_narwhals_backends() -> List[str]:
    """Retrieve all backends available through Narwhals.

    :return: List of Narwhals-supported backend names in lowercase.
    :rtype: List[str]
    """
    return [backend.name.lower() for backend in Implementation]


def get_default_backend_cfg() -> Dict[str, List[str]]:
    """Retrieve the default application configuration for available backends.

    :return: Dictionary with a single key 'BACKENDS' containing a list of all Narwhals-supported backends.
    :rtype: Dict[str, List[str]]
    """
    available_backends = get_narwhals_backends()
    return {"BACKENDS": available_backends}


def get_temporalscope_backends() -> List[str]:
    """Retrieve the subset of Narwhals-supported backends compatible with TemporalScope.

    :return: List of backend names supported by TemporalScope.
    :rtype: List[str]
    """
    available_backends = get_narwhals_backends()
    return [backend for backend in available_backends if backend in TEMPORALSCOPE_CORE_BACKENDS]


def validate_backend(backend_name: str) -> None:
    """Validate that a backend is supported by TemporalScope and Narwhals.

    :param backend_name: Name of the backend to validate.
    :type backend_name: str
    :raises UnsupportedBackendError: If the backend is not in supported or optional backends.
    :raises UserWarning: If the backend is in the optional set but not installed.
    """
    # Assume TEMPORALSCOPE_CORE_BACKENDS and TEMPORALSCOPE_OPTIONAL_BACKENDS are sets
    available_backends = TEMPORALSCOPE_CORE_BACKENDS | TEMPORALSCOPE_OPTIONAL_BACKENDS

    if backend_name in available_backends:
        if backend_name in TEMPORALSCOPE_OPTIONAL_BACKENDS:
            # Check if the optional backend is installed
            if importlib.util.find_spec(backend_name) is None:
                warnings.warn(
                    f"The '{backend_name}' backend is optional and requires additional setup. "
                    f"Please install it (e.g., using Conda).",
                    UserWarning,
                )
        return
    else:
        raise UnsupportedBackendError(
            f"Backend '{backend_name}' is not supported by TemporalScope. "
            f"Supported backends are: {', '.join(sorted(available_backends))}."
        )


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


def convert_to_backend(df: pd.DataFrame, backend: str, npartitions: int = 1) -> Any:
    """Convert a Pandas DataFrame to the specified backend format.

    :param df: The DataFrame to convert.
    :type df: pd.DataFrame
    :param backend: The target backend ('pandas', 'modin', 'polars', 'pyarrow', 'dask').
    :type backend: str
    :param npartitions: Number of partitions to use if backend is Dask. Default is 1.
    :type npartitions: int
    :return: The DataFrame in the specified backend format.
    :rtype: Backend-specific DataFrame type
    :raises ValueError: If the backend is not supported or implemented.
    """
    if backend == "pandas":
        return df
    elif backend == "modin":
        return mpd.DataFrame(df)
    elif backend == "polars":
        return pl.from_pandas(df)
    elif backend == "pyarrow":
        return pa.Table.from_pandas(df)
    elif backend == "dask":
        return dd.from_pandas(df, npartitions=npartitions)
    else:
        raise ValueError(f"Backend '{backend}' is not supported or has not been implemented.")
