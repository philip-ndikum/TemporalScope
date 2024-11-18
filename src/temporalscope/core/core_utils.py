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
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations under the License.

"""TemporalScope/src/temporalscope/core/core_utils.py.

This module provides essential utility functions for the TemporalScope package,
including support for:
- Backend validation (Narwhals).
- Checking for nulls, NaNs, and handling mixed frequency issues in time series
  data.
- Managing different modes (Single-step vs. Multi-step) for machine learning and
  deep learning workflows.

Type safety and validation follow Linux Foundation security standards to ensure
robust backend interoperability.

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

.. seealso::

Example Visualization:
----------------------
Here is a visual demonstration of the datasets generated for single-step and multi-step
modes, including the shape of input (`X`) and target (`Y`) data compatible with most
popular ML frameworks like TensorFlow, PyTorch, and SHAP.

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

DataFrame Types:
----------------
TemporalScope handles various DataFrame types throughout the data processing pipeline. The following table
illustrates the supported DataFrame types and validation cases:

+------------------------+-------------------------------------------------------+---------------------------+
| DataFrame Type         | Description                                           | Example                   |
+------------------------+-------------------------------------------------------+---------------------------+
| Narwhalified          | DataFrames wrapped by Narwhals for backend-agnostic   | @nw.narwhalify decorated   |
| (FrameT)              | operations. These are validated first to ensure        | functions create these    |
|                       | consistent handling across backends.                   | during operations.        |
+------------------------+-------------------------------------------------------+---------------------------+
| Native DataFrames     | Core DataFrame implementations from supported          | pd.DataFrame,             |
|                       | backends. These are validated directly against         | pl.DataFrame,             |
|                       | TEMPORALSCOPE_CORE_BACKEND_TYPES.                      | pa.Table                  |
+------------------------+-------------------------------------------------------+---------------------------+
| DataFrame Subclasses  | Custom or specialized DataFrames that inherit from     | TimeSeriesDataFrame       |
|                       | native types. Common in:                               | (pd.DataFrame),           |
|                       | - Custom DataFrame implementations                     | dask.dataframe.DataFrame  |
|                       | - Backend optimizations (e.g. lazy evaluation)         | (inherits from pandas)    |
|                       | - Backend compatibility layers                         |                           |
+------------------------+-------------------------------------------------------+---------------------------+
| Intermediate States   | DataFrames in the middle of narwhalify operations      | LazyDataFrame during      |
|                       | or backend conversions. These may be temporary         | backend conversion        |
|                       | subclasses used for optimization or compatibility.     | operation chaining        |
+------------------------+-------------------------------------------------------+---------------------------+

.. note::
   The validation system (is_valid_temporal_dataframe) handles all these cases to ensure consistent
   behavior across the entire data processing pipeline. This is particularly important when:
   - Converting between backends
   - Applying narwhalified operations
   - Working with custom DataFrame implementations
   - Handling intermediate states during operations
"""

# Standard Library Imports
import os
import warnings
from importlib import util
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dask.dataframe as dd  # Import dask.dataframe as dd again
import modin.pandas as mpd

# Narwhals Imports
import narwhals as nw

# Third-Party Imports
import pandas as pd
import polars as pl
import pyarrow as pa

# Loading Environment variables
from dotenv import load_dotenv
from narwhals.typing import FrameT, IntoDataFrame
from narwhals.utils import Implementation

# TemporalScope Imports
from temporalscope.core.exceptions import UnsupportedBackendError

# Load environment variables from the .env file
load_dotenv()

# Constants
# ---------
# Define constants for TemporalScope-supported modes
MODE_SINGLE_STEP = "single_step"
MODE_MULTI_STEP = "multi_step"
VALID_MODES = [MODE_SINGLE_STEP, MODE_MULTI_STEP]

# Backend constants for TemporalScope
TEMPORALSCOPE_CORE_BACKENDS = {"pandas", "modin", "pyarrow", "polars", "dask"}
# TODO: Add optional backend "cudf" when Conda setup is confirmed
TEMPORALSCOPE_OPTIONAL_BACKENDS = {"cudf"}

# Define a type alias combining Narwhals' FrameT with the supported TemporalScope dataframes
SupportedTemporalDataFrame = Union[
    FrameT,  # narwhals.typing.FrameT - narwhals DataFrame wrapper
    pd.DataFrame,  # pandas.core.frame.DataFrame - actual pandas DataFrame class
    mpd.DataFrame,  # modin.pandas.dataframe.DataFrame - actual modin DataFrame class
    pa.Table,  # pyarrow.lib.Table - actual pyarrow Table class
    pl.DataFrame,  # polars.dataframe.frame.DataFrame - actual polars DataFrame class
    dd.DataFrame,  # dask.dataframe.core.DataFrame - actual dask DataFrame class
]

# Backend type classes for TemporalScope backends
TEMPORALSCOPE_CORE_BACKEND_TYPES: Dict[str, Type] = {
    "pandas": pd.DataFrame,  # Main pandas DataFrame class
    "modin": mpd.DataFrame,  # Main modin DataFrame class
    "pyarrow": pa.Table,  # Main pyarrow Table class
    "polars": pl.DataFrame,  # Main polars DataFrame class
    "dask": dd.DataFrame,  # Main dask DataFrame class
}

# Module paths for DataFrame validation
TEMPORALSCOPE_MODULE_PATHS: Dict[str, Union[str, Tuple[str, ...]]] = {
    "pandas": "pandas.core.frame",
    "modin": "modin.pandas.dataframe",
    "pyarrow": "pyarrow.lib",
    "polars": "polars.dataframe.frame",
    "dask": ("dask.dataframe.core", "dask_expr._collection"),
}

# Backend converters for DataFrame conversion
TEMPORALSCOPE_BACKEND_CONVERTERS: Dict[str, Callable[[pd.DataFrame, int], Any]] = {
    "pandas": lambda df, _: df,
    "modin": lambda df, _: mpd.DataFrame(df),
    "polars": lambda df, _: pl.DataFrame(df),
    "pyarrow": lambda df, _: pa.Table.from_pandas(df),
    "dask": lambda df, npartitions: dd.from_pandas(df, npartitions=npartitions),
}

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------


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


# ---------------------------------------------------------
# Main Functions
# ---------------------------------------------------------


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


def is_valid_temporal_backend(backend_name: str) -> None:
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
            if util.find_spec(backend_name) is None:
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


def is_valid_temporal_dataframe(df: Union[SupportedTemporalDataFrame, Any]) -> Tuple[bool, Optional[str]]:
    """Validate that a DataFrame is supported by TemporalScope and Narwhals.

    Uses TEMPORALSCOPE_CORE_BACKEND_TYPES to validate actual DataFrame instances.
    Handles both native DataFrame types and narwhalified (FrameT) DataFrames.

    :param df: Object to validate, can be any supported DataFrame type or arbitrary object
    :type df: Union[SupportedTemporalDataFrame, Any]
    :return: Tuple of (is_valid, df_type) where df_type is:
            - "narwhals" for FrameT DataFrames
            - "native" for supported DataFrame types
            - None if not valid
    :rtype: Tuple[bool, Optional[str]]

    Example Usage:
    -------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.core.core_utils import is_valid_temporal_dataframe

        df = pd.DataFrame({"col": [1, 2, 3]})
        is_valid, df_type = is_valid_temporal_dataframe(df)
        print(is_valid)  # True
        print(df_type)  # "native"

    .. note::
        Complements is_valid_temporal_backend which works with strings.
        This works with actual DataFrame instances and provides type information
        to avoid duplicate validation logic across the codebase.

        DataFrame validation is performed by checking module paths rather than using isinstance,
        as this is more reliable across different versions and implementations:
        - pandas: pandas.core.frame
        - modin: modin.pandas.dataframe
        - pyarrow: pyarrow.lib
        - polars: polars.dataframe.frame
        - dask: dask.dataframe.core, dask_expr._collection
    """
    try:
        # Case 1: Narwhalified DataFrames
        if hasattr(df, "__class__") and df.__class__.__module__ == "narwhals.dataframe":
            return True, "narwhals"

        # Case 2: Native DataFrames - check module path
        df_type = type(df)
        df_module = df_type.__module__

        # Check module path against all possible paths
        module_paths = []
        for paths in TEMPORALSCOPE_MODULE_PATHS.values():
            if isinstance(paths, str):
                module_paths.append(paths)
            else:
                module_paths.extend(paths)

        if df_module in module_paths:
            return True, "native"

        # Case 3: Intermediate States - check convertibility
        if any(hasattr(df, attr) for attr in ("to_pandas", "__array__", "to_numpy")):
            return True, "native"

        return False, None
    except Exception:
        return False, None


def convert_to_backend(
    df: Union[SupportedTemporalDataFrame, IntoDataFrame], backend: str, npartitions: int = 1
) -> SupportedTemporalDataFrame:
    """Convert a DataFrame to the specified backend format using Narwhals.

    Primary purpose is handling DataFrame conversions through Narwhals. This function
    ensures consistent backend handling by prioritizing narwhalified and native DataFrame
    operations.

    :param df: Input DataFrame in any supported format (pandas, modin, polars, pyarrow, dask)
    :type df: Union[SupportedTemporalDataFrame, IntoDataFrame]
    :param backend: Target backend ('pandas', 'modin', 'polars', 'pyarrow', 'dask').
    :type backend: str
    :param npartitions: Number of partitions for Dask backend. Default is 1.
    :type npartitions: int
    :return: DataFrame in the specified backend format.
    :rtype: IntoDataFrame
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope or if the DataFrame type is unsupported.

    Example Usage:
    --------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.core.core_utils import convert_to_backend

        # Primary case: Converting narwhalified DataFrame
        df_narwhals = nw.from_native(data)
        df_polars = convert_to_backend(df_narwhals, backend="polars")

        # Secondary case: Converting native DataFrame
        data = pd.DataFrame({"time": range(10), "value": range(10)})
        df_modin = convert_to_backend(data, backend="modin")

    .. note::
        While the DataFrame Interchange Protocol (DIP) aims to standardize DataFrame
        conversions, current implementations still require intermediate conversion steps.
        This function uses pandas as an interoperability layer due to its widespread
        support and reliable conversion methods across different DataFrame libraries.

        The conversion process follows two main paths:

        1. Narwhalified DataFrames (Primary Path):
           - DataFrames that have been wrapped by @nw.narwhalify
           - Already backend-agnostic through Narwhals
           - Detected using is_valid_temporal_dataframe
           - Converted directly to target backend

        2. Native DataFrames (Secondary Path):
           - Standard DataFrames from supported backends
           - Require conversion through pandas as interop layer
           - Converted using TEMPORALSCOPE_BACKEND_CONVERTERS
           - Includes handling of LazyFrames (e.g., dask)

        This dual-path approach ensures optimal handling of both narwhalified and
        native DataFrames while maintaining consistent behavior across the data
        processing pipeline.
    """
    # Validate the backend
    is_valid_temporal_backend(backend)

    # First try to get a valid DataFrame
    is_valid, df_type = is_valid_temporal_dataframe(df)
    if not is_valid:
        # Try conversion methods in order
        if hasattr(df, "to_pandas"):
            intermediate_pd_df = df.to_pandas()
        elif hasattr(df, "__array__"):
            intermediate_pd_df = pd.DataFrame(df.__array__())
        elif hasattr(df, "to_numpy"):
            intermediate_pd_df = pd.DataFrame(df.to_numpy())
        else:
            raise UnsupportedBackendError(f"Input DataFrame type '{type(df).__name__}' is not supported")

        # Return pandas DataFrame directly if that's our target
        if backend == "pandas":
            return intermediate_pd_df

        # Otherwise use it as intermediate
        df = intermediate_pd_df

    try:
        # Handle special cases
        if df_type == "narwhals":
            df = df.to_native()

        if hasattr(df, "compute"):
            df = df.compute()
            # Re-validate after computation
            is_valid, df_type = is_valid_temporal_dataframe(df)
            if not is_valid:
                raise UnsupportedBackendError(f"Failed to compute {type(df).__name__}")

        # Convert through pandas using Narwhals
        intermediate_pd_df = nw.from_native(df).to_pandas()

        # Convert to target backend using converter map
        return TEMPORALSCOPE_BACKEND_CONVERTERS[backend](intermediate_pd_df, npartitions)

    except Exception as e:
        raise UnsupportedBackendError(f"Failed to convert DataFrame: {str(e)}")
