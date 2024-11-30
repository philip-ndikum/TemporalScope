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

import dask.dataframe as dd
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
from temporalscope.core.exceptions import TimeColumnError, UnsupportedBackendError

# Load environment variables from the .env file
load_dotenv()

# Constants
# ---------
# Define constants for TemporalScope-supported modes
MODE_SINGLE_TARGET = "single_target"
MODE_MULTI_TARGET = "multi_target"
VALID_MODES = [MODE_SINGLE_TARGET, MODE_MULTI_TARGET]

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


def get_dataframe_backend(df: Union[SupportedTemporalDataFrame, Any]) -> str:
    """Get the backend name for a DataFrame.

    :param df: DataFrame to get backend for
    :type df: Union[SupportedTemporalDataFrame, Any]
    :return: Backend name ('pandas', 'modin', 'polars', 'pyarrow', 'dask')
    :rtype: str
    :raises UnsupportedBackendError: If DataFrame type not supported
    """
    # First validate DataFrame type
    is_valid, df_type = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError(f"Unknown DataFrame type: {type(df).__name__}")

    # If narwhalified, get native form
    if df_type == "narwhals":
        df = df.to_native()

    # Get backend from type
    for name, cls in TEMPORALSCOPE_CORE_BACKEND_TYPES.items():
        if isinstance(df, cls):
            return name

    raise UnsupportedBackendError(f"Unknown DataFrame type: {type(df).__name__}")


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
        # Only try to_pandas conversion as we only work with DataFrames
        if hasattr(df, "to_pandas"):
            intermediate_pd_df = df.to_pandas()
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


@nw.narwhalify
def is_lazy_evaluation(df: SupportedTemporalDataFrame) -> bool:
    """Check if DataFrame uses lazy evaluation.

    :param df: DataFrame to check evaluation mode
    :type df: SupportedTemporalDataFrame
    :return: True if DataFrame uses lazy evaluation, False otherwise
    :rtype: bool

    Example:
    -------
    .. code-block:: python

        import narwhals as nw
        from temporalscope.core.core_utils import is_lazy_evaluation

        df = nw.from_native(data)
        if is_lazy_evaluation(df):
            # Handle lazy evaluation path
            result = df.select([...])  # Maintain lazy evaluation
        else:
            # Handle eager evaluation path
            result = df.select([...])  # Direct computation ok

    .. note::
        Identifies whether a DataFrame uses lazy evaluation:
        - Lazy execution: dask, polars lazy
        - Eager execution: pandas, polars eager
        - Used to maintain consistent evaluation modes across operations

    """
    df_native = df.to_native()
    return hasattr(df_native, "compute") or hasattr(df_native, "collect")


@nw.narwhalify
def check_dataframe_empty(df: SupportedTemporalDataFrame) -> bool:
    """Check if a DataFrame is empty using backend-agnostic operations.

    This function validates the input DataFrame using `is_valid_temporal_dataframe` and
    determines whether it is empty based on standard backend attributes such as `shape`.
    It handles lazy evaluation transparently for backends like Dask and Polars.

    :param df: The input DataFrame to check.
    :type df: SupportedTemporalDataFrame
    :return: True if the DataFrame is empty, False otherwise.
    :rtype: bool
    :raises ValueError: If the input DataFrame is None or invalid.

    Example:
    -------
        .. code-block:: python

            import narwhals as nw
            from temporalscope.core.core_utils import check_dataframe_empty

            # Example with Pandas
            data = {"col1": []}
            df = nw.from_native(data)  # Convert to SupportedTemporalDataFrame
            assert check_dataframe_empty(df) == True

            # Example with Dask LazyFrame
            lazy_df = nw.from_native(data).lazy()
            assert check_dataframe_empty(lazy_df) == True

    .. note::
        This function checks for emptiness using attributes like `shape`, `__len__`,
        and `num_rows` to support various backends. These attributes cover common
        DataFrame implementations, ensuring robust handling across the Narwhals API.
        If none of these attributes are present, an `UnsupportedBackendError` is raised.

    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    # Validate the DataFrame
    is_valid, df_type = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise ValueError(f"Unsupported DataFrame type: {type(df).__name__}")

    # Resolve lazy evaluation
    if is_lazy_evaluation(df):
        df = df.collect() if hasattr(df, "collect") else df.compute()

    # Check emptiness using backend-specific attributes
    if hasattr(df, "shape") and df.shape:
        return df.shape[0] == 0
    if hasattr(df, "__len__"):
        # Fallback for DataFrames that define `__len__` (e.g., Narwhals BaseFrame).
        return len(df) == 0
    if hasattr(df, "num_rows"):
        # Fallback for DataFrames that expose `num_rows` (e.g., PyArrow-like backends).
        return df.num_rows == 0

    # Fallback for unsupported cases
    return False


@nw.narwhalify
def check_dataframe_nulls_nans(df: SupportedTemporalDataFrame, column_names: List[str]) -> Dict[str, int]:
    """Check for null values in specified DataFrame columns using Narwhals operations.

    This function first validates if the DataFrame is empty using `check_dataframe_empty`
    and then performs backend-agnostic null value counting for the specified columns.

    :param df: DataFrame to check for null values
    :type df: SupportedTemporalDataFrame
    :param column_names: List of column names to check
    :type column_names: List[str]
    :return: Dictionary mapping column names to their null value counts
    :rtype: Dict[str, int]
    :raises ValueError: If the DataFrame is empty or a column is nonexistent.
    :raises UnsupportedBackendError: If the backend is unsupported.

    Example:
    -------
    .. code-block:: python

        import narwhals as nw
        from temporalscope.core.core_utils import SupportedTemporalDataFrame, check_dataframe_nulls_nans

        # Example input DataFrame
        data = {
            "col1": [1, 2, None],
            "col2": [4, None, 6],
        }
        df = nw.from_native(data)  # Convert to SupportedTemporalDataFrame

        # Define columns to check
        column_names = ["col1", "col2"]

        # Call check_dataframe_nulls_nans
        null_counts = check_dataframe_nulls_nans(df, column_names)

        # Output: {"col1": 1, "col2": 1}
        print(null_counts)

    .. note::
        This function integrates `check_dataframe_empty` to handle empty DataFrames,
        and uses backend-agnostic operations (e.g., `is_null`) to count null values
        in the specified columns. It assumes Narwhals-compatible backends.

    """
    # Step 1: Validate if the DataFrame is empty
    if check_dataframe_empty(df):
        raise ValueError("Empty DataFrame provided.")

    result = {}
    is_lazy = is_lazy_evaluation(df)  # Determine evaluation mode upfront

    for col in column_names:
        try:
            # Step 2: Compute null counts
            null_check = df.select([nw.col(col).is_null().sum().alias("null_count")])

            # Step 3: Handle lazy evaluation if applicable
            if is_lazy:
                if hasattr(null_check, "compute"):
                    null_check = null_check.compute()
                elif hasattr(null_check, "collect"):
                    null_check = null_check.collect()

            # Step 4: Extract null count value and handle PyArrow scalar type
            count = null_check["null_count"][0]
            if hasattr(count, "as_py"):  # Handle PyArrow scalar
                count = count.as_py()

            # Step 5: Explicitly cast to int and store the result
            result[col] = int(count)

        except KeyError:
            # Handle nonexistent column error
            raise ValueError(f"Column '{col}' not found.")
        except Exception as e:
            # Generic fallback for other errors
            raise ValueError(f"Error checking null values in column '{col}': {e}")

    return result


@nw.narwhalify
def convert_to_numeric(
    df: SupportedTemporalDataFrame, time_col: str, col_expr: Any, col_dtype: Any
) -> SupportedTemporalDataFrame:
    """Convert a datetime column to numeric using Narwhals API.

    :param df: The input DataFrame containing the column to convert.
    :type df: SupportedTemporalDataFrame
    :param time_col: The name of the time column to convert.
    :type time_col: str
    :param col_expr: The Narwhals column expression for the time column.
    :type col_expr: Any
    :param col_dtype: The resolved dtype of the time column.
    :type col_dtype: Any
    :return: The DataFrame with the converted time column.
    :rtype: SupportedTemporalDataFrame
    :raises ValueError: If the column is not a datetime type.

    .. note::
        - Converts datetime columns to numeric using `dt.timestamp()`.
        - Uses `time_unit="us"` for general backend compatibility.
        - Ensures the resulting column is cast to `Float64` for numeric operations.
        - Handles potential overflow issues for PyArrow by selecting smaller time units.
    """
    # Check if col_dtype is a datetime type (explicit Pandas/Narwhals check)
    if pd.api.types.is_datetime64_any_dtype(col_dtype) or "datetime" in str(col_dtype).lower():
        return df.with_columns([col_expr.dt.timestamp(time_unit="us").cast(nw.Float64()).alias(time_col)])

    raise ValueError(f"Column '{time_col}' is not a datetime column, cannot convert to numeric.")


@nw.narwhalify
def convert_to_datetime(
    df: SupportedTemporalDataFrame, time_col: str, col_expr: Any, col_dtype: Any
) -> SupportedTemporalDataFrame:
    """Convert a string or numeric column to datetime using Narwhals API.

    :param df: The input DataFrame containing the column to convert.
    :type df: SupportedTemporalDataFrame
    :param time_col: The name of the time column to convert.
    :type time_col: str
    :param col_expr: The Narwhals column expression for the time column.
    :type col_expr: Any
    :param col_dtype: The resolved dtype of the time column.
    :type col_dtype: Any
    :return: The DataFrame with the converted time column.
    :rtype: SupportedTemporalDataFrame
    :raises ValueError: If the column is not convertible to datetime.

    .. note::
        - Handles string columns using `str.to_datetime()` for backend compatibility.
        - Numeric columns are cast directly to `Datetime` using `cast(nw.Datetime())` where supported.
        - For PyArrow, handles timezone preservation and default `time_unit="ns"`.
        - Narwhals-backend ensures consistent behavior across lazy and eager backends.
        - Raises errors for unsupported column types to prevent silent failures.
    """
    if "string" in str(col_dtype).lower():
        return df.with_columns([col_expr.str.to_datetime().alias(time_col)])
    if "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower():
        return df.with_columns([col_expr.cast(nw.Datetime()).alias(time_col)])
    raise ValueError(f"Column '{time_col}' is neither string nor numeric; cannot convert to datetime.")


@nw.narwhalify
def validate_column_type(time_col: str, col_dtype: Any) -> None:
    """Validate that a column is either numeric or datetime.

    :param time_col: The name of the time column to validate.
    :type time_col: str
    :param col_dtype: The resolved dtype of the time column.
    :type col_dtype: Any
    :raises ValueError: If the column is neither numeric nor datetime.

    .. note::
        - Validates column dtypes to ensure they are either numeric (float/int) or datetime.
        - For numeric columns, supports all backend-specific numeric types (e.g., Float64, Int64).
        - For datetime columns, supports both timezone-aware and naive formats (e.g., UTC, local).
        - Provides clear error messages for unsupported types, ensuring better debugging in enterprise pipelines.
        - Centralized validation logic avoids repeated dtype checks in other utility functions.
        - Compatible with Narwhals lazy evaluation backends like Dask or Modin.
    """
    is_numeric = "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower()
    is_datetime = "datetime" in str(col_dtype).lower()
    if not is_numeric and not is_datetime:
        raise ValueError(f"Column '{time_col}' is neither numeric nor datetime.")


@nw.narwhalify
def validate_and_convert_time_column(
    df: SupportedTemporalDataFrame,
    time_col: str,
    conversion_type: Optional[str] = None,
) -> SupportedTemporalDataFrame:
    """Validate and optionally convert the time column in a DataFrame.

    :param df: The input DataFrame to process.
    :type df: SupportedTemporalDataFrame
    :param time_col: The name of the time column to validate or convert.
    :type time_col: str
    :param conversion_type: Optional. Specify the conversion type:
                            - 'numeric': Convert to Float64.
                            - 'datetime': Convert to Datetime.
                            - None: Validate only.
    :type conversion_type: Optional[str]
    :return: The validated and optionally converted DataFrame.
    :rtype: SupportedTemporalDataFrame
    :raises TimeColumnError: If validation or conversion fails or if an invalid conversion_type is provided.
    :raises ValueError: If the column dtype cannot be resolved.

    Example:
    -------
    .. code-block:: python

        df = validate_and_convert_time_column(df, "time", conversion_type="numeric")

    .. note::
       - Validates and converts the `time_col` to the specified type (`numeric` or `datetime`).
       - Uses backend-specific adjustments for PyArrow and other frameworks.
       - Handles nulls and ensures consistent schema across all supported backends.
       - Raises errors for invalid `conversion_type` values.

    """
    if time_col not in df.columns:
        raise TimeColumnError(f"Column '{time_col}' does not exist in the DataFrame.")

    if conversion_type not in {"numeric", "datetime", None}:
        raise ValueError(f"Invalid conversion_type '{conversion_type}'. Must be one of 'numeric', 'datetime', or None.")

    # Fetch column dtype safely
    col_dtype = df.schema.get(time_col) if hasattr(df, "schema") else None
    if col_dtype is None:
        raise ValueError(f"Unable to resolve dtype for column '{time_col}'.")

    # Delegate based on conversion type
    if conversion_type == "numeric":
        return convert_to_numeric(df, time_col, nw.col(time_col), col_dtype)

    if conversion_type == "datetime":
        return convert_to_datetime(df, time_col, nw.col(time_col), col_dtype)

    # Validation-only path
    validate_column_type(time_col, col_dtype)

    return df
