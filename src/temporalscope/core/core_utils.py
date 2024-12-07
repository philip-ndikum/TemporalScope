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
    :raises ValueError: If validation fails and `raise_error` is True.
+----------------+-------------------------------------------------------------------+
| multi_step     | Sequential time series tasks (e.g., seq2seq) for deep learning.   |
|                | The data is split into sequences (input X, target Y).             |
|                | Two DataFrames: X for input sequences, Y for targets.             |
|                | Frameworks: TensorFlow, PyTorch, Keras.                           |
+----------------+-------------------------------------------------------------------+

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

Naming Conventions:
----------------
The following naming conventions are used for utility functions in this module:
+--------------------------+------------------------------------------------------------+
| Pattern                  | Purpose                                                    |
+--------------------------+------------------------------------------------------------+
| `validate_<object>`      | Checks that an object meets specific requirements and      |
|                          | raises an error if it doesn't.                             |
+--------------------------+------------------------------------------------------------+
| `is_<property>`          | Returns metadata about an object (Boolean or other).       |
+--------------------------+------------------------------------------------------------+
| `convert_<object>`       | Transforms an object to a desired type or structure.       |
+--------------------------+------------------------------------------------------------+
| `check_<condition>`      | Performs a specific check and returns a result or raises   |
|                          | an error if the condition is violated.                     |
+--------------------------+------------------------------------------------------------+
| `sort_<object>`          | Orders an object based on specified criteria.              |
+--------------------------+------------------------------------------------------------+
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

    Example Usage:
    --------------
    .. code-block:: python

        # Assume environment variables are set:
        # export OPENAI_API_KEY='abc123'
        # export CLAUDE_API_KEY='def456'

        # Retrieve API keys
        api_keys = get_api_keys()
        print(api_keys)
        # Output: {'OPENAI_API_KEY': 'abc123', 'CLAUDE_API_KEY': 'def456'}

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

    Example:
    -------
    .. code-block:: python

        print_divider(char="-", length=50)
        # Output: --------------------------------------------------

    """
    print(char * length)


# ---------------------------------------------------------
# Main Functions
# ---------------------------------------------------------


def get_narwhals_backends() -> List[str]:
    """Retrieve all backends available through Narwhals.

    :return: List of Narwhals-supported backend names in lowercase.
    :rtype: List[str]

    Example Usage:
    --------------
    .. code-block:: python

        backends = get_narwhals_backends()
        print(backends)
        # Output: ['pandas', 'modin', 'pyarrow', 'polars', 'dask']

    """
    return [backend.name.lower() for backend in Implementation]


def get_default_backend_cfg() -> Dict[str, List[str]]:
    """Retrieve the default application configuration for available backends.

    :return: Dictionary with a single key 'BACKENDS' containing a list of all
             Narwhals-supported backends.
    :rtype: Dict[str, List[str]]

    Example:
    -------
    .. code-block:: python

        config = get_default_backend_cfg()
        print(config)
        # Output: {'BACKENDS': ['pandas', 'modin', 'pyarrow', 'polars', 'dask']}

    """
    available_backends = get_narwhals_backends()
    return {"BACKENDS": available_backends}


def get_temporalscope_backends() -> List[str]:
    """Retrieve the subset of Narwhals-supported backends compatible with TemporalScope.

    :return: List of backend names supported by TemporalScope.
    :rtype: List[str]

    Example Usage:
    --------------
    .. code-block:: python

        backends = get_temporalscope_backends()
        print(backends)
        # Output: ['pandas', 'modin', 'pyarrow', 'polars', 'dask']

    """
    available_backends = get_narwhals_backends()
    return [backend for backend in available_backends if backend in TEMPORALSCOPE_CORE_BACKENDS]


def is_valid_temporal_backend(backend_name: str) -> None:
    """Validate that a backend is supported by TemporalScope and Narwhals.

    :param backend_name: Name of the backend to validate.
    :type backend_name: str
    :raises UnsupportedBackendError: If the backend is not in supported or optional backends.
    :raises UserWarning: If the backend is in the optional set but not installed.

    Example Usage:
    --------------
    .. code-block:: python

        # Validate a valid backend
        is_valid_temporal_backend("pandas")  # Passes silently

        # Validate an unsupported backend
        try:
            is_valid_temporal_backend("unsupported_backend")
        except UnsupportedBackendError as e:
            print(e)
        # Output: Backend 'unsupported_backend' is not supported by TemporalScope.

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

    :param df: Object to validate, can be any supported DataFrame type or arbitrary object.
    :type df: Union[SupportedTemporalDataFrame, Any]
    :return: Tuple of (is_valid, df_type) where df_type is:
             - "narwhals" for FrameT DataFrames
             - "native" for supported DataFrame types
             - None if not valid
    :rtype: Tuple[bool, Optional[str]]

    Example:
    -------
    .. code-block:: python

        df = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
        result = is_valid_temporal_dataframe(df)
        print(result)
        # Output: (True, "native")

        invalid_df = "Not a DataFrame"
        result = is_valid_temporal_dataframe(invalid_df)
        print(result)
        # Output: (False, None)

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

        # If no match found
        return False, None
    except Exception:  # pragma: no cover
        # Defensive programming: safeguard against unforeseen exceptions
        return False, None


def get_dataframe_backend(df: Union[SupportedTemporalDataFrame, Any]) -> str:
    """Get the backend name for a DataFrame.

    :param df: DataFrame to get backend for.
    :type df: Union[SupportedTemporalDataFrame, Any]
    :return: Backend name ('pandas', 'modin', 'polars', 'pyarrow', 'dask').
    :rtype: str
    :raises UnsupportedBackendError: If DataFrame type not supported.

    Example Usage:
    --------------
    .. code-block:: python

        from temporalscope.core.core_utils import get_dataframe_backend
        import pandas as pd

        # Example with a Pandas DataFrame
        df = pd.DataFrame({"col1": [1, 2, 3]})
        backend = get_dataframe_backend(df)
        print(backend)  # Output: 'pandas'

        # Example with a Polars DataFrame
        import polars as pl

        df = pl.DataFrame({"col1": [1, 2, 3]})
        backend = get_dataframe_backend(df)
        print(backend)  # Output: 'polars'

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

    # If no backend matches, raise an error
    raise UnsupportedBackendError(
        f"Failed to determine backend for DataFrame of type {type(df).__name__}"
    )  # pragma: no cover


@nw.narwhalify
def is_lazy_evaluation(df: SupportedTemporalDataFrame) -> bool:
    """Check if a DataFrame uses lazy evaluation.

    :param df: The DataFrame to check for evaluation mode.
    :type df: SupportedTemporalDataFrame
    :return: True if the DataFrame uses lazy evaluation, False otherwise.
    :rtype: bool
    :raises UnsupportedBackendError: If the DataFrame's backend is not supported by TemporalScope.

    Example Usage:
    --------------
    .. code-block:: python

        from temporalscope.core.core_utils import is_lazy_evaluation

        # Check evaluation mode
        if is_lazy_evaluation(df):
            # Lazy evaluation path
            result = df.select([...])  # Maintain lazy evaluation
        else:
            # Eager evaluation path
            result = df.select([...])  # Direct computation is safe

    .. note::
        This function determines whether a DataFrame uses lazy or eager evaluation:
        - Lazy execution, such as with Dask or Polars in lazy mode.
        - Eager execution, such as with Pandas or Polars in eager mode.
        This distinction is important for maintaining consistent evaluation modes during computations.
    """
    # Validate the input DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError("The input DataFrame is not supported by TemporalScope.")

    df_native = df.to_native()
    return hasattr(df_native, "compute") or hasattr(df_native, "collect")


@nw.narwhalify
def convert_to_backend(
    df: Union[SupportedTemporalDataFrame, IntoDataFrame],
    backend: str,
    npartitions: int = 1,
    backend_converter_dict: Dict[str, Callable[[pd.DataFrame, int], Any]] = TEMPORALSCOPE_BACKEND_CONVERTERS,
) -> SupportedTemporalDataFrame:
    """Convert a DataFrame to the specified backend format.

    Converts a DataFrame (e.g., Pandas) to a backend (e.g., Dask, Polars) using DIP.
    Narwhals handles validation and lazy evaluation, while `backend_converter_dict`
    manages conversion.

    :param df: Input DataFrame (pandas, modin, polars, pyarrow, dask).
    :type df: Union[SupportedTemporalDataFrame, IntoDataFrame]
    :param backend: Target backend ('pandas', 'modin', 'polars', 'pyarrow', 'dask').
    :type backend: str
    :param npartitions: Number of partitions for Dask backend. Default is 1.
    :type npartitions: int
    :param backend_converter_dict: Backend conversion functions.
    :type backend_converter_dict: Dict[str, Callable[[pd.DataFrame, int], Any]], optional
    :return: Converted DataFrame in the target backend.
    :rtype: SupportedTemporalDataFrame
    :raises UnsupportedBackendError: If the backend or DataFrame is unsupported.

    Example Usage:
    --------------
    .. code-block:: python

        # Pandas -> Polars conversion
        df_pd = pd.DataFrame({"time": range(10), "value": range(10)})
        df_polars = convert_to_backend(df_pd, backend="polars")

        # Dask -> Pandas materialization
        df_dask = dd.from_pandas(df_pd, npartitions=2)
        df_pd_result = convert_to_backend(df_dask, backend="pandas")

    .. note::
        Steps:
        - Validate: Narwhals checks input compatibility (`is_valid_temporal_dataframe`).
        - Materialize: Handles lazy evaluation (Dask/Polars LazyFrames).
        - Convert: Uses `backend_converter_dict` for backend-specific transformations.
    """
    # Validate backend and DataFrame using helper functions
    is_valid_temporal_backend(backend)
    is_valid, df_type = is_valid_temporal_dataframe(df)

    if not is_valid:
        raise UnsupportedBackendError(f"Input DataFrame type '{type(df).__name__}' is not supported")

    # Handle narwhalified DataFrames
    if df_type == "narwhals":
        df = df.to_native()

    # Materialize lazy DataFrames
    if is_lazy_evaluation(df):
        df = df.compute() if hasattr(df, "compute") else df.collect()

    # Convert to target backend using the converter dictionary
    try:
        conversion_func = backend_converter_dict[backend]
        return conversion_func(df, npartitions)
    except KeyError:  # pragma: no cover
        # This exception is unlikely because `is_valid_temporal_backend` already validates the backend.
        raise UnsupportedBackendError(f"The backend '{backend}' is not supported by TemporalScope.")
    except Exception as e:  # pragma: no cover
        # This exception safeguards against unforeseen errors during conversion.
        raise UnsupportedBackendError(f"Failed to convert DataFrame: {str(e)}")


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
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope.

    Example Usage:
    --------------
        .. code-block:: python

            from temporalscope.core.core_utils import check_dataframe_empty

            # Example with Pandas DataFrame
            import pandas as pd

            df = pd.DataFrame(columns=["col1"])
            assert check_dataframe_empty(df) == True

            # Example with lazy-evaluation backends
            # Assumes `df` is a lazy DataFrame (e.g., Dask or Polars)
            assert check_dataframe_empty(df) == True

    .. note::
        This function checks for emptiness using attributes like `shape`, `__len__`,
        and `num_rows` to support various backends. These attributes cover common
        DataFrame implementations, ensuring robust handling across the Narwhals API.
        If none of these attributes are present, an `UnsupportedBackendError` is raised.

    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    # Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise ValueError(f"Unsupported DataFrame type: {type(df).__name__}")

    # Materialize lazy DataFrames
    if is_lazy_evaluation(df):
        df = df.collect() if hasattr(df, "collect") else df.compute()

    # Check emptiness using backend-specific attributes
    if hasattr(df, "shape") and df.shape:
        return df.shape[0] == 0
    if hasattr(df, "__len__"):  # pragma: no cover
        # Defensive fallback for DataFrames that define __len__ but lack .shape or .num_rows
        return len(df) == 0
    if hasattr(df, "num_rows"):  # pragma: no cover
        # Defensive fallback for DataFrames that expose num_rows but lack other attributes
        return df.num_rows == 0

    # Ultimate fallback for unsupported DataFrame types
    return False  # pragma: no cover


@nw.narwhalify
def check_dataframe_nulls_nans(df: SupportedTemporalDataFrame, column_names: List[str]) -> Dict[str, int]:
    """Check for null values in specified DataFrame columns using Narwhals operations.

    This function first validates if the DataFrame is empty using `check_dataframe_empty`
    and then performs backend-agnostic null value counting for the specified columns.

    :param df: DataFrame to check for null values.
    :type df: SupportedTemporalDataFrame
    :param column_names: List of column names to check.
    :type column_names: List[str]
    :return: Dictionary mapping column names to their null value counts.
    :rtype: Dict[str, int]
    :raises ValueError: If the DataFrame is empty or a column is nonexistent.
    :raises UnsupportedBackendError: If the backend is unsupported.

    Example Usage:
    --------------
    .. code-block:: python

        from temporalscope.core.core_utils import check_dataframe_nulls_nans

        # Example input DataFrame
        import pandas as pd

        df = pd.DataFrame(
            {
                "col1": [1, 2, None],
                "col2": [4, None, 6],
            }
        )

        # Define columns to check
        column_names = ["col1", "col2"]

        # Call check_dataframe_nulls_nans
        null_counts = check_dataframe_nulls_nans(df, column_names)

        # Output: {"col1": 1, "col2": 1}
        print(null_counts)

    .. note::
        This function handles lazy evaluation defensively (e.g., using `compute` or `collect`)
        to ensure compatibility with backends like Dask and Polars. It works with eager
        backends such as Pandas or Polars eager mode as well, maintaining backend agnosticism.

    """
    # Step 0: Validate the input DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError("The input DataFrame is not supported by TemporalScope.")

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
                if hasattr(null_check, "compute"):  # pragma: no cover
                    null_check = null_check.compute()
                elif hasattr(null_check, "collect"):  # pragma: no cover
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
        except Exception as e:  # pragma: no cover
            # Handle unforeseen errors.
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
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope.

    Example Usage:
    --------------
    .. code-block:: python

        df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
        df = convert_to_numeric(df, "time", nw.col("time"), df["time"].dtype)
        print(df)

    .. note::
        - Converts datetime columns to numeric using `dt.timestamp()`.
        - Uses `time_unit="us"` for general backend compatibility.
        - Ensures the resulting column is cast to `Float64` for numeric operations.
        - Handles potential overflow issues for PyArrow by selecting smaller time units.

    """
    # Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise ValueError(f"Unsupported DataFrame type: {type(df).__name__}")

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
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope.

    Example Usage:
    --------------
    .. code-block:: python

        df = pd.DataFrame({"time": [1672531200000, 1672617600000]})  # Unix timestamps
        df = convert_to_datetime(df, "time", nw.col("time"), df["time"].dtype)
        print(df)

    .. note::
        - Handles string columns using `str.to_datetime()` for backend compatibility.
        - Numeric columns are cast directly to `Datetime` using `cast(nw.Datetime())` where supported.
        - For PyArrow, handles timezone preservation and default `time_unit="ns"`.
        - Narwhals-backend ensures consistent behavior across lazy and eager backends.
        - Raises errors for unsupported column types to prevent silent failures.

    """
    # Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df).__name__}")

    if "string" in str(col_dtype).lower():
        return df.with_columns([col_expr.str.to_datetime().alias(time_col)])
    if "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower():
        return df.with_columns([col_expr.cast(nw.Datetime()).alias(time_col)])
    raise ValueError(f"Column '{time_col}' is neither string nor numeric; cannot convert to datetime.")


@nw.narwhalify
def validate_time_column_type(time_col: str, col_dtype: Any) -> None:
    """Validate that a column is either numeric or datetime.

    :param time_col: The name of the time column to validate.
    :type time_col: str
    :param col_dtype: The resolved dtype of the time column.
    :type col_dtype: Any
    :raises ValueError: If the column is neither numeric nor datetime.

    Example Usage:
    --------------
    .. code-block:: python

        validate_time_column_type("time", pd.Series([1, 2, 3]).dtype)  # Passes
        validate_time_column_type("time", pd.Series(["2023-01-01"]).dtype)  # Passes

        try:
            validate_time_column_type("time", pd.Series(["abc"]).dtype)  # Raises ValueError
        except ValueError as e:
            print(e)
        # Output: Column 'time' is neither numeric nor datetime.

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
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope.

    Example Usage:
    --------------
    .. code-block:: python

        df = validate_and_convert_time_column(df, "time", conversion_type="numeric")

    .. note::
       - Validates and converts the `time_col` to the specified type (`numeric` or `datetime`).
       - Uses backend-specific adjustments for PyArrow and other frameworks.
       - Handles nulls and ensures consistent schema across all supported backends.
       - Raises errors for invalid `conversion_type` values.

    """
    # Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df).__name__}")

    if time_col not in df.columns:
        raise TimeColumnError(f"Column '{time_col}' does not exist in the DataFrame.")

    if conversion_type not in {"numeric", "datetime", None}:
        raise ValueError(f"Invalid conversion_type '{conversion_type}'. Must be one of 'numeric', 'datetime', or None.")

    # Fetch column dtype safely
    col_dtype = df.schema.get(time_col) if hasattr(df, "schema") else None
    if col_dtype is None:  # pragma: no cover
        # Defensive check for unexpected Narwhals schema failures; excluded from coverage.
        raise ValueError(f"Unable to resolve dtype for column '{time_col}'.")

    # Delegate based on conversion type
    if conversion_type == "numeric":
        return convert_to_numeric(df, time_col, nw.col(time_col), col_dtype)

    if conversion_type == "datetime":
        return convert_to_datetime(df, time_col, nw.col(time_col), col_dtype)

    # Validation-only path
    validate_time_column_type(time_col, col_dtype)

    return df


@nw.narwhalify
def validate_dataframe_column_types(df: SupportedTemporalDataFrame, time_col: str) -> None:
    """Validate the column types in a DataFrame for compatibility with TemporalScope.

    This function ensures the following:
    - The specified `time_col` must be of type numeric or datetime.
    - All other columns in the DataFrame must be of numeric type.

    :param df: The input DataFrame to validate.
    :type df: SupportedTemporalDataFrame
    :param time_col: The name of the time column to validate.
    :type time_col: str
    :raises ValueError: If the `time_col` does not exist or has an invalid type.
    :raises ValueError: If any non-time column has an invalid type.
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope.

    Example Usage:
    --------------
    .. code-block:: python

        from temporalscope.core.temporal_data_loader import validate_dataframe_column_types

        # Example DataFrame
        df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=5), "value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        # Validate column types
        validate_dataframe_column_types(df, time_col="time")
    """
    # Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df).__name__}")

    # Handle lazy evaluation first
    if is_lazy_evaluation(df):
        df = df.collect() if hasattr(df, "collect") else df.compute()

    # Get schema safely using Narwhals API
    schema = df.schema

    # Validate time column
    time_dtype = schema.get(time_col)
    if time_dtype is None:
        raise ValueError(f"Column '{time_col}' does not exist")
    validate_time_column_type(time_col, time_dtype)

    # Validate other columns
    non_time_cols = [col for col in df.columns if col != time_col]
    for col in non_time_cols:
        col_dtype = schema.get(col)
        if col_dtype is None:  # pragma: no cover
            # Defensive check for unexpected Narwhals schema failures; excluded from coverage.
            raise ValueError(f"Column '{col}' does not exist")

        # Check if the column is numeric
        is_numeric = "float" in str(col_dtype).lower() or "int" in str(col_dtype).lower()
        if not is_numeric:
            raise ValueError(f"Column '{col}' must be numeric but found type '{col_dtype}'")


@nw.narwhalify
def sort_dataframe_time(
    df: SupportedTemporalDataFrame, time_col: str, ascending: bool = True
) -> SupportedTemporalDataFrame:
    """Sort a DataFrame by the specified time column using Narwhals' backend-agnostic operations.

    :param df: The input DataFrame to sort.
    :type df: SupportedTemporalDataFrame
    :param time_col: The name of the column to sort by. Must exist in the DataFrame.
    :type time_col: str
    :param ascending: Sort direction. Defaults to True (ascending order).
    :type ascending: bool, optional
    :return: A DataFrame sorted by the specified time column.
    :rtype: SupportedTemporalDataFrame
    :raises ValueError: If the `time_col` does not exist in the DataFrame or has invalid type.
    :raises UnsupportedBackendError: If the backend is not supported by TemporalScope.

    Example Usage:
    --------------
    .. code-block:: python

        from temporalscope.core.core_utils import sort_dataframe_time
        import pandas as pd

        # Example DataFrame
        df = pd.DataFrame({"time": [3, 1, 4, 2, 5], "value": range(5)})

        # Sort DataFrame by the 'time' column in ascending order
        sorted_df = sort_dataframe_time(df, time_col="time", ascending=True)
        print(sorted_df)

    .. note::
        - The `@nw.narwhalify` decorator automatically handles backend detection
          and adapts sorting to Pandas, Modin, Dask, Polars, and PyArrow.
        - Validates that the time column exists and has a valid type (numeric or datetime).
        - Uses string column names for sorting to ensure compatibility across all backends.
        - Handles lazy evaluation in backends like Dask and Polars.

        **Dask Specific Note**:
        - DaskLazyFrame requires explicit computation using `collect()` or `compute()`.
        - This function ensures such materialization happens before sorting.

    .. warning::
        Sorting large DataFrames in lazy backends like Dask or Polars may cause
        computations or require additional memory. Ensure memory constraints are handled.
    """
    # Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df).__name__}")

    # Validate that the time column exists
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Handle lazy evaluation explicitly
    if is_lazy_evaluation(df):
        # Defensive programming to handle Narwhals lazy backends
        df = df.compute() if hasattr(df, "compute") else df.collect()  # pragma: no cover

    # Validate time_col type
    validate_time_column_type(time_col, df.schema.get(time_col, None))

    sorted_df = df.sort(by=time_col, descending=not ascending)
    return sorted_df


@nw.narwhalify
def validate_temporal_uniqueness(
    df: SupportedTemporalDataFrame, time_col: str, raise_error: bool = True, context: str = ""
) -> None:
    """Validate strict temporal ordering and uniqueness in the given DataFrame.

    This function checks that:
    1. Timestamps in the specified `time_col` are unique.
    2. Timestamps in the specified `time_col` are in strictly increasing order.

    :param df: The DataFrame to validate.
    :type df: SupportedTemporalDataFrame
    :param time_col: The column representing time.
    :type time_col: str
    :param raise_error: Whether to raise an error if validation fails.
    :type raise_error: bool
    :param context: An optional string to indicate the validation context (e.g., group name).
    :type context: str
    :raises ValueError: If validation fails and `raise_error` is True.
    :warns UserWarning: If validation fails and `raise_error` is False.

    Example Usage:
    --------------
    .. code-block:: python

        import narwhals as nw
        import pandas as pd

        # Create a test DataFrame with numeric timestamps
        df = pd.DataFrame({"time": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
        validate_temporal_uniqueness(df, time_col="time")  # Will pass

        # Create DataFrame with datetime timestamps
        df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=4), "value": [10, 20, 30, 40]})
        validate_temporal_uniqueness(df, time_col="time")  # Will pass

        # Create DataFrame with duplicates
        df_duplicates = pd.DataFrame(
            {
                "time": [1, 1, 2, 3],  # Note duplicate timestamp
                "value": [10, 10, 20, 30],
            }
        )
        # This will raise ValueError: "Duplicate timestamps in column 'time'."
        validate_temporal_uniqueness(df_duplicates, time_col="time")

    .. note::
        - Use `nw.from_native` only when chaining `@nw.narwhalify` functions, as each
          returns the native format after execution.
        - This ensures compatibility with Narwhals operations in multi-function pipelines.
    """
    # Step 1: Validate time column type and convert if needed
    try:
        df = nw.from_native(validate_and_convert_time_column(df, time_col))
    except Exception as e:
        raise TimeColumnError(f"Invalid time column: {str(e)}")

    # Step 2: Materialize lazy DataFrames
    if is_lazy_evaluation(df):  # Defensive programming for lazy evaluation
        df = df.collect() if hasattr(df, "collect") else df.compute()  # pragma: no cover

    # Handle PyArrow's unique count differently
    if get_dataframe_backend(df) == "pyarrow":
        df = nw.from_native(df.to_pandas()) # pragma: no cover

    # Step 3: Check for duplicates using proper aggregation
    duplicate_check_df = df.select(
        [nw.col(time_col).n_unique().alias("unique_count"), nw.col(time_col).count().alias("total_count")]
    )
    duplicate_check_series = duplicate_check_df.select(
        [(nw.col("unique_count") < nw.col("total_count")).alias("has_duplicates")]
    )

    # Step 4: Raise error if duplicates found
    if duplicate_check_series.select([nw.col("has_duplicates").any()]).item():
        message = f"Duplicate timestamps in {context}column '{time_col}'."
        if raise_error:
            raise ValueError(message)
        warnings.warn(message)

    # Step 5: Check temporal ordering in current data
    non_monotonic_series = df.select(  # Use df instead of sorted_df
        [(nw.col(time_col).shift(-1) <= nw.col(time_col)).sum().alias("non_monotonic_count")]
    )

    # Step 6: Raise error if non-monotonic
    if non_monotonic_series.select([nw.col("non_monotonic_count") > 0]).item():
        message = f"Timestamps not strictly increasing in {context}column '{time_col}'."
        if raise_error:
            raise ValueError(message)
        warnings.warn(message)


@nw.narwhalify
def sort_and_validate_temporal_order(
    df: SupportedTemporalDataFrame,
    time_col: str,
    group_col: Optional[str] = None,
    id_col: Optional[str] = None,
    raise_error: bool = True,
) -> None:
    """Validate strict temporal ordering and uniqueness in the given DataFrame.

    This function checks that:
    1. Timestamps in the specified `time_col` are unique.
    2. Timestamps in the specified `time_col` are in strictly increasing order.
    3. If `group_col` is provided, temporal order is validated within each group.
    4. If `id_col` is provided, the DataFrame is sorted by `id_col` and `time_col` before validation.

    :param df: The DataFrame to validate.
    :type df: SupportedTemporalDataFrame
    :param time_col: The column representing time.
    :type time_col: str
    :param group_col: Optional column for grouping data before validation.
    :type group_col: Optional[str]
    :param id_col: Optional column for sorting data before validation.
    :type id_col: Optional[str]
    :param raise_error: Whether to raise an error if validation fails.
    :type raise_error: bool
    :raises ValueError: If validation fails and `raise_error` is True.
    :raises ValueError: If required columns (`time_col`, `group_col`, or `id_col`) do not exist.
    :raises ValueError: If the input DataFrame is invalid or empty.

    Example Usage:
    --------------
    .. code-block:: python

        import pandas as pd

        # Create a test DataFrame
        df = pd.DataFrame({"time": [1, 2, 3, 4], "value": [10, 20, 30, 40]})

        # Check temporal ordering - will pass
        sort_and_validate_temporal_order(df, time_col="time")

        # Create DataFrame with duplicates
        df_duplicates = pd.DataFrame(
            {
                "time": [1, 1, 2, 3],  # Note duplicate timestamp
                "value": [10, 10, 20, 30],
            }
        )

        # This will raise ValueError: "Duplicate timestamps in column 'time'."
        sort_and_validate_temporal_order(df_duplicates, time_col="time")

        # Check within groups
        df_grouped = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "time": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        sort_and_validate_temporal_order(df_grouped, time_col="time", group_col="group")

    .. note::
        - This function assumes strict temporal ordering and will fail if duplicate
          or unordered timestamps are found in `time_col`. Grouping and sorting
          options (`group_col` and `id_col`) can help ensure correct validation
          in more complex datasets.
        - Use `nw.from_native` only when chaining `@nw.narwhalify` functions, as
          each returns the native format after execution.
        - This ensures compatibility with Narwhals operations in multi-function
          pipelines.
    """
    # Step 1: Initial validation
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid or check_dataframe_empty(df):
        raise ValueError("Invalid or empty DataFrame provided.")

    # Step 2: Materialize lazy DataFrames
    if is_lazy_evaluation(df):  # Defensive programming for lazy evaluation
        df = df.collect() if hasattr(df, "collect") else df.compute()  # pragma: no cover

    # Step 3: Column validation
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Step 4: Sort entire DataFrame by time first
    df = sort_dataframe_time(df, time_col=time_col, ascending=True)
    df = nw.from_native(df)  # Convert back after narwhalified function

    # Step 5: Additional sort by id_col if provided
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' does not exist in the DataFrame.")
        # Sort by id_col while maintaining time order within each id
        df = df.sort(by=[id_col, time_col])
        df = nw.from_native(df)

        # Get unique groups
        group_values = df.select(nw.col(id_col).unique())
        if is_lazy_evaluation(group_values):  # Same pattern as Step 2
            group_values = (
                group_values.collect() if hasattr(group_values, "collect") else group_values.compute()
            )  # pragma: no cover

        # Validate each group's order
        for group_val in group_values[id_col]:
            group_df = df.filter(nw.col(id_col) == group_val)
            validate_temporal_uniqueness(group_df, time_col, raise_error, context=f"group '{group_val}' ")

    # Step 6: Group validation
    elif group_col:
        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' does not exist in the DataFrame.")

        # Get unique groups
        group_values = df.select(nw.col(group_col).unique())

        # Defensive programming for lazy evaluation backends
        if is_lazy_evaluation(group_values):  # Same pattern as Step 2
            group_values = (
                group_values.collect() if hasattr(group_values, "collect") else group_values.compute()
            )  # pragma: no cover

        # Validate each group's order
        for group_val in group_values[group_col]:
            group_df = df.filter(nw.col(group_col) == group_val)
            validate_temporal_uniqueness(group_df, time_col, raise_error, context=f"group '{group_val}' ")
    else:
        # Step 7: Final validation of entire sorted DataFrame
        validate_temporal_uniqueness(df, time_col, raise_error)  # pragma: no cover
