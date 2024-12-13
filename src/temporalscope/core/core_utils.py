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


| Mode | Description |
|------|-------------|
| single_step | General machine learning tasks with scalar targets. Each row is a single time step, and the target is scalar. Single DataFrame: each row is an observation. |
| multi_step | Sequential time series tasks (e.g., seq2seq) for deep learning. The data is split into sequences (input X, target Y). Two DataFrames: X for input sequences, Y for targets. Frameworks: TensorFlow, PyTorch, Keras. |

Example Visualization:
----------------------
Here is a visual demonstration of the datasets generated for single-step and multi-step
modes, including the shape of input (`X`) and target (`Y`) data compatible with most
popular ML frameworks like TensorFlow, PyTorch, and SHAP.

**Single-step mode**:
    - Input shape: X (num_samples, num_features)
    - Target shape: Y (num_samples, 1)  # Scalar target for each time step

    Example:
    | time       | feature_1 | feature_2 | feature_3 | target |
    |------------|-----------|-----------|-----------|--------|
    | 2023-01-01 | 0.15      | 0.67      | 0.89      | 0.33   |
    | 2023-01-02 | 0.24      | 0.41      | 0.92      | 0.28   |

**Multi-step mode (with vectorized targets)**:
    Input shape: X (num_samples, num_features)
    Target shape: Y (num_samples, sequence_length)  # Vectorized target for each input sequence

    Example:
    | time       | feature_1 | feature_2 | feature_3 | target       |
    |------------|-----------|-----------|-----------|--------------|
    | 2023-01-01 | 0.15      | 0.67      | 0.89      | [0.3, 0.4]   |
    | 2023-01-02 | 0.24      | 0.41      | 0.92      | [0.5, 0.6]   |

DataFrame Types:
----------------
TemporalScope handles various DataFrame types throughout the data processing pipeline. The following table
illustrates the supported DataFrame types and validation cases:

| DataFrame Type | Description | Example |
|---------------|--------------|---------|
| Narwhalified (FrameT) | DataFrames wrapped by Narwhals for backend-agnostic operations. These are validated first to ensure consistent handling across backends. | @nw.narwhalify decorated functions create these during operations. |
| Native DataFrames | Core DataFrame implementations from supported backends. These are validated directly against TEMPORALSCOPE_CORE_BACKEND_TYPES. | pd.DataFrame, pl.DataFrame, pa.Table |
| DataFrame Subclasses | Custom or specialized DataFrames that inherit from native types. Common in: - Custom DataFrame implementations - Backend optimizations (e.g. lazy evaluation) - Backend compatibility layers | TimeSeriesDataFrame (pd.DataFrame), dask.dataframe.DataFrame (inherits from pandas) |
| Intermediate States | DataFrames in the middle of narwhalify operations or backend conversions. These may be temporary subclasses used for optimization or compatibility. | LazyDataFrame during backend conversion operation chaining |
Naming Conventions:
-------------------
The following naming conventions are used for utility functions in this module:

| Pattern | Purpose |
|---------|---------|
| `validate_<object>` | Checks that an object meets specific requirements and raises an error if it doesn't. |
| `is_<property>` | Returns metadata about an object (Boolean or other). |
| `convert_<object>` | Transforms an object to a desired type or structure. |
| `check_<condition>` | Performs a specific check and returns a result or raises an error if the condition is violated. |
| `sort_<object>` | Orders an object based on specified criteria. |

"""

import os
import warnings
from importlib import util
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import dask.dataframe as dd
import modin.pandas as mpd
import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa
from dotenv import load_dotenv
from narwhals.typing import FrameT, IntoDataFrame
from narwhals.utils import Implementation

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

    Returns
    -------
        Dict[str, Optional[str]]

    Examples
    --------
    .. code-block:: python

        # Assume environment variables are set:
        # export OPENAI_API_KEY='abc123'
        # export CLAUDE_API_KEY='def456'

        # Retrieve API keys
        api_keys = get_api_keys()
        print(api_keys)
        # Output: {'OPENAI_API_KEY': 'abc123', 'CLAUDE_API_KEY': 'def456'}

    Returns
    -------
    Dict[str, Optional[str]]
        A dictionary containing the API keys, or None if not found.

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

    Parameters
    ----------
        char : str, optional
            The character to use for the divider, defaults to '='
        length : int, optional

    Examples
    --------
    .. code-block:: python

        print_divider(char="-", length=50)
        # Output: --------------------------------------------------

    Returns
    -------
    None

    """
    print(char * length)


# ---------------------------------------------------------
# Main Functions
# ---------------------------------------------------------


def get_narwhals_backends() -> List[str]:
    """Retrieve all backends available through Narwhals.

    Returns
    -------
        List[str]

    Examples
    --------
    .. code-block:: python

        backends = get_narwhals_backends()
        print(backends)
        # Output: ['pandas', 'modin', 'pyarrow', 'polars', 'dask']

    Returns
    -------
    List[str]
        List of Narwhals-supported backend names.

    """
    return [backend.name.lower() for backend in Implementation]


def get_default_backend_cfg() -> Dict[str, List[str]]:
    """Retrieve the default application configuration for available backends.

    Returns
    -------
        Dict[str, List[str]]

    Examples
    --------
    .. code-block:: python

        config = get_default_backend_cfg()
        print(config)
        # Output: {'BACKENDS': ['pandas', 'modin', 'pyarrow', 'polars', 'dask']}

    Returns
    -------
    Dict[str, List[str]]
        Dictiona a single key 'BACKENDS' containing a list of all
        Narwhals-supported backends.

    """
    available_backends = get_narwhals_backends()
    return {"BACKENDS": available_backends}


def get_temporalscope_backends() -> List[str]:
    """Retrieve the subset of Narwhals-supported backends compatible with TemporalScope.

    Returns
    -------
        List[str]

    Examples
    --------
    .. code-block:: python

        backends = get_temporalscope_backends()
        print(backends)
        # Output: ['pandas', 'modin', 'pyarrow', 'polars', 'dask']

    Returns
    -------
    List[str]
        List of backend names supported by TemporalScope.

    """
    available_backends = get_narwhals_backends()
    return [backend for backend in available_backends if backend in TEMPORALSCOPE_CORE_BACKENDS]


def is_valid_temporal_backend(backend_name: str) -> None:
    """Validate that a backend is supported by TemporalScope and Narwhals.

    Parameters
    ----------
    backend_name : str
        Name of the backend to validate.
    backend_name: str :


    Returns
    -------
    None

    Raises
    ------
    UnsupportedBackendError
        If the backend is not in supported or optional backends.
    UserWarning
        If the backend is in the optional set but not installed.

    Examples
    --------
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

    Parameters
    ----------
        df : Union[SupportedTemporalDataFrame, Any]
            Object to validate, can be any supported DataFrame type or arbitrary object.
        df: Union[SupportedTemporalDataFrame, Any]

    Returns
    -------
        Tuple[bool, Optional[str]]

    Examples
    --------
    .. code-block:: python

        df = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
        result = is_valid_temporal_dataframe(df)
        print(result)
        # Output: (True, "native")

        invalid_df = "Not a DataFrame"
        result = is_valid_temporal_dataframe(invalid_df)
        print(result)
        # Output: (False, None)

    Returns
    -------
    Tuple[bool, Optional[str]]
        Tuple of (is_valid, df_type) where df_type is:
        - "narwhals" for FrameT DataFrames
        - "native" for supported DataFrame types
        - None if not valid

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

    Parameters
    ----------
    df : Union[SupportedTemporalDataFrame, Any]
        DataFrame to get backend for.

    Returns
    -------
    str
        Backend name ('pandas', 'modin', 'polars', 'pyarrow', 'dask').

    Raises
    ------
    UnsupportedBackendError
        If DataFrame type not supported.

    Examples
    --------
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

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The DataFrame to check for evaluation mode.

    Returns
    -------
    bool
        True if the DataFrame uses lazy evaluation, False otherwise.

    Raises
    ------
    UnsupportedBackendError
        If the DataFrame's backend is not supported by TemporalScope.

    Examples
    --------
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

    Parameters
    ----------
    df : Union[SupportedTemporalDataFrame, IntoDataFrame]
        Input DataFrame (pandas, modin, polars, pyarrow, dask).
    backend : str
        Target backend ('pandas', 'modin', 'polars', 'pyarrow', 'dask').
    npartitions : int
        Number of partitions for Dask backend. Default is 1.
    backend_converter_dict : Dict[str, Callable[[pd.DataFrame, int], Any]], optional
        Backend conversion functions.
    df: Union[SupportedTemporalDataFrame, IntoDataFrame]
    backend: str
    npartitions: int :
         (Default value = 1)
    backend_converter_dict: Dict[str, Callable[[pd.DataFrame, int, Any]] :
         (Default value = TEMPORALSCOPE_BACKEND_CONVERTERS)

    Returns
    -------
    SupportedTemporalDataFrame
        Converted DataFrame in the target backend.

    Raises
    ------
    UnsupportedBackendError
        If the backend or DataFrame is unsupported.

    Examples
    --------
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
        conversion_func = TEMPORALSCOPE_BACKEND_CONVERTERS[backend]
        converted_df = conversion_func(df, npartitions)

        # Ensure Pandas compatibility if converting to Pandas
        if backend == "pandas":
            # Import inline to avoid unnecessary global imports
            if "pyarrow" in str(type(converted_df)):  # Check if it's a pyarrow object
                import pyarrow  # pragma: no cover

                if isinstance(converted_df, (pyarrow.Table, pyarrow.lib.ChunkedArray)):  # pragma: no cover
                    converted_df = converted_df.to_pandas()  # pragma: no cover
            elif "polars" in str(type(converted_df)):  # Check if it's a polars object
                import polars  # pragma: no cover

                if isinstance(converted_df, polars.DataFrame):  # pragma: no cover
                    converted_df = converted_df.to_pandas()  # pragma: no cover

        return converted_df

    except KeyError:  # pragma: no cover
        raise UnsupportedBackendError(f"The backend '{backend}' is not supported by TemporalScope.")
    except Exception as e:  # pragma: no cover
        raise UnsupportedBackendError(f"Failed to convert DataFrame: {str(e)}")


@nw.narwhalify
def check_dataframe_empty(df: SupportedTemporalDataFrame) -> bool:
    """Check if a DataFrame is empty using backend-agnostic operations.

    This function validates the input DataFrame using `is_valid_temporal_dataframe` and
    determines whether it is empty based on standard backend attributes such as `shape`.
    It handles lazy evaluation transparently for backends like Dask and Polars.

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The input DataFrame to check.
    df: SupportedTemporalDataFrame :


    Returns
    -------
    bool
        True if the DataFrame is empty, False otherwise.

    Raises
    ------
    ValueError
        If the input DataFrame is None or invalid.
    UnsupportedBackendError
        If the backend is not supported by TemporalScope.

    Examples
    --------
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

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        DataFrame to check for null values.
    column_names : List[str]
        List of column names to check.
    df: SupportedTemporalDataFrame :

    column_names: List[str] :


    Returns
    -------
    Dict[str, int]
        Dictionary mapping column names to their null value counts.

    Raises
    ------
    ValueError
        If the DataFrame is empty or a column is nonexistent.
    UnsupportedBackendError
        If the backend is unsupported.

    Examples
    --------
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

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The input DataFrame containing the column to convert.
    time_col : str
        The name of the time column to convert.
    col_expr : Any
        The Narwhals column expression for the time column.
    col_dtype : Any
        The resolved dtype of the time column.

    Returns
    -------
    SupportedTemporalDataFrame
        The DataFrame with the converted time column.

    Raises
    ------
    ValueError
        If the column is not a datetime type.
    UnsupportedBackendError
        If the backend is not supported by TemporalScope.

    Examples
    --------
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
def convert_datetime_column_to_numeric(
    df: SupportedTemporalDataFrame, time_col: str, time_unit: Literal["us", "ms", "ns"] = "us"
) -> SupportedTemporalDataFrame:
    """Convert a datetime column to a numeric representation.

    This function converts the specified datetime column into a numeric
    representation (e.g., Unix timestamp) with precision control using
    microseconds ("us") by default. It ensures compatibility across all
    Narwhals-supported backends.

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        Input DataFrame containing the datetime column.
    time_col : str
        Name of the column to convert. Must be of datetime type.
    time_unit : Literal["us", "ms", "ns"]
        Time unit for conversion ("us", "ms", "ns"). Default is "us".
        The choice of "us" provides optimal compatibility across
        Pandas, Polars, and PyArrow backends.

    Returns
    -------
    SupportedTemporalDataFrame
        DataFrame with the converted time column.

    Raises
    ------
    UnsupportedBackendError
        If the DataFrame's backend is not supported.
    ValueError
        If the specified column is not a datetime type or does not exist.

    Examples
    --------
    .. code-block:: python

        from temporalscope.core.core_utils import convert_datetime_column_to_numeric
        import pandas as pd

        # Create example DataFrame with a datetime column
        df = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=3, freq="D"), "value": [10, 20, 30]})

        # Convert 'time' column to numeric (microseconds precision)
        df = convert_datetime_column_to_numeric(df, time_col="time", time_unit="us")
        print(df)

    .. note::
        - Supports microseconds ("us"), milliseconds ("ms"), and nanoseconds ("ns").
        - Preserves timezone-aware datetimes during conversion.
        - Handles null values consistently across all supported backends.
        - Does not enforce monotonicity or data sorting. Use a sorting utility if required.
        - When using `"ns"` precision, values are cast to `Int64` to avoid overflow issues with large timestamps.

    """
    # Step 1: Validate the DataFrame
    is_valid, _ = is_valid_temporal_dataframe(df)
    if not is_valid:
        raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df).__name__}")

    # Step 2: Validate the time column exists
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Step 3: Check if the column is already numeric
    col_dtype = df.schema.get(time_col) if hasattr(df, "schema") else df[time_col].dtype
    if "int" in str(col_dtype).lower() or "float" in str(col_dtype).lower():
        return df  # No conversion needed for already numeric columns

    # Step 4: Check if DataFrame is empty before checking nulls
    if check_dataframe_empty(df):
        raise ValueError("Null or NaN values detected")

    # Now safely call check_dataframe_nulls_nans, since we know DF is not empty
    null_counts = check_dataframe_nulls_nans(df, [time_col])
    if null_counts.get(time_col, 0) > 0:
        raise ValueError(f"Null or NaN values detected in column '{time_col}'.")

    # Step 5: Ensure it is strictly a datetime column
    if "datetime" not in str(col_dtype).lower():
        raise ValueError(f"Column '{time_col}' must specifically be a datetime type to convert.")

    # Step 6: Perform the conversion to numeric with specified time unit
    # For ns, use Int64 due to range issues. For us/ms, Float64 is safe.
    target_dtype = nw.Int64() if time_unit == "ns" else nw.Float64()
    return df.with_columns([nw.col(time_col).dt.timestamp(time_unit=time_unit).cast(target_dtype).alias(time_col)])


@nw.narwhalify
def convert_time_column_to_datetime(
    df: SupportedTemporalDataFrame, time_col: str, col_expr: Any, col_dtype: Any
) -> SupportedTemporalDataFrame:
    """Convert a string or numeric column to datetime using Narwhals API.

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The input DataFrame containing the column to convert.
    time_col : str
        The name of the time column to convert.
    col_expr : Any
        The Narwhals column expression for the time column.
    col_dtype : Any
        The resolved dtype of the time column.

    Returns
    -------
    SupportedTemporalDataFrame
        The DataFrame with the converted time column.

    Raises
    ------
    ValueError
        If the column is not convertible to datetime.
    UnsupportedBackendError
        If the backend is not supported by TemporalScope.

    Examples
    --------
    .. code-block:: python

        df = pd.DataFrame({"time": [1672531200000, 1672617600000]})  # Unix timestamps
        df = convert_time_column_to_datetime(df, "time", nw.col("time"), df["time"].dtype)
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

    Parameters
    ----------
    time_col : str
        The name of the time column to validate.
    col_dtype : Any
        The resolved dtype of the time column.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the column is neither numeric nor datetime.

    Examples
    --------
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

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The input DataFrame to process.
    time_col : str
        The name of the time column to validate or convert.
    conversion_type : Optional[str]
        Optional. Specify the conversion type:
        - 'numeric': Convert to Float64.
        - 'datetime': Convert to Datetime.
        - None: Validate only.

    Returns
    -------
    SupportedTemporalDataFrame
        The validated and optionally converted DataFrame.

    Raises
    ------
    TimeColumnError
        If validation or conversion fails or if an invalid conversion_type is provided.
    ValueError
        If the column dtype cannot be resolved.
    UnsupportedBackendError
        If the backend is not supported by TemporalScope.

    Examples
    --------
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
        return convert_time_column_to_datetime(df, time_col, nw.col(time_col), col_dtype)

    # Validation-only path
    validate_time_column_type(time_col, col_dtype)

    return df


@nw.narwhalify
def validate_dataframe_column_types(df: SupportedTemporalDataFrame, time_col: str) -> None:
    """Validate the column types in a DataFrame for compatibility with TemporalScope.

    This function ensures the following:
    - The specified `time_col` must be of type numeric or datetime.
    - All other columns in the DataFrame must be of numeric type.

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The input DataFrame to validate.
    time_col : str
        The name of the time column to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the `time_col` does not exist or has an invalid type.
    ValueError
        If any non-time column has an invalid type.
    UnsupportedBackendError
        If the backend is not supported by TemporalScope.

    Examples
    --------
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

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The input DataFrame to sort.
    time_col : str
        The name of the column to sort by. Must exist in the DataFrame.
    ascending : bool, optional
        Sort direction. Defaults to True (ascending order).

    Returns
    -------
    SupportedTemporalDataFrame
        A DataFrame sorted by the specified time column.

    Raises
    ------
    ValueError
        If the `time_col` does not exist in the DataFrame or has invalid type.
    UnsupportedBackendError
        If the backend is not supported by TemporalScope.

    Examples
    --------
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

    Warnings
    --------
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
    df: SupportedTemporalDataFrame, time_col: str, raise_error: bool = True, id_col: str = ""
) -> None:
    """Validate temporal uniqueness in the given DataFrame.

    This function checks that timestamps in the specified `time_col` are unique within
    the groups defined by the `id_col` (e.g., within a patient's records). It does not enforce ordering,
    allowing for mixed-frequency data and flexible temporal patterns.

    Parameters
    ----------
    df : SupportedTemporalDataFrame
        The DataFrame to validate.
    time_col : str
        The column representing time.
    raise_error : bool
        Whether to raise an error if validation fails.
    id_col : str
        An optional string to indicate the grouping identifier (e.g., group name).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If validation fails and `raise_error` is True.
        :warns UserWarning: If validation fails and `raise_error` is False.

    Examples
    --------
    .. code-block:: python

        import narwhals as nw
        import pandas as pd

        # Create insurance claims data with patient visits
        df = pd.DataFrame(
            {
                "patient_id": [1, 1, 1, 2, 2],
                "time": [
                    "2023-01-01",
                    "2023-02-15",
                    "2023-04-01",  # Patient 1's visits
                    "2023-01-01",
                    "2023-03-15",
                ],  # Patient 2's visits
                "claim_amount": [100.0, 250.0, 150.0, 300.0, 200.0],
            }
        )

        # Validate timestamps within each patient's records
        for patient in df["patient_id"].unique():
            patient_records = df[df["patient_id"] == patient]
            validate_temporal_uniqueness(
                patient_records, time_col="time", id_col=f"patient {patient} "
            )  # Will pass - each patient has unique visit dates

        # Example with duplicate timestamps
        df_invalid = pd.DataFrame(
            {
                "patient_id": [1, 1, 1],
                "time": ["2023-01-01", "2023-01-01", "2023-02-15"],  # Duplicate visit date
                "claim_amount": [100.0, 150.0, 200.0],
            }
        )

        # This will raise ValueError: "Duplicate timestamps in patient 1 column 'time'."
        validate_temporal_uniqueness(df_invalid, time_col="time", id_col="patient 1 ")

    .. note::
        - This function only validates uniqueness within the given `id_col` (e.g., per patient).
        - It does not enforce temporal ordering, allowing for mixed-frequency data.
        - Different `id_col` groups (e.g., different patients) can have events on the same dates.

    """
    # Step 1: Validate time column type and convert if needed
    try:
        df = nw.from_native(validate_and_convert_time_column(df, time_col))
    except Exception as e:
        raise TimeColumnError(f"Invalid time column: {str(e)}")

    # Step 2: Check for existence of id_col
    if id_col and id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' does not exist.")

    # Step 3: Check for null values in the time column
    null_counts = check_dataframe_nulls_nans(df, [time_col])
    if null_counts.get(time_col, 0) > 0:
        message = f"Null values found in {id_col}column '{time_col}'."
        if raise_error:
            raise ValueError(message)
        warnings.warn(message)  # pragma: no cover

    # Step 4: Materialize lazy DataFrames
    if is_lazy_evaluation(df):
        df = df.collect() if hasattr(df, "collect") else df.compute()

    # Handle PyArrow's unique count differently
    if get_dataframe_backend(df) == "pyarrow":
        df = nw.from_native(df.to_pandas())

    # Step 5: Check for duplicates using proper aggregation and raise error if found
    if id_col:
        # Count occurrences of each timestamp within each group
        duplicate_check_df = df.group_by([id_col, time_col], drop_null_keys=True).agg(
            [nw.col(time_col).count().alias("count")]
        )
        # If any timestamp appears more than once in its group, it's a duplicate
        has_duplicates = duplicate_check_df.select([(nw.col("count") > 1).any()]).item()
        if has_duplicates:
            message = f"Duplicate timestamps in id_col '{id_col}' column '{time_col}'."
            if raise_error:
                raise ValueError(message)
            warnings.warn(message)  # pragma: no cover
    else:
        # Global duplicate check
        duplicate_check_df = df.select(
            [nw.col(time_col).n_unique().alias("unique_count"), nw.col(time_col).count().alias("total_count")]
        )
        if (
            duplicate_check_df.select([(nw.col("unique_count") < nw.col("total_count")).alias("has_duplicates")])
            .select([nw.col("has_duplicates").any()])
            .item()
        ):
            message = f"Duplicate timestamps in id_col '' column '{time_col}'."
            if raise_error:
                raise ValueError(message)
            warnings.warn(message)
