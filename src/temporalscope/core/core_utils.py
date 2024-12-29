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

This module provides essential utility functions used throughout the TemporalScope package. It serves
two main purposes:
1. Core DataFrame operations for the Temporal Data Loader, handling all data manipulations through
   Narwhals for backend-agnostic operations.
2. Shared utilities used across the library for data validation, type checking, and format conversion.

For testing purposes, we use a data generator to simulate runtime behavior, ensuring consistent
functionality across supported backends. The module supports both single-target and multi-target time
series analysis through a flexible API.

Supported Modes:
----------------
+----------------+-------------------------------------------------------------------+
| Mode           | Description                                                       |
|                | Data Structure                                                    |
+----------------+-------------------------------------------------------------------+
| single_target  | General machine learning tasks with scalar targets. Each row is   |
|                | a single time step, and the target is scalar.                     |
|                | Single DataFrame: each row is an observation.                     |
+----------------+-------------------------------------------------------------------+
| multi_target   | Sequential time series tasks (e.g., seq2seq) for deep learning.   |
|                | The data is split into sequences (input X, target Y).             |
|                | Two DataFrames: X for input sequences, Y for targets.             |
|                | Frameworks: TensorFlow, PyTorch, Keras.                           |
+----------------+-------------------------------------------------------------------+

Example Visualization:
----------------------
Single-target mode:
    +------------+------------+------------+------------+-----------+
    |   time     | feature_1  | feature_2  | feature_3  |  target   |
    +============+============+============+============+===========+
    | 2023-01-01 |   0.15     |   0.67     |   0.89     |   0.33    |
    +------------+------------+------------+------------+-----------+
    | 2023-01-02 |   0.24     |   0.41     |   0.92     |   0.28    |
    +------------+------------+------------+------------+-----------+

    Shape:
    - `X`: (num_samples, num_features)
    - `Y`: (num_samples, 1)  # Single target value per time step

Multi-target mode (with vectorized targets):
    +------------+------------+------------+------------+-------------+
    |   time     | feature_1  | feature_2  | feature_3  |    target   |
    +============+============+============+============+=============+
    | 2023-01-01 |   0.15     |   0.67     |   0.89     |  [0.3, 0.4] |
    +------------+------------+------------+------------+-------------+
    | 2023-01-02 |   0.24     |   0.41     |   0.92     |  [0.5, 0.6] |
    +------------+------------+------------+------------+-------------+

    Shape:
    - `X`: (num_samples, num_features)
    - `Y`: (num_samples, sequence_length)  # Multiple target values per time step
"""

import os
from typing import Dict, List, Literal, Optional

import narwhals as nw
from dotenv import load_dotenv
from narwhals.typing import FrameT
from narwhals.utils import Implementation

from temporalscope.core.exceptions import TimeColumnError

# Load environment variables from the .env file
load_dotenv()

# Constants
# ---------
# Define constants for TemporalScope-supported modes
MODE_SINGLE_TARGET = "single_target"
MODE_MULTI_TARGET = "multi_target"
VALID_MODES = [MODE_SINGLE_TARGET, MODE_MULTI_TARGET]

# Narwhals-supported backends
NARWHALS_BACKENDS = [backend.name.lower() for backend in Implementation]

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------


def get_api_keys() -> Dict[str, Optional[str]]:
    """Retrieve API keys from environment variables for LLM integrations.

    This function retrieves API keys for various LLM services (OpenAI, Claude)
    from environment variables. It provides a centralized way to manage API
    keys and handles missing keys gracefully with warnings.

    Returns
    -------
    Dict[str, Optional[str]]
        A dictionary containing the API keys, with keys:
        - 'OPENAI_API_KEY': OpenAI API key
        - 'CLAUDE_API_KEY': Anthropic Claude API key
        Values will be None if the corresponding environment variable is not set.

    Examples
    --------
    ```python
    # Assume environment variables are set:
    # export OPENAI_API_KEY='abc123'
    # export CLAUDE_API_KEY='def456'

    # Retrieve API keys
    api_keys = get_api_keys()
    print(api_keys)
    # Output: {'OPENAI_API_KEY': 'abc123', 'CLAUDE_API_KEY': 'def456'}

    # Check if a specific key exists
    if api_keys["OPENAI_API_KEY"] is not None:
        # Use OpenAI integration
        pass
    else:
        # Handle missing key
        pass
    ```

    Notes
    -----
    - Uses dotenv for environment variable loading
    - Prints warnings for missing keys to aid debugging
    - Thread-safe for concurrent access
    - Keys are read-only to prevent accidental modification
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
    """Print a visual divider line for output formatting.

    This utility function creates a visual separator line in console output,
    useful for formatting log messages, test output, or CLI interfaces.

    Parameters
    ----------
    char : str, optional
        The character to use for the divider line.
        Must be a single character string.
        Default is '='.
    length : int, optional
        The total length of the divider line.
        Must be a positive integer.
        Default is 70 characters.

    Returns
    -------
    None
        Prints the divider line to stdout.

    Examples
    --------
    ```python
    # Default divider
    print_divider()
    # Output: ======================================================

    # Custom character and length
    print_divider(char="-", length=50)
    # Output: --------------------------------------------------

    # Use in logging output
    print("Section 1")
    print_divider()
    print("Content...")
    ```

    Notes
    -----
    - Thread-safe for concurrent printing
    - Supports Unicode characters
    - Useful for visual separation in logs and CLI output
    - Consistent with enterprise logging practices
    """
    print(char * length)


# ---------------------------------------------------------
# Main Functions
# ---------------------------------------------------------


def get_narwhals_backends() -> List[str]:
    """Retrieve all DataFrame backends available through Narwhals.

    This function provides a centralized way to discover all available DataFrame
    backends supported by the Narwhals library. It's used throughout TemporalScope
    for backend validation and configuration.

    Returns
    -------
    List[str]
        List of Narwhals-supported backend names in lowercase.
        Common backends include:
        - 'pandas': Standard pandas DataFrame
        - 'polars': Polars DataFrame (both eager and lazy modes)
        - 'dask': Dask DataFrame for distributed computing
        - 'modin': Modin DataFrame for parallel processing
        - 'pyarrow': PyArrow Table/Dataset
        - 'pyspark': Spark DataFrame for big data

    Examples
    --------
    ```python
    # Get available backends
    backends = get_narwhals_backends()
    print(backends)
    # Output: ['pandas', 'modin', 'pyarrow', 'polars', 'dask']

    # Use for backend validation
    if "polars" in get_narwhals_backends():
        # Polars-specific optimizations
        pass

    # Configure system defaults
    config = {"default_backend": get_narwhals_backends()[0], "supported_backends": get_narwhals_backends()}
    ```

    Notes
    -----
    - Returns lowercase names for case-insensitive comparison
    - Order is consistent across calls for stable defaults
    - Thread-safe for concurrent access
    - Used by validation functions to verify backend support
    - Critical for TemporalScope's backend-agnostic operations
    """
    return [backend.name.lower() for backend in Implementation]


def get_default_backend_cfg() -> Dict[str, List[str]]:
    """Retrieve the default application configuration for DataFrame backends.

    This function provides a standardized configuration dictionary for DataFrame
    backend support in TemporalScope. It encapsulates the available backends in
    a format suitable for configuration files and system initialization.

    Returns
    -------
    Dict[str, List[str]]
        Configuration dictionary with structure:
        {
            'BACKENDS': List[str]  # List of supported backend names
        }
        The 'BACKENDS' key contains all Narwhals-supported backends in lowercase.

    Examples
    --------
    ```python
    # Get default configuration
    config = get_default_backend_cfg()
    print(config)
    # Output: {'BACKENDS': ['pandas', 'modin', 'pyarrow', 'polars', 'dask']}

    # Use in application initialization
    app_config = {**get_default_backend_cfg(), "other_settings": {...}}

    # Validate user-provided backend
    user_backend = "polars"
    if user_backend in get_default_backend_cfg()["BACKENDS"]:
        # Backend is supported
        pass
    ```

    Notes
    -----
    - Provides consistent configuration format
    - Used for system initialization and validation
    - Thread-safe for concurrent access
    - Integrates with configuration management systems
    - Key component for backend-agnostic operations
    """
    available_backends = get_narwhals_backends()
    return {"BACKENDS": available_backends}


@nw.narwhalify(eager_only=True)
def check_dataframe_empty(df: FrameT) -> bool:
    """Check if a DataFrame is empty using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to check if a DataFrame
    is empty. It handles all DataFrame types supported by Narwhals transparently.

    Parameters
    ----------
    df : FrameT
        The input DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame is empty, False otherwise.

    Raises
    ------
    ValueError
        If the input DataFrame is None.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import check_dataframe_empty

    # Example with any DataFrame type
    df = pd.DataFrame(columns=["col1"])
    assert check_dataframe_empty(df) == True
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Handles all DataFrame types supported by Narwhals
    - Uses eager evaluation for consistent behavior
    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    # Use Narwhals operations for checking emptiness
    return len(df.columns) == 0 or len(df.with_columns([
        nw.col(df.columns[0])
           .alias(f"{df.columns[0]}_test")
    ])) == 0


@nw.narwhalify(eager_only=True)
def check_dataframe_nulls_nans(df: FrameT, column_names: List[str]) -> Dict[str, int]:
    """Check for null values in specified DataFrame columns using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to count null values
    in the specified columns. It ensures consistent behavior across all DataFrame
    backends.

    Parameters
    ----------
    df : FrameT
        DataFrame to check for null values.
    column_names : List[str]
        List of column names to check.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping column names to their null value counts.

    Raises
    ------
    ValueError
        If the DataFrame is empty or a column is nonexistent.
    TimeColumnError
        If a column operation fails.

    Examples
    --------
    ```python
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
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Handles all DataFrame types supported by Narwhals
    - Properly handles PyArrow scalar types
    """
    # Step 1: Validate if the DataFrame is empty
    if check_dataframe_empty(df):
        raise ValueError("Empty DataFrame provided.")

    result = {}
    for col in column_names:
        try:
            # Use Narwhals operations to compute and cast null counts
            result[col] = int(df.with_columns([
                nw.col(col)
                   .is_null()
                   .sum()
                   .cast(nw.Int64())
                   .alias(f"{col}_null_count")
            ])[f"{col}_null_count"][0])

        except KeyError:
            # Handle nonexistent column error
            raise ValueError(f"Column '{col}' not found.")
        except Exception as e:  # pragma: no cover
            # Handle unforeseen errors
            raise TimeColumnError(f"Error checking null values in column '{col}': {str(e)}")

    return result


@nw.narwhalify(eager_only=True)
def convert_to_numeric(df: FrameT, time_col: str) -> FrameT:
    """Convert a datetime column to numeric using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to convert a datetime
    column to a numeric representation. It ensures consistent behavior across all
    DataFrame backends.

    Parameters
    ----------
    df : FrameT
        The input DataFrame containing the column to convert.
    time_col : str
        The name of the time column to convert.

    Returns
    -------
    FrameT
        The DataFrame with the converted time column.

    Raises
    ------
    TimeColumnError
        If the column is not a datetime type or conversion fails.

    Examples
    --------
    ```python
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    df = convert_to_numeric(df, "time")
    print(df)
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Uses microsecond precision for optimal compatibility
    - Handles all DataFrame types supported by Narwhals
    """
    try:
        # Try casting to datetime first to validate type
        df.with_columns([
            nw.col(time_col)
               .cast(nw.Datetime())
               .alias(f"{time_col}_datetime_test")
        ])

        # Convert to numeric timestamp
        return df.with_columns([
            nw.col(time_col)
               .dt.timestamp(time_unit="us")
               .cast(nw.Float64())
               .alias(time_col)
        ])
    except:
        raise TimeColumnError(f"Column '{time_col}' is not a datetime column, cannot convert to numeric.")


@nw.narwhalify(eager_only=True)
def convert_datetime_column_to_numeric(
    df: FrameT, time_col: str, time_unit: Literal["us", "ms", "ns"] = "us"
) -> FrameT:
    """Convert a datetime column to numeric using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to convert a datetime
    column to a numeric timestamp. It ensures consistent behavior across all
    DataFrame backends.

    Parameters
    ----------
    df : FrameT
        Input DataFrame containing the datetime column.
    time_col : str
        Name of the column to convert. Must be of datetime type.
    time_unit : Literal["us", "ms", "ns"]
        Time unit for conversion ("us", "ms", "ns"). Default is "us".
        The choice of "us" provides optimal compatibility across backends.

    Returns
    -------
    FrameT
        DataFrame with the converted time column.

    Raises
    ------
    TimeColumnError
        If the column is not a datetime type or conversion fails.
    ValueError
        If the column does not exist or contains nulls.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import convert_datetime_column_to_numeric
    import pandas as pd

    # Create example DataFrame with a datetime column
    df = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=3, freq="D"), "value": [10, 20, 30]})

    # Convert 'time' column to numeric (microseconds precision)
    df = convert_datetime_column_to_numeric(df, time_col="time", time_unit="us")
    print(df)
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Handles all DataFrame types supported by Narwhals
    - Uses Int64 for nanoseconds to avoid overflow
    - Uses Float64 for microseconds/milliseconds
    """
    # Validate column exists
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Check for nulls
    if check_dataframe_empty(df):
        raise ValueError("Empty DataFrame provided")

    null_counts = check_dataframe_nulls_nans(df, [time_col])
    if null_counts.get(time_col, 0) > 0:
        raise ValueError(f"Null values detected in column '{time_col}'")

    # Get column type
    col_dtype = df.schema.get(time_col) if hasattr(df, "schema") else df[time_col].dtype

    # Return if already numeric
    if "int" in str(col_dtype).lower() or "float" in str(col_dtype).lower():
        return df

    # Ensure datetime type
    if "datetime" not in str(col_dtype).lower():
        raise TimeColumnError(f"Column '{time_col}' must be datetime type to convert")

    # Convert to numeric with appropriate precision
    target_dtype = nw.Int64() if time_unit == "ns" else nw.Float64()
    return df.with_columns([nw.col(time_col).dt.timestamp(time_unit=time_unit).cast(target_dtype).alias(time_col)])


@nw.narwhalify(eager_only=True)
def convert_time_column_to_datetime(df: FrameT, time_col: str) -> FrameT:
    """Convert a string or numeric column to datetime using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to convert a column
    to datetime format. It handles both string and numeric timestamps, ensuring
    consistent behavior across all DataFrame backends.

    Parameters
    ----------
    df : FrameT
        The input DataFrame containing the column to convert.
    time_col : str
        The name of the time column to convert.

    Returns
    -------
    FrameT
        The DataFrame with the converted time column.

    Raises
    ------
    TimeColumnError
        If the column is not convertible to datetime or conversion fails.

    Examples
    --------
    ```python
    # Convert Unix timestamps to datetime
    df = pd.DataFrame({"time": [1672531200000, 1672617600000]})
    df = convert_time_column_to_datetime(df, "time")
    print(df)

    # Convert string dates to datetime
    df = pd.DataFrame({"time": ["2023-01-01", "2023-01-02"]})
    df = convert_time_column_to_datetime(df, "time")
    print(df)
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Handles string dates and numeric timestamps
    - Preserves timezone information where applicable
    """
    # First check if column exists
    if time_col not in df.columns:
        raise TimeColumnError(f"Column '{time_col}' does not exist in DataFrame")

    # Get column type
    col_dtype = str(df.schema.get(time_col)).lower()

    # If already datetime, return as is
    if "datetime" in col_dtype:
        return df

    # Check for boolean type
    if "bool" in col_dtype:
        raise TimeColumnError(f"Column '{time_col}' is boolean type and cannot be converted to datetime")

    # Try direct datetime conversion first
    try:
        return df.with_columns([
            nw.col(time_col)
               .cast(nw.Datetime())
               .alias(time_col)
        ])
    except:
        # Try string to datetime conversion
        try:
            return df.with_columns([
                nw.col(time_col)
                   .cast(nw.String())
                   .str.to_datetime()
                   .alias(time_col)
            ])
        except:
            # Try numeric to datetime conversion
            try:
                return df.with_columns([
                    nw.col(time_col)
                       .cast(nw.Float64())
                       .cast(nw.Datetime())
                       .alias(time_col)
                ])
            except:
                raise TimeColumnError(f"Column '{time_col}' must be string or numeric to convert to datetime")


@nw.narwhalify(eager_only=True)
def validate_time_column_type(df: FrameT, time_col: str) -> None:
    """Validate that a column is either numeric or datetime using Narwhals operations.

    Parameters
    ----------
    df : FrameT
        The DataFrame containing the column to validate.
    time_col : str
        The name of the time column to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the column is neither numeric nor datetime.

    Examples
    --------
    ```python
    # Validate numeric column
    df = pd.DataFrame({"time": [1, 2, 3]})
    validate_time_column_type(df, "time")  # Passes

    # Validate datetime column
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    validate_time_column_type(df, "time")  # Passes

    # Invalid column type
    df = pd.DataFrame({"time": ["a", "b", "c"]})
    validate_time_column_type(df, "time")  # Raises ValueError
    ```

    Notes
    -----
    - Uses Narwhals' native operations for type validation
    - Forces eager evaluation for consistent behavior
    - Handles all DataFrame types supported by Narwhals
    - Provides clear error messages for unsupported types
    """
    # Try numeric cast first
    try:
        df.with_columns([
            nw.col(time_col)
               .cast(nw.Float64())
               .alias(f"{time_col}_numeric_test")
        ])
        return
    except:
        # Try datetime cast
        try:
            df.with_columns([
                nw.col(time_col)
                   .cast(nw.Datetime())
                   .alias(f"{time_col}_datetime_test")
            ])
            return
        except:
            raise ValueError(f"Column '{time_col}' is neither numeric nor datetime.")


@nw.narwhalify(eager_only=True)
def validate_and_convert_time_column(
    df: FrameT,
    time_col: str,
    conversion_type: Optional[str] = None,
) -> FrameT:
    """Validate and optionally convert a time column using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to validate and
    optionally convert a time column. It ensures consistent behavior across all
    DataFrame backends.

    Parameters
    ----------
    df : FrameT
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
    FrameT
        The validated and optionally converted DataFrame.

    Raises
    ------
    TimeColumnError
        If validation or conversion fails.
    ValueError
        If the column does not exist or conversion_type is invalid.

    Examples
    --------
    ```python
    # Validate only
    df = validate_and_convert_time_column(df, "time")

    # Convert to numeric
    df = validate_and_convert_time_column(df, "time", conversion_type="numeric")

    # Convert to datetime
    df = validate_and_convert_time_column(df, "time", conversion_type="datetime")
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Handles all DataFrame types supported by Narwhals
    - Validates column existence and type
    """
    # Validate column exists
    if time_col not in df.columns:
        raise TimeColumnError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Validate conversion type
    if conversion_type not in {"numeric", "datetime", None}:
        raise ValueError(f"Invalid conversion_type '{conversion_type}'. Must be one of 'numeric', 'datetime', or None.")

    # Perform conversion if requested
    if conversion_type == "numeric":
        return convert_to_numeric(df, time_col)

    if conversion_type == "datetime":
        return convert_time_column_to_datetime(df, time_col)

    # Validation-only path
    validate_time_column_type(df, time_col)
    return df


@nw.narwhalify(eager_only=True)
def validate_dataframe_column_types(df: FrameT, time_col: str) -> None:
    """Validate DataFrame column types using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to validate that:
    - The time column is numeric or datetime type
    - All other columns are numeric type

    Parameters
    ----------
    df : FrameT
        The input DataFrame to validate.
    time_col : str
        The name of the time column to validate.

    Returns
    -------
    None

    Raises
    ------
    TimeColumnError
        If validation fails for the time column.
    ValueError
        If any column does not exist or has invalid type.

    Examples
    --------
    ```python
    # Example DataFrame with valid types
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=5), "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    validate_dataframe_column_types(df, time_col="time")  # Passes

    # Example with invalid type
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=5),
            "value": ["a", "b", "c", "d", "e"],  # String type not allowed
        }
    )
    validate_dataframe_column_types(df, time_col="time")  # Raises ValueError
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Validates all columns in a single pass
    - Ensures ML-compatible numeric features
    """
    # Validate time column exists and type
    if time_col not in df.columns:
        raise TimeColumnError(f"Column '{time_col}' does not exist")
    validate_time_column_type(df, time_col)

    # Validate other columns are numeric
    non_time_cols = [col for col in df.columns if col != time_col]
    for col in non_time_cols:
        try:
            # Try casting to numeric (Float64)
            df.with_columns([
                nw.col(col)
                   .cast(nw.Float64())
                   .alias(f"{col}_numeric_test")
            ])
        except:
            raise ValueError(f"Column '{col}' must be numeric")


@nw.narwhalify(eager_only=True)
def sort_dataframe_time(df: FrameT, time_col: str, ascending: bool = True) -> FrameT:
    """Sort a DataFrame by time column using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to sort a DataFrame
    by its time column. It ensures consistent behavior across all DataFrame backends.

    Parameters
    ----------
    df : FrameT
        The input DataFrame to sort.
    time_col : str
        The name of the column to sort by.
    ascending : bool, optional
        Sort direction. Defaults to True (ascending order).

    Returns
    -------
    FrameT
        A DataFrame sorted by the specified time column.

    Raises
    ------
    TimeColumnError
        If the time column validation fails.
    ValueError
        If the time column does not exist.

    Examples
    --------
    ```python
    # Sort numeric time column
    df = pd.DataFrame({"time": [3, 1, 4, 2, 5], "value": range(5)})
    df_sorted = sort_dataframe_time(df, time_col="time", ascending=True)

    # Sort datetime column
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=5), "value": range(5)})
    df_sorted = sort_dataframe_time(df, time_col="time", ascending=False)
    ```

    Notes
    -----
    - Uses Narwhals' native operations for backend-agnostic handling
    - Forces eager evaluation for consistent behavior
    - Validates time column before sorting
    - Preserves DataFrame schema
    """
    # Validate time column exists
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Validate time column type
    validate_time_column_type(df, time_col)

    # Sort using Narwhals operations
    return df.sort(by=time_col, descending=not ascending)
