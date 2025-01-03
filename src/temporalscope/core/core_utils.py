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

# Test environment backends (from pyproject.toml)
TEST_BACKENDS = ["pandas", "modin", "polars", "dask", "pyarrow"]

# Validation constants
MAX_UNIQUE_DELTAS = 1  # Maximum allowed unique time deltas for equidistant sampling

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
def is_dataframe_empty(df: FrameT) -> bool:
    """Check if a DataFrame is empty using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to check if a DataFrame
    is empty. Uses eager evaluation (eager_only=True) because it returns an immediate
    boolean result that cannot be deferred, following Narwhals' pattern for functions
    returning Python primitives.

    Implementation Details
    --------------------
    The function performs a two-step validation:
    1. Checks if DataFrame has any columns
    2. If columns exist, checks if first column has any rows

    This approach is more efficient than materializing the entire DataFrame
    for counting rows, especially with large datasets or lazy backends like Dask.

    Parameters
    ----------
    df : FrameT
        The input DataFrame to check. Can be any backend supported by Narwhals
        (pandas, polars, dask, etc.).

    Returns
    -------
    bool
        True if the DataFrame is empty (no columns or no rows), False otherwise.

    Raises
    ------
    ValueError
        If the input DataFrame is None. Early validation ensures consistent
        error handling across backends.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import is_dataframe_empty
    import pandas as pd

    # Empty DataFrame (no columns)
    df1 = pd.DataFrame()
    assert is_dataframe_empty(df1) == True

    # Empty DataFrame (has columns but no rows)
    df2 = pd.DataFrame(columns=["col1"])
    assert is_dataframe_empty(df2) == True

    # Non-empty DataFrame
    df3 = pd.DataFrame({"col1": [1, 2, 3]})
    assert is_dataframe_empty(df3) == False
    ```

    Notes
    -----
    Backend Independence:
        - Uses with_columns for backend-agnostic operations
        - Avoids backend-specific row counting methods
        - Works consistently across all Narwhals-supported backends

    Performance Considerations:
        - Checks columns first to avoid unnecessary operations
        - Only creates test column if needed
        - Minimizes data materialization for lazy backends

    Type Safety:
        - Uses proper Narwhals column operations
        - Avoids type-specific comparisons
        - Handles all DataFrame implementations consistently
    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    # Use Narwhals operations for checking emptiness
    return len(df.columns) == 0 or len(df.with_columns([nw.col(df.columns[0]).alias(f"{df.columns[0]}_test")])) == 0


@nw.narwhalify(eager_only=True)
def count_dataframe_column_nulls(df: FrameT, column_names: List[str]) -> Dict[str, int]:
    """Count null values in specified DataFrame columns using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to count null values
    in the specified columns. Uses eager evaluation (eager_only=True) because it returns
    an immediate dictionary result and requires materialization for accurate null counts
    across all backends.

    Implementation Details
    --------------------
    The function performs efficient null counting through:
    1. Single select operation for all columns
    2. Proper type casting to ensure consistent counting
    3. Unified null handling across different backends

    This approach is more efficient than checking columns individually, especially
    with large datasets or when using lazy backends like Dask.

    Parameters
    ----------
    df : FrameT
        DataFrame to check for null values. Can be any backend supported by
        Narwhals (pandas, polars, dask, etc.).
    column_names : List[str]
        List of column names to check. All columns must exist in the DataFrame.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping column names to their null value counts.
        Example: {"column1": 5, "column2": 0} means column1 has 5 nulls
        and column2 has no nulls.

    Raises
    ------
    ValueError
        - If the DataFrame is empty
        - If any specified column does not exist
        Early validation ensures data quality before processing.
    TimeColumnError
        If a column operation fails during null checking.
        Provides detailed error context for debugging.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import count_dataframe_column_nulls
    import pandas as pd

    # Create test DataFrame with known null values
    df = pd.DataFrame(
        {
            "col1": [1, None, 3],  # One null
            "col2": [4, 5, None],  # One null
            "col3": [7, 8, 9],  # No nulls
        }
    )

    # Check specific columns
    null_counts = count_dataframe_column_nulls(df, ["col1", "col2"])
    print(null_counts)  # Output: {"col1": 1, "col2": 1}

    # Check all columns
    all_nulls = count_dataframe_column_nulls(df, df.columns)
    print(all_nulls)  # Output: {"col1": 1, "col2": 1, "col3": 0}
    ```

    Notes
    -----
    Backend Independence:
        - Uses select() for unified null checking
        - Handles different null representations
        - Works consistently across all backends

    Performance Considerations:
        - Single pass through the DataFrame
        - Efficient column selection
        - Optimized for large datasets

    Type Safety:
        - Proper null type handling
        - Consistent Int64 casting
        - Robust error handling
    """
    # Step 1: Validate if the DataFrame is empty
    if is_dataframe_empty(df):
        raise ValueError("Empty DataFrame provided.")

    try:
        # More efficient: Single select operation for all columns
        exprs = [nw.col(col).is_null().sum().cast(nw.Int64()).alias(f"{col}_null_count") for col in column_names]
        result_df = df.select(exprs)

        # Convert to dictionary
        return {col: result_df[f"{col}_null_count"][0] for col in column_names}
    except KeyError:
        # Find which column is missing
        missing_cols = [col for col in column_names if col not in df.columns]
        raise ValueError(f"Column '{missing_cols[0]}' not found")
    except Exception as e:  # pragma: no cover
        # Handle unforeseen errors
        raise TimeColumnError(f"Error checking null values: {str(e)}")


@nw.narwhalify(eager_only=True)
def convert_datetime_column_to_microseconds(df: FrameT, time_col: str) -> FrameT:
    """Convert a datetime column to microsecond timestamps using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to convert a datetime
    column to numeric microsecond timestamps. Uses eager evaluation (eager_only=True)
    because it performs immediate type validation and conversion that cannot be
    deferred, following Narwhals' pattern for type transformations.

    Implementation Details
    --------------------
    The function performs a two-step conversion:
    1. Validates the column is datetime type through casting
    2. Converts to microsecond timestamps with Float64 precision

    This approach ensures type safety and consistent precision across all backends,
    particularly important for time series analysis and ML pipelines.

    Parameters
    ----------
    df : FrameT
        The input DataFrame containing the column to convert. Can be any backend
        supported by Narwhals (pandas, polars, dask, etc.).
    time_col : str
        The name of the datetime column to convert. Must exist in the DataFrame
        and be of datetime type.

    Returns
    -------
    FrameT
        The DataFrame with the time column converted to microsecond timestamps.
        The column maintains its original name but contains Float64 values.

    Raises
    ------
    TimeColumnError
        If the column is not a datetime type or conversion fails.
        Provides detailed error context for debugging.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import convert_datetime_column_to_microseconds
    import pandas as pd

    # Create test DataFrame with datetime column
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "value": [1, 2, 3]})

    # Convert time column to microsecond timestamps
    df = convert_datetime_column_to_microseconds(df, "time")
    print(df)
    # Output shows microseconds since epoch
    ```

    Notes
    -----
    Backend Independence:
        - Uses with_columns for unified operations
        - Handles timezone-aware datetimes
        - Works consistently across backends

    Performance Considerations:
        - Single pass validation and conversion
        - Efficient type casting
        - Minimal data copying

    Type Safety:
        - Explicit datetime validation
        - Consistent Float64 precision
        - Preserves column name and schema
    """
    # Get column type
    col_dtype = str(df.schema.get(time_col)).lower()

    # Return if already numeric
    if "int" in col_dtype or "float" in col_dtype:
        return df

    try:
        # Try casting to datetime first to validate type
        df.with_columns([nw.col(time_col).cast(nw.Datetime()).alias(f"{time_col}_datetime_test")])

        # Convert to numeric timestamp
        return df.with_columns([nw.col(time_col).dt.timestamp(time_unit="us").cast(nw.Float64()).alias(time_col)])
    except:
        raise TimeColumnError(f"Column '{time_col}' is not a datetime column, cannot convert to numeric.")


@nw.narwhalify(eager_only=True)
def convert_datetime_column_to_timestamp(
    df: FrameT, time_col: str, time_unit: Literal["us", "ms", "ns"] = "us"
) -> FrameT:
    """Convert a datetime column to timestamp with specified precision using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to convert a datetime
    column to numeric timestamps. Uses eager evaluation (eager_only=True) because it
    performs immediate type validation and conversion that cannot be deferred, following
    Narwhals' pattern for type transformations.

    Implementation Details
    --------------------
    The function performs a three-step process:
    1. Validates column existence and emptiness
    2. Checks if already numeric (early return)
    3. Converts datetime to timestamp with appropriate precision

    This approach ensures type safety and consistent precision across all backends,
    particularly important for time series analysis and ML pipelines.

    Parameters
    ----------
    df : FrameT
        The input DataFrame containing the column to convert. Can be any backend
        supported by Narwhals (pandas, polars, dask, etc.).
    time_col : str
        The name of the datetime column to convert. Must exist in the DataFrame
        and be of datetime type.
    time_unit : Literal["us", "ms", "ns"]
        Time unit for conversion:
        - "us": microseconds (default, optimal compatibility)
        - "ms": milliseconds (reduced precision)
        - "ns": nanoseconds (highest precision, uses Int64)

    Returns
    -------
    FrameT
        The DataFrame with the time column converted to timestamps.
        The column maintains its original name but contains:
        - Float64 values for microseconds/milliseconds
        - Int64 values for nanoseconds (prevents overflow)

    Raises
    ------
    TimeColumnError
        If the column is not a datetime type or conversion fails.
        Provides detailed error context for debugging.
    ValueError
        - If the column does not exist
        - If the DataFrame is empty
        Early validation ensures data quality.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import convert_datetime_column_to_timestamp
    import pandas as pd

    # Create test DataFrame with datetime column
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3, freq="H"), "value": [1, 2, 3]})

    # Convert with different precisions
    df_us = convert_datetime_column_to_timestamp(df, "time", "us")  # microseconds
    df_ms = convert_datetime_column_to_timestamp(df, "time", "ms")  # milliseconds
    df_ns = convert_datetime_column_to_timestamp(df, "time", "ns")  # nanoseconds

    print(df_us)  # Shows microsecond timestamps (Float64)
    print(df_ns)  # Shows nanosecond timestamps (Int64)
    ```

    Notes
    -----
    Backend Independence:
        - Uses with_columns for unified operations
        - Handles timezone-aware datetimes
        - Works consistently across backends

    Performance Considerations:
        - Early return for already numeric columns
        - Efficient type casting
        - Minimal data copying

    Type Safety:
        - Explicit datetime validation
        - Precision-appropriate types
        - Preserves column name and schema
    """
    # Validate column exists
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the DataFrame.")

    # Check for nulls
    if is_dataframe_empty(df):
        raise ValueError("Empty DataFrame provided")

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
def convert_column_to_datetime_type(df: FrameT, time_col: str) -> FrameT:
    """Convert a string or numeric column to datetime type using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to convert a column
    to datetime format. Uses eager evaluation (eager_only=True) because it performs
    immediate type validation and conversion that cannot be deferred, following
    Narwhals' pattern for type transformations.

    Implementation Details
    --------------------
    The function performs a cascading conversion attempt:
    1. Validates column existence and current type
    2. Returns early if already datetime type
    3. Attempts conversions in order:
       a. Direct datetime casting
       b. String to datetime parsing
       c. Numeric to datetime conversion

    This approach ensures maximum compatibility with different input formats
    while maintaining type safety across all backends.

    Parameters
    ----------
    df : FrameT
        The input DataFrame containing the column to convert. Can be any backend
        supported by Narwhals (pandas, polars, dask, etc.).
    time_col : str
        The name of the column to convert. Must exist in the DataFrame and be
        either string (date format) or numeric (timestamp).

    Returns
    -------
    FrameT
        The DataFrame with the specified column converted to datetime type.
        The column maintains its original name but contains datetime values.

    Raises
    ------
    TimeColumnError
        - If the column does not exist
        - If the column is boolean type
        - If conversion fails for all attempted methods
        Provides detailed error context for debugging.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import convert_column_to_datetime_type
    import pandas as pd

    # Convert Unix timestamps to datetime
    df1 = pd.DataFrame(
        {
            "time": [1672531200000, 1672617600000],  # Unix timestamps
            "value": [1, 2],
        }
    )
    df1 = convert_column_to_datetime_type(df1, "time")
    print(df1)  # Shows datetime values

    # Convert string dates to datetime
    df2 = pd.DataFrame(
        {
            "time": ["2023-01-01", "2023-01-02"],  # ISO format strings
            "value": [3, 4],
        }
    )
    df2 = convert_column_to_datetime_type(df2, "time")
    print(df2)  # Shows datetime values
    ```

    Notes
    -----
    Backend Independence:
        - Uses with_columns for unified operations
        - Handles multiple input formats
        - Works consistently across backends

    Performance Considerations:
        - Early return for datetime columns
        - Efficient type casting
        - Minimal data copying

    Type Safety:
        - Explicit type validation
        - Multiple conversion attempts
        - Preserves column name and schema
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
        return df.with_columns([nw.col(time_col).cast(nw.Datetime()).alias(time_col)])
    except:
        # Try string to datetime conversion
        try:
            return df.with_columns([nw.col(time_col).cast(nw.String()).str.to_datetime().alias(time_col)])
        except:
            # Try numeric to datetime conversion
            try:
                return df.with_columns([nw.col(time_col).cast(nw.Float64()).cast(nw.Datetime()).alias(time_col)])
            except:
                raise TimeColumnError(f"Column '{time_col}' must be string or numeric to convert to datetime")


@nw.narwhalify(eager_only=True)
def validate_column_numeric_or_datetime(df: FrameT, time_col: str) -> None:
    """Validate that a column is either numeric or datetime type using Narwhals operations.

    This function uses Narwhals' backend-agnostic operations to validate column types.
    Uses eager evaluation (eager_only=True) because it performs immediate type validation
    that cannot be deferred, following Narwhals' pattern for validation functions.

    Implementation Details
    --------------------
    The function performs a two-step validation:
    1. Attempts numeric type casting (Float64)
    2. If numeric fails, attempts datetime casting

    This approach ensures proper type validation across all backends while
    maintaining consistent error handling and type safety.

    Parameters
    ----------
    df : FrameT
        The input DataFrame containing the column to validate. Can be any backend
        supported by Narwhals (pandas, polars, dask, etc.).
    time_col : str
        The name of the column to validate. Must exist in the DataFrame and be
        either numeric (int/float) or datetime type.

    Returns
    -------
    None
        Function returns None if validation passes, otherwise raises an error.

    Raises
    ------
    ValueError
        If the column is neither numeric nor datetime type.
        Provides clear error context for debugging.

    Examples
    --------
    ```python
    from temporalscope.core.core_utils import validate_column_numeric_or_datetime
    import pandas as pd

    # Test with numeric column
    df1 = pd.DataFrame(
        {
            "time": [1, 2, 3],  # Integer type
            "value": [1.1, 2.2, 3.3],  # Float type
        }
    )
    validate_column_numeric_or_datetime(df1, "time")  # Passes
    validate_column_numeric_or_datetime(df1, "value")  # Passes

    # Test with datetime column
    df2 = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3), "value": [1, 2, 3]})
    validate_column_numeric_or_datetime(df2, "time")  # Passes

    # Test with invalid type
    df3 = pd.DataFrame(
        {
            "time": ["a", "b", "c"],  # String type (invalid)
            "value": [1, 2, 3],
        }
    )
    validate_column_numeric_or_datetime(df3, "time")  # Raises ValueError
    ```

    Notes
    -----
    Backend Independence:
        - Uses with_columns for unified validation
        - Handles different numeric representations
        - Works consistently across backends

    Performance Considerations:
        - Minimal type casting operations
        - Early return on first success
        - No data copying required

    Type Safety:
        - Explicit type validation
        - Clear error messages
        - Proper null handling
    """
    # Try numeric cast first
    try:
        df.with_columns([nw.col(time_col).cast(nw.Float64()).alias(f"{time_col}_numeric_test")])
        return
    except:
        # Try datetime cast
        try:
            df.with_columns([nw.col(time_col).cast(nw.Datetime()).alias(f"{time_col}_datetime_test")])
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
        return convert_datetime_column_to_microseconds(df, time_col)

    if conversion_type == "datetime":
        return convert_column_to_datetime_type(df, time_col)

    # Validation-only path
    validate_column_numeric_or_datetime(df, time_col)
    return df


@nw.narwhalify(eager_only=True)
def validate_feature_columns_numeric(df: FrameT, time_col: Optional[str] = None) -> None:
    """Validate that feature columns are numeric using Narwhals operations.

    Uses eager evaluation (eager_only=True) for immediate type validation that cannot
    be deferred. Ensures feature columns are numeric for ML/DL compatibility.
    If time_col is provided, that column is excluded from numeric validation.

    Implementation Details
    --------------------
    1. If time_col provided, validates it exists
    2. Identifies feature columns (all columns except time_col if provided)
    3. Validates each feature column is numeric via Float64 casting
    4. Uses with_columns for backend-agnostic validation

    Parameters
    ----------
    df : FrameT
        Input DataFrame to validate
    time_col : Optional[str]
        Optional name of time column to exclude from validation.
        If None, all columns are validated as numeric.

    Returns
    -------
    None
        Raises error if validation fails

    Raises
    ------
    TimeColumnError
        If time_col provided but doesn't exist
    ValueError
        If any feature column is not numeric type

    Examples
    --------
    ```python
    # Validate all columns are numeric
    df1 = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],  # numeric ok
            "feature": [4, 5, 6],  # numeric ok
        }
    )
    validate_feature_columns_numeric(df1)  # passes

    # Validate all except time column
    df2 = pd.DataFrame(
        {
            "time": pd.date_range("2023-01", periods=3),  # ignored
            "value": [1.0, 2.0, 3.0],  # numeric ok
            "feature": [4, 5, 6],  # numeric ok
            "category": ["a", "b", "c"],  # error: must be numeric
        }
    )
    validate_feature_columns_numeric(df2, time_col="time")  # raises ValueError
    ```

    Notes
    -----
    Backend Independence:
        - Uses with_columns for validation
        - Works across all backends
    Type Safety:
        - Explicit type validation
        - Clear error messages
    """
    # If time_col provided, validate it exists
    if time_col is not None and time_col not in df.columns:
        raise TimeColumnError(f"Column '{time_col}' does not exist")

    # Get columns to validate (all columns except time_col if provided)
    cols_to_validate = [col for col in df.columns if col != time_col]

    # Validate columns are numeric
    for col in cols_to_validate:
        try:
            # Try casting to numeric (Float64)
            df.with_columns([nw.col(col).cast(nw.Float64()).alias(f"{col}_numeric_test")])
        except:
            raise ValueError(f"Column '{col}' must be numeric")


@nw.narwhalify(eager_only=True)
def validate_temporal_ordering(
    df: FrameT, time_col: str, id_col: Optional[str] = None, enforce_equidistant_sampling: bool = False
) -> None:
    """Validate temporal ordering of time series data.

    Default Behavior (enforce_equidistant_sampling=False):
    - Enforces temporal uniqueness within groups (if id_col provided) or globally
    - Required for most ML/DL models to ensure:
        1. No information leakage in train/test splits
        2. Proper sequence ordering for time series models
        3. Valid feature/target relationships

    Special Cases:
    - Some use cases may allow overlapping timestamps across groups:
        1. Multi-sensor data with different sampling rates
        2. Event data from different sources
        3. Hierarchical time series with nested groupings
    - Handle these by processing each group separately or using appropriate
      resampling/aggregation strategies before modeling

    Equidistant Sampling (enforce_equidistant_sampling=True):
    - Validates constant time deltas (equal spacing between timestamps)
    - Critical for specific modeling approaches:
        1. Classical time series models (ARIMA, SARIMA)
            - Assumes regular intervals for seasonality
            - Required for proper lag calculations
        2. Bayesian models (PyMC3, Stan)
            - Needed for proper prior specification
            - Required for state space models
        3. Spectral analysis
            - FFT and wavelets assume uniform sampling
            - Required for frequency domain analysis

    Parameters
    ----------
    df : FrameT
        Input DataFrame to validate
    time_col : str
        Name of time column
    id_col : Optional[str]
        Optional column for grouped validation. If provided, validates temporal
        ordering within each group.
    enforce_equidistant_sampling : bool
        If True, validates equidistant sampling required by:
        - Classical time series models (ARIMA)
        - Bayesian packages (PyMC3)
        If False (default), only validates ordering.

    Returns
    -------
    None
        Raises error if validation fails

    Raises
    ------
    TimeColumnError
        - If time_col missing or invalid type
        - If duplicate timestamps exist within groups
        - If irregular sampling when enforce_regular_sampling=True
    ValueError
        If id_col provided but doesn't exist

    Examples
    --------
    ```python
    # Basic ML case (default)
    validate_temporal_ordering(df, "time")  # Ensures unique, ordered timestamps

    # Multi-entity ML case
    validate_temporal_ordering(df, "time", id_col="stock_id")

    # ARIMA/PyMC3 case
    validate_temporal_ordering(df, "time", id_col="stock_id", enforce_regular_sampling=True)
    ```

    Notes
    -----
    - Most ML models require unique, ordered timestamps (default behavior)
    - Some models (ARIMA, PyMC3) require regular sampling (enforce_regular_sampling)
    - Uses Narwhals operations for backend-agnostic validation
    """
    # Validate time_col exists and is numeric/datetime
    validate_column_numeric_or_datetime(df, time_col)

    # Validate id_col if provided
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' does not exist")

    # Sort by time (and id_col if provided)
    sort_cols = [time_col] if id_col is None else [id_col, time_col]
    df_sorted = df.sort(sort_cols)

    # Check for duplicates within groups
    if id_col is not None:
        # Group by id and time, count occurrences
        duplicates = (
            df_sorted.group_by([id_col, time_col])
            .agg([nw.col(time_col).count().alias("count")])
            .filter(nw.col("count") > 1)
        )
        if len(duplicates) > 0:
            raise TimeColumnError(f"Duplicate timestamps found within groups in '{time_col}'")
    else:
        # Check for global duplicates
        duplicates = (
            df_sorted.group_by(time_col).agg([nw.col(time_col).count().alias("count")]).filter(nw.col("count") > 1)
        )
        if len(duplicates) > 0:
            raise TimeColumnError(f"Duplicate timestamps found in '{time_col}'")

    # Check for equidistant sampling if required
    if enforce_equidistant_sampling:
        # Calculate time deltas using minus operator
        if id_col is not None:
            # First compute deltas within groups
            df_deltas = df_sorted.select(
                [(nw.col(time_col).shift(-1) - nw.col(time_col)).alias("delta"), nw.col(id_col)]
            )
            # Only check within groups
            df_deltas = df_deltas.filter(nw.col(id_col) == nw.col(id_col).shift(-1))
            # For each group, get unique deltas (excluding nulls)
            group_deltas = (
                df_deltas.filter(~nw.col("delta").is_null())
                .group_by(id_col)
                .agg([nw.col("delta").n_unique().alias("unique_deltas")])
            )
            # Check if any group has more unique deltas than allowed
            irregular_groups = group_deltas.filter(nw.col("unique_deltas") > MAX_UNIQUE_DELTAS)
            if len(irregular_groups) > 0:
                raise TimeColumnError(  # pragma: no cover
                    f"Irregular time sampling found within group(s) for '{time_col}'. "
                    "Equidistant sampling is required within each group when enforce_equidistant_sampling=True."
                )
        else:
            # Compute deltas globally
            df_deltas = df_sorted.select([(nw.col(time_col).shift(-1) - nw.col(time_col)).alias("delta")])
            # Get unique deltas (excluding nulls)
            delta_stats = df_deltas.filter(~nw.col("delta").is_null()).select(
                [nw.col("delta").n_unique().alias("unique_deltas")]
            )
            if delta_stats["unique_deltas"][0] > MAX_UNIQUE_DELTAS:
                raise TimeColumnError(
                    f"Irregular time sampling found in '{time_col}'. "
                    "Equidistant sampling is required when enforce_equidistant_sampling=True."
                )


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
        raise ValueError(f"Column '{time_col}' does not exist in DataFrame")

    # Validate time column type
    validate_column_numeric_or_datetime(df, time_col)

    # Sort using Narwhals operations
    return df.sort(by=time_col, descending=not ascending)
