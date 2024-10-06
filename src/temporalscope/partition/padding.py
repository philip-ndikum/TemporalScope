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

"""TemporalScope/src/temporalscope/partition/padding.py

This module provides utility functions for padding and partitioning time-series DataFrames.
It supports multiple backends such as Pandas, Modin, and Polars, ensuring that data is sorted
by a `time_col` (if provided) before applying transformations such as padding.

The design of this module aligns with TensorFlow, PyTorch, and Darts for universal ML & DL time-series workflows.
It integrates with the central `TimeFrame` concept in TemporalScope, ensuring compatibility with temporal XAI
and partitioning workflows.

Core Functionality:
-------------------
Each padding function ensures the DataFrame is sorted based on a `time_col` (if provided) before applying
the selected padding scheme.

Padding is designed to:
1. Expand Data: When datasets have insufficient data points (e.g., missing timestamps), padding fills in the gaps.
2. Fix Data Shapes: Many ML/DL architectures require fixed input shapes, and padding ensures uniformity across batches or partitions.
3. Maintain Backend Consistency: Padding respects the original backend of the DataFrame (Pandas, Modin, or Polars).
4. Preserve Precision Consistency: Padding operations ensure that data types (e.g., `Float32`, `Int64`) are retained, avoiding unnecessary conversions and ensuring precision consistency throughout the pipeline.

Design Constraint:
------------------
For categorical columns, users **must** handle encoding (e.g., label encoding, one-hot encoding) before using any
partitioning or padding utilities. This module focuses only on numerical and time columns. The only special handling
occurs for the `time_col` (if specified), which can be a timestamp or a numeric column.

Examples
--------
    .. code-block:: python

        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "time": pd.date_range("20210101", periods=3)})
        >>> padded_df = zero_pad(df, target_len=5, time_col="time")
        >>> print(padded_df)
           a  b       time
        0  0  0 2021-01-04
        1  0  0 2021-01-05

        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "time": ["2021-01-01", "2021-01-02", "2021-01-03"]})
        >>> padded_df = pad_dataframe(df, target_len=5, mode="zero", padding="post", time_col="time")
        >>> print(padded_df)

.. seealso::
    1. Dwarampudi, M. and Reddy, N.V., 2019. Effects of padding on LSTMs and CNNs. arXiv preprint arXiv:1903.07288.
    2. Lafabregue, B., Weber, J., Gançarski, P. and Forestier, G., 2022. End-to-end deep representation learning for time series clustering: a comparative study. Data Mining and Knowledge Discovery, 36(1), pp.29-81.

"""

import warnings
from typing import Optional, Union

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.core_utils import SupportedBackendDataFrame

# Define numeric types for each backend
PANDAS_NUMERIC_TYPES = ["number"]
MODIN_NUMERIC_TYPES = ["number"]  # Same as Pandas since Modin mimics Pandas' behavior
POLARS_NUMERIC_TYPES = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]

# Constants for time padding strategies
TIME_PAD_STRATEGIES = {
    "nat": pd.NaT,  # Use NaT for missing time values
    "fill_forward": "fill_forward",  # Forward-fill the missing time values
}

# List of padding schemes
PAD_SCHEMES = ["zero", "forward_fill", "backward_fill", "mean_fill"]


def validate_dataframe(df: SupportedBackendDataFrame) -> None:
    """Validates the type and emptiness of the DataFrame.

    This function raises exceptions if the DataFrame is not one of the supported
    backends (Pandas, Modin, Polars) or if it is empty.

    :param df: The DataFrame (Pandas, Modin, or Polars) to validate.
    :raises TypeError: If the DataFrame type is unsupported.
    :raises ValueError: If the DataFrame is empty.
    """
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        if df.empty:
            raise ValueError("Cannot operate on an empty DataFrame.")
    elif isinstance(df, pl.DataFrame):
        if df.is_empty():
            raise ValueError("Cannot operate on an empty DataFrame.")
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def sort_dataframe(df: SupportedBackendDataFrame, time_col: str, ascending: bool = True) -> SupportedBackendDataFrame:
    """Sort the DataFrame by a time column.

    :param df: The DataFrame to sort (supports Pandas, Modin, or Polars).
    :param time_col: The column name to sort by (for time-based sorting).
    :param ascending: Whether to sort in ascending or descending order.
    :return: The sorted DataFrame.

    :raises ValueError: If the DataFrame is empty or the time_col is missing.
    :raises TypeError: If the DataFrame type is unsupported.
    :raises ValueError: If time_col is not found in the DataFrame.
    :warning: If `time_col` is not numeric or datetime.
    """
    validate_dataframe(df)

    # Ensure time_col is provided and exists in the DataFrame
    if not time_col or time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' must be provided and exist in the DataFrame.")

    # Issue a warning if time_col is not recognized as numeric or datetime
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        if not pd.api.types.is_numeric_dtype(df[time_col]) and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            warnings.warn(f"Column '{time_col}' is neither numeric nor datetime. Ensure it is processed properly.")
        # Sort using Pandas/Modin
        return df.sort_values(by=time_col, ascending=ascending)

    elif isinstance(df, pl.DataFrame):
        # Polars-specific types: handle proper type-checking without over-constraining
        valid_dtypes = [pl.Int64, pl.Int32, pl.Float64, pl.Float32, pl.Datetime]
        if df[time_col].dtype not in valid_dtypes:
            warnings.warn(f"Column '{time_col}' in Polars DataFrame is neither numeric nor datetime.")
        # Sort using Polars
        return df.sort(by=time_col, descending=not ascending)

    # Shouldn't be reachable, but for safety
    raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def ensure_type_consistency(  # noqa: PLR0912
    df: SupportedBackendDataFrame, pad_df: SupportedBackendDataFrame
) -> SupportedBackendDataFrame:
    """Ensure the column types of `pad_df` match the column types of `df`.

    This is crucial when padding time-series data to ensure type consistency across numeric
    columns, especially in ML/DL workflows where precision must be maintained. Without ensuring
    consistency, you risk precision loss or unintended data type conversions (e.g., `float32` to `float64`)
    when padding data, which could affect downstream neural networks or XAI models like SHAP.

    :param df: The original DataFrame (Pandas, Modin, or Polars).
    :param pad_df: The DataFrame to pad with.
    :return: `pad_df` with columns cast to match the types of `df`.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from temporalscope.partition.padding import ensure_type_consistency

        # Original DataFrame
        df = pd.DataFrame({"a": pd.Series([1.0, 2.0], dtype="float32"), "b": pd.Series([3, 4], dtype="int64")})

        # Padded DataFrame
        pad_df = pd.DataFrame({"a": [0.0, 0.0], "b": [0, 0]})

        # Ensure type consistency between df and pad_df
        pad_df = ensure_type_consistency(df, pad_df)
        print(pad_df.dtypes)

    .. note::
        - This function is especially useful when working with frameworks like TensorFlow or PyTorch,
          where maintaining precision (e.g., `float32` vs. `float64`) is essential to avoid issues like
          gradient explosion or vanishing during training.
        - We convert Modin DataFrames to Pandas temporarily to ensure type consistency because Modin’s internal
          `astype()` can sometimes cause issues when working with mixed data types or `bool` columns. After
          consistency is ensured, we convert the DataFrame back to Modin to maintain backend consistency.

    """
    # If df is a Modin DataFrame, convert to Pandas if possible
    is_modin_df = False
    if isinstance(df, mpd.DataFrame):
        is_modin_df = True
        if hasattr(df, "_to_pandas"):
            df = df._to_pandas()  # Convert to Pandas DataFrame
        if hasattr(pad_df, "_to_pandas"):
            pad_df = pad_df._to_pandas()  # Same for pad_df

    # Handle Pandas DataFrame casting
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            if col in pad_df.columns:
                if df[col].dtype == "bool":
                    # Convert boolean columns to int64 for consistency
                    pad_df[col] = pad_df[col].astype("int64")
                else:
                    # Cast column to the original dtype
                    pad_df[col] = pad_df[col].astype(df[col].dtype)

        # Ensure conversion back to Modin happens if pad_df was converted to Pandas
        if is_modin_df and isinstance(pad_df, pd.DataFrame):
            pad_df = mpd.DataFrame(pad_df)  # Convert back to Modin

        return pad_df

    # Handle Polars DataFrame casting
    elif isinstance(df, pl.DataFrame):
        for col in df.columns:
            if col in pad_df.columns:
                pad_df = pad_df.with_columns(pad_df[col].cast(df[col].dtype))
        return pad_df

    # If the DataFrame type is unsupported, raise an error
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def zero_pad(  # noqa: PLR0911, PLR0912
    df: SupportedBackendDataFrame,
    target_len: int,
    time_col: Optional[str] = None,
    padding: str = "post",
    ascending: bool = True,
    pad_value: Union[int, float] = 0,
) -> SupportedBackendDataFrame:
    """Apply padding by adding rows filled with a specified value (default is zero).

    This function only handles numeric columns. If `time_col` is provided, the DataFrame will
    be sorted by that column before applying the padding scheme.

    :param df: The DataFrame (Pandas, Modin, or Polars) to pad.
    :param target_len: The target number of rows after padding.
    :param time_col: Optional. The time column to sort by before padding.
    :param padding: Whether to pad 'pre' (before) or 'post' (after).
    :param ascending: Whether to sort the data in ascending or descending order.
    :param pad_value: The value to use for padding numeric columns (default is 0).
    :return: A DataFrame padded with rows filled with the specified value for numeric columns.

    :raises ValueError: If target_len is less than the current DataFrame length.
    :raises ValueError: If the DataFrame is empty.
    :raises TypeError: If an unsupported DataFrame type is provided.
    """
    validate_dataframe(df)

    # Ensure the target length is greater than the current DataFrame length
    if target_len < len(df):
        raise ValueError("target_len must be greater than the current DataFrame length.")

    # Sort the DataFrame if time_col is provided
    df = sort_dataframe(df, time_col, ascending) if time_col else df
    num_to_pad = target_len - len(df)

    # Create the padding DataFrame
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        # Handle numeric columns: Fill with pad_value
        numeric_cols = df.select_dtypes(include=PANDAS_NUMERIC_TYPES).columns
        pad_df = pd.DataFrame(pad_value, index=range(num_to_pad), columns=numeric_cols)

        # Check for non-numeric columns (excluding time_col) and raise warnings
        non_numeric_cols = df.select_dtypes(exclude=PANDAS_NUMERIC_TYPES).columns.difference([time_col])
        if not non_numeric_cols.empty:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Padding them with NA (null).")

        # Add missing columns to pad_df
        missing_cols = set(df.columns) - set(pad_df.columns)
        for col in missing_cols:
            if col in non_numeric_cols:
                pad_df[col] = pd.NA  # Fill non-numeric columns with NA (null)
            elif col == time_col:
                pad_df[col] = pd.NaT  # Fill time column with NaT for datetime consistency
            else:
                pad_df[col] = pad_value

        # Ensure type consistency
        pad_df = ensure_type_consistency(df, pad_df)

    elif isinstance(df, pl.DataFrame):
        # Handle numeric columns: Fill with pad_value
        numeric_cols = df.select(pl.col(POLARS_NUMERIC_TYPES)).columns
        pad_df = pl.DataFrame({col: [pad_value] * num_to_pad for col in numeric_cols})

        # Check for non-numeric columns (excluding time_col) and raise warnings
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != time_col]
        if non_numeric_cols:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Padding them with None (null).")

        # Add missing columns to pad_df
        missing_cols = {col for col in df.columns if col not in pad_df.columns}
        for col in missing_cols:
            if col in non_numeric_cols:
                pad_df = pad_df.with_columns(
                    pl.lit(None).cast(df[col].dtype).alias(col)
                )  # Fill non-numeric columns with None
            elif col == time_col:
                pad_df = pad_df.with_columns(
                    pl.lit(None).cast(pl.Datetime).alias(col)
                )  # Ensure time column is datetime
            else:
                pad_df = pad_df.with_columns(pl.lit(pad_value).alias(col))

        # Ensure type consistency
        pad_df = ensure_type_consistency(df, pad_df)

    # Concatenate padding DataFrame
    if padding == "post":
        if isinstance(df, pd.DataFrame):
            return pd.concat([df, pad_df], ignore_index=True)
        elif isinstance(df, mpd.DataFrame):
            return mpd.concat([df, pad_df], ignore_index=True)
        elif isinstance(df, pl.DataFrame):
            return df.vstack(pad_df)
    elif padding == "pre":
        if isinstance(df, pd.DataFrame):
            return pd.concat([pad_df, df], ignore_index=True)
        elif isinstance(df, mpd.DataFrame):
            return mpd.concat([pad_df, df], ignore_index=True)
        elif isinstance(df, pl.DataFrame):
            return pad_df.vstack(df)
    else:
        raise ValueError(f"Invalid padding option: {padding}. Use 'pre' or 'post'.")

    return df


def forward_fill_pad(  # noqa: PLR0911, PLR0912
    df: SupportedBackendDataFrame,
    target_len: int,
    end: int,
    reverse: bool,
    padding: str = "post",
    time_col: Optional[str] = None,
    ascending: bool = True,
) -> SupportedBackendDataFrame:
    """Apply forward-fill padding by repeating the last or first row. Data will be sorted by `time_col` if provided.

    :param df: The DataFrame (Pandas, Modin, or Polars) to pad.
    :param target_len: The target number of rows after padding.
    :param end: The index indicating the last valid row for padding.
    :param reverse: If True, fill from the start of the DataFrame.
    :param padding: Whether to pad 'pre' (before) or 'post' (after). Must be one of ['pre', 'post'].
    :param time_col: Optional. The time column to sort by before padding.
    :param ascending: Whether to sort the data in ascending or descending order.
    :return: A DataFrame padded by forward fill.

    :raises ValueError: If target_len is less than the current DataFrame length.
    :raises ValueError: If the DataFrame is empty.
    :raises ValueError: If `padding` is not one of ['pre', 'post'].
    :raises TypeError: If an unsupported DataFrame type is provided.
    :raises ValueError: If non-time columns are not numeric.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from temporalscope.partition.padding import forward_fill_pad

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "time": pd.date_range("2021-01-01", periods=2)})
        padded_df = forward_fill_pad(df, target_len=5, end=len(df), reverse=False, time_col="time")
        print(padded_df)

    .. note::
        Forward-fill padding is useful in scenarios where missing data is best approximated by the last known
        valid value, such as financial data or sensor readings in IoT applications.

    """
    # Validate the padding option
    if padding not in ["pre", "post"]:
        raise ValueError(f"Invalid padding option: {padding}. Use 'pre' or 'post'.")

    validate_dataframe(df)

    # Ensure the target length is greater than the current DataFrame length
    if target_len < len(df):
        raise ValueError("target_len must be greater than the current DataFrame length.")

    # Sort the DataFrame by the time column if provided (includes warning for non-numeric/datetime time_col)
    df = sort_dataframe(df, time_col, ascending) if time_col else df

    # Validate that all non-time columns are numeric, raise warning if not
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        non_numeric_cols = df.select_dtypes(exclude=PANDAS_NUMERIC_TYPES).columns.difference([time_col])
        if not non_numeric_cols.empty:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Only numeric columns will be padded.")
    elif isinstance(df, pl.DataFrame):
        non_numeric_cols = [col for col in df.columns if col not in POLARS_NUMERIC_TYPES and col != time_col]
        if non_numeric_cols:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Only numeric columns will be padded.")

    # Calculate how many rows to pad
    num_to_pad = target_len - len(df)

    # Create the padding DataFrame by repeating the last or first row
    if isinstance(df, pd.DataFrame):
        pad_row = df.iloc[[end - 1]] if not reverse else df.iloc[[0]]
        pad_df = pd.concat([pad_row] * num_to_pad, ignore_index=True)
    elif isinstance(df, mpd.DataFrame):
        pad_row = df.iloc[[end - 1]] if not reverse else df.iloc[[0]]
        pad_df = mpd.concat([pad_row] * num_to_pad, ignore_index=True)  # Use Modin's concat
    elif isinstance(df, pl.DataFrame):
        pad_row = df.slice(end - 1, 1) if not reverse else df.slice(0, 1)
        pad_df = pl.concat([pad_row] * num_to_pad)

    # Ensure type consistency after padding
    pad_df = ensure_type_consistency(df, pad_df)

    # Append or prepend the padding DataFrame to the original DataFrame
    if padding == "post":
        if isinstance(df, pd.DataFrame):
            return pd.concat([df, pad_df], ignore_index=True)
        elif isinstance(df, mpd.DataFrame):
            return mpd.concat([df, pad_df], ignore_index=True)  # Use Modin's concat
        elif isinstance(df, pl.DataFrame):
            return df.vstack(pad_df)
    elif padding == "pre":
        if isinstance(df, pd.DataFrame):
            return pd.concat([pad_df, df], ignore_index=True)
        elif isinstance(df, mpd.DataFrame):
            return mpd.concat([pad_df, df], ignore_index=True)  # Use Modin's concat
        elif isinstance(df, pl.DataFrame):
            return pad_df.vstack(df)

    return df


def backward_fill_pad(  # noqa: PLR0912
    df: SupportedBackendDataFrame,
    target_len: int,
    end: int,
    reverse: bool,
    padding: str = "post",
    time_col: Optional[str] = None,
    ascending: bool = True,
) -> SupportedBackendDataFrame:
    """Apply backward-fill padding by repeating the first or last row.

    Data will be sorted by `time_col` if provided.

    :param df: The DataFrame (Pandas, Modin, or Polars) to pad.
    :param target_len: The target number of rows after padding.
    :param end: The index indicating the last valid row for padding.
    :param reverse: If True, fill from the start of the DataFrame.
    :param padding: Whether to pad 'pre' (before) or 'post' (after).
    :param time_col: Optional. The time column to sort by before padding.
    :param ascending: Whether to sort the data in ascending or descending order.
    :return: A DataFrame padded by backward fill.

    :raises ValueError: If target_len is less than the current DataFrame length.
    :raises ValueError: If the DataFrame is empty.
    :raises TypeError: If an unsupported DataFrame type is provided.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from temporalscope.partition.padding import backward_fill_pad

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "time": pd.date_range("2021-01-01", periods=2)})
        padded_df = backward_fill_pad(df, target_len=5, end=len(df), reverse=False, time_col="time")
        print(padded_df)

    .. note::
        Backward-fill padding is often applied when future values are unknown and it's reasonable to assume that
        the first valid observation represents future unknowns, which is useful in cases like predictive modeling.

    """
    validate_dataframe(df)

    # Ensure the target length is greater than the current DataFrame length
    if target_len < len(df):
        raise ValueError("target_len must be greater than the current DataFrame length.")

    # Sort the DataFrame by the time column if provided (includes warning for non-numeric/datetime time_col)
    df = sort_dataframe(df, time_col, ascending) if time_col else df

    # Validate that all non-time columns are numeric, raise warning if not
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        non_numeric_cols = df.select_dtypes(exclude=PANDAS_NUMERIC_TYPES).columns.difference([time_col])
        if not non_numeric_cols.empty:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Only numeric columns will be padded.")
    elif isinstance(df, pl.DataFrame):
        non_numeric_cols = [col for col in df.columns if col not in POLARS_NUMERIC_TYPES and col != time_col]
        if non_numeric_cols:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Only numeric columns will be padded.")

    # Calculate how many rows to pad
    num_to_pad = target_len - len(df)

    # Create the padding DataFrame by repeating the first or last row
    if isinstance(df, pd.DataFrame):
        pad_row = df.iloc[[0]] if not reverse else df.iloc[[end - 1]]
        pad_df = pd.concat([pad_row] * num_to_pad, ignore_index=True)
    elif isinstance(df, mpd.DataFrame):
        pad_row = df.iloc[[0]] if not reverse else df.iloc[[end - 1]]
        pad_df = mpd.concat([pad_row] * num_to_pad, ignore_index=True)  # Use Modin's concat
    elif isinstance(df, pl.DataFrame):
        pad_row = df.slice(0, 1) if not reverse else df.slice(end - 1, 1)
        pad_df = pl.concat([pad_row] * num_to_pad)

    # Ensure type consistency after padding
    pad_df = ensure_type_consistency(df, pad_df)

    # Append or prepend the padding DataFrame to the original DataFrame
    if padding == "post":
        if isinstance(df, pd.DataFrame):
            return pd.concat([df, pad_df], ignore_index=True)
        elif isinstance(df, mpd.DataFrame):
            return mpd.concat([df, pad_df], ignore_index=True)  # Use Modin's concat
        elif isinstance(df, pl.DataFrame):
            return df.vstack(pad_df)
    elif padding == "pre":
        if isinstance(df, pd.DataFrame):
            return pd.concat([pad_df, df], ignore_index=True)
        elif isinstance(df, mpd.DataFrame):
            return mpd.concat([pad_df, df], ignore_index=True)  # Use Modin's concat
        elif isinstance(df, pl.DataFrame):
            return pad_df.vstack(df)
    else:
        raise ValueError(f"Invalid padding option: {padding}. Use 'pre' or 'post'.")

    # This line ensures that MyPy sees a return in all cases, although it's unreachable.
    raise RuntimeError("This should never be reached")


def mean_fill_pad(  # noqa: PLR0912
    df: SupportedBackendDataFrame,
    target_len: int,
    end: int,
    reverse: bool,
    padding: str = "post",
    time_col: Optional[str] = None,
    ascending: bool = True,
) -> SupportedBackendDataFrame:
    """Apply mean-fill padding by filling numeric columns with their mean values.

    Data will be sorted by `time_col` if provided.

    :param df: The DataFrame (Pandas, Modin, or Polars) to pad.
    :param target_len: The target number of rows after padding.
    :param end: The index indicating the last valid row for padding.
    :param reverse: If True, fill from the start of the DataFrame.
    :param padding: Whether to pad 'pre' (before) or 'post' (after).
    :param time_col: Optional. The time column to sort by before padding.
    :param ascending: Whether to sort the data in ascending or descending order.
    :return: A DataFrame padded by mean fill.

    :raises ValueError: If target_len is less than the current DataFrame length.
    :raises ValueError: If the DataFrame is empty.
    :raises TypeError: If an unsupported DataFrame type is provided.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from temporalscope.partition.padding import mean_fill_pad

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "time": pd.date_range("2021-01-01", periods=2)})
        padded_df = mean_fill_pad(df, target_len=5, end=len(df), reverse=False, time_col="time")
        print(padded_df)

    .. note::
        Mean-fill padding is useful when you want to fill gaps in the data with the mean of the numeric columns.
        It is commonly used in time-series forecasting and analytics when you want to smooth over missing values.

    """
    validate_dataframe(df)

    # Ensure the target length is greater than the current DataFrame length
    if target_len < len(df):
        raise ValueError("target_len must be greater than the current DataFrame length.")

    # Sort the DataFrame by the time column if provided (includes warning for non-numeric/datetime time_col)
    df = sort_dataframe(df, time_col, ascending) if time_col else df

    # Validate that all non-time columns are numeric, raise warning if not
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        non_numeric_cols = df.select_dtypes(exclude=PANDAS_NUMERIC_TYPES).columns.difference([time_col])
        if not non_numeric_cols.empty:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Only numeric columns will be padded.")
    elif isinstance(df, pl.DataFrame):
        non_numeric_cols = [col for col in df.columns if col not in POLARS_NUMERIC_TYPES and col != time_col]
        if non_numeric_cols:
            warnings.warn(f"Non-numeric columns found: {non_numeric_cols}. Only numeric columns will be padded.")

    num_to_pad = target_len - len(df)

    # Handle Pandas and Modin DataFrames
    if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
        numeric_cols = df.select_dtypes(include=PANDAS_NUMERIC_TYPES).columns
        mean_values = df[numeric_cols].mean()
        pad_df = pd.DataFrame([mean_values] * num_to_pad, columns=numeric_cols)

        # Handle non-numeric columns (nearest row padding)
        non_numeric_cols = df.select_dtypes(exclude=PANDAS_NUMERIC_TYPES).columns
        if not non_numeric_cols.empty:
            nearest_row = df.iloc[[end - 1]] if not reverse else df.iloc[[0]]
            for col in non_numeric_cols:
                pad_df[col] = nearest_row[col].values[0]

        # Ensure column types match
        pad_df = ensure_type_consistency(df, pad_df)

        # Concatenate the DataFrame
        if isinstance(df, mpd.DataFrame):
            return (
                mpd.concat([df, mpd.DataFrame(pad_df)], ignore_index=True)
                if padding == "post"
                else mpd.concat([mpd.DataFrame(pad_df), df], ignore_index=True)
            )
        else:
            return (
                pd.concat([df, pad_df], ignore_index=True)
                if padding == "post"
                else pd.concat([pad_df, df], ignore_index=True)
            )

    # Handle Polars DataFrame
    elif isinstance(df, pl.DataFrame):
        numeric_cols = df.select(pl.col(POLARS_NUMERIC_TYPES)).columns
        mean_values = {col: df[col].mean() for col in numeric_cols}
        pad_df = pl.DataFrame({col: [mean_values[col]] * num_to_pad for col in numeric_cols})

        # Handle non-numeric columns (nearest row padding)
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        if non_numeric_cols:
            nearest_row = df.slice(end - 1, 1) if not reverse else df.slice(0, 1)
            for col in non_numeric_cols:
                pad_df = pad_df.with_columns(pl.lit(nearest_row[col][0]).alias(col))

        # Ensure column types match
        pad_df = ensure_type_consistency(df, pad_df)

        # Ensure complete padding
        if len(pad_df) != num_to_pad:
            raise ValueError(f"Padding mismatch: expected {num_to_pad}, but got {len(pad_df)}")

        # Return padded Polars DataFrame
        return df.vstack(pad_df) if padding == "post" else pad_df.vstack(df)

    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

    # This return statement satisfies MyPy's expectation, but should not actually be reachable.
    raise RuntimeError("This should never be reached")


def pad_dataframe(
    df: SupportedBackendDataFrame,
    target_len: int,
    mode: str,
    padding: str = "post",
    time_col: Optional[str] = None,
    ascending: bool = True,
    pad_value: Union[int, float, None] = None,
    end: Optional[int] = None,
    reverse: bool = False,
) -> SupportedBackendDataFrame:
    """Apply a padding scheme to a DataFrame, ensuring it's sorted by `time_col` if provided.

    :param df: The DataFrame (Pandas, Modin, or Polars) to pad.
    :param target_len: Target number of rows after padding.
    :param mode: Padding mode to use. Options are: "zero", "forward_fill", "backward_fill", "mean_fill".
    :param padding: Direction to apply padding ('pre' or 'post'). Default is 'post'.
    :param time_col: Optional column name to sort by (for time-based sorting).
    :param ascending: Whether to sort data in ascending order (default is True).
    :param pad_value: Custom value to use for padding in the "zero" mode. Default is None. Ignored for other modes.
    :param end: The index indicating the last valid row for padding. Required for forward_fill, backward_fill, and mean_fill modes.
    :param reverse: If True, fill from the start of the DataFrame. Required for forward_fill, backward_fill, and mean_fill modes.
    :return: The padded DataFrame.

    :raises ValueError: If mode is unknown, if `target_len` is less than the current DataFrame length, or if DataFrame is empty.
    :raises TypeError: If an unsupported DataFrame type is provided.

    Examples
    --------
    .. code-block:: python

        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "time": pd.date_range("20210101", periods=2)})
        >>> pad_dataframe(df, target_len=4, mode="zero", pad_value=0, time_col="time")
           a  b       time
        0  1  3 2021-01-01
        1  2  4 2021-01-02
        2  0  0        NaT
        3  0  0        NaT

        >>> pad_dataframe(df, target_len=4, mode="mean_fill", time_col="time")
           a    b       time
        0  1.0  3.0 2021-01-01
        1  2.0  4.0 2021-01-02
        2  1.5  3.5        NaT
        3  1.5  3.5        NaT

    """
    validate_dataframe(df)

    if target_len < len(df):
        raise ValueError("target_len must be greater than the current DataFrame length.")

    # Ensure the mode is valid
    if mode not in PAD_SCHEMES:
        raise ValueError(f"Unknown padding mode: {mode}. Available modes: {', '.join(PAD_SCHEMES)}")

    # Sort the DataFrame by the time column if provided (includes warning for non-numeric/datetime time_col)
    df = sort_dataframe(df, time_col, ascending) if time_col else df

    # Handle zero padding with a default pad_value of 0
    if mode == "zero":
        if pad_value is None:
            pad_value = 0  # Default pad_value for zero padding
        return zero_pad(df, target_len, time_col=time_col, padding=padding, pad_value=pad_value)

    # Ensure `end` is not None for other modes
    if end is None:
        raise ValueError(f"`end` parameter is required for {mode} mode.")

    # Dynamically select and call the padding function
    if mode == "forward_fill":
        return forward_fill_pad(
            df, target_len, end=end, reverse=reverse, time_col=time_col, padding=padding, ascending=ascending
        )
    elif mode == "backward_fill":
        return backward_fill_pad(
            df, target_len, end=end, reverse=reverse, time_col=time_col, padding=padding, ascending=ascending
        )
    elif mode == "mean_fill":
        return mean_fill_pad(
            df, target_len, end=end, reverse=reverse, time_col=time_col, padding=padding, ascending=ascending
        )

    # This should never be reached, but included as a safety net
    raise ValueError(f"Invalid padding mode: {mode}")
