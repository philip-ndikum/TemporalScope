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

# TemporalScope/test/unit/test_partition_padding.py


import pytest
import numpy as np
import pandas as pd
import modin.pandas as mpd
import polars as pl
from temporalscope.partition.padding import (
    zero_pad, 
    forward_fill_pad, 
    backward_fill_pad, 
    mean_fill_pad, 
    pad_dataframe, 
    sort_dataframe, 
    ensure_type_consistency
)
from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
)

from temporalscope.core.core_utils import SupportedBackendDataFrame

np.random.seed(42)  # Set a seed for reproducibility



def generate_test_data(backend, num_samples=5):
    """Generate test data with consistent column names across all backends."""
    start_date = pd.to_datetime("2021-01-01")
    data = {
        "feature_1": range(1, num_samples + 1),
        "feature_2": range(num_samples, 0, -1),
        "target": [i * 10 for i in range(1, num_samples + 1)],
        "ds": pd.date_range(start_date, periods=num_samples)  # Ensure 'ds' is a date column
    }

    if backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
        df['ds'] = df['ds'].astype('datetime64[ns]')  # Ensure ds is in datetime64[ns]
        return df

    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)
        df['ds'] = df['ds'].astype('datetime64[ns]')  # Modin relies on Pandas dtype system
        return df

    elif backend == BACKEND_POLARS:
        df = pl.DataFrame({
            "feature_1": data["feature_1"],
            "feature_2": data["feature_2"],
            "target": data["target"],
            "ds": [d for d in data["ds"]]  # Keep `ds` as a date column
        })
        return df.with_columns(pl.col("ds").cast(pl.Datetime))  # Cast ds to Polars datetime

    else:
        raise ValueError(f"Unsupported backend: {backend}")


@pytest.fixture
def test_data():
    return {
        BACKEND_PANDAS: generate_test_data(BACKEND_PANDAS),
        BACKEND_MODIN: generate_test_data(BACKEND_MODIN),
        BACKEND_POLARS: generate_test_data(BACKEND_POLARS),
    }


# Utility function to generate empty DataFrame
def get_empty_dataframe(backend):
    if backend == BACKEND_PANDAS:
        return pd.DataFrame()
    elif backend == BACKEND_MODIN:
        return mpd.DataFrame()
    elif backend == BACKEND_POLARS:
        return pl.DataFrame()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
        
def generate_mixed_data(num_samples: int = 5) -> pd.DataFrame:
    """Generates a DataFrame with mixed data types (numeric, categorical, datetime).

    This can be used for parametrized tests to check how functions handle different
    column types.

    :param num_samples: Number of rows to generate in the DataFrame.
    :return: A DataFrame with mixed data types.
    """
    start_date = pd.to_datetime("2021-01-01")
    data = {
        "numeric_col": range(1, num_samples + 1),
        "category_col": ["A", "B", "C", "D", "E"][:num_samples],
        "datetime_col": pd.date_range(start_date, periods=num_samples),
        "mixed_col": ["A", 1, pd.NaT, None, 5][:num_samples],  # Mixed types
    }
    return pd.DataFrame(data)



def check_monotonicity(df: SupportedBackendDataFrame, time_col: str, ascending: bool = True) -> bool:
    if isinstance(df, pl.DataFrame):
        # Handle Polars DataFrame
        diffs = df.select(pl.col(time_col).diff()).select(pl.col(time_col).drop_nulls())  # Handle nulls
        if ascending:
            return diffs.select(pl.col(time_col).gt(pl.lit(0))).to_series().all()  # Use Polars comparison
        else:
            return diffs.select(pl.col(time_col).lt(pl.lit(0))).to_series().all()
    else:
        # Handle Pandas and Modin (already handled correctly)
        diffs = df[time_col].diff().dropna()  # For Pandas/Modin, dropna() works fine
        if pd.api.types.is_timedelta64_dtype(diffs):
            zero_timedelta = pd.Timedelta(0)
            if ascending:
                return diffs.gt(zero_timedelta).all()
            else:
                return diffs.lt(zero_timedelta).all()
        else:
            if ascending:
                return diffs.gt(0).all()
            else:
                return diffs.lt(0).all()



# Parametrize tests for ascending and descending order
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_dataframe(test_data, backend, ascending):
    df = test_data[backend]
    sorted_df = sort_dataframe(df, time_col="ds", ascending=ascending)

    # Check sorting for each backend
    assert check_monotonicity(sorted_df, "ds", ascending=ascending)


# Test for invalid time column in sort_dataframe
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_sort_dataframe_invalid_time_col(test_data, backend):
    df = test_data[backend]
    with pytest.raises(ValueError):
        sort_dataframe(df, time_col="invalid_col")


# Test sorting for empty DataFrame
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_sort_dataframe_empty_dataframe(backend):
    empty_df = get_empty_dataframe(backend)
    with pytest.raises(ValueError):
        sort_dataframe(empty_df, time_col="ds")


# Test raising TypeError for unsupported input type
def test_sort_dataframe_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported DataFrame type"):
        sort_dataframe([], time_col="ds")  # List is an unsupported type


# Test warning when `time_col` is neither numeric nor datetime
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN])
def test_sort_dataframe_warning(test_data, backend):
    df = test_data[backend]
    df["non_time_col"] = ["a", "b", "c", "d", "e"]

    # Ensure warning is raised when time_col is non-numeric and non-datetime
    with pytest.warns(UserWarning, match="is neither numeric nor datetime"):
        sort_dataframe(df, time_col="non_time_col", ascending=True)

    # Continue with checking valid sorting after warning
    sorted_df = sort_dataframe(df, time_col="ds", ascending=True)
    assert check_monotonicity(sorted_df, "ds", ascending=True)



        

# Padding function tests with Modin and Polars compatibility
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
@pytest.mark.parametrize("padding_func", [zero_pad, forward_fill_pad, backward_fill_pad, mean_fill_pad])
def test_padding_functions(test_data, backend, padding_func):
    df = test_data[backend]

    if padding_func == zero_pad:
        padded_df = padding_func(df, target_len=7, time_col="ds", pad_value=0)
    else:
        padded_df = padding_func(df, target_len=7, end=5, reverse=False, time_col="ds")

    assert len(padded_df) == 7


# Ensure the 'ds' column is used consistently across backends in pad_dataframe
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
@pytest.mark.parametrize("mode", ["zero", "forward_fill", "backward_fill", "mean_fill"])
def test_pad_dataframe(test_data, backend, mode):
    df = test_data[backend]

    if mode == "zero":
        padded_df = pad_dataframe(df, target_len=7, mode=mode, pad_value=0, time_col="ds")
    else:
        padded_df = pad_dataframe(df, target_len=7, mode=mode, end=5, reverse=False, time_col="ds")

    assert len(padded_df) == 7


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_empty_dataframe(backend):
    if backend == BACKEND_PANDAS:
        df = pd.DataFrame()
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame()
    elif backend == BACKEND_POLARS:
        df = pl.DataFrame()

    with pytest.raises(ValueError):
        zero_pad(df, target_len=5, time_col="ds")


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_invalid_time_col(test_data, backend):
    df = test_data[backend]

    with pytest.raises(ValueError):
        zero_pad(df, target_len=7, time_col="invalid_col")


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_target_len_less_than_current_len(test_data, backend):
    df = test_data[backend]

    with pytest.raises(ValueError):
        zero_pad(df, target_len=3, time_col="ds")

        
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
def test_sort_dataframe_edge_cases(test_data, backend):
    df = test_data[backend]

    # Add non-numeric, non-datetime column to test sorting warnings
    if backend == BACKEND_POLARS:
        df = df.with_columns(pl.Series("non_numeric", ["a", "b", "c", "d", "e"]))
    else:
        df["non_numeric"] = ["a", "b", "c", "d", "e"]

    # Ensure warning is raised when time_col is non-numeric and non-datetime
    with pytest.warns(UserWarning, match="is neither numeric nor datetime"):
        sort_dataframe(df, time_col="non_numeric", ascending=True)

    # Continue with existing tests
    sorted_df = sort_dataframe(df, time_col="ds", ascending=True)
    if backend == BACKEND_POLARS:
        assert sorted_df["ds"].is_sorted()
    else:
        assert sorted_df["ds"].is_monotonic_increasing


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
@pytest.mark.parametrize("padding_func", [zero_pad, forward_fill_pad, backward_fill_pad, mean_fill_pad])
def test_padding_functions_with_warnings(test_data, backend, padding_func):
    df = test_data[backend]

    # Add non-numeric columns
    if backend == BACKEND_POLARS:
        df = df.with_columns(pl.Series("non_numeric", ["a", "b", "c", "d", "e"]))
        pad_df = pad_dataframe(df, target_len=7, mode="zero", time_col="ds")  # Add mode here
        pad_df = pad_df.with_columns(pl.lit(None).alias("non_numeric"))  # Ensure "non_numeric" exists in pad_df
    else:
        df["non_numeric"] = ["a", "b", "c", "d", "e"]

    if padding_func == zero_pad:
        with pytest.warns(UserWarning, match="Non-numeric columns found"):
            padded_df = padding_func(df, target_len=7, time_col="ds", pad_value=0)
    else:
        padded_df = padding_func(df, target_len=7, end=5, reverse=False, time_col="ds")

    assert len(padded_df) == 7


@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS])
@pytest.mark.parametrize("mode", ["zero", "forward_fill", "backward_fill", "mean_fill"])
def test_pad_dataframe_type_consistency(test_data, backend, mode):
    df = test_data[backend]

    # Add non-numeric column
    if backend == BACKEND_POLARS:
        df = df.with_columns(pl.Series("non_numeric", ["x", "y", "z", "w", "v"]))
    else:
        df["non_numeric"] = ["x", "y", "z", "w", "v"]

    if mode == "zero":
        with pytest.warns(UserWarning, match="Non-numeric columns found"):
            padded_df = pad_dataframe(df, target_len=7, mode=mode, pad_value=0, time_col="ds")
    else:
        with pytest.warns(UserWarning, match="Non-numeric columns found"):
            padded_df = pad_dataframe(df, target_len=7, mode=mode, end=5, reverse=False, time_col="ds")

    assert len(padded_df) == 7

    # Ensure types are consistent
    assert padded_df["feature_1"].dtype == df["feature_1"].dtype
    assert padded_df["feature_2"].dtype == df["feature_2"].dtype
    
@pytest.mark.parametrize("backend", [BACKEND_PANDAS, BACKEND_MODIN])
def test_pad_dataframe_boolean_to_int64(test_data, backend):
    """Test that boolean columns in the DataFrame are correctly cast to int64."""
    df = test_data[backend]

    # Add a boolean column to the DataFrame
    if backend == BACKEND_PANDAS:
        df["bool_col"] = [True, False, True, False, True]
    elif backend == BACKEND_MODIN:
        df["bool_col"] = mpd.Series([True, False, True, False, True])

    # Create a padding DataFrame with the same columns
    pad_df = pd.DataFrame({
        "bool_col": [False, False]  # Padding with False values (should become 0)
    })

    # Ensure type consistency (bool -> int64)
    consistent_df = ensure_type_consistency(df, pad_df)

    # Check that the boolean column is converted to int64
    assert consistent_df["bool_col"].dtype == "int64"
    assert (consistent_df["bool_col"] == 0).all()  # All padded values should be 0

    
@pytest.mark.parametrize("backend", [BACKEND_MODIN])
def test_pad_dataframe_conversion_to_modin(test_data, backend):
    """Test that pad_df is correctly converted back to Modin after type consistency check."""
    df = test_data[backend]

    # Create a padding DataFrame with mismatched types
    pad_df = pd.DataFrame({
        "feature_1": [0.0, 0.0],
        "feature_2": [0, 0],
        "target": [0, 0],
        "ds": [pd.Timestamp("1970-01-01"), pd.Timestamp("1970-01-01")]
    })

    # Ensure type consistency (pad_df starts as Pandas DataFrame)
    consistent_df = ensure_type_consistency(df, pad_df)

    # Ensure pad_df is converted back to Modin if df was Modin
    assert isinstance(consistent_df, mpd.DataFrame), "pad_df should be converted back to Modin"
