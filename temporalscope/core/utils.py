"""temporalscope/core/utils.py

This module provides utility functions that can be used throughout the TemporalScope package.
It includes methods for printing dividers, checking for nulls and NaNs, and validating the backend.
"""

from typing import Union
import polars as pl
import pandas as pd

# Constants
AVAILABLE_BACKENDS = ["polars", "pandas"]


def print_divider(char="=", length=70):
    """Prints a divider line made of a specified character and length."""
    print(char * length)


def check_nulls(df: Union[pl.DataFrame, pd.DataFrame], backend="polars") -> bool:
    """Check for null values in the DataFrame.

    :param df: Input DataFrame (Polars or Pandas).
    :param backend: The backend used for the DataFrame ('polars' or 'pandas'). Default is 'polars'.
    :return: True if there are null values, False otherwise.
    """
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends: {AVAILABLE_BACKENDS}"
        )

    if backend == "pandas":
        return df.isnull().values.any()
    elif backend == "polars":
        return df.null_count().sum().sum() > 0


def check_nans(df: Union[pl.DataFrame, pd.DataFrame], backend="polars") -> bool:
    """Check for NaN values in the DataFrame.

    :param df: Input DataFrame (Polars or Pandas).
    :param backend: The backend used for the DataFrame ('polars' or 'pandas'). Default is 'polars'.
    :return: True if there are NaN values, False otherwise.
    """
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends: {AVAILABLE_BACKENDS}"
        )

    if backend == "pandas":
        return df.isna().values.any()
    elif backend == "polars":
        return (df == pl.lit(float("nan"))).sum().sum() > 0
