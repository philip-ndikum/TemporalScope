"""Utility functions for the TemporalScope package.

This module provides various utility functions that can be used throughout the
TemporalScope package. It includes methods for printing dividers, checking for
null and NaN values in DataFrames, and validating the backend.
"""

from typing import cast

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.conf import validate_backend


def print_divider(char: str = "=", length: int = 70) -> None:
    """Print a divider line made of a specified character and length."""
    print(char * length)


def check_nulls(df: pl.DataFrame | pd.DataFrame | mpd.DataFrame, backend: str) -> bool:
    """Check for null values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for null values.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend used for the DataFrame ('polars', 'pandas', 'modin').
    :type backend: str
    :return: True if there are null values, False otherwise.
    :rtype: bool
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(backend)

    if backend == "pd":
        # Convert NumPy result to Python bool
        return bool(cast(pd.DataFrame, df).isnull().values.any())
    elif backend == "pl":
        # Polars-specific null check: sum the null counts and return a boolean
        polars_df = cast(pl.DataFrame, df)
        null_count = polars_df.null_count().select(pl.col("*").sum()).to_numpy().sum()
        return bool(null_count > 0)
    elif backend == "mpd":
        # Convert NumPy result to Python bool
        return bool(cast(mpd.DataFrame, df).isnull().values.any())
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")


def check_nans(df: pl.DataFrame | pd.DataFrame | mpd.DataFrame, backend: str) -> bool:
    """Check for NaN values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for NaN values.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend used for the DataFrame
                    ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :return: True if there are NaN values, False otherwise.
    :rtype: bool
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(backend)

    if backend == "pd":
        # Convert NumPy result to Python bool
        return bool(cast(pd.DataFrame, df).isna().values.any())
    elif backend == "pl":
        # Polars-specific NaN check: check if there are any NaNs
        polars_df = cast(pl.DataFrame, df)
        nan_count = polars_df.select((polars_df == float("nan")).sum()).to_numpy().sum()
        return bool(nan_count > 0)
    elif backend == "mpd":
        # Convert NumPy result to Python bool
        return bool(cast(mpd.DataFrame, df).isna().values.any())
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")
