"""temporalscope/core/utils.py

This module provides utility functions that can be used throughout the TemporalScope package.
It includes methods for printing dividers, checking for nulls and NaNs, and validating the backend.
"""

from typing import Union
import polars as pl
import pandas as pd
import modin.pandas as mpd
from temporalscope.config import validate_backend


def print_divider(char="=", length=70):
    """Prints a divider line made of a specified character and length."""
    print(char * length)


def check_nulls(
    df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], backend: str
) -> bool:
    """Check for null values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for null values.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend used for the DataFrame ('polars', 'pandas', 'modin').
    :type backend: str
    :return: True if there are null values, False otherwise.
    :rtype: bool
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(
        backend
    )  # This will raise ValueError if the backend is unsupported

    if backend == "pd":
        return df.isnull().values.any()
    elif backend == "pl":
        return df.null_count().sum().sum() > 0
    elif backend == "mpd":
        return df.isnull().values.any()  # Assuming Modin's API is similar to Pandas
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")


def check_nans(
    df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], backend: str
) -> bool:
    """Check for NaN values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for NaN values.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend used for the DataFrame ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :return: True if there are NaN values, False otherwise.
    :rtype: bool
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(backend)  # Validate backend using the centralized function

    if backend == "pd":
        return df.isna().values.any()
    elif backend == "pl":
        return (df == pl.lit(float("nan"))).sum().sum() > 0
    elif backend == "mpd":
        return df.isna().values.any()  # Assuming Modin's API is similar to Pandas
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")
