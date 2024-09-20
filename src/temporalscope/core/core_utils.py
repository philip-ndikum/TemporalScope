""" TemporalScope/src/temporalscope/core/core_utils.py

This module provides utility functions that can be used throughout the TemporalScope package.
It includes methods for printing dividers, checking for nulls and NaNs, and validating the backend.

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Union, cast, Dict, Optional, NoReturn
import os
from dotenv import load_dotenv
import polars as pl
import pandas as pd
import modin.pandas as mpd

# Load environment variables from the .env file
load_dotenv()

# Backend abbreviations
BACKEND_POLARS = "pl"
BACKEND_PANDAS = "pd"
BACKEND_MODIN = "mpd"

# Mapping of backend keys to their full names or module references
BACKENDS = {
    BACKEND_POLARS: "polars",
    BACKEND_PANDAS: "pandas",
    BACKEND_MODIN: "modin",
}

TF_DEFAULT_CFG = {
    "BACKENDS": BACKENDS,
}

# Define a type alias for DataFrames that support Pandas, Modin, and Polars backends
SupportedBackendDataFrame = Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]

def get_default_backend_cfg() -> Dict[str, Dict[str, str]]:
    """Retrieve the application configuration settings.

    :return: A dictionary of configuration settings.
    :rtype: Dict[str, Dict[str, str]]
    """
    return TF_DEFAULT_CFG.copy()


def validate_backend(backend: str) -> None:
    """Validate the backend against the supported backends in the configuration.

    :param backend: The backend to validate ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :raises ValueError: If the backend is not supported.
    """
    if backend not in TF_DEFAULT_CFG["BACKENDS"]:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends are: "
            f"{', '.join(TF_DEFAULT_CFG['BACKENDS'].keys())}."
        )


def raise_invalid_backend(backend: str) -> NoReturn:
    """Raise a ValueError for an invalid backend.

    :param backend: The backend to validate ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :raises ValueError: If the backend is not supported.
    """
    raise ValueError(f"Unsupported backend: {backend}")


def validate_input(
    df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], backend: str
) -> None:
    """Validate that the DataFrame matches the expected type for the specified backend.

    :param df: The DataFrame to validate.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend against which to validate the DataFrame's type ('pl', 'pd', 'mpd').
    :type backend: str
    :raises TypeError: If the DataFrame does not match the expected type for the backend.
    """
    if backend == BACKEND_POLARS and not isinstance(df, pl.DataFrame):
        raise TypeError("Expected a Polars DataFrame.")
    elif backend == BACKEND_PANDAS and not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")
    elif backend == BACKEND_MODIN and not isinstance(df, mpd.DataFrame):
        raise TypeError("Expected a Modin DataFrame.")


def validate_and_convert_input(
    df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], backend: str
) -> Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]:
    """Validates and converts the input DataFrame to the specified backend type.

    :param df: The input DataFrame to validate and convert.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The desired backend type ('pl', 'pd', or 'mpd').
    :type backend: str
    :return: The DataFrame converted to the specified backend type.
    :rtype: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :raises TypeError: If the input DataFrame type doesn't match the specified backend or conversion fails.
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(backend)  # Validates if backend is supported

    if backend == BACKEND_POLARS:
        if isinstance(df, pl.DataFrame):
            return df
        elif isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)  # Convert Pandas to Polars
        elif isinstance(df, mpd.DataFrame):
            return pl.from_pandas(df._to_pandas())  # Modin to Pandas to Polars
    elif backend == BACKEND_PANDAS:
        if isinstance(df, pd.DataFrame):
            return df
        elif isinstance(df, pl.DataFrame):
            return df.to_pandas()  # Convert Polars to Pandas
        elif isinstance(df, mpd.DataFrame):
            return df._to_pandas()  # Convert Modin to Pandas
    elif backend == BACKEND_MODIN:
        if isinstance(df, mpd.DataFrame):
            return df
        elif isinstance(df, pd.DataFrame):
            return mpd.DataFrame(df)  # Convert Pandas to Modin
        elif isinstance(df, pl.DataFrame):
            return mpd.DataFrame(df.to_pandas())  # Polars to Pandas to Modin

    # If none of the types match, raise a TypeError
    raise TypeError(
        f"Input DataFrame type {type(df)} does not match the specified backend '{backend}'"
    )


def get_api_keys() -> Dict[str, Optional[str]]:
    """Retrieve API keys from environment variables.

    :return: A dictionary containing the API keys, or None if not found.
    :rtype: Dict[str, Optional[str]]
    """
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY"),
    }

    # Print warnings if keys are missing
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


def check_nulls(
    df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], backend: str
) -> bool:
    """Check for null values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for null values.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend used for the DataFrame ('pl', 'pd', 'mpd').
    :type backend: str
    :return: True if there are null values, False otherwise.
    :rtype: bool
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(backend)

    if backend == BACKEND_PANDAS:
        return bool(cast(pd.DataFrame, df).isnull().values.any())
    elif backend == BACKEND_POLARS:
        polars_df = cast(pl.DataFrame, df)
        null_count = polars_df.null_count().select(pl.col("*").sum()).to_numpy().sum()
        return bool(null_count > 0)
    elif backend == BACKEND_MODIN:
        return bool(cast(mpd.DataFrame, df).isnull().values.any())
    else:
        raise_invalid_backend(backend)


def check_nans(
    df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame], backend: str
) -> bool:
    """Check for NaN values in the DataFrame using the specified backend.

    :param df: The DataFrame to check for NaN values.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param backend: The backend used for the DataFrame ('pl', 'pd', 'mpd').
    :type backend: str
    :return: True if there are NaN values, False otherwise.
    :rtype: bool
    :raises ValueError: If the backend is not supported.
    """
    validate_backend(backend)

    if backend == BACKEND_PANDAS:
        return bool(cast(pd.DataFrame, df).isna().values.any())
    elif backend == BACKEND_POLARS:
        polars_df = cast(pl.DataFrame, df)
        nan_count = polars_df.select((polars_df == float("nan")).sum()).to_numpy().sum()
        return bool(nan_count > 0)
    elif backend == BACKEND_MODIN:
        return bool(cast(mpd.DataFrame, df).isna().values.any())
    else:
        raise_invalid_backend(backend)
