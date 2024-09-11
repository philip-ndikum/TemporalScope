"""
TemporalScope/temporalscope/conf.py

Package-level configurations and utilities.

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

import os

import modin.pandas as mpd
import pandas as pd
import polars as pl
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Supported backend configuration
TF_DEFAULT_CFG = {
    "BACKENDS": {"pl": "polars", "pd": "pandas", "mpd": "modin"},
}


def get_default_backend_cfg() -> dict[str, dict[str, str]]:
    """
    Get the default backend configuration.

    Returns
    -------
    dict[str, dict[str, str]]
        The default backend configuration.
    """
    return TF_DEFAULT_CFG.copy()


def validate_backend(backend: str) -> None:
    """
    Validate the backend against the supported backends in the configuration.

    Parameters
    ----------
    backend : str
        The backend to validate ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).

    Raises
    ------
    ValueError
        If the backend is not supported.
    """

    if backend not in TF_DEFAULT_CFG["BACKENDS"]:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends are: "
            f"{', '.join(TF_DEFAULT_CFG['BACKENDS'].keys())}."
        )


def validate_input(
    df: pl.DataFrame | pd.DataFrame | mpd.DataFrame, backend: str
) -> None:
    """
    Validate the input DataFrame to ensure it matches the specified backend.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        The DataFrame to validate.
    backend : str
        The backend against which to validate the DataFrame's type
        ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).

    Raises
    ------
    TypeError
        If the DataFrame does not match the expected type for the backend.
    """
    if backend == "pl" and not isinstance(df, pl.DataFrame):
        raise TypeError("Expected a Polars DataFrame.")
    elif backend == "pd" and not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")
    elif backend == "mpd" and not isinstance(df, mpd.DataFrame):
        raise TypeError("Expected a Modin DataFrame.")


def get_api_keys() -> dict[str, str | None]:
    """
    Retrieve API keys from environment variables.

    Returns
    -------
    Dict[str, Optional[str]]
        A dictionary containing the API keys, or None if not found.
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
