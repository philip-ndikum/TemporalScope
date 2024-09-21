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

"""TemporalScope/test/unit/test_core_utils.py

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

import warnings
from typing import Optional, Tuple, Union
from unittest.mock import patch

import modin.pandas as mpd
import numpy as np
import pandas as pd
import polars as pl
import pytest

from temporalscope.core.core_utils import (
    check_nans,
    check_nulls,
    get_api_keys,
    get_default_backend_cfg,
    print_divider,
    validate_and_convert_input,
    validate_backend,
    validate_input,
)

warnings.filterwarnings("ignore", message=".*defaulting to pandas.*")

# Mock API key constants
MOCK_OPENAI_API_KEY = "mock_openai_key"
MOCK_CLAUDE_API_KEY = "mock_claude_key"


# --- Data Generation Functions ---
def create_sample_data(num_samples: int = 100, with_nulls=False, with_nans=False):
    """Create sample data with options for introducing nulls and NaNs."""
    data = {
        "feature_1": np.random.rand(num_samples).tolist(),
        "feature_2": np.random.rand(num_samples).tolist(),
        "feature_3": np.random.rand(num_samples).tolist(),
    }

    if with_nans:
        for i in range(0, num_samples, 10):
            data["feature_2"][i] = float("nan")  # Every 10th value is NaN

    if with_nulls:
        for i in range(0, num_samples, 15):
            data["feature_3"][i] = None  # Every 15th value is Null

    return data


# Unified fixture for data with nulls and NaNs
@pytest.fixture
def sample_df_with_conditions():
    """Fixture for creating DataFrames for each backend.

    Provides a function to generate sample DataFrames with optional nulls or NaNs.

    :return: A function that generates a DataFrame and backend identifier based on the specified conditions.
    :rtype: Callable[[Optional[str], bool, bool], Tuple[Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], str]]
    """

    def _create_sample_df(
        backend: Optional[str] = None, with_nulls: bool = False, with_nans: bool = False
    ) -> Tuple[Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], str]:
        """Creates a sample DataFrame for the specified backend with optional nulls and NaNs.

        :param backend: The backend to use ('pd', 'pl', 'mpd'). Defaults to 'pd' if None.
        :type backend: Optional[str]
        :param with_nulls: Whether to include null values in the data. Defaults to False.
        :type with_nulls: bool
        :param with_nans: Whether to include NaN values in the data. Defaults to False.
        :type with_nans: bool
        :return: A tuple containing the DataFrame and the backend string.
        :rtype: Tuple[Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], str]
        :raises ValueError: If an unsupported backend is specified.
        """
        data = create_sample_data(with_nulls=with_nulls, with_nans=with_nans)
        if backend is None:
            backend = "pd"  # Default to pandas for backward compatibility
        if backend == "pd":
            return pd.DataFrame(data), "pd"
        elif backend == "pl":
            return pl.DataFrame(data), "pl"
        elif backend == "mpd":
            return mpd.DataFrame(data), "mpd"
        else:
            raise ValueError(f"Unsupported backend '{backend}'")

    return _create_sample_df


# --- Tests ---


def test_get_api_keys():
    """Test that get_api_keys retrieves environment variables correctly."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": MOCK_OPENAI_API_KEY, "CLAUDE_API_KEY": MOCK_CLAUDE_API_KEY}):
        api_keys = get_api_keys()
        assert api_keys["OPENAI_API_KEY"] == MOCK_OPENAI_API_KEY
        assert api_keys["CLAUDE_API_KEY"] == MOCK_CLAUDE_API_KEY

    with patch.dict("os.environ", {}, clear=True):
        api_keys = get_api_keys()
        assert api_keys["OPENAI_API_KEY"] is None
        assert api_keys["CLAUDE_API_KEY"] is None


@pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
@pytest.mark.parametrize("with_nans", [True, False])
def test_check_nans(backend, sample_df_with_conditions, with_nans):
    """Test check_nans for both NaNs present and no NaNs across backends."""
    df, _ = sample_df_with_conditions(backend=backend, with_nans=with_nans)
    result = check_nans(df, backend)
    expected = with_nans  # True if NaNs were introduced, else False
    assert result == expected, f"Expected {expected} but got {result} for backend {backend}"


def test_get_default_backend_cfg():
    """Test that the default backend configuration is returned correctly."""
    expected_cfg = {"BACKENDS": {"pl": "polars", "pd": "pandas", "mpd": "modin"}}
    result = get_default_backend_cfg()
    assert result == expected_cfg


@pytest.mark.parametrize("backend", ["pl", "pd", "mpd"])
def test_validate_backend_supported(backend):
    """Test that supported backends are validated successfully."""
    validate_backend(backend)


@pytest.mark.parametrize("invalid_backend", ["tf", "spark", "unknown"])
def test_validate_backend_unsupported(invalid_backend):
    """Test that unsupported backends raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        validate_backend(invalid_backend)


@pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
@pytest.mark.parametrize("target_backend", ["pl", "pd", "mpd"])
def test_validate_and_convert_input(sample_df_with_conditions, backend, target_backend):
    """Test that DataFrame conversion between backends works correctly."""
    df, _ = sample_df_with_conditions(backend=backend, with_nulls=False)
    result = validate_and_convert_input(df, target_backend)

    if target_backend == "pd":
        assert isinstance(result, pd.DataFrame), f"Expected Pandas DataFrame but got {type(result)}"
    elif target_backend == "pl":
        assert isinstance(result, pl.DataFrame), f"Expected Polars DataFrame but got {type(result)}"
    elif target_backend == "mpd":
        assert isinstance(result, mpd.DataFrame), f"Expected Modin DataFrame but got {type(result)}"


@pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
def test_validate_and_convert_input_invalid_type(backend):
    """Test that validate_and_convert_input raises TypeError when given an invalid DataFrame type."""
    invalid_df = "This is not a DataFrame"

    with pytest.raises(TypeError, match="Input DataFrame type"):
        validate_and_convert_input(invalid_df, backend)


def test_print_divider(capsys):
    """Test the print_divider function outputs the correct string."""
    print_divider("-", 50)
    captured = capsys.readouterr()
    assert captured.out == "-" * 50 + "\n"


def test_check_nans_invalid_backend(sample_df_with_conditions):
    """Test that an unsupported backend raises a ValueError in check_nans."""
    df, _ = sample_df_with_conditions(with_nans=True)
    with pytest.raises(ValueError, match="Unsupported backend"):
        check_nans(df, "invalid_backend")


@pytest.mark.parametrize(
    "backend, expected_type",
    [
        ("pl", pl.DataFrame),
        ("pd", pd.DataFrame),
        ("mpd", mpd.DataFrame),
    ],
)
def test_validate_input_correct_backend(sample_df_with_conditions, backend, expected_type):
    """Test that validate_input passes when the DataFrame matches the backend."""
    df, _ = sample_df_with_conditions(backend=backend, with_nulls=False)
    validate_input(df, backend)


@pytest.mark.parametrize("df_backend", ["pd", "pl", "mpd"])
@pytest.mark.parametrize("validate_backend", ["pd", "pl", "mpd"])
def test_validate_input_mismatched_backend(sample_df_with_conditions, df_backend, validate_backend):
    """Test that validate_input raises TypeError when the DataFrame does not match the backend."""
    df, _ = sample_df_with_conditions(backend=df_backend, with_nulls=False)

    if df_backend != validate_backend:
        # Expect TypeError when backends don't match
        with pytest.raises(TypeError, match="Expected a"):
            validate_input(df, validate_backend)
    else:
        # Should pass when backends match
        validate_input(df, validate_backend)


@pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
@pytest.mark.parametrize("with_nulls", [True, False])
def test_check_nulls(backend, sample_df_with_conditions, with_nulls):
    """Test check_nulls for both nulls present and no nulls across backends."""
    df, _ = sample_df_with_conditions(backend=backend, with_nulls=with_nulls)
    result = check_nulls(df, backend)
    expected = with_nulls  # True if nulls were introduced, else False
    assert result == expected, f"Expected {expected} but got {result} for backend {backend}"


# Test for invalid backend handling
def test_check_nulls_invalid_backend(sample_df_with_conditions):
    """Test that check_nulls raises ValueError when given an unsupported backend."""
    df, _ = sample_df_with_conditions(with_nulls=True)
    with pytest.raises(ValueError, match="Unsupported backend"):
        check_nulls(df, "invalid_backend")
