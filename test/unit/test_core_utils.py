""" TemporalScope/test/unit/test_core_utils.py

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

from unittest.mock import patch
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest
from temporalscope.core.core_utils import (
    get_api_keys,
    get_default_backend_cfg,
    validate_backend,
    validate_input,
    validate_and_convert_input,
    check_nans,
    check_nulls,
    print_divider
)
import warnings

warnings.filterwarnings("ignore", message=".*defaulting to pandas.*")

# Define mock API key constants
MOCK_OPENAI_API_KEY = "mock_openai_key"
MOCK_CLAUDE_API_KEY = "mock_claude_key"

def create_sample_data():
    """Create a sample data dictionary."""
    return {"a": [1, 2, 3]}

@pytest.fixture(params=["pd", "pl", "mpd"])
def sample_df(request):
    """Fixture for creating sample DataFrames for each backend."""
    data = create_sample_data()
    backend = request.param
    if backend == "pd":
        return pd.DataFrame(data), backend
    elif backend == "pl":
        return pl.DataFrame(data), backend
    elif backend == "mpd":
        return mpd.DataFrame(data), backend

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

def test_validate_input_valid(sample_df):
    """Test that valid DataFrame types are accepted based on the backend."""
    df, backend = sample_df
    validate_input(df, backend)

@pytest.mark.parametrize("backend, df_type", [
    ("pl", pd.DataFrame), ("pd", pl.DataFrame), ("mpd", pd.DataFrame)
])
def test_validate_input_invalid(backend, df_type):
    """Test that invalid DataFrame types raise a TypeError based on the backend."""
    with pytest.raises(TypeError, match="Expected a .* DataFrame"):
        validate_input(df_type(create_sample_data()), backend)

@pytest.mark.parametrize("output_backend", ["pd", "pl", "mpd"])
def test_validate_and_convert_input(sample_df, output_backend):
    """Test that validate_and_convert_input correctly converts DataFrames."""
    df, input_backend = sample_df
    result = validate_and_convert_input(df, output_backend)
    
    if output_backend == "pd":
        assert isinstance(result, pd.DataFrame)
    elif output_backend == "pl":
        assert isinstance(result, pl.DataFrame)
    elif output_backend == "mpd":
        assert isinstance(result, mpd.DataFrame)
    
    assert result.shape == df.shape
    
    if output_backend == "pl":
        assert result["a"].to_list() == [1, 2, 3]
    else:
        assert result["a"].tolist() == [1, 2, 3]

def test_validate_and_convert_input_invalid_backend(sample_df):
    """Test that an invalid backend raises a ValueError."""
    df, _ = sample_df
    with pytest.raises(ValueError, match="Unsupported backend"):
        validate_and_convert_input(df, "invalid_backend")

def test_validate_and_convert_input_invalid_df_type():
    """Test that an invalid DataFrame type raises a TypeError."""
    with pytest.raises(TypeError, match="Input DataFrame type .* does not match the specified backend"):
        validate_and_convert_input({"a": [1, 2, 3]}, "pd")  # Not a DataFrame
        



# Test data for check_nulls
test_nulls_data = [
    ("pd", pd.DataFrame({"FEATURE_1": [1, None, 3]}), True),
    (
        "pl",
        pl.DataFrame({"FEATURE_1": [1, None, 3]}, schema={"FEATURE_1": pl.Float64}),
        True,
    ),
    ("mpd", mpd.DataFrame({"FEATURE_1": [1, None, 3]}), True),
    ("pd", pd.DataFrame({"FEATURE_1": [1, 2, 3]}), False),
    (
        "pl",
        pl.DataFrame({"FEATURE_1": [1, 2, 3]}, schema={"FEATURE_1": pl.Float64}),
        False,
    ),
    ("mpd", mpd.DataFrame({"FEATURE_1": [1, 2, 3]}), False),
    ("pd", pd.DataFrame(), False),  # Empty DataFrame for Pandas
    (
        "pl",
        pl.DataFrame({"FEATURE_1": []}, schema={"FEATURE_1": pl.Float64}),
        False,
    ),  # Empty DataFrame for Polars
    ("mpd", mpd.DataFrame(), False),  # Empty DataFrame for Modin
]

# Test data for check_nans
test_nans_data = [
    ("pd", pd.DataFrame({"FEATURE_1": [1, float("nan"), 3]}), True),
    (
        "pl",
        pl.DataFrame(
            {"FEATURE_1": [1, float("nan"), 3]},
            schema={"FEATURE_1": pl.Float64},
        ),
        True,
    ),
    ("mpd", mpd.DataFrame({"FEATURE_1": [1, float("nan"), 3]}), True),
    ("pd", pd.DataFrame({"FEATURE_1": [1, 2, 3]}), False),
    (
        "pl",
        pl.DataFrame({"FEATURE_1": [1, 2, 3]}, schema={"FEATURE_1": pl.Float64}),
        False,
    ),
    ("mpd", mpd.DataFrame({"FEATURE_1": [1, 2, 3]}), False),
    ("pd", pd.DataFrame(), False),  # Empty DataFrame for Pandas
    (
        "pl",
        pl.DataFrame({"FEATURE_1": []}, schema={"FEATURE_1": pl.Float64}),
        False,
    ),  # Empty DataFrame for Polars
    ("mpd", mpd.DataFrame(), False),  # Empty DataFrame for Modin
]


def test_print_divider(capsys):
    """Test that print_divider prints without error."""
    print_divider()
    captured = capsys.readouterr()
    # Ensure that print was called and output is non-empty
    assert len(captured.out.strip()) > 0


@pytest.mark.parametrize("backend, df, expected", test_nulls_data)
def test_check_nulls(backend, df, expected):
    """Test that check_nulls detects null values correctly across backends."""
    if backend == "pl":
        # Polars-specific null check: check if there are any null values
        null_count = df.null_count().select(pl.col("*").sum()).to_numpy()[0][0]
        assert (null_count > 0) == expected
    else:
        # Pandas/Modin null check using the utils function
        result = check_nulls(df, backend)
        assert result == expected


@pytest.mark.parametrize("backend, df, expected", test_nans_data)
def test_check_nans(backend, df, expected):
    """Test that check_nans detects NaN values correctly across backends."""
    if backend == "pl":
        # Polars-specific NaN check: convert to boolean, count NaN values
        nan_count = df.select(pl.col("FEATURE_1").is_nan()).sum().to_numpy()[0][0]
        assert (nan_count > 0) == expected
    else:
        # Pandas/Modin NaN check using the utils function
        result = check_nans(df, backend)
        assert result == expected


@pytest.mark.parametrize("backend", ["unsupported_backend"])
def test_check_nulls_invalid_backend(backend):
    """Test that check_nulls raises ValueError for unsupported backends."""
    df = pd.DataFrame({"FEATURE_1": [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsupported backend"):
        check_nulls(df, backend)


@pytest.mark.parametrize("backend", ["unsupported_backend"])
def test_check_nans_invalid_backend(backend):
    """Test that check_nans raises ValueError for unsupported backends."""
    df = pd.DataFrame({"FEATURE_1": [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsupported backend"):
        check_nans(df, backend)
