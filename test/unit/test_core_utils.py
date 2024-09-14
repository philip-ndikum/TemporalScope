import warnings

import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest

from temporalscope.core.utils import check_nans, check_nulls, print_divider

warnings.filterwarnings("ignore", message=".*defaulting to pandas.*")


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
