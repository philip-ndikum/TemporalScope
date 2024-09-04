""" temporalscope/tests/unit/test_temporal_data_loader.py
"""

import pytest
import polars as pl
import pandas as pd
import modin.pandas as mpd
from temporalscope.core.temporal_data_loader import TimeFrame


@pytest.fixture
def sample_pandas_df():
    """Fixture for creating a sample Pandas DataFrame."""
    return pd.DataFrame(
        {
            "time": pd.date_range(start="2021-01-01", periods=10, freq="D"),
            "value": range(10),
        }
    )


@pytest.fixture
def sample_polars_df():
    """Fixture for creating a sample Polars DataFrame."""
    end_date = pl.Series(
        "time", pd.date_range(start="2021-01-01", periods=10, freq="D")
    )
    return pl.DataFrame({"time": end_date, "value": range(10)})


@pytest.fixture
def sample_modin_df():
    """Fixture for creating a sample Modin DataFrame."""
    return mpd.DataFrame(
        {
            "time": pd.date_range(start="2021-01-01", periods=10, freq="D"),
            "value": range(10),
        }
    )


def test_initialize_pandas(sample_pandas_df):
    """Test TimeFrame initialization with Pandas backend."""
    tf = TimeFrame(sample_pandas_df, time_col="time", target_col="value", backend="pd")
    if tf.backend != "pd" or tf.time_col != "time" or tf.target_col != "value":
        pytest.fail("Initialization with Pandas backend failed.")


def test_initialize_polars(sample_polars_df):
    """Test TimeFrame initialization with Polars backend."""
    tf = TimeFrame(sample_polars_df, time_col="time", target_col="value", backend="pl")
    if tf.backend != "pl" or tf.time_col != "time" or tf.target_col != "value":
        pytest.fail("Initialization with Polars backend failed.")


def test_initialize_modin(sample_modin_df):
    """Test TimeFrame initialization with Modin backend."""
    tf = TimeFrame(sample_modin_df, time_col="time", target_col="value", backend="mpd")
    if tf.backend != "mpd" or tf.time_col != "time" or tf.target_col != "value":
        pytest.fail("Initialization with Modin backend failed.")


def test_invalid_backend(sample_pandas_df):
    """Test TimeFrame with an invalid backend."""
    with pytest.raises(ValueError):
        TimeFrame(
            sample_pandas_df,
            time_col="time",
            target_col="value",
            backend="invalid_backend",
        )


def test_missing_columns(sample_pandas_df):
    """Test TimeFrame initialization with missing required columns."""
    # Missing time column
    with pytest.raises(ValueError):
        TimeFrame(
            sample_pandas_df.drop(columns=["time"]),
            time_col="time",
            target_col="value",
            backend="pd",
        )
    # Missing target column
    with pytest.raises(ValueError):
        TimeFrame(
            sample_pandas_df.drop(columns=["value"]),
            time_col="time",
            target_col="value",
            backend="pd",
        )


def test_duplicate_time_entries(sample_pandas_df):
    """Test handling of duplicate time entries."""
    sample_pandas_df.loc[1, "time"] = sample_pandas_df.loc[0, "time"]
    tf = TimeFrame(sample_pandas_df, time_col="time", target_col="value", backend="pd")
    try:
        tf.check_duplicates()
        pytest.fail("Duplicate time entries should raise a ValueError.")
    except ValueError:
        pass


def test_get_data(sample_pandas_df):
    """Test get_data method."""
    tf = TimeFrame(sample_pandas_df, time_col="time", target_col="value", backend="pd")
    df = tf.get_data()
    if not isinstance(df, pd.DataFrame) or df.shape != sample_pandas_df.shape:
        pytest.fail("get_data method failed to return correct DataFrame.")


def test_rename_target_column(sample_pandas_df):
    """Test renaming of target column."""
    tf = TimeFrame(
        sample_pandas_df,
        time_col="time",
        target_col="value",
        backend="pd",
        rename_target=True,
    )
    df = tf.get_data()
    if "y" not in df.columns or "value" in df.columns:
        pytest.fail("Renaming of target column failed.")
