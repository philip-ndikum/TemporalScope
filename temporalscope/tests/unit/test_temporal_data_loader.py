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
    assert tf.backend == "pandas"
    assert tf.time_col == "time"
    assert tf.target_col == "value"


def test_initialize_polars(sample_polars_df):
    """Test TimeFrame initialization with Polars backend."""
    tf = TimeFrame(sample_polars_df, time_col="time", target_col="value", backend="pl")
    assert tf.backend == "polars"
    assert tf.time_col == "time"
    assert tf.target_col == "value"


def test_initialize_modin(sample_modin_df):
    """Test TimeFrame initialization with Modin backend."""
    tf = TimeFrame(sample_modin_df, time_col="time", target_col="value", backend="mpd")
    assert tf.backend == "modin"
    assert tf.time_col == "time"
    assert tf.target_col == "value"


def test_invalid_backend(sample_pandas_df):
    """Test TimeFrame with an invalid backend."""
    with pytest.raises(ValueError):
        TimeFrame(
            sample_pandas_df,
            time_col="time",
            target_col="value",
            backend="invalid_backend",
        )


def test_time_col_missing(sample_pandas_df):
    """Test TimeFrame initialization with a missing time column."""
    with pytest.raises(ValueError):
        TimeFrame(
            sample_pandas_df.drop(columns=["time"]),
            time_col="time",
            target_col="value",
            backend="pd",
        )


def test_target_col_missing(sample_pandas_df):
    """Test TimeFrame initialization with a missing target column."""
    with pytest.raises(ValueError):
        TimeFrame(
            sample_pandas_df.drop(columns=["value"]),
            time_col="time",
            target_col="value",
            backend="pd",
        )


def test_duplicate_time_entries(sample_pandas_df):
    """Test duplicate time entries handling."""
    # Introduce a duplicate time entry
    sample_pandas_df.loc[1, "time"] = sample_pandas_df.loc[0, "time"]
    with pytest.raises(ValueError):
        TimeFrame(sample_pandas_df, time_col="time", target_col="value", backend="pd")


def test_get_data(sample_pandas_df):
    """Test get_data method."""
    tf = TimeFrame(sample_pandas_df, time_col="time", target_col="value", backend="pd")
    df = tf.get_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_pandas_df.shape


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
    assert "y" in df.columns
    assert "value" not in df.columns
