"""tests/timeseriesdata/test_pandas_backend.py

This module contains pytest tests for the TimeSeriesData class using the Pandas backend.
It includes systematic testing with synthetic datasets of varying frequencies and sizes.
"""

# import pandas as pd
# import pytest
# from temporalscope.timeseriesdata import TimeSeriesData


# @pytest.fixture
# def sample_pandas_df():
#     data = {
#         "time": pd.date_range(start="1/1/2020", periods=5, freq="D"),
#         "value": [1, 2, 3, 4, 5],
#         "id": ["A", "A", "B", "B", "B"],
#     }
#     return pd.DataFrame(data)


# def test_initialization(sample_pandas_df):
#     ts_data = TimeSeriesData(
#         df=sample_pandas_df, time_col="time", id_col="id", backend="pandas"
#     )
#     assert ts_data.backend == "pandas"
#     assert ts_data.time_col == "time"
#     assert ts_data.id_col == "id"


# def test_groupby(sample_pandas_df):
#     ts_data = TimeSeriesData(
#         df=sample_pandas_df, time_col="time", id_col="id", backend="pandas"
#     )
#     grouped = ts_data.groupby()
#     assert len(grouped) == 2  # Two groups: 'A' and 'B'


# def test_run_method(sample_pandas_df):
#     def dummy_method(ts_data, *args, **kwargs):
#         return ts_data.df["value"].sum()

#     ts_data = TimeSeriesData(
#         df=sample_pandas_df, time_col="time", id_col="id", backend="pandas"
#     )
#     result = ts_data.run_method(dummy_method)
#     assert result == 15  # Sum of values [1, 2, 3, 4, 5]
