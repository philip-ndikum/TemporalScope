""" temporalscope/core/temporal_data_loader.py

This module implements the TimeFrame class, designed to provide a flexible and scalable interface for handling
time series data using multiple backends. It supports Polars as the default backend, Pandas as a secondary option
for users who prefer the traditional data processing framework, and Modin for users needing scalable Pandas-like
data processing with distributed computing.

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

from typing import Union, Optional, cast
import polars as pl
from polars import Expr
import pandas as pd
import modin.pandas as mpd
from temporalscope.conf import (
    get_default_backend_cfg,
    validate_input,
    validate_backend,
)


class TimeFrame:
    """Handles time series data with support for various backends like Polars, Pandas, and Modin.

    This class provides functionalities to manage time series data with optional grouping,
    available masks, and backend flexibility. It can handle large datasets efficiently.
    The class is intended for Machine & Deep Learning time series forecasting not classical
    time series forecasting. The implementation assumes one-step ahead but other classes
    & modules can be utilized for partioning for pre-trained multi-step DL models that are
    compatible with SHAP & related tools e.g. PyTorch or Tensorflow forecasting models.

    :param df: The input DataFrame.
    :type df: Union[pl.DataFrame, pd.DataFrame, modin.pandas.DataFrame]
    :param time_col: The column representing time in the DataFrame.
    :type time_col: str
    :param target_col: The column representing the target variable in the DataFrame.
    :type target_col: str
    :param id_col: Optional. The column representing the ID for grouping. Default is None.
    :type id_col: Optional[str]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, or 'mpd' for Modin). Default is 'pl'.
    :type backend: str
    :param sort: Optional. Sort the data by time_col (and id_col if provided) in ascending order. Default is True.
    :type sort: bool

    .. note::

       The default assumption for the `TimeFrame` class is that the dataset is cleaned and prepared for one-step-ahead
       forecasting, where the `target_col` directly corresponds to the label. The `id_col` is included for grouping and
       sorting purposes but is not used in the default model-building process.

    .. warning::

       Ensure that the `time_col` is properly formatted as a datetime type to avoid issues with sorting and grouping.

    .. rubric:: Examples

    .. code-block:: python

       # Example of creating a TimeFrame with Polars DataFrame
       data = pl.DataFrame({
           'time': pl.date_range(start='2021-01-01', periods=100, interval='1d'),
           'value': range(100)
       })
       tf = TimeFrame(data, time_col='time', target_col='value')

       # Accessing the data
       print(tf.get_data().head())

       # Example of creating a TimeFrame with Modin DataFrame
       import modin.pandas as mpd
       df = mpd.DataFrame({
           'time': pd.date_range(start='2021-01-01', periods=100, freq='D'),
           'value': range(100)
       })
       tf = TimeFrame(df, time_col='time', target_col='value', backend='mpd')

       # Accessing the data
       print(tf.get_data().head())
    """

    def __init__(
        self,
        df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame],
        time_col: str,
        target_col: str,
        id_col: Optional[str] = None,
        backend: str = "pl",
        sort: bool = True,
    ):
        self._cfg = get_default_backend_cfg()
        self._backend = backend
        self._df = df
        self._time_col = time_col
        self._target_col = target_col
        self._id_col = id_col
        self._sort = sort

        # Setup TimeFrame including renaming, sorting, and validations
        self.setup_timeframe()

    def setup_timeframe(self) -> None:
        """Sets up the TimeFrame object by validating and preparing data as required."""
        # Validate the columns are present and correct after potential renaming
        self.validate_columns()

        # Now sort data, assuming all columns are correct and exist
        if self._sort:
            self.sort_data(ascending=True)

        # Final validations after setup
        validate_backend(self._backend)
        validate_input(self._df, self._backend)

    @property
    def backend(self) -> str:
        """Return the backend used ('pl' for Polars, 'pd' for Pandas, or 'mpd' for Modin)."""
        return self._backend

    @property
    def time_col(self) -> str:
        """Return the column name representing time."""
        return self._time_col

    @property
    def target_col(self) -> str:
        """Return the column name representing the target variable."""
        return self._target_col

    @property
    def id_col(self) -> Optional[str]:
        """Return the column name used for grouping or None if not set."""
        return self._id_col

    def validate_columns(self) -> None:
        """Validate the presence and types of required columns in the DataFrame."""
        # Check for the presence of required columns, ignoring None values
        required_columns = [self.time_col, self._target_col] + (
            [self.id_col] if self.id_col else []
        )
        missing_columns = [
            col for col in required_columns if col and col not in self._df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def sort_data(self, ascending: bool = True) -> None:
        """Sorts the DataFrame based on the backend.

        :param ascending: Specifies whether to sort in ascending order.
        :type ascending: bool
        """
        sort_key = [self.id_col, self.time_col] if self.id_col else [self.time_col]

        if self._backend == "pl":
            # For Polars, sort the DataFrame
            if isinstance(self._df, pl.DataFrame):
                if ascending:
                    # Sort in ascending order
                    self._df = self._df.sort(sort_key)
                else:
                    # Sort in descending order using tuples with Polars' SORT_DESCENDING
                    sort_key_desc = [(col, pl.SORT_DESCENDING) for col in sort_key]
                    self._df = self._df.sort(sort_key_desc)

        elif self._backend in ["pd", "mpd"]:
            # For Pandas/Modin, ensure we have a DataFrame before sorting
            if isinstance(self._df, (pd.DataFrame, mpd.DataFrame)):
                self._df = self._df.sort_values(by=sort_key, ascending=ascending)

    def check_duplicates(self) -> None:
        """Check for duplicate time entries within groups, handling different data backends.

        :raises ValueError: If duplicate entries are found.
        """
        if self._backend == "pl":
            # Polars specific check: Use boolean masks
            if self._id_col:
                # Create unique expressions for id and time columns
                id_duplicated_expr: Expr = pl.col(self._id_col).is_duplicated()
                time_duplicated_expr: Expr = pl.col(self._time_col).is_duplicated()
                # Combine expressions
                combined_expr: Expr = id_duplicated_expr | time_duplicated_expr
                duplicates = self._df.filter(combined_expr)  # type: ignore
            else:
                # Only check the time column for duplicates
                duplicates = self._df.filter(pl.col(self._time_col).is_duplicated())  # type: ignore
            # Check for duplicates by inspecting the number of rows
            if duplicates.height > 0:
                raise ValueError("Duplicate time entries found within the same group.")
        elif self._backend in ["pd", "mpd"]:
            # Cast to Pandas DataFrame for Pandas/Modin specific check
            pandas_df = cast(pd.DataFrame, self._df)
            duplicates = pandas_df.duplicated(
                subset=(
                    [self._id_col, self._time_col] if self._id_col else [self._time_col]
                )
            )

            if duplicates.any():
                raise ValueError("Duplicate time entries found within the same group.")

    def get_data(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """Return the DataFrame in its current state.

        :return: The DataFrame managed by the TimeFrame instance.
        :rtype: Union[pl.DataFrame, pd.DataFrame]
        """
        return self._df

    def get_grouped_data(self) -> Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]:
        """Return the grouped DataFrame if an ID column is provided.

        :return: Grouped DataFrame by the ID column if it is set, otherwise returns the original DataFrame.
        :rtype: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        :raises ValueError: If the ID column is not set or an unsupported backend is provided.
        :raises TypeError: If the DataFrame type does not match the expected type for the specified backend.
        """
        if not self.id_col:
            raise ValueError("ID column is not set; cannot group data.")

        if self._backend == "pl":
            # Polars specific group_by with aggregation
            if isinstance(self._df, pl.DataFrame):
                return self._df.group_by(self.id_col).agg(
                    pl.all()
                )  # Polars uses `group_by`
            else:
                raise TypeError(f"Expected Polars DataFrame but got {type(self._df)}.")
        elif self._backend == "pd":
            # Pandas specific groupby
            if isinstance(self._df, pd.DataFrame):
                return self._df.groupby(self.id_col).apply(lambda x: x)
            else:
                raise TypeError(f"Expected Pandas DataFrame but got {type(self._df)}.")
        elif self._backend == "mpd":
            # Modin uses the same API as Pandas for this operation
            if isinstance(self._df, mpd.DataFrame):
                return self._df.groupby(self.id_col).apply(lambda x: x)
            else:
                raise TypeError(f"Expected Modin DataFrame but got {type(self._df)}.")
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")