""" temporalscope/core/temporal_data_loader.py

This module implements the TimeFrame class, designed to provide a flexible and scalable interface for handling
time series data using multiple backends. It supports Polars as the default backend, Pandas as a secondary option
for users who prefer the traditional data processing framework, and Modin for users needing scalable Pandas-like
data processing with distributed computing.

"""

from typing import Union, Optional, List, Tuple, Callable
import polars as pl
import pandas as pd
import modin.pandas as mpd
from temporalscope.config import (
    get_default_backend_cfg,
    validate_input,
    validate_backend,
)
from temporalscope.partition.base_temporal_partioner import BaseTemporalPartitioner


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
    :param rename_target: Optional. Whether to rename the target_col to 'y'. Default is False.
    :type rename_target: bool

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
        rename_target: bool = False,
    ):
        self._cfg = get_default_backend_cfg()
        self._backend = backend
        self._df = df
        self._time_col = time_col
        self._target_col = target_col
        self._id_col = id_col
        self._sort = sort
        self._rename_target = rename_target
        self._rename_target_column()

        if self._sort:
            self.sort_data(ascending=True)

        validate_backend(self._backend)
        validate_input(self._df, self._backend)

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

    def validate_columns(self):
        """Validate the presence and types of required columns in the DataFrame."""
        # Check for the presence of required columns
        required_columns = [self.time_col, self.target_col] + (
            [self.id_col] if self.id_col else []
        )
        missing_columns = [
            col for col in required_columns if col not in self._df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Ensure time_col is properly formatted as datetime for sorting and operations
        if self._backend in ["pd", "mpd"]:
            if not pd.api.types.is_datetime64_any_dtype(self._df[self.time_col]):
                self._df[self.time_col] = pd.to_datetime(self._df[self.time_col])
        elif self._backend == "pl":
            if self._df[self.time_col].dtype != pl.Datetime:
                self._df = self._df.with_columns(
                    pl.col(self.time_col).str.strptime(pl.Datetime)
                )

    def _rename_target_column(self):
        """Rename the target column to 'y' if rename_target is True."""
        if self._rename_target:
            if self._backend == "pl":
                self._df = self._df.rename({self.target_col: "y"})
            elif self._backend in ["pd", "mpd"]:
                self._df.rename(columns={self.target_col: "y"}, inplace=True)

    def sort_data(self, ascending: bool = True):
        """Sorts the DataFrame based on the backend.

        :param ascending: Specifies whether to sort in ascending order.
        :type ascending: bool
        """
        sort_key = [self.id_col, self.time_col] if self.id_col else [self.time_col]
        if self._backend == "pl":
            # Polars sorting - use reverse which is the opposite of ascending
            self._df = self._df.sort(sort_key, reverse=not ascending)
        elif self._backend == "pd":
            # Pandas sorting
            self._df = self._df.sort_values(by=sort_key, ascending=ascending)
        elif self._backend == "mpd":
            # Modin uses the same API as Pandas for sorting
            self._df = self._df.sort_values(by=sort_key, ascending=ascending)

    def check_duplicates(self):
        """Check for duplicate time entries within groups, handling different data backends."""
        if self._backend == "pl":
            # Polars specific check
            duplicates = self._df.filter(
                self._df.select(
                    [self._id_col, self._time_col] if self._id_col else [self._time_col]
                ).is_duplicated()
            )
            if duplicates.height > 0:
                raise ValueError("Duplicate time entries found within the same group.")
        elif self._backend == "pd":
            # Pandas specific check
            duplicates = self._df.duplicated(
                subset=(
                    [self._id_col, self._time_col] if self._id_col else [self._time_col]
                )
            )
            if duplicates.any():
                raise ValueError("Duplicate time entries found within the same group.")
        elif self._backend == "mpd":
            # Modin uses the same API as Pandas for this operation
            duplicates = self._df.duplicated(
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
        """
        if not self.id_col:
            raise ValueError("ID column is not set; cannot group data.")

        if self._backend == "pl":
            return self._df.groupby(self.id_col).agg(pl.all())
        elif self._backend == "pd":
            return self._df.groupby(self.id_col).apply(lambda x: x)
        elif self._backend == "mpd":
            return self._df.groupby(self.id_col).apply(
                lambda x: x
            )  # Modin uses pandas-like syntax

        raise ValueError("Unsupported backend specified.")

    def apply_partitioning(self, partitioner: BaseTemporalPartitioner):
        """Apply a partitioning strategy and update the partition configuration."""
        self._partition_cfg["partitions"] = partitioner.get_partitions()
        self._partition_cfg["partitioner"] = partitioner

    def get_partitioned_data(self) -> List[Union[pd.DataFrame, pl.DataFrame]]:
        """Return the partitioned data based on the current partition configuration."""
        if self._partition_cfg["partitions"] is None:
            raise ValueError("No partitioning strategy has been applied.")
        partitioner = self._partition_cfg["partitioner"]
        return partitioner.get_partitioned_data()

    def get_partition_indices(self) -> List[Tuple[int, int]]:
        """Return the partition indices."""
        if self._partition_cfg["partitions"] is None:
            raise ValueError("No partitioning strategy has been applied.")
        return self._partition_cfg["partitions"]

    def run_method(self, method: Callable, *args, **kwargs):
        """Run an analytical method on the data.

        :param method: The method to run.
        :type method: Callable
        :param args: Additional arguments to pass to the method.
        :param kwargs: Additional keyword arguments to pass to the method.
        :return: The result of the analytical method.
        :rtype: Any
        """
        return method(self, *args, **kwargs)
