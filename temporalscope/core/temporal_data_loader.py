""" temporalscope/core/temporal_data_loader.py

This module implements the TimeFrame class, designed to provide a flexible and scalable interface for handling
time series data using multiple backends. It supports Polars as the default backend, Pandas as a secondary option
for users who prefer the traditional data processing framework, and Modin for users needing scalable Pandas-like
data processing with distributed computing.

"""

from typing import Union, Optional, List, Tuple, Callable
import polars as pl
import pandas as pd
import warnings
from temporalscope.partition.base_temporal_partioner import BaseTemporalPartitioner

TF_DEFAULT_CFG = {
    "BACKENDS": {"pl": "polars", "pd": "pandas", "md": "modin"},
}


class TimeFrame:
    """Handles time series data with support for various backends like Polars, Pandas, and Modin.

    This class provides functionalities to manage time series data with optional grouping,
    available masks, and backend flexibility. It can handle large datasets efficiently.

    :param df: The input DataFrame.
    :type df: Union[pl.DataFrame, pd.DataFrame, modin.pandas.DataFrame]
    :param time_col: The column representing time in the DataFrame.
    :type time_col: str
    :param target_col: The column representing the target variable in the DataFrame.
    :type target_col: str
    :param id_col: Optional. The column representing the ID for grouping. Default is None.
    :type id_col: Optional[str]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, or 'md' for Modin). Default is 'pl'.
    :type backend: str
    :param sort: Optional. Whether to sort the data by time_col (and id_col if provided). Default is True.
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
       import modin.pandas as pd
       data = pd.DataFrame({
           'time': pd.date_range(start='2021-01-01', periods=100, freq='D'),
           'value': range(100)
       })
       tf = TimeFrame(data, time_col='time', target_col='value', backend='md')

       # Accessing the data
       print(tf.get_data().head())
    """

    def __init__(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],
        time_col: str,
        target_col: str,
        id_col: Optional[str] = None,
        backend: str = "pl",
        sort: bool = True,
        rename_target: bool = False,
    ):
        self._cfg = TF_DEFAULT_CFG.copy()
        self._df = df
        self._time_col = time_col
        self._target_col = target_col
        self._id_col = id_col

        # Handle both short forms and full names
        backend_lower = backend.lower()
        if backend_lower in self._cfg["BACKENDS"]:
            self._backend = self._cfg["BACKENDS"][backend_lower]
        elif backend_lower in self._cfg["BACKENDS"].values():
            self._backend = backend_lower
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported backends are: "
                f"{', '.join(self._cfg['BACKENDS'].keys())}, "
                f"{', '.join(self._cfg['BACKENDS'].values())}"
            )

        self._sort = sort
        self._rename_target = rename_target
        self._partition_cfg = {"partitions": None, "partitioner": None}

        self._validate_config()
        self._validate_input()
        self._apply_backend_logic()
        self._rename_target_column()

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

    def _validate_config(self):
        """Validate the instance configuration against the default configuration."""
        for key in TF_DEFAULT_CFG.keys():
            if key not in self._cfg:
                raise ValueError(f"Missing configuration key: {key}")
            if type(self._cfg[key]) != type(TF_DEFAULT_CFG[key]):
                raise TypeError(
                    f"Incorrect type for configuration key: {key}. Expected {type(TF_DEFAULT_CFG[key])}, got {type(self._cfg[key])}."
                )


    @property
    def backend(self) -> str:
        """Return the backend used ('polars', 'pandas', or 'modin')."""
        return self._backend


    def _validate_input(self) -> None:
        """Validate the input DataFrame and ensure required columns are present."""
        # Validate the DataFrame type based on the backend
        if self._backend == "polars":
            if not isinstance(self._df, pl.DataFrame):
                raise TypeError("Expected a Polars DataFrame.")
        elif self._backend == "pandas":
            if not isinstance(self._df, pd.DataFrame):
                raise TypeError("Expected a Pandas DataFrame.")
        elif self._backend == "modin":
            import modin.pandas as mpd

            if not isinstance(self._df, mpd.DataFrame):
                raise TypeError("Expected a Modin DataFrame.")

        # Validate required columns
        if self.time_col not in self._df.columns:
            raise ValueError(
                f"Time column '{self.time_col}' must be in the DataFrame columns"
            )

        if self.target_col not in self._df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' must be in the DataFrame columns"
            )

        if self.id_col and self.id_col not in self._df.columns:
            raise ValueError(
                f"ID column '{self.id_col}' must be in the DataFrame columns if provided"
            )

        # Ensure time_col is datetime for proper sorting
        if self._backend in ["pandas", "modin"]:
            if not pd.api.types.is_datetime64_any_dtype(self._df[self.time_col]):
                self._df[self.time_col] = pd.to_datetime(self._df[self.time_col])
        elif self._backend == "polars":
            if self._df[self.time_col].dtype != pl.Datetime:
                self._df = self._df.with_columns(
                    pl.col(self.time_col).str.strptime(pl.Datetime)
                )

    def _apply_backend_logic(self):
        """Apply backend-specific logic such as sorting and grouping."""
        if self._backend == "polars":
            self._polars_logic()
        elif self._backend == "pandas":
            self._pandas_logic()
        elif self._backend == "modin":
            self._modin_logic()

    def _polars_logic(self):
        """Apply Polars-specific logic like sorting and handling duplicates."""
        if self._sort:
            self._df = self._df.sort(
                by=[self.id_col, self.time_col] if self.id_col else [self.time_col]
            )
        self._check_duplicates()

    def _pandas_logic(self):
        """Apply Pandas-specific logic like sorting and handling duplicates."""
        if self._sort:
            self._df = self._df.sort_values(
                by=[self.id_col, self.time_col] if self.id_col else [self.time_col]
            )
        self._check_duplicates()

    def _modin_logic(self):
        """Apply Modin-specific logic like sorting and handling duplicates."""
        if self._sort:
            self._df = self._df.sort_values(
                by=[self.id_col, self.time_col] if self.id_col else [self.time_col]
            )
        self._check_duplicates()

    def _check_duplicates(self):
        """Check for duplicate time entries within groups."""
        if self._backend == "polars":
            if self.id_col:
                duplicates = self._df.filter(
                    self._df[[self.id_col, self.time_col]].is_duplicated()
                )
            else:
                duplicates = self._df.filter(self._df[self.time_col].is_duplicated())
        elif self._backend in ["pandas", "modin"]:
            if self.id_col:
                duplicates = self._df.duplicated(subset=[self.id_col, self.time_col])
            else:
                duplicates = self._df.duplicated(subset=[self.time_col])

        if len(duplicates) > 0 if self._backend == "polars" else duplicates.any():
            raise ValueError("Duplicate time entries found within the same group.")

    def _apply_available_mask(self):
        """Generate an available mask column if not present."""
        if "available_mask" not in self._df.columns:
            if self._backend == "polars":
                self._df = self._df.with_columns(pl.lit(1.0).alias("available_mask"))
            else:  # for pandas and modin
                self._df["available_mask"] = 1.0
            warnings.warn(
                "No available_mask column found, assuming all data is available."
            )

    def _rename_target_column(self):
        """Rename the target column to 'y' if rename_target is True."""
        if self._rename_target:
            if self._backend == "polars":
                self._df = self._df.rename({self.target_col: "y"})
            elif self._backend in ["pandas", "modin"]:
                self._df.rename(columns={self.target_col: "y"}, inplace=True)

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

    def get_data(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """Return the DataFrame in its current state.

        :return: The DataFrame managed by the TimeFrame instance.
        :rtype: Union[pl.DataFrame, pd.DataFrame]
        """
        return self._df

    def get_grouped_data(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """Return the grouped DataFrame if an ID column is provided.

        :return: Grouped DataFrame by the ID column if it is set, otherwise returns the original DataFrame.
        :rtype: Union[pl.DataFrame, pd.DataFrame]
        """
        if not self.id_col:
            raise ValueError("ID column is not set; cannot group data.")

        if self._backend == self._cfg["BACKENDS"]["pl"]:
            return self._df.groupby(self.id_col).agg(pl.all())
        elif self._backend == self._cfg["BACKENDS"]["pd"]:
            return self._df.groupby(self.id_col).apply(lambda x: x)

        return self._df
