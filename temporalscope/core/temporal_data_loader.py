"""temporalscope/temporal_data_loader.py

This module implements the TimeFrame class, designed to provide a flexible and scalable interface for handling
time series data using multiple backends. It supports Polars as the default backend and Pandas as a secondary option
for users who prefer the traditional data processing framework.

Design Considerations:
1. Backend Flexibility: Users can switch between Polars and Pandas depending on their dataset and computational requirements.
2. Optional Grouping: Grouping by an ID column (e.g., stock ID, item ID) is optional. The class ensures no duplicate time entries within groups if grouping is applied.
3. Partitioning and Algorithmic Flexibility: Facilitates the use of partitioning schemes (e.g., sliding windows) and methods like SHAP for feature importance analysis, with the ability to easily extend with custom methods.
"""

from typing import Union, Optional, Callable, List, Tuple
import polars as pl
import pandas as pd
import warnings

class TimeFrame:
    """ Handles time series data with support for various backends like Polars and Pandas.

    This class provides functionalities to manage time series data with optional static features,
    available masks, and backend flexibility. It also supports partitioning schemes, feature
    importance methods, and can handle large datasets efficiently.

    :param df: The input DataFrame.
    :type df: Union[pl.DataFrame, pd.DataFrame]
    :param time_col: The column representing time in the DataFrame.
    :type time_col: str
    :param target_col: The column representing the target variable in the DataFrame.
    :type target_col: str
    :param id_col: Optional. The column representing the ID for grouping. Default is None.
    :type id_col: Optional[str]
    :param static_cols: Optional. List of columns representing static features. Default is None.
    :type static_cols: Optional[list[str]]
    :param backend: The backend to use ('polars' or 'pandas'). Default is 'polars'.
    :type backend: str
    :param sort: Optional. Whether to sort the data by time_col (and id_col if provided). Default is True.
    :type sort: bool
    :param rename_target: Optional. Whether to rename the target_col to 'y'. Default is False.
    :type rename_target: bool
    """

    def __init__(self, df, time_col, target_col, id_col=None, static_cols=None, backend="polars", sort=True, rename_target=False):
        self._df = df
        self._time_col = time_col
        self._target_col = target_col
        self._id_col = id_col
        self._static_cols = static_cols
        self._backend = backend.lower()
        self._sort = sort
        self._rename_target = rename_target
        self._validate_input()
        self._apply_backend_logic()
        self._apply_static_features()
        self._rename_target_column()

    @property
    def backend(self) -> str:
        """Return the backend used ('polars' or 'pandas')."""
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

    @property
    def static_cols(self) -> Optional[list[str]]:
        """Return the list of static columns if provided."""
        return self._static_cols

    def _validate_input(self) -> None:
        """Validate the input DataFrame and ensure required columns are present."""
        if self.backend == "polars":
            if not isinstance(self._df, pl.DataFrame):
                raise TypeError("Expected a Polars DataFrame")
        elif self.backend == "pandas":
            if not isinstance(self._df, pd.DataFrame):
                raise TypeError("Expected a Pandas DataFrame")
        else:
            raise ValueError("Unsupported backend. Use 'polars' or 'pandas'.")

        if self.time_col not in self._df.columns:
            raise ValueError(f"Time column '{self.time_col}' must be in the DataFrame columns")

        if self.target_col not in self._df.columns:
            raise ValueError(f"Target column '{self.target_col}' must be in the DataFrame columns")

        if self.id_col and self.id_col not in self._df.columns:
            raise ValueError(
                f"ID column '{self.id_col}' must be in the DataFrame columns if provided"
            )

        # Ensure time_col is datetime for proper sorting
        if self.backend == "pandas":
            if not pd.api.types.is_datetime64_any_dtype(self._df[self.time_col]):
                self._df[self.time_col] = pd.to_datetime(self._df[self.time_col])
        elif self.backend == "polars":
            if self._df[self.time_col].dtype != pl.Datetime:
                self._df = self._df.with_columns(pl.col(self.time_col).str.strptime(pl.Datetime))

    def _apply_backend_logic(self):
        """Apply backend-specific logic such as sorting and grouping."""
        if self.backend == "polars":
            self._polars_logic()
        elif self.backend == "pandas":
            self._pandas_logic()

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

    def _check_duplicates(self):
        """Check for duplicate time entries within groups."""
        if self.backend == "polars":
            if self.id_col:
                duplicates = self._df.filter(
                    self._df[[self.id_col, self.time_col]].is_duplicated()
                )
            else:
                duplicates = self._df.filter(
                    self._df[self.time_col].is_duplicated()
                )
        elif self.backend == "pandas":
            if self.id_col:
                duplicates = self._df.duplicated(subset=[self.id_col, self.time_col])
            else:
                duplicates = self._df.duplicated(subset=[self.time_col])

        if len(duplicates) > 0 if self.backend == "polars" else duplicates.any():
            raise ValueError("Duplicate time entries found within the same group.")

    def _apply_static_features(self):
        """Process static features if provided."""
        if self.static_cols:
            if not all(col in self._df.columns for col in self.static_cols):
                raise ValueError("Some static columns are not found in the DataFrame")
            # Additional logic for processing static features could go here

    def _apply_available_mask(self):
        """Generate an available mask column if not present."""
        if "available_mask" not in self._df.columns:
            if self.backend == "polars":
                self._df = self._df.with_columns(pl.lit(1.0).alias("available_mask"))
            else:
                self._df["available_mask"] = 1.0
            warnings.warn(
                "No available_mask column found, assuming all data is available."
            )

    def _rename_target_column(self):
        """Rename the target column to 'y' if rename_target is True."""
        if self._rename_target:
            if self.backend == "polars":
                self._df = self._df.rename({self.target_col: 'y'})
            elif self.backend == "pandas":
                self._df.rename(columns={self.target_col: 'y'}, inplace=True)

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

        if self.backend == "polars":
            return self._df.groupby(self.id_col).agg(pl.all())
        elif self.backend == "pandas":
            return self._df.groupby(self.id_col).apply(lambda x: x)

        return self._df
