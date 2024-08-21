"""temporalscope/timeseriesdata.py

This module implements the TimeSeriesData class, designed to provide a flexible and scalable interface for handling
time series data using multiple backends. It supports Polars as the default backend and Pandas as a secondary option
for users who prefer the traditional data processing framework.

Design Considerations:
1. Backend Flexibility: Users can switch between Polars and Pandas depending on their dataset and computational requirements.
2. Optional Grouping: Grouping by an ID column (e.g., stock ID, item ID) is optional. The class ensures no duplicate time entries within groups if grouping is applied.
3. Partitioning and Algorithmic Flexibility: Facilitates the use of partitioning schemes (e.g., sliding windows) and methods like SHAP for feature importance analysis, with the ability to easily extend with custom methods.
"""

from typing import Union, Optional, Callable
import polars as pl
import pandas as pd
import warnings


class TimeSeriesData:
    """
    Handles time series data with support for various backends like Polars and Pandas.

    This class provides functionalities to manage time series data with optional static features,
    available masks, and backend flexibility. It also supports partitioning schemes, feature
    importance methods, and can handle large datasets efficiently.

    :param df: The input DataFrame.
    :type df: Union[pl.DataFrame, pd.DataFrame]
    :param time_col: The column representing time in the DataFrame.
    :type time_col: str
    :param id_col: Optional. The column representing the ID for grouping. Default is None.
    :type id_col: Optional[str]
    :param static_cols: Optional. List of columns representing static features. Default is None.
    :type static_cols: Optional[list[str]]
    :param backend: The backend to use ('polars' or 'pandas'). Default is 'polars'.
    :type backend: str
    """

    def __init__(
        self,
        df: Union[pl.DataFrame, pd.DataFrame],
        time_col: str,
        id_col: Optional[str] = None,
        static_cols: Optional[list[str]] = None,
        backend: str = "polars",
    ):
        self.df = df
        self.time_col = time_col
        self.id_col = id_col
        self.static_cols = static_cols
        self.backend = backend.lower()
        self._validate_input()
        self._apply_backend_logic()
        self._apply_static_features()
        self._apply_available_mask()

    def _validate_input(self) -> None:
        """Validate the input DataFrame and ensure required columns are present."""
        if self.backend == "polars":
            assert isinstance(self.df, pl.DataFrame), "Expected a Polars DataFrame"
        elif self.backend == "pandas":
            assert isinstance(self.df, pd.DataFrame), "Expected a Pandas DataFrame"
        else:
            raise ValueError("Unsupported backend. Use 'polars' or 'pandas'.")

        if self.time_col not in self.df.columns:
            raise ValueError(f"{self.time_col} must be in the DataFrame columns")

        if self.id_col and self.id_col not in self.df.columns:
            raise ValueError(
                f"{self.id_col} must be in the DataFrame columns if provided"
            )

    def _apply_backend_logic(self):
        """Apply backend-specific logic such as sorting and grouping."""
        if self.backend == "polars":
            self._polars_logic()
        elif self.backend == "pandas":
            self._pandas_logic()

    def _polars_logic(self):
        """Apply Polars-specific logic like sorting and handling duplicates."""
        self.df = self.df.sort(
            by=[self.id_col, self.time_col] if self.id_col else [self.time_col]
        )
        self._check_duplicates()

    def _pandas_logic(self):
        """Apply Pandas-specific logic like sorting and handling duplicates."""
        self.df = self.df.sort_values(
            by=[self.id_col, self.time_col] if self.id_col else [self.time_col]
        )
        self._check_duplicates()

    def _check_duplicates(self):
        """Check for duplicate time entries within groups."""
        if self.id_col:
            duplicates = self.df.duplicated(subset=[self.id_col, self.time_col])
        else:
            duplicates = self.df.duplicated(subset=[self.time_col])

        if duplicates.any():
            raise ValueError("Duplicate time entries found within the same group.")

    def _apply_static_features(self):
        """Process static features if provided."""
        if self.static_cols:
            if not all(col in self.df.columns for col in self.static_cols):
                raise ValueError("Some static columns are not found in the DataFrame")
            # Additional logic for processing static features could go here

    def _apply_available_mask(self):
        """Generate an available mask column if not present."""
        if "available_mask" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.lit(1.0).alias("available_mask")
                if self.backend == "polars"
                else self.df.assign(available_mask=1.0)
            )
            warnings.warn(
                "No available_mask column found, assuming all data is available."
            )

    def groupby(self):
        """Group the DataFrame by the ID column if provided."""
        if self.id_col:
            return (
                self.df.groupby(self.id_col)
                if self.backend == "polars"
                else self.df.groupby(self.id_col)
            )
        return self.df

    def run_method(self, method: Callable, *args, **kwargs):
        """
        Run an analytical method on the data.

        :param method: The method to run.
        :type method: Callable
        :param args: Additional arguments to pass to the method.
        :param kwargs: Additional keyword arguments to pass to the method.
        :return: The result of the analytical method.
        :rtype: Any
        """
        return method(self, *args, **kwargs)
