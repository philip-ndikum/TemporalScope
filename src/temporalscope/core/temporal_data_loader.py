"""TemporalScope/src/temporalscope/core/temporal_data_loader.py

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
from temporalscope.core.core_utils import (
    validate_and_convert_input,
    validate_backend,
    get_default_backend_cfg,
)


class TimeFrame:
    """Central class for the TemporalScope package, designed to manage time series data
    across various backends such as Polars, Pandas, and Modin. This class enables
    modular and flexible workflows for machine learning, deep learning, and time
    series explainability (XAI) methods like temporal SHAP.

    The `TimeFrame` class supports workflows where the target variable can be either 1D scalar data,
    typical in classical machine learning, or 3D tensor data, more common in deep learning contexts.
    It is an essential component for temporal data analysis, including but not limited to explainability pipelines
    like Temporal SHAP and concept drift analysis.

    Designed to be the core data handler in a variety of temporal analysis scenarios, the `TimeFrame` class
    integrates seamlessly with other TemporalScope modules and can be extended for more advanced use cases.

    :param df: The input DataFrame.
    :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    :param time_col: The column representing time in the DataFrame.
    :type time_col: str
    :param target_col: The column representing the target variable in the DataFrame.
    :type target_col: str
    :param id_col: Optional. The column representing the ID for grouping. Default is None.
    :type id_col: Optional[str]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, or 'mpd' for Modin). Default is 'pl'.
    :type backend: str
    :param sort: Optional. Sort the data by `time_col` (and `id_col` if provided) in ascending order. Default is True.
    :type sort: bool

    .. note::
       The `TimeFrame` class is designed for workflows where the target label has already been generated.
       If your workflow requires generating the target label, consider using the `TemporalTargetShifter` class
       from the `TemporalScope` package to shift the target variable appropriately for tasks like forecasting.

    Example Usage:
    --------------

    .. code-block:: python

       # Example of creating a TimeFrame with a Polars DataFrame
       data = pl.DataFrame({
           'time': pl.date_range(start='2021-01-01', periods=100, interval='1d'),
           'value': range(100)
       })
       tf = TimeFrame(data, time_col='time', target_col='value')

       # Accessing the data
       print(tf.get_data().head())

       # Example of creating a TimeFrame with a Modin DataFrame
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
        self._time_col = time_col
        self._target_col = target_col
        self._id_col = id_col
        self._sort = sort

        # Convert, validate, and set up the DataFrame
        self.df = self._setup_timeframe(df)

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

    def _validate_columns(
        self, df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    ) -> None:
        """Validate the presence and types of required columns in the DataFrame.

        :param df: The DataFrame to validate.
        :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        :raises ValueError: If required columns are missing.
        """
        required_columns = [self.time_col, self._target_col] + (
            [self.id_col] if self.id_col else []
        )
        missing_columns = [
            col for col in required_columns if col and col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def _sort_data(
        self, df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    ) -> Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]:
        """Internal method to sort the DataFrame based on the backend.

        :param df: The DataFrame to sort.
        :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        :return: The sorted DataFrame.
        :rtype: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        """
        sort_key = [self.id_col, self.time_col] if self.id_col else [self.time_col]

        # Polars backend sorting
        if self._backend == "pl":
            if isinstance(df, pl.DataFrame):
                return df.sort(sort_key)
            else:
                raise TypeError("Expected a Polars DataFrame for the Polars backend.")

        # Pandas or Modin backend sorting
        elif self._backend in ["pd", "mpd"]:
            if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
                return df.sort_values(by=sort_key)
            else:
                raise TypeError(
                    "Expected a Pandas or Modin DataFrame for the Pandas or Modin backend."
                )

        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def _setup_timeframe(
        self, df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    ) -> Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]:
        """Sets up the TimeFrame object by converting, validating, and preparing data as required.

        :param df: The input DataFrame to be processed.
        :type df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        :return: The processed DataFrame.
        :rtype: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        """
        # Convert and validate the input DataFrame
        df = validate_and_convert_input(df, self._backend)

        # Validate the presence of required columns
        self._validate_columns(df)

        # Sort data if required
        if self._sort:
            df = self._sort_data(df)

        return df

    def sort_data(self, ascending: bool = True) -> None:
        """Public method to sort the DataFrame by the time column (and ID column if present).

        :param ascending: If True, sort in ascending order; if False, sort in descending order.
        :type ascending: bool
        """
        self._sort_data(ascending)

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
                duplicates = self.df.filter(combined_expr)  # type: ignore
            else:
                # Only check the time column for duplicates
                duplicates = self.df.filter(pl.col(self._time_col).is_duplicated())  # type: ignore
            # Check for duplicates by inspecting the number of rows
            if duplicates.height > 0:
                raise ValueError("Duplicate time entries found within the same group.")
        elif self._backend in ["pd", "mpd"]:
            # Cast to Pandas DataFrame for Pandas/Modin specific check
            pandas_df = cast(pd.DataFrame, self.df)
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
        return self.df

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
            if isinstance(self.df, pl.DataFrame):
                return self.df.group_by(self.id_col).agg(
                    pl.all()
                )  # Polars uses `group_by`
            else:
                raise TypeError(f"Expected Polars DataFrame but got {type(self.df)}.")
        elif self._backend == "pd":
            # Pandas specific groupby
            if isinstance(self.df, pd.DataFrame):
                return self.df.groupby(self.id_col).apply(lambda x: x)
            else:
                raise TypeError(f"Expected Pandas DataFrame but got {type(self.df)}.")
        elif self._backend == "mpd":
            # Modin uses the same API as Pandas for this operation
            if isinstance(self.df, mpd.DataFrame):
                return self.df.groupby(self.id_col).apply(lambda x: x)
            else:
                raise TypeError(f"Expected Modin DataFrame but got {type(self.df)}.")
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def update_data(
        self, new_df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
    ) -> None:
        """Updates the internal DataFrame with the provided new DataFrame.

        :param new_df: The new DataFrame to replace the existing one.
        :type new_df: Union[pl.DataFrame, pd.DataFrame, mpd.DataFrame]
        """
        self.df = new_df

    def update_target_col(
        self, new_target_col: Union[pl.Series, pd.Series, mpd.Series]
    ) -> None:
        """Updates the target column in the internal DataFrame with the provided new target column.

        :param new_target_col: The new target column to replace the existing one.
        :type new_target_col: Union{pl.Series, pd.Series, mpd.Series}
        """
        if self._backend == "pl":
            if isinstance(self.df, pl.DataFrame):
                self.df = self.df.with_columns([new_target_col.alias(self._target_col)])
            else:
                raise TypeError("Expected Polars DataFrame for Polars backend.")
        elif self._backend in ["pd", "mpd"]:
            if isinstance(self.df, (pd.DataFrame, mpd.DataFrame)):
                self.df[self._target_col] = new_target_col
            else:
                raise TypeError(
                    "Expected Pandas or Modin DataFrame for respective backend."
                )
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")
