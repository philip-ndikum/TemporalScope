"""TemporalScope/src/temporalscope/core/temporal_target_shifter.py.

This module provides a transformer-like class to shift the target variable in time series data, either
to a scalar value (for classical machine learning) or to an array (for deep learning).
It is designed to work with the TimeFrame class, supporting multiple backends.

.. seealso::

    1. Torres, J.F., Hadjout, D., Sebaa, A., Martínez-Álvarez, F., & Troncoso, A. (2021). Deep learning for time series forecasting: a survey. Big Data, 9(1), 3-21. https://doi.org/10.1089/big.2020.0159
    2. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209
    3. Tang, Y., Song, Z., Zhu, Y., Yuan, H., Hou, M., Ji, J., Tang, C., & Li, J. (2022). A survey on machine learning models for financial time series forecasting. Neurocomputing, 512, 363-380. https://doi.org/10.1016/j.neucom.2022.09.078

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

import warnings
from typing import Optional, Union

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
    SupportedBackendDataFrame,
    validate_backend,
)
from temporalscope.core.temporal_data_loader import TimeFrame


class TemporalTargetShifter:
    """A class for shifting the target variable in time series data for machine learning or deep learning.

    This class works with the `TimeFrame` and partitioned datasets (e.g., from `SlidingWindowPartitioner`)
    to shift the target variable by a specified number of lags (time steps). It supports multiple backends
    (Polars, Pandas, Modin) and can generate output suitable for both machine learning models (scalar)
    and deep learning models (sequence).

    Assumptions:
    ------------
    1. The class applies time shifting globally, without grouping by entities (e.g., tickers or SKUs).
       Users should handle any entity-specific grouping outside of this class.
    2. The time shifting is applied to the target column, which may have varying data structures
       depending on the backend (Polars, Pandas, Modin).

    Examples
    --------
    **Using `TimeFrame`:**

    .. code-block:: python

        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.core.temporal_target_shifter import TemporalTargetShifter

        # Create a sample Pandas DataFrame
        data = {
            'time': pd.date_range(start='2022-01-01', periods=100),
            'target': np.random.rand(100),
            'feature_1': np.random.rand(100)
        }
        df = pd.DataFrame(data)

        # Create a TimeFrame object
        tf = TimeFrame(df, time_col="time", target_col="target", backend="pd")

        # Apply target shifting
        shifter = TemporalTargetShifter(n_lags=1, target_col="target")
        shifted_df = shifter.fit_transform(tf)

    **Using `SlidingWindowPartitioner`:**

    .. code-block:: python

        from temporalscope.partition.sliding_window import SlidingWindowPartitioner
        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.core.temporal_target_shifter import TemporalTargetShifter

        # Create a sample TimeFrame
        tf = TimeFrame(df, time_col="time", target_col="target", backend="pd")

        # Create a SlidingWindowPartitioner
        partitioner = SlidingWindowPartitioner(tf=tf, window_size=10, stride=1)

        # Apply TemporalTargetShifter on each partition
        shifter = TemporalTargetShifter(n_lags=1, target_col="target")
        for partition in partitioner.fit_transform():
            shifted_partition = shifter.fit_transform(partition)

    :param n_lags: Number of lags (time steps) to shift the target variable. Default is 1.
    :type n_lags: int
    :param mode: Mode of operation: "machine_learning" for scalar or "deep_learning" for sequences.
                 Default is "machine_learning".
    :type mode: str
    :param sequence_length: (Deep Learning Mode Only) The length of the input sequences. Required if mode is "deep_learning".
    :type sequence_length: Optional[int]
    :param target_col: The column representing the target variable (mandatory).
    :type target_col: str
    :param drop_target: Whether to drop the original target column after shifting. Default is True.
    :type drop_target: bool
    :param verbose: If True, prints information about the number of dropped rows during transformation.
    :type verbose: bool
    :raises ValueError: If the backend is unsupported or if validation checks fail.

    """

    MODE_MACHINE_LEARNING = "machine_learning"
    MODE_DEEP_LEARNING = "deep_learning"

    def __init__(
        self,
        n_lags: int = 1,
        mode: str = MODE_MACHINE_LEARNING,
        sequence_length: Optional[int] = None,
        target_col: Optional[str] = None,
        drop_target: bool = True,
        verbose: bool = False,
    ):
        """Initialize the TemporalTargetShifter.

        :param n_lags: Number of lags (time steps) to shift the target variable. Default is 1.
        :param mode: Mode of operation: "machine_learning" or "deep_learning". Default is "machine_learning".
        :param sequence_length: (Deep Learning Mode Only) Length of the input sequences. Required if mode is
            "deep_learning".
        :param target_col: Column representing the target variable (mandatory).
        :param drop_target: Whether to drop the original target column after shifting. Default is True.
        :param verbose: Whether to print detailed information about transformations.
        :raises ValueError: If the target column is not provided or if an invalid mode is selected.
        """
        if mode not in [self.MODE_MACHINE_LEARNING, self.MODE_DEEP_LEARNING]:
            raise ValueError(f"`mode` must be '{self.MODE_MACHINE_LEARNING}' or '{self.MODE_DEEP_LEARNING}'.")

        if target_col is None:
            raise ValueError("`target_col` must be explicitly provided for TemporalTargetShifter.")

        if n_lags <= 0:
            raise ValueError("`n_lags` must be greater than 0.")

        self.n_lags = n_lags
        self.mode = mode
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.drop_target = drop_target
        self.verbose = verbose
        self.backend: Optional[str] = None  # Backend will be set during fit

        if self.mode == self.MODE_DEEP_LEARNING and not self.sequence_length:
            raise ValueError("`sequence_length` must be provided when mode is 'deep_learning'.")

    def _infer_backend(self, df: SupportedBackendDataFrame) -> str:
        """Infer the backend from the DataFrame type.

        :param df: The input DataFrame.
        :type df: SupportedBackendDataFrame
        :return: The inferred backend ('pl', 'pd', or 'mpd').
        :raises ValueError: If the DataFrame type is unsupported.
        """
        if isinstance(df, pl.DataFrame):
            return BACKEND_POLARS
        elif isinstance(df, pd.DataFrame):
            return BACKEND_PANDAS
        elif isinstance(df, mpd.DataFrame):
            return BACKEND_MODIN
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df)}")

    def _set_backend(self, df: SupportedBackendDataFrame) -> None:
        """Set or infer the backend based on the DataFrame.

        :param df: The input DataFrame.
        :type df: SupportedBackendDataFrame
        :raises ValueError: If the backend is not supported.
        """
        if self.backend is None:
            self.backend = self._infer_backend(df)
        validate_backend(self.backend)

    def _validate_data(self, tf: SupportedBackendDataFrame) -> None:
        """Validate the TimeFrame or partitioned data for consistency.

        :param tf: The `TimeFrame` object or a DataFrame (`pandas`, `modin`, or `polars`) that contains the time series data.
        :type tf: SupportedBackendDataFrame
        :raises ValueError: If the data is invalid or empty.
        """
        if isinstance(tf, TimeFrame):
            df = tf.get_data()
        else:
            df = tf

        # Check if the DataFrame is empty based on the backend
        if isinstance(df, (pd.DataFrame, mpd.DataFrame)):  # Merge the `isinstance` calls for `pd` and `mpd`
            if df is None or df.empty:
                raise ValueError("Input DataFrame is empty.")
        elif isinstance(df, pl.DataFrame):
            if df is None or df.is_empty():
                raise ValueError("Input DataFrame is empty.")
        else:
            raise ValueError("Unsupported DataFrame type.")

    def _shift_polars(self, df: pl.DataFrame, target_col: str) -> pl.DataFrame:
        """Shift the target variable in a Polars DataFrame.

        :param df: The Polars DataFrame containing the time series data.
        :type df: pl.DataFrame
        :param target_col: The column representing the target variable.
        :type target_col: str
        :return: The Polars DataFrame with the shifted target variable.
        :rtype: pl.DataFrame
        :raises ValueError: If `sequence_length` or `n_lags` are not properly set.
        """
        if self.mode == self.MODE_DEEP_LEARNING:
            if not isinstance(self.sequence_length, int):
                raise ValueError("`sequence_length` must be an integer.")
            shifted_columns = [
                df[target_col].shift(-i).alias(f"{target_col}_shift_{i}") for i in range(self.sequence_length)
            ]
            df = df.with_columns(shifted_columns)
            df = df.with_columns(
                pl.concat_list([pl.col(f"{target_col}_shift_{i}") for i in range(self.sequence_length)]).alias(
                    f"{target_col}_sequence"
                )
            )
            df = df.drop([f"{target_col}_shift_{i}" for i in range(self.sequence_length)])
            df = df.drop_nulls()
            df = df.slice(0, len(df) - self.sequence_length + 1)
        else:
            df = df.with_columns(df[target_col].shift(-self.n_lags).alias(f"{target_col}_shift_{self.n_lags}"))
            df = df.drop_nulls()

        if df.is_empty():
            raise ValueError("DataFrame is empty after shifting operation.")

        if self.drop_target:
            df = df.drop(target_col)

        return df

    def _shift_pandas_modin(
        self, df: Union[pd.DataFrame, mpd.DataFrame], target_col: str
    ) -> Union[pd.DataFrame, mpd.DataFrame]:
        """Shift the target variable in a Pandas or Modin DataFrame.

        :param df: The Pandas or Modin DataFrame containing the time series data.
        :type df: Union[pd.DataFrame, mpd.DataFrame]
        :param target_col: The column representing the target variable.
        :type target_col: str
        :return: The DataFrame with the shifted target variable.
        :rtype: Union[pd.DataFrame, mpd.DataFrame]
        :raises ValueError: If `sequence_length` or `n_lags` are not properly set.
        """
        if self.mode == self.MODE_DEEP_LEARNING:
            if not isinstance(self.sequence_length, int):
                raise ValueError("`sequence_length` must be an integer.")
            shifted_columns = [df[target_col].shift(-i) for i in range(self.sequence_length)]
            df[f"{target_col}_sequence"] = list(zip(*shifted_columns))
            df = df.dropna()
            df = df.iloc[: -self.sequence_length + 1]
        else:
            df[f"{target_col}_shift_{self.n_lags}"] = df[target_col].shift(-self.n_lags)
            df = df.dropna()

        if df.empty:
            raise ValueError("DataFrame is empty after shifting operation.")

        if self.drop_target:
            df = df.drop(columns=[target_col])

        return df

    def _transform_pandas_modin(self, df: Union[pd.DataFrame, mpd.DataFrame]) -> Union[pd.DataFrame, mpd.DataFrame]:
        """Handle shifting for Pandas or Modin backends.

        :param df: The input DataFrame (Pandas or Modin).
        :type df: Union[pd.DataFrame, mpd.DataFrame]
        :return: The transformed DataFrame with the target column shifted.
        :rtype: Union[pd.DataFrame, mpd.DataFrame]
        :raises ValueError: If `target_col` is not set.
        """
        # Ensure target_col is not None
        if self.target_col is None:
            raise ValueError("`target_col` must be set before transformation.")

        df = self._shift_pandas_modin(df, self.target_col)

        rows_before = len(df)
        df = df.dropna()  # Handle missing values
        rows_after = len(df)

        if rows_after == 0:
            raise ValueError("All rows were dropped during transformation.")

        self._print_dropped_rows(rows_before, rows_after)
        return df

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle shifting for Polars backend.

        :param df: The input Polars DataFrame.
        :type df: pl.DataFrame
        :return: The transformed Polars DataFrame with the target column shifted.
        :rtype: pl.DataFrame
        :raises ValueError: If `target_col` is not set.
        """
        # Ensure target_col is not None
        if self.target_col is None:
            raise ValueError("`target_col` must be set before transformation.")

        df = self._shift_polars(df, self.target_col)

        rows_before = df.shape[0]
        df = df.drop_nulls()
        rows_after = df.shape[0]

        if rows_after == 0:
            raise ValueError("All rows were dropped during transformation.")

        self._print_dropped_rows(rows_before, rows_after)
        return df

    def _print_dropped_rows(self, rows_before: int, rows_after: int) -> None:
        """Print information about dropped rows if verbose mode is enabled.

        :param rows_before: Number of rows before dropping nulls.
        :type rows_before: int
        :param rows_after: Number of rows after dropping nulls.
        :type rows_after: int
        """
        if self.verbose:
            rows_dropped = rows_before - rows_after
            print(f"Rows before shift: {rows_before}; Rows after shift: {rows_after}; Rows dropped: {rows_dropped}")

    def fit(self, tf: SupportedBackendDataFrame) -> "TemporalTargetShifter":
        """Validate and prepare the target data for transformation based on the specified backend.

        The `fit` method initializes the backend and validates the input data, ensuring the target column is consistent with the input data.
        It does not alter the data but sets up the necessary configuration for later transformations.

        :param tf: The `TimeFrame` object, or a DataFrame (`pandas`, `modin`, or `polars`) that contains the time series data.
                   The DataFrame must have a target column defined or the `target_col` attribute set during initialization.
        :type tf: SupportedBackendDataFrame, optional
        :raises ValueError: If the target column is not provided, the data is invalid, or the backend is unsupported.
        :raises Warning: If the target column provided in `TemporalTargetShifter` differs from the one in the `TimeFrame`.
        :return: The fitted `TemporalTargetShifter` instance, ready for transforming the data.
        :rtype: TemporalTargetShifter

        Example Usage:
        --------------
        .. code-block:: python

            shifter = TemporalTargetShifter(n_lags=2, target_col="target")
            shifter.fit(time_frame)
        """
        self._validate_data(tf)

        if isinstance(tf, TimeFrame):
            # Set backend and handle target column for TimeFrame input
            self.backend = tf.backend
            if not self.target_col:
                self.target_col = tf._target_col
            elif self.target_col != tf._target_col:
                warnings.warn(
                    f"The `target_col` in TemporalTargetShifter ('{self.target_col}') "
                    f"differs from the TimeFrame's target_col ('{tf._target_col}').",
                    UserWarning,
                )
        elif tf is not None:
            # Infer backend for non-TimeFrame input
            self.backend = self._infer_backend(tf)
        else:
            raise ValueError("Input data is None.")

        return self

    def transform(self, tf: SupportedBackendDataFrame) -> SupportedBackendDataFrame:
        """Transform the input time series data by shifting the target variable according to the specified number of lags.

        The `transform` method shifts the target variable in the input data according to the `n_lags` or `sequence_length` set during initialization.
        This method works directly on either a `TimeFrame` or a raw DataFrame (Pandas, Modin, or Polars), applying the appropriate backend-specific transformation.

        :param tf: The `TimeFrame` object or a DataFrame (Pandas, Modin, or Polars) that contains the time series data to be transformed.
                   The data should contain a target column that will be shifted.
        :type tf: SupportedBackendDataFrame, optional
        :raises ValueError: If the input data is invalid, unsupported, or lacks columns.
        :raises ValueError: If the backend is unsupported or data validation fails.
        :return: A transformed DataFrame or `TimeFrame` with the target variable shifted by the specified lags or sequence length.
                 If a `TimeFrame` is provided, the returned object will be a `TimeFrame`. Otherwise, a DataFrame will be returned.
        :rtype: SupportedBackendDataFrame

        Example Usage:
        --------------
        .. code-block:: python

            shifter = TemporalTargetShifter(n_lags=2, target_col="target")
            transformed_data = shifter.transform(time_frame)
        """
        if isinstance(tf, TimeFrame):
            tf.sort_data()  # Ensure the data is sorted before shifting
            df = tf.get_data()
            if not self.target_col:
                self.target_col = tf._target_col
            self.backend = tf.backend
        elif tf is not None:
            df = tf
            if not self.target_col:
                if hasattr(df, "columns"):
                    self.target_col = df.columns[-1]
                else:
                    raise ValueError("The input DataFrame does not have columns.")
            self._set_backend(df)
        else:
            raise ValueError("Input data is None.")

        # Delegate the transformation to backend-specific methods
        if self.backend == BACKEND_PANDAS or self.backend == BACKEND_MODIN:
            transformed_df = self._transform_pandas_modin(df)
        elif self.backend == BACKEND_POLARS:
            transformed_df = self._transform_polars(df)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # If the input was a TimeFrame, return a transformed TimeFrame
        if isinstance(tf, TimeFrame):
            return TimeFrame(
                transformed_df,
                time_col=tf.time_col,
                target_col=(
                    f"{self.target_col}_shift_{self.n_lags}"
                    if self.mode == self.MODE_MACHINE_LEARNING
                    else f"{self.target_col}_sequence"
                ),
                backend=self.backend,
            )

        return transformed_df

    def fit_transform(self, tf: SupportedBackendDataFrame) -> SupportedBackendDataFrame:
        """Fit and transform the input data in a single step.

        This method combines the functionality of the `fit` and `transform` methods. It first validates and prepares the input data (fitting),
        then applies the target variable shifting (transformation) based on the `n_lags` or `sequence_length` specified during initialization.

        :param tf: The `TimeFrame` object or a DataFrame (`pandas`, `modin`, or `polars`) to be transformed.
                   The data should contain a target column that will be shifted according to the `n_lags` or `sequence_length`.
        :type tf: SupportedBackendDataFrame, optional
        :raises ValueError: If the input data is invalid or the backend is unsupported.
        :raises ValueError: If the target column is not set, or is incompatible with the data.
        :return: A transformed DataFrame or TimeFrame with the target variable shifted by the specified lags or sequence length.
        :rtype: SupportedBackendDataFrame

        Example Usage:
        --------------
        .. code-block:: python

            shifter = TemporalTargetShifter(n_lags=2, target_col="target")
            shifted_data = shifter.fit_transform(time_frame)
        """
        self.fit(tf)
        transformed = self.transform(tf)

        # Return TimeFrame if input was TimeFrame, otherwise return DataFrame
        if isinstance(tf, TimeFrame):
            return TimeFrame(
                transformed,
                time_col=tf.time_col,
                target_col=(
                    f"{self.target_col}_shift_{self.n_lags}"
                    if self.mode == self.MODE_MACHINE_LEARNING
                    else f"{self.target_col}_sequence"
                ),
                backend=self.backend,
            )
        return transformed
