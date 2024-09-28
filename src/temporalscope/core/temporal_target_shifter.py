# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""TemporalScope/src/temporalscope/core/temporal_target_shifter.py.

This module provides a transformer-like class to shift the target variable in time series data, either
to a scalar value (for classical machine learning) or to an array (for deep learning).
It is designed to work with the TimeFrame class, supporting multiple backends.

.. seealso::

    1. Torres, J.F., Hadjout, D., Sebaa, A., Martínez-Álvarez, F., & Troncoso, A. (2021). Deep learning for time series forecasting: a survey. Big Data, 9(1), 3-21. https://doi.org/10.1089/big.2020.0159
    2. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209
    3. Tang, Y., Song, Z., Zhu, Y., Yuan, H., Hou, M., Ji, J., Tang, C., & Li, J. (2022). A survey on machine learning models for financial time series forecasting. Neurocomputing, 512, 363-380. https://doi.org/10.1016/j.neucom.2022.09.078
"""

import warnings
from typing import Optional, Union, cast

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
    MODE_MACHINE_LEARNING,
    MODE_DEEP_LEARNING,
    SupportedBackendDataFrame,
    validate_backend,
)
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.core.temporal_data_loader import TimeFrameCompatibleData


class TemporalTargetShifter:
    """A class for shifting the target variable in time series data for machine learning or deep learning.

    This class works with `TimeFrame` objects or raw DataFrame types (Pandas, Modin, Polars) to shift the target variable
    by a specified number of lags (time steps). It supports multiple backends and can generate output suitable for
    machine learning models (scalar) or deep learning models (sequences).

    Design:
    -------
    The `TemporalTargetShifter` follows a strategy pattern, where the data format (backend) is either inferred from the
    input or set explicitly. This enables flexible support for different DataFrame libraries. The class ensures that
    input type consistency is maintained, and it returns the same data type that is provided. For instance, if the input
    is a `TimeFrame`, the output will be a `TimeFrame`. If a raw DataFrame is provided, the output will be a raw
    DataFrame of the same type.

    Assumptions:
    ------------
    1. Time shifting is applied globally, meaning the data is not grouped by entities (e.g., tickers or SKUs). Users
       should handle such grouping outside of this class.
    2. The time shifting is applied to a target column, which may have varying data structures depending on the backend
       (Polars, Pandas, Modin).

    :param target_col: The column representing the target variable (mandatory).
    :type target_col: str
    :param n_lags: Number of lags (time steps) to shift the target variable. Default is 1.
    :type n_lags: int
    :param mode: Mode of operation: "machine_learning" for scalar or "deep_learning" for sequences.
                 Default is "machine_learning".
    :type mode: str
    :param sequence_length: (Deep Learning Mode Only) The length of the input sequences. Required if mode is "deep_learning".
    :type sequence_length: Optional[int]
    :param drop_target: Whether to drop the original target column after shifting. Default is True.
    :type drop_target: bool
    :param verbose: If True, prints information about the number of dropped rows during transformation.
    :type verbose: bool
    :raises ValueError: If the backend is unsupported or if validation checks fail.

    Examples
    --------
    **Using TimeFrame:**

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
        shifter = TemporalTargetShifter(target_col="target", n_lags=1)
        shifted_df = shifter.fit_transform(tf)

    **Using SlidingWindowPartitioner:**

    .. code-block:: python

        from temporalscope.partition.sliding_window import SlidingWindowPartitioner
        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.core.temporal_target_shifter import TemporalTargetShifter

        # Create a sample TimeFrame
        tf = TimeFrame(df, time_col="time", target_col="target", backend="pd")

        # Create a SlidingWindowPartitioner
        partitioner = SlidingWindowPartitioner(tf=tf, window_size=10, stride=1)

        # Apply TemporalTargetShifter on each partition
        shifter = TemporalTargetShifter(target_col="target", n_lags=1)
        for partition in partitioner.fit_transform():
            shifted_partition = shifter.fit_transform(partition)
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        n_lags: int = 1,
        mode: str = MODE_MACHINE_LEARNING,
        sequence_length: Optional[int] = None,
        drop_target: bool = True,
        verbose: bool = False,
    ):
        """Initialize the TemporalTargetShifter.

        :param target_col: Column representing the target variable (mandatory).
        :param n_lags: Number of lags (time steps) to shift the target variable. Default is 1.
        :param mode: Mode of operation: "machine_learning" or "deep_learning". Default is "machine_learning".
        :param sequence_length: (Deep Learning Mode Only) Length of the input sequences. Required if mode is
            "deep_learning".
        :param drop_target: Whether to drop the original target column after shifting. Default is True.
        :param verbose: Whether to print detailed information about transformations.
        :raises ValueError: If the target column is not provided or if an invalid mode is selected.

        Note:
        The data_format is set to None during initialization and will be inferred in the fit() method based on
        the type of input data (TimeFrame or SupportedBackendDataFrame).
        """
        # Validate the mode (should be machine learning or deep learning)
        if mode not in [MODE_MACHINE_LEARNING, MODE_DEEP_LEARNING]:
            raise ValueError(f"`mode` must be '{MODE_MACHINE_LEARNING}' or '{MODE_DEEP_LEARNING}'.")

        # Ensure the target column is provided
        if target_col is None:
            raise ValueError("`target_col` must be explicitly provided for TemporalTargetShifter.")

        # Validate n_lags (should be greater than 0)
        if n_lags <= 0:
            raise ValueError("`n_lags` must be greater than 0.")

        # Handle deep learning mode, ensure sequence length is set
        if mode == MODE_DEEP_LEARNING and sequence_length is None:
            raise ValueError("`sequence_length` must be provided when mode is 'deep_learning'.")

        # Assign instance attributes
        self.target_col = target_col
        self.n_lags = n_lags
        self.mode = mode
        self.sequence_length = sequence_length
        self.drop_target = drop_target
        self.verbose = verbose

        # The data format will be inferred later during fit()
        self.data_format: Optional[str] = None

        # Print a verbose message if required
        if verbose:
            print(f"Initialized TemporalTargetShifter with target_col={target_col}, mode={mode}, n_lags={n_lags}")

    def _infer_data_format(self, df: SupportedBackendDataFrame) -> str:
        """Infer the backend from the DataFrame type.

        :param df: The input DataFrame.
        :type df: SupportedBackendDataFrame
        :return: The inferred backend ('BACKEND_POLARS', 'BACKEND_PANDAS', or 'BACKEND_MODIN').
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

    def _set_or_infer_data_format(self, tf: TimeFrameCompatibleData) -> None:
        """Set or infer the data format based on the input type.

        This method checks if the input is a TimeFrame and uses its data format.
        If the input is a raw DataFrame (Pandas, Modin, or Polars), it infers the data format.
        """
        if isinstance(tf, TimeFrame):
            self.data_format = tf.dataframe_backend
        else:
            # Infer the data format using the existing _infer_data_format method
            self.data_format = self._infer_data_format(tf)

        if self.data_format is None:
            raise ValueError("Data format could not be inferred or is not set.")

        validate_backend(self.data_format)

    def _validate_data(self, tf: TimeFrameCompatibleData) -> None:
        """Validate the TimeFrame or DataFrame input for consistency.

        This method ensures that the input data is valid and non-empty, regardless of whether it is a TimeFrame or a raw DataFrame.

        :param tf: The `TimeFrame` object or a raw DataFrame (Pandas, Modin, or Polars) to be validated.
        :type tf: TimeFrameCompatibleData
        :raises ValueError: If the input data is empty or invalid.
        """
        if isinstance(tf, TimeFrame):
            df = tf.get_data()
        else:
            df = tf

        # Check if the DataFrame is empty
        if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
            if df is None or df.empty:
                raise ValueError("Input DataFrame is empty.")
        elif isinstance(df, pl.DataFrame):
            if df.is_empty():
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
        if self.mode == MODE_DEEP_LEARNING:
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
        if self.mode == MODE_DEEP_LEARNING:
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

    def fit(self, tf: TimeFrameCompatibleData) -> "TemporalTargetShifter":
        """Validate and prepare the target data for transformation based on the inferred data format (backend).

        The `fit` method initializes the data format (whether it's a `TimeFrame` or a raw DataFrame) and validates the input data.
        It ensures the target column is consistent with the input data and sets the backend (`data_format`), which will be used
        in subsequent transformations.

        :param tf: The `TimeFrame` object or a raw DataFrame (`pandas`, `modin`, or `polars`) that contains the time series data.
                   The data should contain a target column that will be shifted.
        :type tf: TimeFrameCompatibleData
        :raises ValueError: If the target column is not provided, the data is invalid, or the backend format is unsupported.
        :raises Warning: If the target column provided in `TemporalTargetShifter` differs from the one in the `TimeFrame`.
        :return: The fitted `TemporalTargetShifter` instance, ready for transforming the data.
        :rtype: TemporalTargetShifter

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_target_shifter import TemporalTargetShifter
            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd
            import numpy as np

            # Create a sample Pandas DataFrame
            data = {
                'time': pd.date_range(start='2022-01-01', periods=100),
                'target': np.random.rand(100),
                'feature_1': np.random.rand(100)
            }
            df = pd.DataFrame(data)

            # Create a TimeFrame object
            tf = TimeFrame(df, time_col="time", target_col="target", backend="pd")

            # Create a TemporalTargetShifter instance
            shifter = TemporalTargetShifter(n_lags=2, target_col="target")

            # Fit the shifter to the TimeFrame
            shifter.fit(tf)
        """
        # Validate the input data (whether it's TimeFrame or DataFrame)
        self._validate_data(tf)

        # If input is a TimeFrame, set the backend using the @property method and manage the target column
        if isinstance(tf, TimeFrame):
            self.data_format = tf.dataframe_backend  # Using the @property to access the backend
            if not self.target_col:
                self.target_col = tf._target_col  # If target_col not set in the shifter, use TimeFrame's target_col
            elif self.target_col != tf._target_col:
                warnings.warn(
                    f"The `target_col` in TemporalTargetShifter ('{self.target_col}') differs from the TimeFrame's "
                    f"target_col ('{tf._target_col}').",
                    UserWarning,
                )
        # If input is a raw DataFrame (pandas, modin, or polars), infer the backend
        elif tf is not None:
            self.data_format = self._infer_data_format(tf)
        else:
            raise ValueError("Input data is None.")

        # Return the instance after fitting
        return self

    def transform(self, tf: TimeFrameCompatibleData) -> TimeFrameCompatibleData:
        """Transform the input time series data by shifting the target variable according to the specified number of lags.

        The `transform` method shifts the target variable in the input data according to the `n_lags` or `sequence_length`
        set during initialization. This method works directly on either a `TimeFrame` or a raw DataFrame (Pandas, Modin,
        or Polars), applying the appropriate backend-specific transformation.

        Design:
        -------
        The method returns the same type as the input: If a `TimeFrame` object is passed in, a `TimeFrame` object is returned.
        If a raw DataFrame (Pandas, Modin, or Polars) is passed in, the same type of DataFrame is returned. This ensures that
        the transformation remains consistent in pipeline workflows where the type of data object is important.

        :param tf: The `TimeFrame` object or a DataFrame (Pandas, Modin, or Polars) that contains the time series data
                   to be transformed. The data should contain a target column that will be shifted.
        :type tf: TimeFrameCompatibleData
        :raises ValueError: If the input data is invalid, unsupported, or lacks columns.
        :raises ValueError: If the backend is unsupported or data validation fails.
        :return: A transformed `TimeFrame` if the input was a `TimeFrame`, otherwise a DataFrame of the same type as the input.
        :rtype: TimeFrameCompatibleData

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_target_shifter import TemporalTargetShifter
            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a sample Pandas DataFrame
            data = {
                'time': pd.date_range(start='2022-01-01', periods=100),
                'target': np.random.rand(100),
                'feature_1': np.random.rand(100)
            }
            df = pd.DataFrame(data)

            # Create a TimeFrame object
            tf = TimeFrame(df, time_col="time", target_col="target", backend="pd")

            # Initialize TemporalTargetShifter
            shifter = TemporalTargetShifter(n_lags=2, target_col="target")

            # Fit the shifter and transform the data
            shifter.fit(tf)
            transformed_data = shifter.transform(tf)

        """
        # Handle TimeFrame input: sort data and retrieve the DataFrame
        if isinstance(tf, TimeFrame):
            tf.sort_data()  # Ensure data is sorted before shifting
            df = tf.get_data()

            # If target_col isn't set in the shifter, retrieve it from TimeFrame
            if not self.target_col:
                self.target_col = tf._target_col

            # Assign the backend from TimeFrame
            self.data_format = tf.dataframe_backend

        # Handle raw DataFrame input
        elif tf is not None:
            df = tf

            # Infer the target column from the input if not already set
            if not self.target_col:
                if hasattr(df, "columns"):
                    self.target_col = df.columns[-1]
                else:
                    raise ValueError("The input DataFrame does not have columns.")

            # Set or infer the backend for the DataFrame
            self._set_or_infer_data_format(df)
        else:
            raise ValueError("Input data is None.")

        # Delegate transformation to backend-specific methods
        if self.data_format == BACKEND_PANDAS or self.data_format == BACKEND_MODIN:
            transformed_df = self._transform_pandas_modin(df)
        elif self.data_format == BACKEND_POLARS:
            transformed_df = self._transform_polars(df)
        else:
            raise ValueError(f"Unsupported backend: {self.data_format}")

        # If the input was a TimeFrame, return a transformed TimeFrame
        if isinstance(tf, TimeFrame):
            return TimeFrame(
                transformed_df,
                time_col=tf.time_col,
                target_col=(
                    f"{self.target_col}_shift_{self.n_lags}"
                    if self.mode == MODE_MACHINE_LEARNING
                    else f"{self.target_col}_sequence"
                ),
                dataframe_backend=self.data_format,
            )

        return transformed_df

    def fit_transform(self, tf: TimeFrameCompatibleData) -> TimeFrameCompatibleData:
        """Fit and transform the input data in a single step.

        This method combines the functionality of the `fit` and `transform` methods. It first validates and prepares the input
        data (fitting), then applies the target variable shifting (transformation) based on the `n_lags` or `sequence_length`
        specified during initialization.

        Design:
        -------
        The output type mirrors the input type. If a `TimeFrame` is provided, a `TimeFrame` is returned. If a raw DataFrame
        (Pandas, Modin, or Polars) is provided, the output will be a DataFrame of the same type. This ensures that the
        transformation remains consistent with the input, making it easier to work with in pipeline workflows.

        :param tf: The `TimeFrame` object or a DataFrame (`pandas`, `modin`, or `polars`) to be transformed.
        :type tf: TimeFrameCompatibleData
        :raises ValueError: If the input data is invalid or the backend is unsupported.
        :raises ValueError: If the target column is not set, or is incompatible with the data.
        :return: A transformed `TimeFrame` if the input was a `TimeFrame`, otherwise a DataFrame of the same type as the input.
        :rtype: TimeFrameCompatibleData

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_target_shifter import TemporalTargetShifter
            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a sample Pandas DataFrame
            data = {
                'time': pd.date_range(start='2022-01-01', periods=100),
                'target': np.random.rand(100),
                'feature_1': np.random.rand(100)
            }
            df = pd.DataFrame(data)

            # Create a TimeFrame object
            tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend="pd")

            # Initialize TemporalTargetShifter
            shifter = TemporalTargetShifter(n_lags=2, target_col="target")

            # Fit and transform in a single step
            shifted_data = shifter.fit_transform(tf)
        """
        # Fit the data (infers backend and validates input)
        self.fit(tf)

        # Apply the transformation (delegates to backend-specific methods)
        transformed = self.transform(tf)

        # If the input was a TimeFrame, return a new TimeFrame with the transformed DataFrame
        if isinstance(tf, TimeFrame):
            tf_casted = cast(TimeFrame, tf)
            return TimeFrame(
                transformed,  # Pass the transformed DataFrame directly
                time_col=tf_casted.time_col,
                target_col=(
                    f"{self.target_col}_shift_{self.n_lags}"
                    if self.mode == MODE_MACHINE_LEARNING
                    else f"{self.target_col}_sequence"
                ),
                dataframe_backend=tf_casted.dataframe_backend,  # Ensure we use the original backend from the input
            )

        # Otherwise, return the transformed raw DataFrame
        return transformed
