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

"""TemporalScope/src/temporalscope/target_shifters/single_step.py.

This module provides essential functionality for shifting target variables in time series data for
single-step prediction tasks. It works in conjunction with the TimeFrame class to enable
consistent target shifting operations across different DataFrame backends through Narwhals.

Engineering Design
------------------
1. Lightweight scikit-learn integration enabling pipeline and grid search compatibility
  while preserving Narwhals as the core transformation engine.

2. Backend-agnostic design using Narwhals operations, supporting all DataFrame types
  and TimeFrame operations with zero backend-specific optimizations.

3. Type-safe unified interface with consistent handling of both DataFrame and TimeFrame
  inputs through a familiar scikit-learn style API.

.. note::
   Currently implements single-step prediction only. For multi-step sequence prediction,
   see the planned MultiStepTargetShifter in temporalscope.target_shifters.multi_step.

.. seealso::

   1. Torres, J.F., Hadjout, D., Sebaa, A., Martínez-Álvarez, F., & Troncoso, A. (2021). Deep learning for time series forecasting: a survey. Big Data, 9(1), 3-21. https://doi.org/10.1089/big.2020.0159
   2. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209
   3. Tang, Y., Song, Z., Zhu, Y., Yuan, H., Hou, M., Ji, J., Tang, C., & Li, J. (2022). A survey on machine learning models for financial time series forecasting. Neurocomputing, 512, 363-380. https://doi.org/10.1016/j.neucom.2022.09.078
"""

from typing import Optional, Union

import narwhals as nw
import pandas as pd

from temporalscope.core.core_utils import (
    MODE_SINGLE_STEP,
    SupportedTemporalDataFrame,
)
from temporalscope.core.temporal_data_loader import TimeFrame


class TemporalTargetShifter:
    """A transformer-like class for shifting target variables in time series data.

    This class provides target shifting functionality for single-step prediction tasks,
    working with both TimeFrame objects and raw DataFrames through Narwhals' backend-agnostic
    operations.

    Engineering Design Assumptions
    ----------------------------
    1. Preprocessed Data:
       - Input data is clean and preprocessed
       - Missing values are handled before shifting
       - Features are appropriately scaled/normalized

    2. Single-step Mode:
       - Each row represents one time step
       - Target variable is shifted by specified lag
       - Compatible with traditional ML frameworks
       - Supports scalar target prediction tasks

    3. Backend Agnostic:
       - Operations use Narwhals for consistency
       - Works across supported DataFrame types
       - Preserves input backend in output

    .. note::
        Multi-step mode is currently not implemented due to limitations across DataFrame
        backends (Modin, Polars) which don't natively support vectorized targets.
        A future interoperability layer is planned to support sequence-based targets.

    :param target_col: Column name to shift (required)
    :type target_col: str
    :param n_lags: Number of steps to shift target, must be > 0
    :type n_lags: int
    :param drop_target: Whether to remove original target column
    :type drop_target: bool
    :param verbose: Enable progress/debug logging
    :type verbose: bool
    :raises ValueError: If target_col is None or n_lags ≤ 0

    Example Usage
    ------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.core.temporal_target_shifter import TemporalTargetShifter

        # Create sample data
        data = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=100), "target": range(100), "feature": range(100)})

        # Initialize TimeFrame
        tf = TimeFrame(data, time_col="time", target_col="target")

        # Create shifter and transform
        shifter = TemporalTargetShifter(target_col="target", n_lags=1)
        shifted_df = shifter.fit_transform(tf)
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        n_lags: int = 1,
        drop_target: bool = True,
        verbose: bool = False,
    ):
        """Initialize the shifter with target column and lag settings.

        :param target_col: Column to shift
        :param n_lags: Steps to shift target
        :param drop_target: Remove original target
        :param verbose: Enable logging
        :raises ValueError: Invalid parameters
        """
        if target_col is None:
            raise ValueError("`target_col` must be explicitly provided")

        if n_lags <= 0:
            raise ValueError("`n_lags` must be greater than 0")

        self.target_col = target_col
        self.n_lags = n_lags
        self.mode = MODE_SINGLE_STEP  # Only single-step supported
        self.drop_target = drop_target
        self.verbose = verbose

        if verbose:
            print(f"Initialized TemporalTargetShifter with target_col={target_col}, n_lags={n_lags}")

    @nw.narwhalify
    def _transform_df(self, df: SupportedTemporalDataFrame) -> SupportedTemporalDataFrame:
        """Transform DataFrame by shifting target variable using Narwhals operations.

        Performs backend-agnostic target shifting while preserving data integrity and handling
        null values appropriately.

        :param df: Input DataFrame to transform
        :type df: SupportedTemporalDataFrame
        :return: Transformed DataFrame with shifted target
        :rtype: SupportedTemporalDataFrame
        :raises ValueError: If transformation results in empty DataFrame

        Example:
        -------
        .. code-block:: python

            # Inside TemporalTargetShifter instance
            transformed_df = self._transform_df(input_df)
            # Returns DataFrame with shifted target column

        .. note::
            Key implementation patterns:
            - Uses Narwhals column operations for backend-agnostic shifting
            - Handles null values through drop_nulls() operation
            - Preserves original column names with clear suffixes
            - Tracks row counts for data integrity checks
            - Maintains consistent error reporting

        See Also:
            - transform: Public method using this internal transformer
            - TimeFrame: Main class for temporal data operations
            - Narwhals operations documentation

        """
        rows_before = len(df)

        # Single-step shift
        df = df.with_columns(
            [nw.col(self.target_col).shift(-self.n_lags).alias(f"{self.target_col}_shift_{self.n_lags}")]
        )
        df = df.drop_nulls()

        if self.drop_target:
            df = df.drop([self.target_col])

        if df.is_empty():
            raise ValueError("All rows were dropped during transformation")

        if self.verbose:
            rows_after = len(df)
            print(f"Rows before: {rows_before}; Rows after: {rows_after}; Dropped: {rows_before - rows_after}")

        return df

    @nw.narwhalify
    def fit(self, df: Union[TimeFrame, SupportedTemporalDataFrame]) -> "TemporalTargetShifter":
        """Prepare shifter for transformation (primarily for scikit-learn compatibility).

        This method is stateless but maintains scikit-learn's fit/transform pattern
        and handles target column inference from TimeFrame inputs.

        :param df: Input TimeFrame or DataFrame to analyze
        :type df: Union[TimeFrame, SupportedTemporalDataFrame]
        :return: Self for method chaining
        :rtype: TemporalTargetShifter

        Example:
        -------
        .. code-block:: python

            # Initialize and fit shifter
            shifter = TemporalTargetShifter()
            fitted_shifter = shifter.fit(timeframe)
            # Ready for transformation

        .. note::
            Key patterns:
            - Maintains scikit-learn compatibility
            - Infers target column from TimeFrame if needed
            - Enables method chaining
            - No state modification beyond target column inference

        See Also:
            - transform: Apply the shifting operation
            - fit_transform: Combined fitting and transformation

        """
        if isinstance(df, TimeFrame):
            if not self.target_col:
                self.target_col = df._target_col
        return self

    @nw.narwhalify
    def transform(
        self, df: Union[TimeFrame, SupportedTemporalDataFrame]
    ) -> Union[TimeFrame, SupportedTemporalDataFrame]:
        """Transform input data by applying target shifting operation.

        Maintains input type consistency (TimeFrame or DataFrame) while applying
        the shifting operation using Narwhals backend-agnostic functions.

        :param df: TimeFrame or DataFrame containing the time series data
        :type df: Union[TimeFrame, SupportedTemporalDataFrame]
        :return: Transformed data with shifted target column
        :rtype: Union[TimeFrame, SupportedTemporalDataFrame]
        :raises ValueError: If transformation fails or results in invalid state

        Example:
        -------
        .. code-block:: python

            import pandas as pd
            from temporalscope.core.temporal_target_shifter import TemporalTargetShifter

            # Create shifter and transform data
            data = pd.DataFrame({"time": range(10), "target": range(10)})
            shifter = TemporalTargetShifter(target_col="target", n_lags=1)
            shifted_data = shifter.transform(data)

        .. note::
            Key implementation patterns:
            - Preserves input type (TimeFrame -> TimeFrame, DataFrame -> DataFrame)
            - Uses Narwhals operations for backend-agnostic transformation
            - Maintains column naming conventions across transformations
            - Handles TimeFrame metadata consistently

        See Also:
            - _transform_df: Internal transformation method
            - fit_transform: Combined fitting and transformation
            - TimeFrame: Container class for temporal data

        """
        if isinstance(df, TimeFrame):
            if not self.target_col:
                self.target_col = df._target_col
            transformed_df = self._transform_df(df.df)
            return TimeFrame(
                transformed_df,
                time_col=df._time_col,
                target_col=f"{self.target_col}_shift_{self.n_lags}",
                dataframe_backend=df.backend,
            )
        return self._transform_df(df)

    @nw.narwhalify
    def fit_transform(
        self, df: Union[TimeFrame, SupportedTemporalDataFrame]
    ) -> Union[TimeFrame, SupportedTemporalDataFrame]:
        """Fit shifter and transform data in a single operation.

        Convenience method combining fit() and transform() while maintaining
        proper type signatures and error handling.

        :param df: Input data to transform
        :type df: Union[TimeFrame, SupportedTemporalDataFrame]
        :return: Transformed data with shifted target
        :rtype: Union[TimeFrame, SupportedTemporalDataFrame]

        Example:
        -------
        .. code-block:: python

            # Single-step transformation
            shifted_data = shifter.fit_transform(input_data)

        .. note::
            Key implementation patterns:
            - Combines fit and transform efficiently
            - Preserves input/output type consistency
            - Maintains proper error propagation
            - Follows scikit-learn's API pattern

        See Also:
            - fit: Preparation step
            - transform: Transformation step

        """
        return self.fit(df).transform(df)
