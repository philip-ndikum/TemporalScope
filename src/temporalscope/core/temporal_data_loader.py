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

"""TemporalScope/src/temporalscope/core/temporal_data_loader.py.

This module provides a flexible data loader for time series forecasting, allowing users to define their own
preprocessing, loss functions, and explainability workflows. The core assumption is that features are organized
in a context window prior to the target column, making the system compatible with SHAP and other explainability methods.
Given the variance in pre-processing techniques, meta-learning & loss-functions TemporalScope explicitly does not
impose constraints on the end-user in the engineering design.

.. seealso::

    1. Van Ness, M., Shen, H., Wang, H., Jin, X., Maddix, D.C., & Gopalswamy, K. (2023). Cross-Frequency Time Series Meta-Forecasting. arXiv preprint arXiv:2302.02077.
    2. Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024). Unified training of universal time series forecasting transformers. arXiv preprint arXiv:2402.02592.
    3. Trirat, P., Shin, Y., Kang, J., Nam, Y., Na, J., Bae, M., Kim, J., Kim, B., & Lee, J.-G. (2024). Universal time-series representation learning: A survey. arXiv preprint arXiv:2401.03717.

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the License.
"""

from typing import Optional, Union

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
    SupportedBackendDataFrame,
    get_default_backend_cfg,
    validate_and_convert_input,
    validate_backend,
    validate_input,
)


class TimeFrame:
    """Central class for the TemporalScope package.

    Designed to manage time series data across various backends such as
    Polars, Pandas, and Modin. This class enables modular and flexible workflows for machine learning, deep learning,
    and time series explainability (XAI) methods like temporal SHAP.

    The `TimeFrame` class supports workflows where the target variable can be either 1D scalar data,
    typical in classical machine learning, or 3D tensor data, more common in deep learning contexts.
    It is an essential component for temporal data analysis, including but not limited to explainability pipelines
    like Temporal SHAP and concept drift analysis.

    Designed to be the core data handler in a variety of temporal analysis scenarios, the `TimeFrame` class
    integrates seamlessly with other TemporalScope modules and can be extended for more advanced use cases.

    Assumptions:
    --------------
    - This package does not impose constraints on grouping or handling duplicates.
    - We assume users will build universal models and handle preprocessing (e.g., grouping, deduplication) with
      TemporalScope modules or external methods.
    - The only requirement is that features are arranged in a context window prior to the target column.

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
       tf = TimeFrame(df, time_col='time', target_col='value', backend=BACKEND_MODIN)

       # Accessing the data
       print(tf.get_data().head())
    """

    def __init__(
        self,
        df: SupportedBackendDataFrame,
        time_col: str,
        target_col: str,
        backend: Optional[str] = None,
        sort: bool = True,
    ):
        """Initialize a TimeFrame object.

        :param df: The input DataFrame.
        :type df: SupportedBackendDataFrame
        :param time_col: The name of the column representing time in the DataFrame.
        :type time_col: str
        :param target_col: The name of the column representing the target variable in the DataFrame.
        :type target_col: str
        :param backend: The backend to use. If not provided, it will be inferred from the DataFrame type.
                        Supported backends are:
                        - `BACKEND_POLARS` ('pl') for Polars
                        - `BACKEND_PANDAS` ('pd') for Pandas
                        - `BACKEND_MODIN` ('mpd') for Modin
                        Default is to infer from the DataFrame.
        :type backend: Optional[str]
        :param sort: Optional. If True, sort the data by `time_col` in ascending order. Default is True.
        :type sort: bool
        :raises ValueError:
            - If `time_col` or `target_col` is not a non-empty string.
            - If required columns are missing in the DataFrame.
            - If the inferred or specified backend is not supported.
        :raises TypeError:
            - If the DataFrame type does not match the specified backend.
        """
        if not isinstance(time_col, str) or not time_col:
            raise ValueError("time_col must be a non-empty string.")
        if not isinstance(target_col, str) or not target_col:
            raise ValueError("target_col must be a non-empty string.")

        # Infer the backend if not explicitly provided
        self._backend = backend or self._infer_backend(df)
        validate_backend(self._backend)

        self._cfg = get_default_backend_cfg()
        self._time_col = time_col
        self._target_col = target_col
        self._sort = sort

        # Convert, validate, and set up the DataFrame
        self.df = self._setup_timeframe(df)

    @property
    def backend(self) -> str:
        """Return the backend used.

        :return: The backend identifier (e.g., 'pl', 'pd', 'mpd').
        :rtype: str
        """
        return self._backend

    @property
    def time_col(self) -> str:
        """Return the column name representing time.

        :return: The name of the time column.
        :rtype: str
        """
        return self._time_col

    @property
    def target_col(self) -> str:
        """Return the column name representing the target variable.

        :return: The name of the target column.
        :rtype: str
        """
        return self._target_col

    def _infer_backend(self, df: SupportedBackendDataFrame) -> str:
        """Infer the backend from the DataFrame type.

        :param df: The input DataFrame.
        :type df: SupportedBackendDataFrame
        :return: The inferred backend ('pl', 'pd', or 'mpd').
        :rtype: str
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

    def _validate_columns(self, df: SupportedBackendDataFrame) -> None:
        """Validate the presence of required columns in the DataFrame.

        :param df: The DataFrame to validate.
        :type df: SupportedBackendDataFrame
        :raises ValueError: If required columns are missing.
        """
        required_columns = [self._time_col, self._target_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def _sort_data(
        self,
        df: SupportedBackendDataFrame,
        ascending: bool = True,
    ) -> SupportedBackendDataFrame:
        """Internal method to sort the DataFrame based on the backend.

        :param df: The DataFrame to sort.
        :type df: SupportedBackendDataFrame
        :param ascending: If True, sort in ascending order; if False, sort in descending order.
        :type ascending: bool
        :return: The sorted DataFrame.
        :rtype: SupportedBackendDataFrame
        :raises TypeError: If the DataFrame type does not match the backend.
        :raises ValueError: If the backend is unsupported.
        """
        # Validate the DataFrame type
        validate_input(df, self._backend)

        sort_key = [self._time_col]

        # Mapping of backends to their sort functions
        sort_functions = {
            BACKEND_POLARS: lambda df: df.sort(by=sort_key, descending=not ascending),
            BACKEND_PANDAS: lambda df: df.sort_values(by=sort_key, ascending=ascending),
            BACKEND_MODIN: lambda df: df.sort_values(by=sort_key, ascending=ascending),
        }

        try:
            return sort_functions[self._backend](df)
        except KeyError:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def _setup_timeframe(self, df: SupportedBackendDataFrame) -> SupportedBackendDataFrame:
        """Sets up the TimeFrame object by converting, validating, and preparing data as required.

        :param df: The input DataFrame to be processed.
        :type df: SupportedBackendDataFrame
        :return: The processed DataFrame.
        :rtype: SupportedBackendDataFrame
        :raises ValueError:
            - If required columns are missing.
            - If the specified backend is not supported.
        :raises TypeError: If the DataFrame type does not match the backend.
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
        """Public method to sort the DataFrame by the time column.

        :param ascending: If True, sort in ascending order; if False, sort in descending order.
        :type ascending: bool
        :raises TypeError: If the DataFrame type does not match the backend.
        :raises ValueError: If the backend is unsupported.
        """
        self.df = self._sort_data(self.df, ascending=ascending)

    def get_data(self) -> SupportedBackendDataFrame:
        """Return the DataFrame in its current state.

        :return: The DataFrame managed by the TimeFrame instance.
        :rtype: SupportedBackendDataFrame
        """
        return self.df

    def update_data(self, new_df: SupportedBackendDataFrame) -> None:
        """Updates the internal DataFrame with the provided new DataFrame.

        :param new_df: The new DataFrame to replace the existing one.
        :type new_df: SupportedBackendDataFrame
        :raises TypeError: If the new DataFrame type does not match the backend.
        :raises ValueError: If required columns are missing in the new DataFrame.
        """
        # Validate and convert the new DataFrame
        new_df = validate_and_convert_input(new_df, self._backend)
        # Validate required columns
        self._validate_columns(new_df)
        self.df = new_df

    def update_target_col(self, new_target_col: Union[pl.Series, pd.Series, mpd.Series]) -> None:
        """Updates the target column in the internal DataFrame with the provided new target column.

        :param new_target_col: The new target column to replace the existing one.
        :type new_target_col: Union[pl.Series, pd.Series, mpd.Series]
        :raises TypeError: If the target column type does not match the backend.
        :raises ValueError: If the length of the new target column does not match the DataFrame.
        """
        # Validate the target column type
        if self._backend == BACKEND_POLARS:
            if not isinstance(new_target_col, pl.Series):
                raise TypeError("Expected a Polars Series for the Polars backend.")
        elif self._backend == BACKEND_PANDAS:
            if not isinstance(new_target_col, pd.Series):
                raise TypeError("Expected a Pandas Series for the Pandas backend.")
        elif self._backend == BACKEND_MODIN:
            if not isinstance(new_target_col, mpd.Series):
                raise TypeError("Expected a Modin Series for the Modin backend.")
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

        # Check if the new target column length matches the DataFrame length
        if len(new_target_col) != len(self.df):
            raise ValueError("The new target column must have the same number of rows as the DataFrame.")

        # Update the target column based on the backend
        if self._backend == BACKEND_POLARS:
            self.df = self.df.with_columns([new_target_col.alias(self._target_col)])
        elif self._backend == BACKEND_PANDAS:
            self.df[self._target_col] = new_target_col.to_numpy()  # Convert to NumPy for Pandas
        elif self._backend == BACKEND_MODIN:
            self.df[self._target_col] = new_target_col.to_numpy()  # Use .to_numpy() for Modin
