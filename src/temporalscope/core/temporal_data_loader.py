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
Given the variance in pre-processing techniques, meta-learning & loss-functions, TemporalScope explicitly does not
impose constraints on the end-user in the engineering design.

Engineering Design
--------------------
.. note::

    TemporalScope is designed with several key assumptions to ensure performance, scalability, and flexibility
    across a wide range of time series forecasting and XAI workflows.

    1. **Preprocessed Data Assumption**:
       TemporalScope assumes that the user provides clean, preprocessed data. This includes handling categorical
       encoding, missing data imputation, and feature scaling prior to using TemporalScope's partitioning and explainability
       methods. Similar assumptions are seen in popular packages such as TensorFlow and GluonTS, which expect the
       user to manage data preprocessing outside of the core workflow.
    2. **Time Column Constraints**:
       The `time_col` must be either a numeric index or a timestamp. TemporalScope relies on this temporal ordering for
       key operations like sliding window partitioning and temporal explainability workflows (e.g., SHAP). Packages like
       **Facebook Prophet** and **Darts** also require proper temporal ordering as a baseline assumption for modeling time
       series data.
    3. **Numeric Features Requirement**:
       Aside from the `time_col`, all other features in the dataset must be numeric. This ensures compatibility with machine
       learning and deep learning models that require numeric inputs. As seen in frameworks like TensorFlow and
       Prophet, users are expected to preprocess categorical features (e.g., one-hot encoding or embeddings) before
       applying modeling or partitioning algorithms.
    4. **Modular Design for Explainability**:
       TemporalScope assumes a modular, window-based design that is naturally compatible with model-agnostic explainability
       methods like SHAP and LIME. Features are expected to be structured in a temporal context for efficient partitioning
       and explainability. This mirrors the design of frameworks like Darts, which use similar assumptions for time
       series forecasting and explainability workflows.

    By enforcing these constraints, TemporalScope focuses on its core purpose—time series partitioning, explainability,
    and scalability—while leaving more general preprocessing tasks to the user. This follows industry standards seen in
    popular time series libraries.

.. seealso::

    1. Van Ness, M., Shen, H., Wang, H., Jin, X., Maddix, D.C., & Gopalswamy, K. (2023). Cross-Frequency Time Series Meta-Forecasting. arXiv preprint arXiv:2302.02077.
    2. Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024). Unified training of universal time series forecasting transformers. arXiv preprint arXiv:2402.02592.
    3. Trirat, P., Shin, Y., Kang, J., Nam, Y., Na, J., Bae, M., Kim, J., Kim, B., & Lee, J.-G. (2024). Universal time-series representation learning: A survey. arXiv preprint arXiv:2401.03717.
    4. Xu, Q., Zhuo, X., Jiang, C. and Liu, Y., 2019. An artificial neural network for mixed frequency data. Expert Systems with Applications, 118, pp.127-139.4
    5. Filho, L.L., de Oliveira Werneck, R., Castro, M., Ribeiro Mendes Júnior, P., Lustosa, A., Zampieri, M., Linares, O., Moura, R., Morais, E., Amaral, M. and Salavati, S., 2024. A multi-modal approach for mixed-frequency time series forecasting. Neural Computing and Applications, pp.1-25.
"""

import warnings
from typing import Optional, Union
from datetime import datetime, timedelta, date

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.exceptions import (
    TimeColumnError,
    MixedTypesWarning,
    MixedTimezonesWarning,
    MixedFrequencyWarning,
)

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

# Define alias with forward reference
TimeFrameCompatibleData = Union["TimeFrame", SupportedBackendDataFrame]


class TimeFrame:
    """Central class for the TemporalScope package.

    The `TimeFrame` class is designed to handle time series data across various backends, including Polars, Pandas,
    and Modin. It facilitates workflows for machine learning, deep learning, and explainability methods, while abstracting
    away backend-specific implementation details.

    This class automatically infers the appropriate backend, validates the data, and sorts it by time. It ensures
    compatibility with temporal XAI techniques (SHAP, Boruta-SHAP, LIME etc) supporting larger data workflows in
    production.

    Backend Handling
    ----------------
    - If a `dataframe_backend` is explicitly provided, it takes precedence over backend inference.
    - If no backend is specified, the class infers the backend from the DataFrame type, supporting Polars, Pandas, and Modin.

    Engineering Design Assumptions
    ------------------
    - Universal Models: This class is designed assuming the user has pre-processed their data for compatibility with
      deep learning models. Across the TemporalScope utilities (e.g., target shifter, padding, partitioning algorithms),
      it is assumed that preprocessing tasks, such as categorical feature encoding, will be managed by the user or
      upstream modules. Thus the model will learn global weights and will not groupby categorical variables.
    - Mixed Time Frequency supported: Given the flexibility of deep learning models to handle various time frequencies,
      this class allows `time_col` to contain mixed frequency data, assuming the user will manage any necessary preprocessing
      or alignment outside of this class.
    - The `time_col` should be either numeric or timestamp-like for proper temporal ordering. Any mixed or invalid data
      types will raise validation errors.
    - All non-time columns are expected to be numeric. Users are responsible for handling non-numeric features
      (e.g., encoding categorical features).

    Example Usage
    -------------
    .. code-block:: python

       import polars as pl
       data = pl.DataFrame({
           'time': pl.date_range(start='2021-01-01', periods=100, interval='1d'),
           'value': range(100)
       })
       tf = TimeFrame(data, time_col='time', target_col='value')
       print(tf.get_data().head())

    .. seealso::
       - `polars` documentation: https://pola-rs.github.io/polars/
       - `pandas` documentation: https://pandas.pydata.org/
       - `modin` documentation: https://modin.readthedocs.io/
    """

    def __init__(
        self,
        df: SupportedBackendDataFrame,
        time_col: str,
        target_col: str,
        dataframe_backend: Optional[str] = None,
        sort: bool = True,
    ) -> None:
        """Initialize a TimeFrame object with required validations and backend handling.

        This constructor validates the provided DataFrame and performs checks on the required columns (`time_col`,
        `target_col`). It also ensures compatibility between the DataFrame and the specified or inferred backend.

        :param df: The input DataFrame, which can be any supported backend (e.g., Polars, Pandas, Modin).
        :type df: SupportedBackendDataFrame
        :param time_col: The name of the column representing time. Should be numeric or timestamp-like for sorting.
        :type time_col: str
        :param target_col: The column representing the target variable. Must be a valid column in the DataFrame.
        :type target_col: str
        :param dataframe_backend: The backend to use. If provided, the DataFrame will be converted to the appropriate backend.
            If not provided, it will be inferred from the DataFrame type.
            Supported backends are:
            - `BACKEND_POLARS` ('pl') for Polars
            - `BACKEND_PANDAS` ('pd') for Pandas
            - `BACKEND_MODIN` ('mpd') for Modin
        :type dataframe_backend: Optional[str]
        :param sort: If True, the data will be sorted by `time_col` in ascending order. Default is True.
        :type sort: bool

        :raises ValueError:
            - If `time_col` or `target_col` is not a valid non-empty string.
            - If the input DataFrame is missing required columns or is empty.
            - If the inferred or provided backend is unsupported.
        :raises TypeError:
            - If the DataFrame type does not match the specified backend.

        .. note::
            - The `time_col` must be numeric or timestamp-like to ensure proper temporal ordering.
            - Sorting is automatically performed by `time_col` unless disabled via `sort=False`.
            - If `dataframe_backend` is provided, the DataFrame will be converted to the corresponding backend format.
            - If `dataframe_backend` is not provided, it will be inferred based on the DataFrame type.

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pl.DataFrame({
                'time': pl.date_range(start='2021-01-01', periods=5, interval='1d'),
                'value': range(5)
            })

            tf = TimeFrame(data, time_col='time', target_col='value')
            print(tf.get_data().head())
        """

        # Ensure time_col and target_col are valid strings
        if not isinstance(time_col, str) or not time_col:
            raise ValueError("`time_col` must be a non-empty string.")
        if not isinstance(target_col, str) or not target_col:
            raise ValueError("`target_col` must be a non-empty string.")

        # Set class variables
        self._time_col = time_col
        self._target_col = target_col
        self._sort = sort

        # Use the centralized setup method to handle backend inference, validation, and sorting
        self._setup_timeframe(df, dataframe_backend)

    @property
    def dataframe_backend(self) -> str:
        """Return the backend used."""
        return self._dataframe_backend

    @property
    def time_col(self) -> str:
        """Return the column name representing time."""
        return self._time_col

    @property
    def target_col(self) -> str:
        """Return the column name representing the target variable."""
        return self._target_col

    @property
    def df(self) -> SupportedBackendDataFrame:
        """Return the DataFrame in its current state."""
        return self._df

    @df.setter
    def df(self, dataframe: SupportedBackendDataFrame):
        """Set the internal DataFrame."""
        self._df = dataframe

    def _setup_timeframe(self, df: SupportedBackendDataFrame, dataframe_backend: Optional[str]) -> None:
        """Centralized method to set up the TimeFrame instance.

        This method handles backend inference, data validation, and sorting.
        It ensures consistency between the initialization and update processes.

        :param df: The input DataFrame to be set up.
        :type df: SupportedBackendDataFrame
        :param dataframe_backend: The backend to use. If None, it will be inferred.
        :type dataframe_backend: Optional[str]

        :raises ValueError: If required validations fail (e.g., missing columns, unsupported backend).
        """

        # Infer backend if not provided
        self._dataframe_backend = dataframe_backend or self._infer_dataframe_backend(df)

        # Set the DataFrame
        self.df = validate_and_convert_input(df, self._dataframe_backend)

        # Validate data (e.g., columns, types)
        self.validate_data()

        # Sort the data if necessary
        if self._sort:
            self.sort_data()

    def _infer_dataframe_backend(self, df: SupportedBackendDataFrame) -> str:
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

    def _validate_numeric_features(self) -> None:
        """Validate that all features, except for the time column, are numeric.

        This method checks that all columns, other than the `time_col`, contain numeric data, which is a requirement
        for machine learning and deep learning workflows.

        :raises ValueError: If any feature column is not numeric.
        """
        df = self.get_data()

        # Backend-specific handling for numeric validation
        if self.dataframe_backend in [BACKEND_PANDAS, BACKEND_MODIN]:
            non_numeric_columns = [
                col for col in df.columns if col != self.time_col and not pd.api.types.is_numeric_dtype(df[col])
            ]
        elif self.dataframe_backend == BACKEND_POLARS:
            non_numeric_columns = [col for col in df.columns if col != self.time_col and not df[col].dtype.is_numeric()]
        else:
            raise ValueError(f"Unsupported backend: {self.dataframe_backend}")

        if non_numeric_columns:
            raise ValueError(
                f"All features except `time_col` must be numeric. Found non-numeric columns: {non_numeric_columns}."
            )

    def _validate_time_column(self) -> None:
        """Validate that the `time_col` in the DataFrame is either numeric or timestamp-like.

        This ensures the `time_col` can be used for temporal operations like sorting
        or partitioning, which are essential for time-series forecasting. The `time_col`
        must be numeric (e.g., integers) or timestamp-like (e.g., datetime). Mixed frequencies
        (e.g., daily and monthly timestamps) are allowed, but mixed data types (e.g., numeric and
        string) are not. String data types in `time_col` are not allowed across any backend.

        :raises TimeColumnError: If `time_col` is missing, contains unsupported types (non-numeric or non-timestamp),
        or has missing values.
        :warns MixedFrequencyWarning: If `time_col` contains mixed frequencies (e.g., daily and monthly timestamps).
        :warns MixedTimezonesWarning: If `time_col` contains mixed timezone-aware and timezone-naive entries.
        """
        df = self.get_data()

        # Ensure the time column exists
        if self.time_col not in df.columns:
            raise TimeColumnError(f"Missing required column: {self.time_col}")

        # Time column could be Pandas/Modin Series or Polars Series
        time_col = df[self.time_col]

        # Narrowing the type to ensure type checking with MyPy
        if isinstance(time_col, (pd.Series, mpd.Series)):
            # Check for missing values in time_col (specific to Pandas/Modin)
            if time_col.isnull().any():
                raise TimeColumnError("Missing values found in `time_col`")

            # Validate if time_col is either numeric or timestamp-like
            is_numeric_col = self._is_numeric(time_col)
            is_timestamp_col = self._is_timestamp_like(time_col)

            if not is_numeric_col and not is_timestamp_col:
                raise TimeColumnError(f"`time_col` must be either numeric or timestamp-like, got {time_col.dtype}")

            # Raise MixedFrequencyWarning if mixed frequencies are detected
            if is_timestamp_col and self._has_mixed_frequencies(time_col):
                warnings.warn("`time_col` contains mixed timestamp frequencies.", MixedFrequencyWarning)

            # Raise MixedTimezonesWarning if mixed timezones are detected
            if is_timestamp_col and self._has_mixed_timezones(time_col):
                warnings.warn(
                    "`time_col` contains mixed timezone-aware and timezone-naive entries.", MixedTimezonesWarning
                )

        elif isinstance(time_col, pl.Series):
            # Check for missing values in Polars
            if time_col.is_null().sum() > 0:
                raise TimeColumnError("Missing values found in `time_col`")

            is_numeric_col = self._is_numeric(time_col)
            is_timestamp_col = self._is_timestamp_like(time_col)

            if not is_numeric_col and not is_timestamp_col:
                raise TimeColumnError(f"`time_col` must be either numeric or timestamp-like, got {time_col.dtype}")

            # Raise MixedFrequencyWarning if mixed frequencies are detected
            if is_timestamp_col and self._has_mixed_frequencies(time_col):
                warnings.warn("`time_col` contains mixed timestamp frequencies.", MixedFrequencyWarning)

            # Raise MixedTimezonesWarning if mixed timezones are detected
            if is_timestamp_col and self._has_mixed_timezones(time_col):
                warnings.warn(
                    "`time_col` contains mixed timezone-aware and timezone-naive entries.", MixedTimezonesWarning
                )

    def _is_timestamp_like(self, time_col: Union[pd.Series, mpd.Series, pl.Series]) -> bool:
        """Check if a time column is timestamp-like based on the backend.

        :param time_col: The time column to check.
        :return: True if the column is timestamp-like, False otherwise.
        """
        if self.dataframe_backend in [BACKEND_PANDAS, BACKEND_MODIN]:
            return pd.api.types.is_datetime64_any_dtype(time_col)
        elif self.dataframe_backend == BACKEND_POLARS:
            return time_col.dtype == pl.Datetime
        return False

    def _is_numeric(self, time_col: Union[pd.Series, mpd.Series, pl.Series]) -> bool:
        """Check if a time column is numeric based on the backend.

        :param time_col: The time column to check.
        :return: True if the column is numeric, False otherwise.
        """
        if self.dataframe_backend in [BACKEND_PANDAS, BACKEND_MODIN]:
            return pd.api.types.is_numeric_dtype(time_col)
        elif self.dataframe_backend == BACKEND_POLARS:
            return time_col.dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        return False

    def _has_mixed_frequencies(self, time_col: Union[pd.Series, mpd.Series, pl.Series]) -> bool:
        """Check for mixed frequencies in the time column.

        :param time_col: The time column to check for mixed frequencies.
        :return: True if mixed frequencies are detected, False otherwise.
        """
        if isinstance(time_col, (pd.Series, mpd.Series)):
            inferred_freq = pd.infer_freq(time_col.dropna())
            return inferred_freq is None
        elif isinstance(time_col, pl.Series):
            inferred_freq = time_col.to_pandas().infer_freq()  # Converts Polars to Pandas for frequency detection
            return inferred_freq is None
        return False

    def _has_mixed_timezones(self, time_col: Union[pd.Series, mpd.Series, pl.Series]) -> bool:
        """Check for mixed timezone-aware and naive timestamps.

        :param time_col: The time column to check for mixed timezones.
        :return: True if mixed timezone-aware and naive timestamps are detected, False otherwise.
        """
        if isinstance(time_col, (pd.Series, mpd.Series)):
            if time_col.dt.tz is not None:
                return time_col.dt.tz.hasnans
        elif isinstance(time_col, pl.Series):
            dtype_str = str(time_col.dtype)
            return "TimeZone" in dtype_str
        return False

    def get_data(self) -> SupportedBackendDataFrame:
        """Return the DataFrame in its current state.

        :return: The DataFrame managed by the TimeFrame instance.
        :rtype: SupportedBackendDataFrame

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            data = {
                'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
                'target': range(5, 0, -1)
            }
            df = pd.DataFrame(data)

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col='time', target_col='target')

            # Retrieve the DataFrame
            data = tf.get_data()
            print(data.head())
        """
        return self.df

    def validate_data(self) -> None:
        """Run validation checks on the TimeFrame data to ensure it meets the required constraints.

        This method runs all internal validation checks to ensure that:
        - The `time_col` is numeric or timestamp-like.
        - All features, except `time_col`, are numeric.
        - There are no missing values in the `time_col` or `target_col`.
        - It checks for mixed frequencies in the `time_col` and raises a warning if detected.

        :raises ValueError: If any of the validation checks fail.

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            data = {
                'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
                'target': range(5, 0, -1)
            }
            df = pd.DataFrame(data)

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col='time', target_col='target')

            # Validate the data
            tf.validate_data()

        """
        # Centralized validation of time column
        self._validate_time_column()

        # Ensure all non-time columns are numeric
        self._validate_numeric_features()

        # Validate that there are no missing values in the time and target columns
        self._validate_no_missing_values()

        # Check for mixed frequencies in the time column
        self._check_mixed_frequencies()

        # Indicate successful validation
        print("Data validation passed successfully.")

    def sort_data(self, ascending: bool = True) -> None:
        """Sort the DataFrame by the time column in place.

        :param ascending: If True, sort in ascending order; if False, sort in descending order.
        :type ascending: bool
        :raises TypeError: If the DataFrame type does not match the backend.
        :raises ValueError: If the backend is unsupported or validation fails.

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            data = {
                'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
                'target': range(5, 0, -1)
            }
            df = pd.DataFrame(data)

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col='time', target_col='target')

            # Sort the DataFrame in ascending order
            tf.sort_data(ascending=True)
            print(tf.df)
        """
        # Validate the DataFrame before sorting
        self.validate_data()

        sort_key = [self._time_col]

        # Mapping of backends to their sort functions, sorting in place
        if self.dataframe_backend == BACKEND_POLARS:
            if isinstance(self.df, pl.DataFrame):
                self.df = self.df.sort(by=sort_key, descending=not ascending)
            else:
                raise TypeError(f"Expected Polars DataFrame but got {type(self.df)}")
        elif self.dataframe_backend == BACKEND_PANDAS:
            if isinstance(self.df, pd.DataFrame):
                self.df.sort_values(by=sort_key, ascending=ascending, inplace=True)
            else:
                raise TypeError(f"Expected Pandas DataFrame but got {type(self.df)}")
        elif self.dataframe_backend == BACKEND_MODIN:
            if isinstance(self.df, mpd.DataFrame):
                self.df.sort_values(by=sort_key, ascending=ascending, inplace=True)
            else:
                raise TypeError(f"Expected Modin DataFrame but got {type(self.df)}")
        else:
            raise ValueError(f"Unsupported dataframe backend {self._dataframe_backend}")

    def update_data(
        self, new_df: SupportedBackendDataFrame, time_col: Optional[str] = None, target_col: Optional[str] = None
    ) -> None:
        """Updates the internal DataFrame with the provided new DataFrame and ensures backend consistency.

        :param new_df: The new DataFrame to replace the existing one.
        :type new_df: SupportedBackendDataFrame
        :param time_col: The name of the column representing time. Should be numeric or timestamp-like for sorting. Optional.
        :type time_col: Optional[str]
        :param target_col: The column representing the target variable. Must be a valid column in the DataFrame. Optional.
        :type target_col: Optional[str]
        :raises TypeError: If the new DataFrame type does not match the backend.
        :raises ValueError: If required columns are missing in the new DataFrame, or validation fails.

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create initial DataFrame
            df = pd.DataFrame({
                'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
                'target': range(5, 0, -1)
            })

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col='time', target_col='target')

            # Create new DataFrame to update
            new_df = pd.DataFrame({
                'time': pd.date_range(start='2021-01-06', periods=5, freq='D'),
                'target': range(1, 6)
            })

            # Update the DataFrame within TimeFrame
            tf.update_data(new_df, time_col='time', target_col='target')
            print(tf.get_data())
        """

        # Infer backend for the new DataFrame if needed
        self._dataframe_backend = self._infer_dataframe_backend(new_df)

        # Validate and update the time_col and target_col if provided
        if time_col:
            if time_col not in new_df.columns:
                raise ValueError(f"`time_col` {time_col} not found in the new DataFrame.")
            self._time_col = time_col

        if target_col:
            if target_col not in new_df.columns:
                raise ValueError(f"`target_col` {target_col} not found in the new DataFrame.")
            self.update_target_col(new_df[target_col])

        # Use _setup_timeframe to centralize backend inference, validation, and sorting
        self._setup_timeframe(new_df, self._dataframe_backend)

    def update_target_col(self, new_target_col: SupportedBackendDataFrame) -> None:
        """Updates the target column in the internal DataFrame with the provided new target column.

        :param new_target_col: The new target column to replace the existing one.
        :type new_target_col: Union[pl.Series, pd.Series, mpd.Series]
        :raises TypeError: If the target column type does not match the backend.
        :raises ValueError: If the length of the new target column does not match the DataFrame, or validation fails.

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            df = pd.DataFrame({
                'time': pd.date_range(start='2021-01-01', periods=5, freq='D'),
                'target': range(5, 0, -1)
            })

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col='time', target_col='target')

            # Update the target column with new values
            new_target = pd.Series([1, 2, 3, 4, 5], name='target')
            tf.update_target_col(new_target)
            print(tf.get_data())
        """
        # Step 1: Validate the target column type
        if self._dataframe_backend == BACKEND_POLARS:
            if not isinstance(new_target_col, pl.Series):
                raise TypeError("Expected a Polars Series for the Polars backend.")
        elif self._dataframe_backend == BACKEND_PANDAS:
            if not isinstance(new_target_col, pd.Series):
                raise TypeError("Expected a Pandas Series for the Pandas backend.")
        elif self._dataframe_backend == BACKEND_MODIN:
            if not isinstance(new_target_col, mpd.Series):
                raise TypeError("Expected a Modin Series for the Modin backend.")
        else:
            raise ValueError(f"Unsupported dataframe_backend {self._dataframe_backend}")

        # Step 2: Check if the new target column length matches the DataFrame length
        if len(new_target_col) != len(self.df):
            raise ValueError("The new target column must have the same number of rows as the DataFrame.")

        # Step 3: Validate the entire DataFrame before making changes
        self.validate_data()

        # Step 4: If all validations pass, proceed with updating the target column
        # Use a temporary copy of the DataFrame for update and commit only after all checks
        temp_df = None  # Declare once without type hints

        if self._dataframe_backend == BACKEND_POLARS:
            temp_df = self.df.clone()  # Polars DataFrame uses `clone()`
        elif self._dataframe_backend == BACKEND_PANDAS and isinstance(self.df, pd.DataFrame):
            temp_df = self.df.copy()  # Pandas DataFrame uses `copy()`
        elif self._dataframe_backend == BACKEND_MODIN and isinstance(self.df, mpd.DataFrame):
            temp_df = self.df.copy()  # Modin DataFrame uses `copy()`
        else:
            raise ValueError(f"Unsupported dataframe_backend {self._dataframe_backend}")

        # Update the target column based on the backend
        if self._dataframe_backend == BACKEND_POLARS:
            temp_df = temp_df.with_columns([new_target_col.alias(self._target_col)])
        elif self._dataframe_backend == BACKEND_PANDAS:
            temp_df[self._target_col] = new_target_col.to_numpy()  # Convert to NumPy for Pandas
        elif self._dataframe_backend == BACKEND_MODIN:
            temp_df[self._target_col] = new_target_col.to_numpy()  # Use .to_numpy() for Modin

        # Step 5: Commit the changes by updating the internal DataFrame
        self.df = temp_df
