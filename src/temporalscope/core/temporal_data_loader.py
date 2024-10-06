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

    TemporalScope is designed with several key assumptions to ensure performance, scalability, and flexibility
    across a wide range of time series forecasting and XAI workflows.

    1. Preprocessed Data Assumption:
       TemporalScope assumes that the user provides clean, preprocessed data. This includes handling categorical
       encoding, missing data imputation, and feature scaling prior to using TemporalScope's partitioning and explainability
       methods. Similar assumptions are seen in popular packages such as TensorFlow and GluonTS, which expect the
       user to manage data preprocessing outside of the core workflow.

    2. Time Column Constraints:
       The `time_col` must be either a numeric index or a timestamp. TemporalScope relies on this temporal ordering for
       key operations like sliding window partitioning and temporal explainability workflows (e.g., SHAP).

    3. Numeric Features Requirement:
       Aside from the `time_col`, all other features in the dataset must be numeric. This ensures compatibility with machine
       learning and deep learning models that require numeric inputs. As seen in frameworks like TensorFlow, users are expected
       to preprocess categorical features (e.g., one-hot encoding or embeddings) before applying modeling or partitioning algorithms.

    4. Universal Model Assumption:
       TemporalScope is designed with the assumption that models trained will operate on the entire dataset without
       automatically applying hidden groupings or segmentations (e.g., for mixed-frequency data). This ensures that users
       can leverage frameworks like SHAP, Boruta-SHAP, and LIME for model-agnostic explainability without limitations.

    5. Supported Data Modes:

       TemporalScope also integrates seamlessly with model-agnostic explainability techniques like SHAP, LIME, and
       Boruta-SHAP, allowing insights to be extracted from most machine learning and deep learning models.

       The following table illustrates the two primary modes supported by TemporalScope and their typical use cases:

       +--------------------+----------------------------------------------------+--------------------------------------------+
       | **Mode**           | **Description**                                    | **Compatible Frameworks**                  |
       +--------------------+----------------------------------------------------+--------------------------------------------+
       | Single-step mode   | Suitable for scalar target machine learning tasks. | Scikit-learn, XGBoost, LightGBM, SHAP      |
       |                    | Each row represents a single time step.            | TensorFlow (for standard regression tasks) |
       +--------------------+----------------------------------------------------+--------------------------------------------+
       | Multi-step mode    | Suitable for deep learning tasks like sequence     | TensorFlow, PyTorch, Keras, SHAP, LIME     |
       |                    | forecasting. Input sequences (`X`) and output      | (for seq2seq models, sequence forecasting) |
       |                    | sequences (`Y`) are handled as separate datasets.  |                                            |
       +--------------------+----------------------------------------------------+--------------------------------------------+


       These modes follow standard assumptions in time series forecasting libraries, allowing for seamless integration
       with different models while requiring the user to manage their own data preprocessing.

    By enforcing these constraints, TemporalScope focuses on its core purpose—time series partitioning, explainability,
    and scalability—while leaving more general preprocessing tasks to the user. This follows industry standards seen in
    popular time series libraries.

.. seealso::

    1. Van Ness, M., Shen, H., Wang, H., Jin, X., Maddix, D.C., & Gopalswamy, K.
       (2023). Cross-Frequency Time Series Meta-Forecasting. arXiv preprint
       arXiv:2302.02077.

    2. Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024).
       Unified training of universal time series forecasting transformers. arXiv
       preprint arXiv:2402.02592.

    3. Trirat, P., Shin, Y., Kang, J., Nam, Y., Na, J., Bae, M., Kim, J., Kim, B., &
       Lee, J.-G. (2024). Universal time-series representation learning: A survey.
       arXiv preprint arXiv:2401.03717.

    4. Xu, Q., Zhuo, X., Jiang, C., & Liu, Y. (2019). An artificial neural network
       for mixed frequency data. Expert Systems with Applications, 118, pp.127-139.

    5. Filho, L.L., de Oliveira Werneck, R., Castro, M., Ribeiro Mendes Júnior, P.,
       Lustosa, A., Zampieri, M., Linares, O., Moura, R., Morais, E., Amaral, M., &
       Salavati, S. (2024). A multi-modal approach for mixed-frequency time series
       forecasting. Neural Computing and Applications, pp.1-25.

.. note::

    - Multi-Step Mode Limitation: Multi-step mode is not fully supported for backends like Modin and Polars
      due to their inability to handle vectorized (sequence) targets in a single cell. This limitation will require
      an interoperability layer for converting datasets into compatible formats (e.g., TensorFlow's `tf.data.Dataset`
      or flattening the target sequences for use in Modin/Polars).
    - Single-step mode: All backends (Pandas, Modin, Polars) work as expected without the need for special handling.
    - Recommendation: For multi-step mode, please use Pandas for now until support is added for other backends.
      Future releases will include an interoperability step to handle vectorized targets across different backends.

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
    check_empty_columns,
    check_nulls,
    infer_backend_from_dataframe,
    is_numeric,
    is_timestamp_like,
    sort_dataframe,
    validate_and_convert_input,
)
from temporalscope.core.exceptions import (
    TimeColumnError,
    UnsupportedBackendError,
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

       data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=100, interval="1d"), "value": range(100)})
       tf = TimeFrame(data, time_col="time", target_col="value")
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

            data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=5, interval="1d"), "value": range(5)})

            tf = TimeFrame(data, time_col="time", target_col="value")
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

        :raises UnsupportedBackendError: If the DataFrame's backend is unsupported.
        :raises TimeColumnError: If the time column fails validation.
        :raises ValueError: If other general validations (e.g., missing columns) fail.
        """
        try:
            # Step 1: Check if a backend was provided, otherwise infer from the DataFrame
            if dataframe_backend is not None:
                self._dataframe_backend = dataframe_backend
            else:
                self._dataframe_backend = infer_backend_from_dataframe(df)

            # Step 2: Validate and convert the input DataFrame to the correct backend format
            self.df = validate_and_convert_input(df, self._dataframe_backend)

            # Step 3: Perform data validation
            self.validate_data()

            # Step 4: Sort the DataFrame if sorting is enabled
            if self._sort:
                self.sort_data()

        except UnsupportedBackendError as e:
            raise UnsupportedBackendError(f"Unsupported backend: {e}")

        except TimeColumnError as e:
            raise TimeColumnError(f"Time column validation failed: {e}")

        except ValueError as e:
            raise ValueError(f"General validation error: {e}")

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
            data = {"time": pd.date_range(start="2021-01-01", periods=5, freq="D"), "target": range(5, 0, -1)}
            df = pd.DataFrame(data)

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col="time", target_col="target")

            # Retrieve the DataFrame
            data = tf.get_data()
            print(data.head())
        """
        return self.df

    def sort_data(self, ascending: bool = True) -> None:
        """Sort the DataFrame by the time column in place.

        This method sorts the DataFrame based on the `time_col` in ascending or descending order.
        The sorting logic is handled based on the backend (Pandas, Polars, or Modin) via the `sort_dataframe` utility.

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
            data = {"time": pd.date_range(start="2021-01-01", periods=5, freq="D"), "target": range(5, 0, -1)}
            df = pd.DataFrame(data)

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col="time", target_col="target")

            # Sort the DataFrame in ascending order
            tf.sort_data(ascending=True)
            print(tf.df)

            # Sort the DataFrame in descending order
            tf.sort_data(ascending=False)
            print(tf.df)
        """
        # Ensure the DataFrame is valid before sorting
        self.validate_data()

        # Use the utility function from core_utils to perform the sort
        self.df = sort_dataframe(self.df, self._time_col, self.dataframe_backend, ascending)

    def update_data(
        self,
        new_df: Optional[SupportedBackendDataFrame] = None,
        new_target_col: Optional[Union[pl.Series, pd.Series, mpd.Series]] = None,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> None:
        """Update the internal DataFrame and columns with new data.

        This method updates the internal DataFrame (`df`) and/or the `target_col` with the new data provided.
        It ensures the backend remains consistent across Polars, Pandas, or Modin. It validates the input
        DataFrame, checks its length, and performs safe updates.

        :param new_df: The new DataFrame to replace the existing one. Optional.
        :type new_df: SupportedBackendDataFrame, optional
        :param new_target_col: The new target column to replace the existing one. Optional.
        :type new_target_col: Union[pl.Series, pd.Series, mpd.Series], optional
        :param time_col: The name of the column representing time. Optional.
        :type time_col: str, optional
        :param target_col: The column representing the target variable. Optional.
        :type target_col: str, optional
        :raises TypeError: If the target column type does not match the backend or is not numeric.
        :raises ValueError: If the length of the new target column or new DataFrame does not match the existing one.
        :raises UnsupportedBackendError: If the backend is unsupported.

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            df = pd.DataFrame({"time": pd.date_range(start="2021-01-01", periods=5, freq="D"), "target": range(5, 0, -1)})

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col="time", target_col="target")

            # Update the DataFrame and target column
            new_target = pd.Series([1, 2, 3, 4, 5], name="target")
            tf.update_data(new_df=None, new_target_col=new_target)
            print(tf.get_data())
        """
        # Update time_col and target_col if provided
        if time_col:
            self._time_col = time_col
        if target_col:
            self._target_col = target_col

        # If a new DataFrame is provided, validate and convert it
        if new_df is not None:
            self.df = validate_and_convert_input(new_df, self._dataframe_backend)

        # If a new target column is provided, validate and update it
        if new_target_col is not None:
            # Ensure the new target column has the same length as the current DataFrame
            if len(new_target_col) != len(self.df):
                raise ValueError("The new target column must have the same number of rows as the existing DataFrame.")

            # Validate that the target column is numeric
            if not is_numeric(self.df, self._target_col):
                raise TypeError(f"The target column '{self._target_col}' must be numeric.")

            # Update the target column using backend-specific logic
            if self._dataframe_backend == BACKEND_POLARS:
                self.df = self.df.with_columns([new_target_col.alias(self._target_col)])
            elif self._dataframe_backend == BACKEND_PANDAS:
                self.df[self._target_col] = new_target_col.to_numpy()  # Convert to NumPy for Pandas
            elif self._dataframe_backend == BACKEND_MODIN:
                self.df[self._target_col] = new_target_col.to_numpy()  # Convert to NumPy for Modin

        # Perform validation of the data after updating
        self.validate_data()

        # Sort the DataFrame if needed
        if self._sort:
            self.sort_data()

    def validate_data(self) -> None:
        """Run validation checks on the TimeFrame data to ensure it meets the required constraints.

        This method performs the following validations:
        - The `time_col` is either numeric or timestamp-like.
        - All columns, except for the `time_col`, are numeric.
        - There are no missing values in the `time_col` and `target_col`.
        - No columns in the DataFrame are entirely empty.

        :raises ValueError: If any validation checks fail.
        :raises UnsupportedBackendError: If the DataFrame backend is unsupported.

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            df = pd.DataFrame({"time": pd.date_range(start="2021-01-01", periods=5, freq="D"), "target": range(5, 0, -1)})

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col="time", target_col="target")

            # Run validation on the TimeFrame
            tf.validate_data()
        """
        # 1. Check if any columns are entirely empty
        if check_empty_columns(self.df, self._dataframe_backend):
            raise ValueError("One or more columns in the DataFrame are entirely empty (all values are NaN or None).")

        # 2. Validate `time_col` is numeric or timestamp-like
        if not is_numeric(self.df, self._time_col) and not is_timestamp_like(self.df, self._time_col):
            raise TimeColumnError(
                f"`time_col` must be numeric or timestamp-like, found {self.df[self._time_col].dtype}"
            )

        # 3. Validate all non-time columns are numeric
        non_numeric_columns = [col for col in self.df.columns if col != self._time_col and not is_numeric(self.df, col)]
        if non_numeric_columns:
            raise ValueError(
                f"All features except `time_col` must be numeric. Non-numeric columns: {non_numeric_columns}"
            )

        # 4. Check for missing values in `time_col` and `target_col`
        if check_nulls(self.df, self._dataframe_backend):
            raise ValueError("Missing values found in `time_col` or `target_col`.")
