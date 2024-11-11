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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""TemporalScope/src/temporalscope/core/temporal_data_loader.py.

This module provides `TimeFrame`, a flexible, universal data loader for time series forecasting, tailored for
state-of-the-art models, including multi-modal and mixed-frequency approaches. Assuming users employ universal
models without built-in grouping or ID-based segmentation, `TimeFrame` enables integration with custom
preprocessing, loss functions, and explainability techniques such as SHAP, LIME, and Boruta-SHAP. TemporalScope
imposes no engineering constraints, allowing users full flexibility in data preparation and model design.

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

    - **Multi-Step Mode Limitation**: Multi-step mode is currently not implemented due to limitations across
      DataFrames like Modin and Polars, which do not natively support vectorized (sequence-based) targets within a single cell.
      A future interoperability layer is planned to convert multi-step datasets into compatible formats, such as TensorFlow’s
      `tf.data.Dataset`, or to flatten target sequences for these backends.
    - **Single-Step Mode Support**: With Narwhals as the backend-agnostic layer, all Narwhals-supported backends (Pandas, Modin, Polars)
      support single-step mode without requiring additional adjustments, ensuring compatibility with workflows using scalar target variables.
    - **Recommendation**: For current multi-step workflows, Pandas is recommended as it best supports the necessary data structures.
      Future releases will include interoperability enhancements to manage vectorized targets across all supported backends.

.. seealso::

   - Narwhals documentation: https://narwhals.readthedocs.io/
   - SHAP documentation: https://shap.readthedocs.io/
   - Boruta-SHAP documentation: https://github.com/Ekeany/Boruta-Shap
   - LIME documentation: https://lime-ml.readthedocs.io/
"""

from typing import Optional, Union
import narwhals as nw
from narwhals.typing import FrameT
from narwhals.dtypes import DType, Datetime, NumericType


from temporalscope.core.core_utils import (
    validate_backend,
    convert_to_backend,
    SupportedTemporalDataFrame,
    MODE_SINGLE_STEP,
    MODE_MULTI_STEP,
)

from temporalscope.core.exceptions import (
    TimeColumnError,
    ModeValidationError,
)


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
        df: SupportedTemporalDataFrame,
        time_col: str,
        target_col: str,
        dataframe_backend: Optional[str] = None,
        sort: bool = True,
        mode: str = MODE_SINGLE_STEP,
    ) -> None:
        """Initialize a TimeFrame object with required validations and backend handling.

        This constructor validates the provided DataFrame and performs checks on the required columns (`time_col`,
        `target_col`). It also ensures compatibility between the DataFrame and the specified or inferred backend.

        :param df: The input DataFrame, which can be any Narwhals-supported backend (e.g., Pandas, Modin, Polars).
        :type df: SupportedTemporalDataFrame
        :param time_col: The name of the column representing time. Should be numeric or timestamp-like for sorting.
        :type time_col: str
        :param target_col: The column representing the target variable. Must be a valid column in the DataFrame.
        :type target_col: str
        :param dataframe_backend: The backend to use. If provided, the DataFrame will be converted to the appropriate backend.
            If not provided, it will be inferred from the DataFrame type. Supported backends are dynamically validated
            using Narwhals’ `validate_backend`.
        :type dataframe_backend: Optional[str]
        :param sort: If True, the data will be sorted by `time_col` in ascending order. Default is True.
        :type sort: bool
        :param mode: The mode of operation, either `MODE_SINGLE_STEP` or `MODE_MULTI_STEP`. Default is `MODE_SINGLE_STEP`.
        :type mode: str

        :raises ValueError:
            - If `time_col` or `target_col` is not a valid non-empty string.
            - If the input DataFrame is missing required columns or is empty.
            - If the inferred or provided backend is unsupported.
        :raises UnsupportedBackendError:
            - If the specified backend is not supported by TemporalScope.
        :raises TypeError:
            - If the DataFrame type does not match the specified backend.
        :raises ModeValidationError:
            - If the specified mode is not one of the valid modes.

        .. note::
            - The `time_col` must be numeric or timestamp-like to ensure proper temporal ordering.
            - Sorting is automatically performed by `time_col` unless disabled via `sort=False`.
            - If `dataframe_backend` is provided, the DataFrame will be validated and converted as necessary.
            - If `dataframe_backend` is not provided, it will be inferred from the DataFrame type using Narwhals.

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=5, interval="1d"), "value": range(5)})

            tf = TimeFrame(data, time_col="time", target_col="value", mode=MODE_SINGLE_STEP)
            print(tf.get_data().head())
        """
        # Validate mode
        if mode not in [MODE_SINGLE_STEP, MODE_MULTI_STEP]:
            raise ModeValidationError(mode)

        # Store metadata about columns
        self._time_col = time_col
        self._target_col = target_col
        self._mode = mode

        # Validate backend and convert if specified
        if dataframe_backend:
            validate_backend(dataframe_backend)  # Check if the backend is supported
            df = convert_to_backend(df, dataframe_backend)  # type: ignore[arg-type]

        # Call setup method to validate and initialize the DataFrame
        self._df, self._original_backend = self._setup_timeframe(df, sort)

    @nw.narwhalify
    def validate_data(self, df: SupportedTemporalDataFrame) -> None:
        """Run validation checks on the DataFrame to ensure it meets required constraints.

        This method validates the following:
        - The `time_col` is numeric or timestamp-like.
        - All columns, except for `time_col`, are numeric.
        - There are no missing values in critical columns (`time_col`, `target_col`).

        :param df: Input DataFrame.
        :type df: SupportedTemporalDataFrame

        :raises ValueError: If required columns are missing, contain null values, or fail type validation.

        Example Usage:
        --------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=5), "value": [1, 2, 3, 4, 5]})

            tf = TimeFrame(data, time_col="time", target_col="value")
            tf.validate_data(data)
        """
        # Step 1: Validate column existence
        if self._time_col not in df.columns or self._target_col not in df.columns:
            raise TimeColumnError(f"Columns `{self._time_col}` and `{self._target_col}` must exist in the DataFrame.")

        # Step 2: Validate `time_col` type
        if not isinstance(df[self._time_col].dtype, (Datetime, NumericType)):
            raise TimeColumnError(
                f"`time_col` must be numeric or timestamp-like. Found type: {df[self._time_col].dtype}."
            )

        # Step 3: Validate non-time columns
        non_time_cols = [col for col in df.columns if col != self._time_col]
        non_numeric_cols = [col for col in non_time_cols if not isinstance(df[col].dtype, NumericType)]
        if non_numeric_cols:
            raise ValueError(
                f"All columns except `{self._time_col}` must be numeric. Non-numeric columns: {non_numeric_cols}."
            )

        # Step 4: Check for missing values
        if df.select([self._time_col, self._target_col]).null_count().gt(0).any():
            raise ValueError(f"Missing values detected in `{self._time_col}` or `{self._target_col}`.")

    @nw.narwhalify
    def _setup_timeframe(
        self, df: SupportedTemporalDataFrame, sort: bool = True
    ) -> tuple[SupportedTemporalDataFrame, str]:
        """Set up and validate the DataFrame.

        This method ensures the DataFrame is compatible with TemporalScope requirements, validates critical columns,
        and sorts the DataFrame if needed. It uses Narwhals to abstract backend-specific operations and
        determines the original backend.

        :param df: The input DataFrame.
        :type df: SupportedTemporalDataFrame
        :param sort: Whether to sort the DataFrame by the `time_col`. Default is True.
        :type sort: bool

        :return: A tuple containing the validated and sorted DataFrame and the original backend.
        :rtype: tuple[SupportedTemporalDataFrame, str]

        Example Usage:
        --------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=5), "value": [1, 2, 3, 4, 5]})

            tf = TimeFrame(data, time_col="time", target_col="value")
            validated_df, backend = tf._setup_timeframe(data)
        """
        # Step 1: Validate the DataFrame
        self.validate_data(df)

        # Step 2: Sort DataFrame if required
        if sort:
            df = df.sort(by=self._time_col)

        # Step 3: Determine backend using Narwhals' `.backend` attribute
        original_backend = getattr(df, 'backend', 'pandas')

        return df, original_backend

    @nw.narwhalify
    def update_data(
        self,
        df: SupportedTemporalDataFrame,
        new_target_col: Optional[SupportedTemporalDataFrame] = None,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        sort: bool = True,
    ) -> None:
        """Update the DataFrame and its columns with new data.

        This method updates the internal DataFrame and its associated metadata (e.g., column names). It validates the data,
        updates the DataFrame, and optionally sorts it.

        :param df: Input DataFrame to update.
        :type df: SupportedTemporalDataFrame
        :param new_target_col: New target column to replace the existing one. Optional.
        :type new_target_col: SupportedTemporalDataFrame, optional
        :param time_col: New time column name. Optional.
        :type time_col: str, optional
        :param target_col: New target column name. Optional.
        :type target_col: str, optional
        :param sort: Whether to sort the DataFrame. Default is True.
        :type sort: bool

        :return: None

        Example Usage:
        --------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=5), "value": [1, 2, 3, 4, 5]})

            tf = TimeFrame(data, time_col="time", target_col="value")

            # Update the DataFrame
            tf.update_data(data, time_col="new_time", target_col="new_target")
        """
        # Step 1: Update column names if provided
        if time_col:
            self._time_col = time_col
        if target_col:
            self._target_col = target_col

        # Step 2: Replace target column if provided
        if new_target_col is not None:
            if len(new_target_col) != len(df):
                raise ValueError("The new target column must have the same number of rows as the existing DataFrame.")
            df[self._target_col] = new_target_col

        # Step 3: Set up and validate the DataFrame
        self._df, self._original_backend = self._setup_timeframe(df, sort=sort)

    @nw.narwhalify
    def sort_data(self, df: SupportedTemporalDataFrame, ascending: bool = True) -> SupportedTemporalDataFrame:
        """Sort the DataFrame by the time column.

        :param df: Input DataFrame.
        :type df: SupportedTemporalDataFrame
        :param ascending: If True, sort in ascending order; if False, sort in descending order.
        :type ascending: bool
        :return: The sorted DataFrame.
        :rtype: SupportedTemporalDataFrame

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create sample data
            data = {"time": pd.date_range(start="2021-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
            df = pd.DataFrame(data)

            # Create TimeFrame instance
            tf = TimeFrame(df, time_col="time", target_col="value")

            # Sort the data
            sorted_df = tf.sort_data(tf.df, ascending=True)
        """
        return df.sort(by=[self._time_col], descending=not ascending)  # Reverse is opposite of ascending


    @property
    def df(self) -> SupportedTemporalDataFrame:
        """Return the DataFrame in its current state.

        :return: The DataFrame managed by the TimeFrame instance.
        :rtype: SupportedTemporalDataFrame

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

            # Access the DataFrame directly
            print(tf.df.head())
        """
        return self._df

    @property
    def mode(self) -> str:
        """Return the mode of the TimeFrame instance.

        :return: The mode of operation, either `MODE_SINGLE_STEP` or `MODE_MULTI_STEP`.
        :rtype: str

        Example Usage:
        --------------
        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            import pandas as pd

            # Create a Pandas DataFrame
            data = {"time": pd.date_range(start="2021-01-01", periods=5, freq="D"), "target": range(5, 0, -1)}
            df = pd.DataFrame(data)

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col="time", target_col="target", mode=MODE_SINGLE_STEP)

            # Access the mode directly
            print(tf.mode)
        """
        return self._mode

    @property
    def backend(self) -> str:
        """Return the backend of the TimeFrame instance.

        :return: The backend of the DataFrame, either specified or inferred.
        :rtype: str

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

            # Access the backend directly
            print(tf.backend)
        """
        return self._original_backend
