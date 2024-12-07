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

This module provides `TimeFrame`, a flexible, universal data loader for time series forecasting, tailored for
state-of-the-art models, including multi-modal and mixed-frequency approaches. Assuming users employ universal
models without built-in grouping or ID-based segmentation, `TimeFrame` enables integration with custom
preprocessing, loss functions, and explainability techniques such as SHAP, LIME, and Boruta-SHAP. TemporalScope
imposes no engineering constraints, allowing users full flexibility in data preparation and model design.

Engineering Design
--------------------

TemporalScope is designed with several key assumptions to ensure performance, scalability, and flexibility
across a wide range of time series forecasting and XAI workflows:

+------------------------+---------------------------------------------------+
| Approach               | Description                                       |
+------------------------+---------------------------------------------------+
| Implicit & Static Time | Time column is treated as a feature for static    |
| Series                 | modeling with ML/DL. Mixed-frequency workflows    |
|                        | are supported. `strict_temporal_order` is False.  |
+------------------------+---------------------------------------------------+
| Strict Time Series     | Temporal ordering and uniqueness are enforced,    |
|                        | suitable for forecasting tasks. Grouped or        |
|                        | segmented validation can be done using `id_col`.  |
+------------------------+---------------------------------------------------+

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

+--------------------+----------------------------------------------------+--------------------------------------------+
| Mode               | Description                                        | Compatible Frameworks                      |
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

    3. Trirat, P., Shin, Y., Kang, J., Nam, Y., Bae, M., Kim, J., Kim, B., &
       Lee, J.-G. (2024). Universal time-series representation learning: A survey.
       arXiv preprint arXiv:2401.03717.

    4. Xu, Q., Zhuo, X., Jiang, C., & Liu, Y. (2019). An artificial neural network
       for mixed frequency data. Expert Systems with Applications, 118, pp.127-139.

    5. Filho, L.L., de Oliveira Werneck, R., Castro, M., Ribeiro Mendes Júnior, P.,
       Lustosa, A., Zampieri, M., Linares, O., Moura, R., Morais, E., Amaral, M., &
       Salavati, S. (2024). A multi-modal approach for mixed-frequency time series
       forecasting. Neural Computing and Applications, pp.1-25.

.. note::

    - Multi-Step Mode Limitation: Multi-step mode is currently not implemented due to limitations across
      DataFrames like Modin and Polars, which do not natively support vectorized (sequence-based) targets within a single cell.
      A future interoperability layer is planned to convert multi-step datasets into compatible formats, such as TensorFlow's
      `tf.data.Dataset`, or to flatten target sequences for these backends.
    - Single-Step Mode Support: With Narwhals as the backend-agnostic layer, all Narwhals-supported backends (Pandas, Modin, Polars)
      support single-step mode without requiring additional adjustments, ensuring compatibility with workflows using scalar target variables.
    - Recommendation: For current multi-step workflows, Pandas is recommended as it best supports the necessary data structures.
      Future releases will include interoperability enhancements to manage vectorized targets across all supported backends.

.. seealso::

   - Narwhals documentation: https://narwhals.readthedocs.io/
   - SHAP documentation: https://shap.readthedocs.io/
   - Boruta-SHAP documentation: https://github.com/Ekeany/Boruta-Shap
   - LIME documentation: https://lime-ml.readthedocs.io/
"""

from typing import Any, Dict, Optional

import narwhals as nw

# Import necessary types and utilities from the Narwhals library for handling temporal data
from temporalscope.core.core_utils import (
    MODE_SINGLE_TARGET,
    VALID_MODES,
    SupportedTemporalDataFrame,
    check_dataframe_nulls_nans,
    check_strict_temporal_ordering,
    get_dataframe_backend,
    is_valid_temporal_backend,
    is_valid_temporal_dataframe,
    sort_dataframe_time,
    validate_dataframe_column_types,
)
from temporalscope.core.exceptions import UnsupportedBackendError


class TimeFrame:
    """Central class for the TemporalScope package.

    The `TimeFrame` class is designed to handle time series data across various backends, including Polars, Pandas,
    and Modin. It facilitates workflows for machine learning, deep learning, and explainability methods, while abstracting
    away backend-specific implementation details.

    This class automatically infers the appropriate backend, validates the data, and sorts it by time. It ensures
    compatibility with temporal XAI techniques (SHAP, Boruta-SHAP, LIME etc) supporting larger data workflows in
    production.

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

    Backend Handling
    ----------------
    - If a `dataframe_backend` is explicitly provided, it takes precedence over backend inference.
    - If no backend is specified, the class infers the backend from the DataFrame type, supporting Polars, Pandas, and Modin.

    Example Usage
    -------------
    .. code-block:: python

       import polars as pl

       data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=100, interval="1d"), "value": range(100)})
       tf = TimeFrame(data, time_col="time", target_col="value")
       print(tf.get_data().head())
    """

    def __init__(
        self,
        df: SupportedTemporalDataFrame,
        time_col: str,
        target_col: str,
        dataframe_backend: Optional[str] = None,
        sort: bool = True,
        ascending: bool = True,
        mode: str = MODE_SINGLE_TARGET,
        time_col_conversion: Optional[str] = None,
        strict_temporal_order: bool = False,  # New parameter
        id_col: Optional[str] = None,  # New parameter
        verbose: bool = False,
    ) -> None:
        """Initialize a TimeFrame object with required validations, backend handling, and optional time column conversion.

        This constructor initializes the `TimeFrame` object, validates the input DataFrame, and performs optional
        sorting based on the specified `time_col`. It also allows for validation and conversion of the `time_col`
        to ensure compatibility with downstream processing. Designed for universal workflows supporting
        state-of-the-art AI models, this class accommodates mixed-frequency time series data.

        There are two common use cases for `TimeFrame`:
        1. Implicit & Static Time Series: For workflows where `time_col` is treated as a feature, such as in
        static modeling for ML/DL applications, `strict_temporal_order` can remain `False` (default). This
        mode emphasizes a universal design, accommodating mixed-frequency data.
        2. Strict Time Series: For workflows requiring strict temporal ordering and uniqueness (e.g., forecasting),
        set `strict_temporal_order=True`. Additionally, specify `id_col` for grouped or segmented validation.

        :param df: The input DataFrame, which can be any TemporalScope-supported backend (e.g., Pandas, Modin, Polars).
        :type df: SupportedTemporalDataFrame
        :param time_col: The name of the column representing time. Must be numeric or timestamp-like for sorting.
        :type time_col: str
        :param target_col: The name of the column representing the target variable.
        :type target_col: str
        :param dataframe_backend: The backend to use. If not provided, it will be inferred based on the DataFrame type.
        :type dataframe_backend: Optional[str]
        :param sort: If True, the data will be sorted by `time_col`. Default is True.
        :type sort: bool
        :param ascending: If sorting, whether to sort in ascending order. Default is True.
        :type ascending: bool
        :param mode: The operation mode, either `MODE_SINGLE_TARGET` (default) or `MODE_MULTI_TARGET`.
        :type mode: str
        :param time_col_conversion: Optional. Specify the conversion type for the `time_col`:
                                    - 'numeric': Convert to Float64.
                                    - 'datetime': Convert to Datetime.
                                    - None: Validate only.
                                    Default is None.
        :type time_col_conversion: Optional[str]
        :param strict_temporal_order: If True, enforces strict temporal ordering and uniqueness validation. Default is False.
        :type strict_temporal_order: bool
        :param id_col: Optional column for grouped or segmented strict temporal validation. Default is None.
        :type id_col: Optional[str]
        :param verbose: If True, enables logging for validation and setup stages.
        :type verbose: bool
        :raises ModeValidationError: If the specified mode is invalid.
        :raises UnsupportedBackendError: If the specified or inferred backend is not supported.
        :raises ValueError: If required columns are missing, invalid, or if the time column conversion fails.

        :ivar _metadata: A private metadata dictionary to allow end-users flexibility in extending the TimeFrame object.
                        This provides storage for any additional attributes or information during runtime.
        :type _metadata: Dict[str, Any]

        Example Usage:
        --------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.core.temporal_data_loader import TimeFrame, MODE_SINGLE_TARGET

            # Example DataFrame
            df = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=10, freq="D"), "value": range(10)})

            # Initialize TimeFrame with automatic time column conversion to numeric
            tf = TimeFrame(df, time_col="time", target_col="value", time_col_conversion="numeric", mode=MODE_SINGLE_TARGET)
            print(tf.df.head())

        .. note::
            - The `mode` parameter must be one of:
                - `"single_target"`: For scalar target predictions (e.g., regression).
                - `"multi_target"`: For sequence forecasting tasks (e.g., seq2seq models).
            - The `time_col_conversion` parameter allows for automatic conversion of the `time_col` to either numeric
            or datetime during initialization.
            - The `_metadata` container follows design patterns similar to SB3, enabling users to manage custom attributes
            and extend functionality for advanced workflows, such as future conversion to TensorFlow or PyTorch types
            in multi-target explainable AI workflows.
        """
        self._time_col = time_col
        self._target_col = target_col
        self._mode = mode
        self._ascending = ascending
        self._sort = sort
        self._verbose = verbose
        self._strict_temporal_order = strict_temporal_order
        self._id_col = id_col
        self._metadata: Dict[str, Any] = {}

        # Validate parameters
        self._validate_parameters(
            time_col, target_col, dataframe_backend, sort, ascending, verbose, time_col_conversion, id_col
        )

        # Initialize backend
        self._backend = self._initialize_backend(df, dataframe_backend)

        # Setup DataFrame
        self._df = self.setup(df, sort=self._sort, ascending=self._ascending)

        if self._verbose:
            print(f"TimeFrame successfully initialized with backend: {self._backend}")

    def _validate_parameters(
        self,
        time_col: str,
        target_col: str,
        dataframe_backend: Optional[str],
        sort: bool,
        ascending: bool,
        verbose: bool,
        time_col_conversion: Optional[str],
        id_col: Optional[str],
    ) -> None:
        """Validate input parameters for the TimeFrame initialization.

        :param time_col: Name of the time column.
        :type time_col: str
        :param target_col: Name of the target column.
        :type target_col: str
        :param dataframe_backend: Backend to use for the DataFrame. Default is None.
        :type dataframe_backend: Optional[str]
        :param sort: Indicates whether the DataFrame should be sorted. Default is True.
        :type sort: bool
        :param ascending: Indicates sorting direction if `sort` is enabled. Default is True.
        :type ascending: bool
        :param verbose: Enables logging during initialization if True. Default is False.
        :type verbose: bool
        :param time_col_conversion: Conversion type for `time_col`: 'numeric', 'datetime', or None.
        :type time_col_conversion: Optional[str]
        :param id_col: Column for grouped validation. Default is None.
        :type id_col: Optional[str]
        :raises TypeError: If any parameter has an invalid type.
        :raises ValueError: If a parameter value is invalid or unsupported.
        """
        if not isinstance(time_col, str):
            raise TypeError(f"`time_col` must be a string. Got {type(time_col).__name__}.")
        if not isinstance(target_col, str):
            raise TypeError(f"`target_col` must be a string. Got {type(target_col).__name__}.")
        if dataframe_backend is not None and not isinstance(dataframe_backend, str):
            raise TypeError(f"`dataframe_backend` must be a string or None. Got {type(dataframe_backend).__name__}.")
        if not isinstance(sort, bool):
            raise TypeError(f"`sort` must be a boolean. Got {type(sort).__name__}.")
        if not isinstance(ascending, bool):
            raise TypeError(f"`ascending` must be a boolean. Got {type(ascending).__name__}.")
        if not isinstance(verbose, bool):
            raise TypeError(f"`verbose` must be a boolean. Got {type(verbose).__name__}.")
        if id_col is not None and not isinstance(id_col, str):
            raise TypeError(f"`id_col` must be a string or None. Got {type(id_col).__name__}.")
        if time_col_conversion not in {None, "numeric", "datetime"}:
            raise ValueError(
                f"Invalid `time_col_conversion` value '{time_col_conversion}'. "
                f"Must be one of {{None, 'numeric', 'datetime'}}."
            )
        if self._mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self._mode}'. Must be one of {VALID_MODES}.")

    def _validate_parameters(
        self,
        time_col: str,
        target_col: str,
        dataframe_backend: Optional[str],
        sort: bool,
        ascending: bool,
        verbose: bool,
        time_col_conversion: Optional[str],
        id_col: Optional[str],
    ) -> None:
        """Validate input parameters for the TimeFrame initialization.

        :param time_col: Name of the time column.
        :type time_col: str
        :param target_col: Name of the target column.
        :type target_col: str
        :param dataframe_backend: Backend to use for the DataFrame. Default is None.
        :type dataframe_backend: Optional[str]
        :param sort: Indicates whether the DataFrame should be sorted. Default is True.
        :type sort: bool
        :param ascending: Indicates sorting direction if `sort` is enabled. Default is True.
        :type ascending: bool
        :param verbose: Enables logging during initialization if True. Default is False.
        :type verbose: bool
        :param time_col_conversion: Conversion type for `time_col`: 'numeric', 'datetime', or None.
        :type time_col_conversion: Optional[str]
        :param id_col: Column for grouped validation. Default is None.
        :type id_col: Optional[str]
        :raises TypeError: If any parameter has an invalid type.
        :raises ValueError: If a parameter value is invalid or unsupported.
        """
        if not isinstance(time_col, str):
            raise TypeError(f"`time_col` must be a string. Got {type(time_col).__name__}.")
        if not isinstance(target_col, str):
            raise TypeError(f"`target_col` must be a string. Got {type(target_col).__name__}.")
        if dataframe_backend is not None and not isinstance(dataframe_backend, str):
            raise TypeError(f"`dataframe_backend` must be a string or None. Got {type(dataframe_backend).__name__}.")
        if not isinstance(sort, bool):
            raise TypeError(f"`sort` must be a boolean. Got {type(sort).__name__}.")
        if not isinstance(ascending, bool):
            raise TypeError(f"`ascending` must be a boolean. Got {type(ascending).__name__}.")
        if not isinstance(verbose, bool):
            raise TypeError(f"`verbose` must be a boolean. Got {type(verbose).__name__}.")
        if id_col is not None and not isinstance(id_col, str):
            raise TypeError(f"`id_col` must be a string or None. Got {type(id_col).__name__}.")
        if time_col_conversion not in {None, "numeric", "datetime"}:
            raise ValueError(
                f"Invalid `time_col_conversion` value '{time_col_conversion}'. "
                f"Must be one of {{None, 'numeric', 'datetime'}}."
            )
        if self._mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self._mode}'. Must be one of {VALID_MODES}.")

    def _initialize_backend(self, df: SupportedTemporalDataFrame, dataframe_backend: Optional[str]) -> str:
        """Determine and validate the backend for the DataFrame.

        :param df: Input DataFrame to initialize the backend.
        :type df: SupportedTemporalDataFrame
        :param dataframe_backend: Backend to use. If None, it is inferred from the DataFrame type.
        :type dataframe_backend: Optional[str]
        :return: Initialized backend for the DataFrame.
        :rtype: str
        :raises UnsupportedBackendError: If the backend is invalid or unsupported.
        """
        if dataframe_backend:
            is_valid_temporal_backend(dataframe_backend)
            return dataframe_backend

        is_valid, _ = is_valid_temporal_dataframe(df)
        if not is_valid:
            raise UnsupportedBackendError(f"Unsupported DataFrame type: {type(df).__name__}")
        return get_dataframe_backend(df)

    @nw.narwhalify
    def sort_dataframe_time(self, df: SupportedTemporalDataFrame, ascending: bool = True) -> SupportedTemporalDataFrame:
        """Sort DataFrame by time column using backend-agnostic Narwhals operations.

        :param df: DataFrame to sort
        :type df: SupportedTemporalDataFrame
        :param ascending: Sort direction, defaults to True
        :type ascending: bool
        :return: Sorted DataFrame
        :rtype: SupportedTemporalDataFrame

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pl.DataFrame({"time": [3, 1, 4, 2, 5], "target": range(5)})
            tf = TimeFrame(data, time_col="time", target_col="target", sort=False)
            sorted_df = tf.sort_dataframe_time(tf.df, ascending=True)
            print(sorted_df)  # Shows data sorted by time column

        .. note::
            Uses the reusable utility function `sort_dataframe_time` for consistency across the codebase.
        """
        return sort_dataframe_time(df, time_col=self._time_col, ascending=ascending)

    @nw.narwhalify
    def validate_dataframe(self, df: SupportedTemporalDataFrame) -> None:
        """Run streamlined validation checks on the DataFrame to ensure it meets required constraints.

        This function validates the DataFrame by:
        1. Ensuring all columns are non-null and contain no NaNs.
        2. Validating the `time_col` and ensuring all non-time columns are numeric.

        The method is designed to support backend-agnostic operations through Narwhals and handles
        different DataFrame backends such as Pandas, Polars, and Modin.

        :param df: Input DataFrame to validate.
        :type df: SupportedTemporalDataFrame
        :raises ValueError: If any columns contain nulls/NaNs or invalid data types.
        :raises UnsupportedBackendError: If the backend is not supported.

        Example Usage:
        --------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.core.temporal_data_loader import TimeFrame

            # Sample DataFrame
            df = pd.DataFrame(
                {
                    "time": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                    "value": range(5),
                }
            )

            # Initialize a TimeFrame object
            tf = TimeFrame(df, time_col="time", target_col="value")

            # Validate the DataFrame
            tf.validate_dataframe(df)

        .. note::
            - This function ensures that `time_col` is valid and optionally convertible.
            - All other columns must be numeric and free from null values.
        """
        # Step 1: Ensure all columns are free of nulls and NaNs
        null_counts = check_dataframe_nulls_nans(df, df.columns)
        null_columns = [col for col, count in null_counts.items() if count > 0]
        if null_columns:
            raise ValueError(f"Missing values detected in columns: {', '.join(null_columns)}")

        # Step 2: Validate column types (time_col and others)
        validate_dataframe_column_types(df, self._time_col)

        # If verbose, notify the user about validation success
        if self._verbose:
            print("Validation completed successfully.")

    @nw.narwhalify
    def setup(
        self, df: SupportedTemporalDataFrame, sort: bool = True, ascending: bool = True
    ) -> SupportedTemporalDataFrame:
        """Initialize and validate a TimeFrame's DataFrame with proper sorting and validation.

        This method performs the necessary validation and sorting operations to prepare
        the input DataFrame for use in TemporalScope workflows. The method is idempotent.

        :param df: Input DataFrame to set up and validate.
        :type df: SupportedTemporalDataFrame
        :param sort: Whether to sort the DataFrame by `time_col`. Defaults to True.
        :type sort: bool
        :param ascending: Sort order if sorting is enabled. Defaults to True.
        :type ascending: bool
        :return: Validated and optionally sorted DataFrame.
        :rtype: SupportedTemporalDataFrame

        Steps:
        ------
        1. Validate the input DataFrame using the `validate_dataframe` method.
        2. Perform strict temporal ordering validation if `strict_temporal_order` is enabled.
        3. Optionally sort the DataFrame by `time_col` in the specified order.

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            # Create example data
            data = pl.DataFrame({"time": [3, 1, 4, 2, 5], "value": range(5)})

            # Initialize a TimeFrame object
            tf = TimeFrame(data, time_col="time", target_col="value", sort=False)

            # Validate and sort the DataFrame
            sorted_df = tf.setup(data, sort=True, ascending=True)
            print(sorted_df)

        .. note::
            - This method is designed to be idempotent, ensuring safe revalidation or reinitialization.
            - Sorting is performed only if explicitly enabled via the `sort` parameter.
            - While this method validates and sorts the DataFrame, it does not modify the TimeFrame's internal state
            unless explicitly used within another method (e.g., `update_dataframe`).
        """
        # Step 1: Validate the DataFrame using validate_dataframe
        self.validate_dataframe(df)

        # Step 2: Perform strict temporal ordering validation if enabled
        if self._strict_temporal_order:
            check_strict_temporal_ordering(
                df=df,
                time_col=self._time_col,
                id_col=self._id_col,
                raise_error=True,  # Ensures invalid data raises an error
            )

        # Step 3: Sort DataFrame by time column if enabled
        if sort:
            df = self.sort_dataframe_time(df, ascending=ascending)

        return df

    @nw.narwhalify
    def update_dataframe(self, df: SupportedTemporalDataFrame) -> None:
        """Update TimeFrame's internal DataFrame with new data.

        Whilst TemporalScope target shifter and padding functions are available, the user must
        handle any data transformations (e.g., changing target columns) within the TimeFrame
        workflow and ensure that they handle pre-processing to be compatible with
        downstream tasks.

        :param df: New DataFrame to use
        :type df: SupportedTemporalDataFrame

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            # Initial TimeFrame setup
            data = pl.DataFrame(
                {
                    "time": pl.date_range(start="2021-01-01", periods=5, interval="1d"),
                    "target": range(5),
                    "feature": range(5),
                }
            )
            tf = TimeFrame(
                data,
                time_col="time",
                target_col="target",
                ascending=True,  # Sort order set at initialization
                sort=True,  # Sort behavior set at initialization
            )

            # Update with new data - uses parameters from initialization
            new_data = pl.DataFrame(
                {
                    "time": pl.date_range(start="2021-01-06", periods=5, interval="1d"),
                    "target": range(5, 10),
                    "feature": range(5, 10),
                }
            )
            tf.update_dataframe(new_data)  # Will use time_col="time", ascending=True, sort=True

        .. note::
            This method uses the parameters set during TimeFrame initialization:
            - Uses the same time_col and target_col
            - Maintains the same sort order (ascending/descending)
            - Keeps the same sorting behavior (enabled/disabled)

            If you need to change these parameters, create a new TimeFrame instance
            with the desired configuration.

        .. seealso::
            - :class:`temporalscope.target_shifters.single_step.SingleStepTargetShifter`
            - :class:`temporalscope.partition.padding.functional`
            For handling target transformations and padding operations.
        """
        self._df = self.setup(df, sort=True, ascending=self._ascending)

    @property
    def df(self) -> SupportedTemporalDataFrame:
        """Return the DataFrame in its current state.

        :return: The DataFrame managed by the TimeFrame instance.
        :rtype: SupportedTemporalDataFrame
        """
        return self._df

    @property
    def mode(self) -> str:
        """Return the mode of the TimeFrame instance.

        :return: The mode of operation, either `MODE_SINGLE_TARGET` or `MODE_MULTI_TARGET`.
        :rtype: str
        """
        return self._mode

    @property
    def backend(self) -> str:
        """Return the backend of the TimeFrame instance.

        :return: The backend of the DataFrame, either specified or inferred.
        :rtype: str
        """
        return self._backend

    @property
    def ascending(self) -> bool:
        """Return the sort order of the TimeFrame instance.

        :return: The sort order, True if ascending, False if descending.
        :rtype: bool
        """
        return self._ascending

    @property
    def metadata(self) -> Dict[str, Any]:
        """Container for storing additional metadata associated with the TimeFrame.

        This property provides a flexible storage mechanism for arbitrary metadata
        related to the TimeFrame, such as configuration details, additional
        annotations, or external data structures. It is designed to support future
        extensions, including multi-target workflows and integration with deep
        learning libraries like TensorFlow or PyTorch.

        Example Usage:
        --------------
        .. code-block:: python

            # Initialize a TimeFrame
            tf = TimeFrame(df, time_col="time", target_col="value")

            # Add custom metadata
            tf.metadata["description"] = "This dataset is for monthly sales forecasting"
            tf.metadata["model_details"] = {"type": "LSTM", "framework": "TensorFlow"}

            # Access metadata
            print(tf.metadata["description"])  # Output: "This dataset is for monthly sales forecasting"

        .. note::
            - This metadata container is designed following patterns seen in deep reinforcement
            learning (DRL) libraries like Stable-Baselines3, where additional metadata is
            stored alongside primary data structures for extensibility.
            - In future releases, this will support multi-target workflows, enabling the storage
            of processed tensor data for deep learning explainability (e.g., SHAP, LIME).

        :return: Dictionary for storing metadata related to the TimeFrame.
        :rtype: Dict[str, Any]
        """
        return self._metadata
