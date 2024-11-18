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

    - **Multi-Step Mode Limitation**: Multi-step mode is currently not implemented due to limitations across
      DataFrames like Modin and Polars, which do not natively support vectorized (sequence-based) targets within a single cell.
      A future interoperability layer is planned to convert multi-step datasets into compatible formats, such as TensorFlow's
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

from datetime import datetime
from typing import Dict, List, Optional

import narwhals as nw
import pandas as pd

# Import necessary types and utilities from the Narwhals library for handling temporal data
from narwhals.dtypes import Float64

from temporalscope.core.core_utils import (
    MODE_MULTI_STEP,
    MODE_SINGLE_STEP,
    SupportedTemporalDataFrame,
    convert_to_backend,
    get_dataframe_backend,
    is_valid_temporal_backend,
    is_valid_temporal_dataframe,
)
from temporalscope.core.exceptions import ModeValidationError, TimeColumnError, UnsupportedBackendError


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
    """

    def __init__(
        self,
        df: SupportedTemporalDataFrame,
        time_col: str,
        target_col: str,
        dataframe_backend: Optional[str] = None,
        sort: bool = True,
        ascending: bool = True,
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
                                using Narwhals' `is_valid_temporal_backend`.
        :type dataframe_backend: Optional[str]
        :param sort: If True, the data will be sorted by `time_col`. Default is True.
        :type sort: bool
        :param ascending: If sorting, whether to sort in ascending order. Default is True.
        :type ascending: bool
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
        # Validate mode first
        if mode not in [MODE_SINGLE_STEP, MODE_MULTI_STEP]:
            raise ModeValidationError(mode)

        # Store metadata about columns
        self._time_col = time_col
        self._target_col = target_col
        self._mode = mode
        self._ascending = ascending
        self._sort = sort

        # Validate optional backend string first if provided
        if dataframe_backend is not None:
            is_valid_temporal_backend(dataframe_backend)

        # Then validate DataFrame type
        is_valid, df_type = is_valid_temporal_dataframe(df)
        if not is_valid:
            raise UnsupportedBackendError(f"Unknown DataFrame type: {type(df).__name__}")

        # Handle backend conversion and inference
        if dataframe_backend:
            df = convert_to_backend(df, dataframe_backend)
            self._original_backend = dataframe_backend
        else:
            self._original_backend = get_dataframe_backend(df)

        # Call setup method to validate and initialize the DataFrame
        self._df = self._setup_timeframe(df, sort=sort, ascending=ascending)

    @nw.narwhalify
    def _check_nulls(self, df: SupportedTemporalDataFrame, column_names: List[str]) -> Dict[str, int]:
        """Check for null values in specified DataFrame columns using Narwhals operations.

        Performs backend-agnostic null checking for every specified column, handling both
        lazy and eager evaluation patterns.

        :param df: DataFrame to check for null values
        :type df: SupportedTemporalDataFrame
        :param column_names: List of column names to check
        :type column_names: List[str]
        :return: Dictionary mapping column names to their null value counts
        :rtype: Dict[str, int]
        """
        result = {}
        for col in column_names:
            # Create expression for null check
            null_expr = nw.col(col).is_null().sum().alias(f"{col}_nulls")

            # Execute null check and collect result
            null_count = df.select([null_expr])
            if hasattr(null_count, "collect"):
                null_count = null_count.collect()

            # Convert to pandas for consistent handling
            if hasattr(null_count, "to_pandas"):
                null_count = null_count.to_pandas()

            # Extract the count value
            count = null_count[f"{col}_nulls"].iloc[0]
            result[col] = int(count)

        return result

    @nw.narwhalify
    def validate_data(self, df: SupportedTemporalDataFrame) -> None:
        """Run validation checks on the DataFrame to ensure it meets required constraints.

        This function performs comprehensive validation of the input DataFrame, ensuring:
        1. Required columns exist
        2. No null values are present
        3. All non-time columns are numeric
        4. Time column is either numeric, datetime, or convertible to datetime

        :param df: Input DataFrame to validate
        :type df: SupportedTemporalDataFrame
        :raises TimeColumnError: If required columns are missing or time column has invalid type
        :raises ValueError: If non-time columns contain nulls or non-numeric values

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pl.DataFrame(
                {
                    "time": pl.date_range(start="2021-01-01", periods=5, interval="1d"),
                    "target": range(5),
                    "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
            tf = TimeFrame(data, time_col="time", target_col="target")
            tf.validate_data(data)  # Validates data meets all requirements

        .. note::
            This method uses Narwhals operations for backend-agnostic validation:
            - Uses nw.col() for column references
            - Handles lazy evaluation through collect()
            - Performs type validation consistently across backends
        """
        # Step 1: Check for required columns
        if self._time_col not in df.columns or self._target_col not in df.columns:
            raise TimeColumnError(f"Columns `{self._time_col}` and `{self._target_col}` must exist in the DataFrame.")

        # Step 2: Ensure all columns do not have any nulls
        null_counts = self._check_nulls(df, df.columns)
        null_columns = [col for col, count in null_counts.items() if count > 0]
        if null_columns:
            raise ValueError(f"Missing values detected in columns: {', '.join(null_columns)}")

        # Step 3: Ensure all columns are numeric except time_col
        for col in df.columns:
            if col != self._time_col:
                try:
                    # Cast using Narwhals for backend-agnostic operation
                    numeric_check = df.select([nw.col(col).cast(Float64)])
                    if hasattr(numeric_check, "collect"):
                        numeric_check = numeric_check.collect()
                except Exception as e:
                    raise ValueError(f"Column '{col}' must be numeric. Error: {str(e)}")

        # Step 4: Get time column values for validation
        time_values = df.select([nw.col(self._time_col)])
        if hasattr(time_values, "collect"):
            time_values = time_values.collect()
        if hasattr(time_values, "to_pandas"):
            time_values = time_values.to_pandas()

        time_value = time_values.iloc[0, 0]

        # Type validation for time column
        if isinstance(time_value, (int, float)):
            return
        if isinstance(time_value, (datetime, pd.Timestamp)):
            return
        if isinstance(time_value, str):
            try:
                pd.to_datetime(time_value)
                return
            except (ValueError, TypeError):
                pass

        raise TimeColumnError(
            f"time_col must be numeric, datetime, or a valid datetime string. Found type: {type(time_value)}"
        )

    @nw.narwhalify
    def _setup_timeframe(
        self, df: SupportedTemporalDataFrame, sort: bool = True, ascending: bool = True
    ) -> SupportedTemporalDataFrame:
        """Initialize and validate a TimeFrame's DataFrame with proper sorting and validation.

        :param df: Input DataFrame to set up and validate
        :type df: SupportedTemporalDataFrame
        :param sort: Whether to sort by time_col, defaults to True
        :type sort: bool
        :param ascending: Sort order if sorting enabled, defaults to True
        :type ascending: bool
        :return: Validated and optionally sorted DataFrame
        :rtype: SupportedTemporalDataFrame
        """
        # Step 1: Validate the DataFrame using validate_data
        self.validate_data(df)

        # Step 2: Sort DataFrame by time column if enabled
        if sort:
            df = self.sort_data(df, ascending=ascending)

        return df

    @nw.narwhalify
    def sort_data(self, df: SupportedTemporalDataFrame, ascending: bool = True) -> SupportedTemporalDataFrame:
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
            sorted_df = tf.sort_data(tf.df, ascending=True)
            print(sorted_df)  # Shows data sorted by time column

        .. note::
            This method uses Narwhals operations for backend-agnostic sorting:
            - Uses nw.col() for column references
            - Handles lazy evaluation through collect()
            - Works consistently across all supported backends
        """
        sorted_df = df.sort(by=[self._time_col], descending=not ascending)

        # Handle lazy evaluation
        if hasattr(sorted_df, "collect"):
            sorted_df = sorted_df.collect()

        return sorted_df

    @nw.narwhalify
    def update_data(
        self,
        df: SupportedTemporalDataFrame,
        new_target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
        sort: bool = True,
    ) -> None:
        """Update TimeFrame's internal DataFrame with new data or column configurations.

        :param df: New DataFrame to use
        :type df: SupportedTemporalDataFrame
        :param new_target_col: New column to use as target, defaults to None
        :type new_target_col: Optional[str]
        :param time_col: New time column name, defaults to None
        :type time_col: Optional[str]
        :param target_col: New target column name, defaults to None
        :type target_col: Optional[str]
        :param sort: Whether to sort the new data, defaults to True
        :type sort: bool

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            # Initial data
            data = pl.DataFrame(
                {
                    "time": pl.date_range(start="2021-01-01", periods=5, interval="1d"),
                    "target": range(5),
                    "new_target": [x * 2 for x in range(5)],
                }
            )
            tf = TimeFrame(data, time_col="time", target_col="target")

            # Update with new target column
            tf.update_data(data, new_target_col="new_target")

        .. note::
            This method uses Narwhals operations for backend-agnostic updates:
            - Uses nw.col() for column references
            - Validates data after updates
            - Maintains sorting if enabled
        """
        # Step 1: Update column names if provided
        if time_col:
            self._time_col = time_col
        if target_col:
            self._target_col = target_col

        # Step 2: Replace target column if provided
        if new_target_col is not None:
            if new_target_col not in df.columns:
                raise ValueError(f"Column '{new_target_col}' does not exist in DataFrame")
            df = df.with_columns([nw.col(new_target_col).alias(self._target_col)])

        # Step 3: Use _setup_timeframe for validation and sorting
        self._df = self._setup_timeframe(df, sort=sort, ascending=self._ascending)

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

        :return: The mode of operation, either `MODE_SINGLE_STEP` or `MODE_MULTI_STEP`.
        :rtype: str
        """
        return self._mode

    @property
    def backend(self) -> str:
        """Return the backend of the TimeFrame instance.

        :return: The backend of the DataFrame, either specified or inferred.
        :rtype: str
        """
        return self._original_backend

    @property
    def ascending(self) -> bool:
        """Return the sort order of the TimeFrame instance.

        :return: The sort order, True if ascending, False if descending.
        :rtype: bool
        """
        return self._ascending
