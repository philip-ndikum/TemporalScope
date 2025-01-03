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

This module provides `TimeFrame`, a universal data loader for time series forecasting that can store metadata
for conversions between DataFrame and PyTorch/TensorFlow types. It supports state-of-the-art models including
multi-modal and mixed-frequency workflows, with integration for explainability tools (SHAP, LIME, Boruta-SHAP).

TimeFrame is designed to support different modeling approaches by allowing users to add columns that generalize
temporal patterns. For example, adding a market regime column to analyze feature importance across different
market conditions, or a treatment phase column to understand changing feature effects during patient care.
TimeFrame enforces minimal restrictions to maintain flexibility:

1. The time column must be numeric or timestamp-like
2. Non-time columns must be numeric (preprocess categorical features)
3. Data can have mixed frequencies and asynchronous records

Supported Modeling Approaches:
------------------------------
+------------------------+-----------------------------------------------+
| Approach               | Description                                   |
+------------------------+-----------------------------------------------+
| Standard Regression    | Basic ML models where Temporal SHAP reveals   |
|                        | how feature importance evolves naturally      |
|                        | over time without enforced constraints.       |
+------------------------+-----------------------------------------------+
| Time Series            | Group-aware models (e.g., by stock_id) where  |
| Regression             | Temporal SHAP shows how features impact       |
|                        | predictions differently across groups and     |
|                        | their unique temporal patterns.               |
+------------------------+-----------------------------------------------+
| Bayesian               | Probabilistic models where Temporal SHAP      |
| Regression             | explains how features drive both predictions  |
|                        | and uncertainty estimates through time.       |
+------------------------+-----------------------------------------------+

Supported Modes:
----------------
+----------------+-------------------------------------------------------------------+
| Mode           | Description                                                       |
|                | Data Structure                                                    |
+----------------+-------------------------------------------------------------------+
| single_target  | General machine learning tasks with scalar targets. Each row is   |
|                | a single time step, and the target is scalar.                     |
|                | Single DataFrame: each row is an observation.                     |
+----------------+-------------------------------------------------------------------+
| multi_target   | Sequential time series tasks (e.g., seq2seq) for deep learning.   |
|                | The data is split into sequences (input X, target Y).             |
|                | Two DataFrames: X for input sequences, Y for targets.             |
|                | Frameworks: TensorFlow, PyTorch, Keras.                           |
+----------------+-------------------------------------------------------------------+

References
----------
1. Van Ness, M., et al. (2023). Cross-Frequency Time Series Meta-Forecasting.
   arXiv:2302.02077.

2. Woo, G., et al. (2024). Unified training of universal time series forecasting
   transformers. arXiv:2402.02592.

3. Trirat, P., et al. (2024). Universal time-series representation learning:
   A survey. arXiv:2401.03717.

4. Xu, Q., et al. (2019). An artificial neural network for mixed frequency data.
   Expert Systems with Applications, 118, pp.127-139.

5. Filho, L.L., et al. (2024). A multi-modal approach for mixed-frequency time
   series forecasting. Neural Computing and Applications, pp.1-25.
"""

from typing import Any, Dict, Optional

import narwhals as nw
from narwhals.typing import FrameT

# Import necessary types and utilities from the Narwhals library for handling temporal data
from temporalscope.core.core_utils import (
    is_dataframe_empty,
    sort_dataframe_time,
    validate_and_convert_time_column,
    validate_column_numeric_or_datetime,
    validate_feature_columns_numeric,
    validate_temporal_ordering,
)

# Constants for temporal data loading
MODE_SINGLE_TARGET = "single_target"
MODE_MULTI_TARGET = "multi_target"
VALID_MODES = [MODE_SINGLE_TARGET, MODE_MULTI_TARGET]


class TimeFrame:
    """Central class for the TemporalScope package.

    The `TimeFrame` class is designed to handle time series data across various backends, including Polars, Pandas,
    and Modin. It facilitates workflows for machine learning, deep learning, and explainability methods, while abstracting
    away backend-specific implementation details.

    This class automatically infers the appropriate backend, validates the data, and sorts it by time. It ensures
    compatibility with temporal XAI techniques (SHAP, Boruta-SHAP, LIME etc) supporting larger data workflows in
    production.

    Engineering Design Assumptions:
    -------------------------------
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

    Backend Handling:
    -----------------
    - If a `dataframe_backend` is explicitly provided, it takes precedence over backend inference.
    - If no backend is specified, the class infers the backend from the DataFrame type, supporting Polars, Pandas, and Modin.

    Examples
    --------
    ```python
    import polars as pl

    data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=100, interval="1d"), "value": range(100)})
    tf = TimeFrame(data, time_col="time", target_col="value")
    print(tf.get_data().head())
    ```

    """

    def __init__(
        self,
        df: FrameT,
        time_col: str,
        target_col: str,
        time_col_conversion: Optional[str] = None,
        sort: bool = True,
        ascending: bool = True,
        mode: str = MODE_SINGLE_TARGET,
        enforce_temporal_uniqueness: bool = False,
        id_col: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize a TimeFrame object with required validations, backend handling, pre-processing.

        This constructor initializes the `TimeFrame` object, validates the input DataFrame,
        and performs optional sorting based on the specified `time_col`. It also allows for
        validation and conversion of the `time_col` to numeric for compatibility with downstream
        processing. Designed for universal workflows supporting state-of-the-art AI models,
        this class accommodates mixed-frequency time series data.

        There are two common use cases for `TimeFrame`:

        1. Implicit & Static Time Series:
        For workflows where `time_col` is treated as a feature, such as in static
        modeling for ML/DL applications, `enforce_temporal_uniqueness` can remain
        `False` (default). This mode emphasizes a universal design, accommodating
        mixed-frequency data.

        2. Strict Time Series:
        For workflows requiring strict temporal ordering and uniqueness (e.g.,
        forecasting), set `enforce_temporal_uniqueness=True`. Additionally,
        specify `id_col` for grouped or segmented validation.

        Parameters
        ----------
        df : SupportedTemporalDataFrame
            The input DataFrame, which can be any TemporalScope-supported backend
            (e.g., Pandas, Modin, Polars).
        time_col : str
            The name of the column representing time. Must be numeric or
            timestamp-like for sorting.
        target_col : str
            The name of the column representing the target variable.
        time_col_conversion : Optional[str], optional
            Specify the conversion type for the `time_col`:
            - 'numeric': Convert to Float64.
            - 'datetime': Convert to Datetime.
            - None: Validate only.
            Default is None.
        dataframe_backend : Optional[str], optional
            The backend to use. If not provided, it will be inferred
            based on the DataFrame type. Default is None.
        sort : bool, optional
            If True, the data will be sorted by `time_col`. Default is True.
        ascending : bool, optional
            If sorting, whether to sort in ascending order. Default is True.
        mode : str, optional
            The operation mode, either `MODE_SINGLE_TARGET` (default) or `MODE_MULTI_TARGET`.
            Default is MODE_SINGLE_TARGET.
        enforce_temporal_uniqueness : bool, optional
            If True, ensures that timestamps in `time_col` are unique within
            each group (defined by `id_col`) or globally if `id_col` is None.
            This setting is essential for workflows requiring temporal
            consistency, such as forecasting or explainability analysis.
            Default is False.
        id_col : Optional[str], optional
            Optional column for grouped or segmented strict temporal validation.
            Default is None.
        verbose : bool, optional
            If True, enables logging for validation and setup stages.
            Default is False.

        Raises
        ------
        ModeValidationError
            If the specified mode is invalid.
        UnsupportedBackendError
            If the specified or inferred backend is not supported.
        ValueError
            If required columns are missing, invalid, or if the time column
            conversion fails.

        Attributes
        ----------
        _metadata : Dict[str, Any]
            A private metadata dictionary to allow end-users flexibility in extending
            the TimeFrame object. This provides storage for any additional attributes
            or information during runtime.

        Examples
        --------
        ```python
        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame, MODE_SINGLE_TARGET

        # Example DataFrame
        df = pd.DataFrame({"time": pd.date_range(start="2023-01-01", periods=10, freq="D"), "value": range(10)})

        # Initialize TimeFrame with automatic time column conversion to numeric
        tf = TimeFrame(df, time_col="time", target_col="value", time_col_convert_numeric=True, mode=MODE_SINGLE_TARGET)
        print(tf.df.head())
        ```

        Warnings
        --------
        - The `mode` parameter must be one of:
            - `"single_target"`: For scalar target predictions (e.g., regression).
            - `"multi_target"`: For sequence forecasting tasks (e.g., seq2seq models).
        - The `time_col_conversion` parameter allows for automatic conversion of the `time_col` to either numeric or datetime during initialization.
        - The `_metadata` container follows design patterns similar to SB3, enabling users to manage custom attributes and extend functionality for advanced workflows, such as
            future conversion to TensorFlow or PyTorch types in multi-target explainable AI workflows.
        """
        # Initialize instance variables
        self._time_col = time_col
        self._target_col = target_col
        self._time_col_conversion = time_col_conversion
        self._mode = mode
        self._ascending = ascending
        self._sort = sort
        self._verbose = verbose
        self._enforce_temporal_uniqueness = enforce_temporal_uniqueness
        self._id_col = id_col
        self._metadata: Dict[str, Any] = {}

        # Validate parameters
        self._validate_parameters()

        # Setup DataFrame
        self._df = self.setup(
            df,
            sort=self._sort,
            ascending=self._ascending,
            time_col_conversion=time_col_conversion,
            enforce_temporal_uniqueness=self._enforce_temporal_uniqueness,
            id_col=self._id_col,
        )

        if self._verbose:
            print("TimeFrame successfully initialized")

    def _validate_parameters(self) -> None:
        """Validate input parameters for the TimeFrame initialization.

        This method performs comprehensive validation of all parameters passed to
        TimeFrame's constructor, ensuring type safety and value validity before
        any data processing begins.

        The validation includes:
        1. Type checking for all parameters
        2. Value validation for enumerated parameters
        3. Consistency checks for related parameters

        Returns
        -------
        None
            Method returns None if validation passes.

        Raises
        ------
        TypeError
            - If time_col is not a string
            - If target_col is not a string
            - If sort is not a boolean
            - If ascending is not a boolean
            - If verbose is not a boolean
            - If id_col is not None or a string
        ValueError
            - If time_col_conversion is not one of {None, 'numeric', 'datetime'}
            - If mode is not one of VALID_MODES

        Examples
        --------
        ```python
        from temporalscope.core.temporal_data_loader import TimeFrame

        # Valid parameters
        tf = TimeFrame(df, time_col="time", target_col="target")  # Passes validation

        # Invalid type
        tf = TimeFrame(df, time_col=123, target_col="target")  # Raises TypeError

        # Invalid value
        tf = TimeFrame(df, time_col="time", target_col="target", mode="invalid")  # Raises ValueError
        ```

        See Also
        --------
        TimeFrame.__init__ : The constructor that uses this validation
        setup : The method that uses validated parameters

        Notes
        -----
        - Called automatically during initialization
        - Validates all instance variables
        - Ensures consistent state before data processing
        """
        if not isinstance(self._time_col, str):
            raise TypeError(f"`time_col` must be a string. Got {type(self._time_col).__name__}.")
        if not isinstance(self._target_col, str):
            raise TypeError(f"`target_col` must be a string. Got {type(self._target_col).__name__}.")
        if not isinstance(self._sort, bool):
            raise TypeError(f"`sort` must be a boolean. Got {type(self._sort).__name__}.")
        if not isinstance(self._ascending, bool):
            raise TypeError(f"`ascending` must be a boolean. Got {type(self._ascending).__name__}.")
        if not isinstance(self._verbose, bool):
            raise TypeError(f"`verbose` must be a boolean. Got {type(self._verbose).__name__}.")
        if self._id_col is not None and not isinstance(self._id_col, str):
            raise TypeError(f"`id_col` must be a string or None. Got {type(self._id_col).__name__}.")
        if self._time_col_conversion not in {None, "numeric", "datetime"}:
            raise ValueError(
                f"Invalid `time_col_conversion` value '{self._time_col_conversion}'. "
                f"Must be one of {{None, 'numeric', 'datetime'}}."
            )
        if self._mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self._mode}'. Must be one of {VALID_MODES}.")

    def sort_dataframe_time(self, df: FrameT, ascending: bool = True) -> FrameT:
        """Sort DataFrame by time column using backend-agnostic Narwhals operations.

        This method provides a consistent way to sort DataFrames by their time column
        across all supported backends (Pandas, Polars, etc.). It delegates to
        core_utils.sort_dataframe_time for the actual sorting operation.

        Parameters
        ----------
        df : FrameT
            DataFrame to sort. Can be any backend supported by Narwhals
            (Pandas, Polars, etc.).
        ascending : bool, optional
            Sort direction. True for ascending (default), False for descending.

        Returns
        -------
        FrameT
            A new DataFrame sorted by the time column.

        Examples
        --------
        ```python
        import polars as pl
        from temporalscope.core.temporal_data_loader import TimeFrame

        # Create TimeFrame with unsorted data
        data = pl.DataFrame({"time": [3, 1, 4, 2, 5], "target": range(5)})
        tf = TimeFrame(data, time_col="time", target_col="target", sort=False)

        # Sort ascending
        sorted_asc = tf.sort_dataframe_time(tf.df, ascending=True)
        print(sorted_asc)  # Shows: 1, 2, 3, 4, 5

        # Sort descending
        sorted_desc = tf.sort_dataframe_time(tf.df, ascending=False)
        print(sorted_desc)  # Shows: 5, 4, 3, 2, 1
        ```

        See Also
        --------
        temporalscope.core.core_utils.sort_dataframe_time : The underlying sorting function

        Notes
        -----
        - Uses core_utils.sort_dataframe_time for consistent sorting across the codebase
        - Preserves DataFrame schema and column types
        - Returns a new DataFrame; does not modify the input DataFrame
        """
        return sort_dataframe_time(df, self._time_col, ascending)

    def validate_dataframe(self, df: FrameT) -> None:
        """Validate DataFrame structure and data types.

        This method performs comprehensive validation of the input DataFrame:
        1. Converts to Narwhals DataFrame for backend-agnostic operations
        2. Checks for empty DataFrame
        3. Validates required columns exist (time_col and target_col)
        4. Validates time column is numeric or datetime
        5. Validates all non-time columns are numeric

        Parameters
        ----------
        df : FrameT
            DataFrame to validate. Can be any backend supported by Narwhals
            (Pandas, Polars, etc.).

        Returns
        -------
        None
            Method returns None if validation passes.

        Raises
        ------
        ValueError
            - If DataFrame is empty
            - If required columns are missing
            - If column types are invalid

        Examples
        --------
        ```python
        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame

        # Create TimeFrame with valid data
        df = pd.DataFrame({"time": [1, 2, 3], "target": [10, 20, 30]})
        tf = TimeFrame(df, time_col="time", target_col="target")

        # Validate new data
        new_df = pd.DataFrame({"time": [4, 5, 6], "target": [40, 50, 60]})
        tf.validate_dataframe(new_df)  # Passes validation

        # Invalid data (missing column)
        invalid_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        tf.validate_dataframe(invalid_df)  # Raises ValueError
        ```

        See Also
        --------
        temporalscope.core.core_utils.validate_column_numeric_or_datetime
        temporalscope.core.core_utils.validate_feature_columns_numeric

        Notes
        -----
        - Uses core_utils functions for consistent validation across the codebase
        - Performs validation without modifying the input DataFrame
        - Supports all DataFrame backends through Narwhals abstraction
        """
        # Convert to Narwhals DataFrame
        df = nw.from_native(df)

        # Validate DataFrame using core_utils functions
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame provided")

        # Validate columns exist
        if self._time_col not in df.columns:
            raise ValueError(f"Column '{self._time_col}' does not exist in DataFrame")
        if self._target_col not in df.columns:
            raise ValueError(f"Column '{self._target_col}' does not exist in DataFrame")

        # Validate using core_utils
        validate_column_numeric_or_datetime(df, self._time_col)
        validate_feature_columns_numeric(df, time_col=self._time_col)

        if self._verbose:
            print("Validation completed successfully.")

    @nw.narwhalify
    def setup(
        self,
        df: FrameT,
        sort: bool = True,
        ascending: bool = True,
        time_col_conversion: Optional[str] = None,
        enforce_temporal_uniqueness: bool = False,
        id_col: Optional[str] = None,
    ) -> FrameT:
        """Initialize and validate a TimeFrame's DataFrame with proper sorting and validation.

        This method performs the necessary validation, conversion, and sorting operations to prepare
        the input DataFrame for use in TemporalScope workflows. The method is idempotent.

        Steps:
        ------
        1. Validate the input DataFrame using the `validate_dataframe` method.
        2. Optionally convert the `time_col` to the specified type (`numeric` or `datetime`).
        3. Perform temporal uniqueness validation within groups if enabled.
        4. Optionally sort the DataFrame by `time_col` in the specified order.

        Parameters
        ----------
        df : SupportedTemporalDataFrame
            Input DataFrame to set up and validate.
        sort : bool
            Whether to sort the DataFrame by `time_col`. Defaults to True.
        ascending : bool
            Sort order if sorting is enabled. Defaults to True.
        time_col_conversion : Optional[str]
            Optional. Specify the conversion type for the `time_col`:
            - 'numeric': Convert to Float64.
        the input DataFrame for use in TemporalScope workflows. The method is idempotent.

        Steps:
        ------
        1. Validate the input DataFrame using the `validate_dataframe` method.
        2. Optionally convert the `time_col` to the specified type (`numeric` or `datetime`).
        3. Perform temporal uniqueness validation within groups if enabled.
        4. Optionally sort the DataFrame by `time_col` in the specified order.

        Parameters
        ----------
        df : SupportedTemporalDataFrame
            Input DataFrame to set up and validate.
        sort : bool
            Whether to sort the DataFrame by `time_col`. Defaults to True.
        ascending : bool
            Sort order if sorting is enabled. Defaults to True.
        time_col_conversion : Optional[str]
            Optional. Specify the conversion type for the `time_col`:
            - 'numeric': Convert to Float64.
            - 'datetime': Convert to Datetime.
            - None: Validate only.
            Default is None.
        enforce_temporal_uniqueness : bool
            If True, validates that timestamps in the `time_col` are
            unique within the groups defined by the `id_col` parameter
            (if specified) or across the entire DataFrame. Default is False.
        id_col : Optional[str]
            An optional column name to define groups for temporal uniqueness validation. If None,
            validation is performed across the entire DataFrame. Default is None.
        df: SupportedTemporalDataFrame :

        sort: bool :
             (Default value = True)
        ascending: bool :
             (Default value = True)
        time_col_conversion: Optional[str] :
             (Default value = None)
        enforce_temporal_uniqueness: bool :
             (Default value = False)
        id_col: Optional[str] :
             (Default value = None)

        Returns
        -------
        SupportedTemporalDataFrame

        Example usage:
        --------------
        ```python
          import pandas as pd
          from temporalscope.core.temporal_data_loader import TimeFrame

          df = pd.DataFrame(
              {
                  "patient_id": [1, 1, 2, 2],
                  "time": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-03"],
                  "value": [10, 20, 30, 40],
              }
          )

          tf = TimeFrame(
              df,
              time_col="time",
              target_col="value",
          )
          sorted_df = tf.setup(df, time_col_conversion="datetime", enforce_temporal_uniqueness=True, id_col="patient_id")
          print(sorted_df)
        ```

        Notes
        -----
        - This method is designed to be idempotent, ensuring safe revalidation or reinitialization.
        - The `time_col_conversion` parameter allows you to convert the `time_col` to a numeric or datetime type.
        - Sorting is performed only if explicitly enabled via the `sort` parameter.
        - While this method validates, converts, and sorts the DataFrame, it does not modify the TimeFrame's
        internal state unless explicitly used within another method (e.g., `update_dataframe`).
        - The `enforce_temporal_uniqueness` parameter can be set dynamically in this method, allowing
        validation of temporal uniqueness to be turned on/off as needed.
        - The `id_col` parameter can also be set dynamically, defining the scope of the temporal uniqueness validation.
        - The `id_col` parameter enables validation of temporal uniqueness within each group's records, ensuring no duplicate
        timestamps exist per group while allowing different groups to have events on the same dates. This is particularly
        useful for multi-entity time series datasets (e.g., patient data, stock prices). Note: Users must check the Apache License
        for the complete terms of use. This software is distributed "AS-IS" and may require adjustments for specific use cases.
        Validated, converted, and optionally sorted DataFrame.

        """
        from temporalscope.core.core_utils import is_dataframe_empty

        # Convert to Narwhals DataFrame
        df = nw.from_native(df)

        # Check if DataFrame is empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame provided")

        # Check if required columns exist
        if self._time_col not in df.columns:
            raise ValueError(f"Column '{self._time_col}' does not exist in DataFrame")
        if self._target_col not in df.columns:
            raise ValueError(f"Column '{self._target_col}' does not exist in DataFrame")

        # First validate DataFrame (includes type checks and null validation)
        self.validate_dataframe(df)

        # Convert time column if requested
        if time_col_conversion:
            df = validate_and_convert_time_column(df, self._time_col, time_col_conversion)
            if self._verbose:
                print(f"Converted column '{self._time_col}' to {time_col_conversion}.")

        # Check temporal uniqueness if required
        if enforce_temporal_uniqueness:
            validate_temporal_ordering(df, self._time_col, id_col=id_col, enforce_equidistant_sampling=False)
            if self._verbose:
                print("Temporal ordering validation successful.")

        # Optional sorting (lazy operation)
        if sort:
            df = self.sort_dataframe_time(df, ascending=ascending)

        return df

    @nw.narwhalify(eager_only=True)
    def update_dataframe(self, df: FrameT) -> None:
        """Update TimeFrame's internal DataFrame with new data.

        This method updates the internal DataFrame with new data, performing all necessary
        validations and conversions to maintain consistency. Uses eager evaluation since
        it modifies internal state.

        The update process includes:
        1. Converting input to Narwhals DataFrame
        2. Running full setup validation and conversion pipeline
        3. Replacing internal DataFrame with validated result

        Parameters
        ----------
        df : FrameT
            The new DataFrame to update with. Must contain the required time and target
            columns with appropriate data types.

        Returns
        -------
        None
            Method updates internal state but returns nothing.

        Raises
        ------
        ValueError
            - If DataFrame is empty
            - If required columns are missing
            - If column types are invalid
            - If temporal validation fails

        Examples
        --------
        ```python
        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame

        # Initialize TimeFrame
        initial_df = pd.DataFrame({"time": [1, 2, 3], "target": [10, 20, 30]})
        tf = TimeFrame(initial_df, time_col="time", target_col="target")

        # Update with new data
        new_df = pd.DataFrame({"time": [4, 5, 6], "target": [40, 50, 60]})
        tf.update_dataframe(new_df)  # Updates internal DataFrame

        # Invalid update (missing column)
        invalid_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        tf.update_dataframe(invalid_df)  # Raises ValueError
        ```

        See Also
        --------
        setup : The underlying validation and setup method
        validate_dataframe : The validation method used

        Notes
        -----
        - Uses eager evaluation to ensure immediate state update
        - Maintains all TimeFrame settings (sorting, conversion, etc.)
        - Performs full validation to ensure data consistency
        """
        # Convert to Narwhals DataFrame first
        df = nw.from_native(df)

        # Setup with all validations and conversions
        self._df = self.setup(
            df,
            sort=self._sort,
            ascending=self._ascending,
            time_col_conversion=self._time_col_conversion,
            enforce_temporal_uniqueness=self._enforce_temporal_uniqueness,
            id_col=self._id_col,
        )

        if self._verbose:
            print("DataFrame successfully updated.")

    @property
    def df(self) -> FrameT:
        """Access the internal DataFrame.

        This property provides read-only access to the TimeFrame's internal DataFrame.
        The DataFrame maintains all validations, conversions, and sorting settings
        applied during initialization or updates.

        Returns
        -------
        FrameT
            The current state of the DataFrame, with all validations and
            transformations applied.

        Examples
        --------
        ```python
        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame

        # Create TimeFrame
        tf = TimeFrame(pd.DataFrame({"time": [1, 2, 3], "target": [10, 20, 30]}), time_col="time", target_col="target")

        # Access DataFrame
        current_df = tf.df
        print(current_df)  # Shows current state
        ```

        See Also
        --------
        update_dataframe : Method to update the internal DataFrame
        setup : Method that prepares the DataFrame

        Notes
        -----
        - Returns a reference to the internal DataFrame
        - Any modifications should be done through update_dataframe
        - Maintains all TimeFrame settings and validations
        """
        return self._df

    @property
    def mode(self) -> str:
        """Get the TimeFrame's operation mode.

        This property indicates whether the TimeFrame is configured for single-target
        or multi-target operations, affecting how data is processed and validated.

        Returns
        -------
        str
            The current operation mode:
            - MODE_SINGLE_TARGET: For scalar target predictions
            - MODE_MULTI_TARGET: For sequence forecasting tasks

        Examples
        --------
        ```python
        from temporalscope.core.temporal_data_loader import TimeFrame, MODE_MULTI_TARGET

        # Create TimeFrame in multi-target mode
        tf = TimeFrame(df, time_col="time", target_col="target", mode=MODE_MULTI_TARGET)

        # Check mode
        if tf.mode == MODE_MULTI_TARGET:
            print("Configured for sequence forecasting")
        ```

        See Also
        --------
        TimeFrame.__init__ : Where mode is set during initialization

        Notes
        -----
        - Mode affects validation and processing behavior
        - Cannot be changed after initialization
        - Determines compatibility with different model types
        """
        return self._mode

    @property
    def ascending(self) -> bool:
        """Get the TimeFrame's sort order setting.

        This property indicates whether time-based sorting is performed in ascending
        or descending order, affecting how data is organized for analysis and modeling.

        Returns
        -------
        bool
            True if sorting is ascending (earlier to later),
            False if descending (later to earlier).

        Examples
        --------
        ```python
        from temporalscope.core.temporal_data_loader import TimeFrame

        # Create TimeFrame with descending sort
        tf = TimeFrame(df, time_col="time", target_col="target", ascending=False)

        # Check sort order
        if not tf.ascending:
            print("Data sorted from latest to earliest")
        ```

        See Also
        --------
        sort_dataframe_time : Method that uses this setting
        setup : Where sorting is applied

        Notes
        -----
        - Affects all sorting operations
        - Set during initialization
        - Used by sort_dataframe_time method
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

        Returns
        -------
        Dict[str, Any]
            Dictionary for storing metadata related to the TimeFrame.

        Examples
        --------
        ```python
        # Initialize a TimeFrame
        tf = TimeFrame(df, time_col="time", target_col="value")

        # Add custom metadata
        tf.metadata["description"] = "This dataset is for monthly sales forecasting"
        tf.metadata["model_details"] = {"type": "LSTM", "framework": "TensorFlow"}

        # Access metadata
        print(tf.metadata["description"])  # Output: "This dataset is for monthly sales forecasting"
        ```

        Notes
        -----
        This metadata container is designed following patterns seen in deep reinforcement
        learning (DRL) libraries like Stable-Baselines3, where additional metadata is
        stored alongside primary data structures for extensibility.

        Future Support
        -------------
        In future releases, this will support multi-target workflows, enabling the storage
        of processed tensor data for deep learning explainability (e.g., SHAP, LIME).
        """
        return self._metadata
