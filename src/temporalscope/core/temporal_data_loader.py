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

# Import necessary types and utilities from the Narwhals library for handling temporal data
from temporalscope.core.core_utils import (
    MODE_SINGLE_TARGET,
    VALID_MODES,
    SupportedTemporalDataFrame,
    check_dataframe_nulls_nans,
    convert_datetime_column_to_numeric,
    convert_time_column_to_datetime,
    get_dataframe_backend,
    is_valid_temporal_backend,
    is_valid_temporal_dataframe,
    sort_dataframe_time,
    validate_dataframe_column_types,
    validate_temporal_uniqueness,
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
        df: SupportedTemporalDataFrame,
        time_col: str,
        target_col: str,
        time_col_conversion: Optional[str] = None,
        dataframe_backend: Optional[str] = None,
        sort: bool = True,
        ascending: bool = True,
        mode: str = MODE_SINGLE_TARGET,
        enforce_temporal_uniqueness: bool = False,
        id_col: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a TimeFrame object with required validations, backend handling, and optional time column conversion.

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

        # Initialize backend
        self._backend = self._initialize_backend(df, dataframe_backend)

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
            print(f"TimeFrame successfully initialized with backend: {self._backend}")

    def _validate_parameters(self) -> None:
        """Validate input parameters for the TimeFrame initialization.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any parameter has an invalid type.
        ValueError
            If a parameter value is invalid or unsupported.

        """
        if not isinstance(self._time_col, str):
            raise TypeError(f"`time_col` must be a string. Got {type(self._time_col).__name__}.")
        if not isinstance(self._target_col, str):
            raise TypeError(f"`target_col` must be a string. Got {type(self._target_col).__name__}.")
        if self._backend is not None and not isinstance(self._backend, str):
            raise TypeError(f"`dataframe_backend` must be a string or None. Got {type(self._backend).__name__}.")
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

    def _initialize_backend(self, df: SupportedTemporalDataFrame, dataframe_backend: Optional[str]) -> str:
        """Determine and validate the backend for the DataFrame.

        Parameters
        ----------
        df : SupportedTemporalDataFrame
            Input DataFrame to initialize the backend.
        dataframe_backend : Optional[str]
            Backend to use. If None, it is inferred from the DataFrame type.
        df: SupportedTemporalDataFrame :

        dataframe_backend: Optional[str] :


        Returns
        -------
        str
            Initialized backend for the DataFrame.

        Raises
        ------
        UnsupportedBackendError
            If the backend is invalid or unsupported.

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

        Parameters
        ----------
        df : SupportedTemporalDataFrame
            DataFrame to sort
        ascending : bool
            Sort direction, defaults to True
        df: SupportedTemporalDataFrame :

        ascending: bool :
             (Default value = True)

        Returns
        -------
        SupportedTemporalDataFrame

        Examples
        --------
        ```python
        import polars as pl
        from temporalscope.core.temporal_data_loader import TimeFrame

        data = pl.DataFrame({"time": [3, 1, 4, 2, 5], "target": range(5)})
        tf = TimeFrame(data, time_col="time", target_col="target", sort=False)
        sorted_df = tf.sort_dataframe_time(tf.df, ascending=True)
        print(sorted_df)  # Shows data sorted by time column
        ```

        Notes
        -----
        Uses the reusable utility function `sort_dataframe_time` for consistency across the codebase.
        Sorted DataFrame

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

        Parameters
        ----------
        df : SupportedTemporalDataFrame
            Input DataFrame to validate.
        df: SupportedTemporalDataFrame :


        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any columns contain nulls/NaNs or invalid data types.
        UnsupportedBackendError
            If the backend is not supported.

        Examples
        --------
        ```python
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
        ```

        Notes
        -----
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
        self,
        df: SupportedTemporalDataFrame,
        sort: bool = True,
        ascending: bool = True,
        time_col_conversion: Optional[str] = None,
        enforce_temporal_uniqueness: bool = False,
        id_col: Optional[str] = None,
    ) -> SupportedTemporalDataFrame:
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
        # Step 1: Basic validation
        self.validate_dataframe(df)

        # Step 2: Time column conversion
        if time_col_conversion == "numeric":
            df = convert_datetime_column_to_numeric(df, self._time_col)
            if self._verbose:
                print(f"Converted column '{self._time_col}' to numeric (Unix timestamp).")
        elif time_col_conversion == "datetime":
            df = convert_time_column_to_datetime(df, self._time_col, nw.col(self._time_col), df.schema[self._time_col])
            if self._verbose:
                print(f"Converted column '{self._time_col}' to datetime.")

        # Step 3: If enforce temporal uniqueness is enabled, validate with `validate_temporal_uniqueness`
        if enforce_temporal_uniqueness:
            validate_temporal_uniqueness(df, self._time_col, raise_error=True, id_col=id_col)

        # Step 4: Optional sorting (user's choice for data organization)
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

        Parameters
        ----------
        df : SupportedTemporalDataFrame

        Example Usage:
        --------------
        ```python
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
        ```

        Notes
        -----
        This method uses the parameters set during TimeFrame initialization:
        - Uses the same time_col and target_col
        - Maintains the same sort order (ascending/descending)
        - Keeps the same sorting behavior (enabled/disabled)

        If you need to change these parameters, create a new TimeFrame instance
        with the desired configuration.

        See Also
        --------
        - :class:`temporalscope.target_shifters.single_step.SingleStepTargetShifter`
        - :class:`temporalscope.partition.padding.functional`
        For handling target transformations and padding operations.
            New DataFrame to use
        df: SupportedTemporalDataFrame :

        Returns
        -------
        None

        """
        self._df = self.setup(df, sort=True, ascending=self._ascending)

    @property
    def df(self) -> SupportedTemporalDataFrame:
        """Return the DataFrame in its current state.

        Returns
        -------
        SupportedTemporalDataFrame
            The DataFrame managed by the TimeFrame instance.

        """
        return self._df

    @property
    def mode(self) -> str:
        """Return the mode of the TimeFrame instance.

        Returns
        -------
        str
            The mode of operation, either `MODE_SINGLE_TARGET` or `MODE_MULTI_TARGET`.

        """
        return self._mode

    @property
    def backend(self) -> str:
        """Return the backend of the TimeFrame instance.

        Returns
        -------
        str
            The backend of the DataFrame, either specified or inferred.

        """
        return self._backend

    @property
    def ascending(self) -> bool:
        """Return the sort order of the TimeFrame instance.

        Returns
        -------
        bool
            The sort order, True if ascending, False if descending.

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

        Examples
        --------
        >>> # Initialize a TimeFrame
        >>>  tf = TimeFrame(df, time_col="time", target_col="value")
        >>> ...
        >>> # Add custom metadata
        >>> tf.metadata["description"] = "This dataset is for monthly sales forecasting"
        >>> tf.metadata["model_details"] = {"type": "LSTM", "framework": "TensorFlow"}
        >>> ...
        >>> # Access metadata
        >>> print(tf.metadata["description"])  # Output: "This dataset is for monthly sales forecasting"

        Notes
        -----
        This metadata container is designed following patterns seen in deep reinforcement
        learning (DRL) libraries like Stable-Baselines3, where additional metadata is
        stored alongside primary data structures for extensibility.
        - In future releases, this will support multi-target workflows, enabling the storage
        of processed tensor data for deep learning explainability (e.g., SHAP, LIME).

        Returns
        -------
        Dict[str, Any]
            Dictionary for storing metadata related to the TimeFrame.

        """
        return self._metadata
