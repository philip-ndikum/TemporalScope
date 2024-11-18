"""TemporalScope/src/temporalscope/target_shifters/single_step.py.

This module provides the SingleStepTargetShifter class for shifting target variables in time series data.
It works in conjunction with the TimeFrame class to enable consistent target shifting operations across
different DataFrame backends through Narwhals. Following the same backend-agnostic design as core_utils.py
and temporal_data_loader.py, it ensures consistent behavior across all supported DataFrame types.

Engineering Design
------------------
The SingleStepTargetShifter follows a clear separation between validation and transformation phases,
designed to work seamlessly with both TimeFrame and raw DataFrame inputs.

+----------------+-------------------------------------------------------------------+
| Component      | Description                                                       |
+----------------+-------------------------------------------------------------------+
| fit()          | Input validation phase that ensures:                              |
|                | - Valid TimeFrame or supported DataFrame type                     |
|                | - Target column is set or can be inferred                         |
|                | - No Narwhals operations at this stage                            |
+----------------+-------------------------------------------------------------------+
| transform()    | Pure Narwhals transformation phase that:                          |
|                | - Uses backend-agnostic operations only                           |
|                | - Shifts target using Narwhals operations                         |
|                | - Preserves TimeFrame metadata if present                         |
+----------------+-------------------------------------------------------------------+

Backend-Specific Patterns
------------------------
The following table outlines key patterns for working with different DataFrame backends
through Narwhals operations:

+----------------+-------------------------------------------------------------------+
| Backend        | Implementation Pattern                                            |
+----------------+-------------------------------------------------------------------+
| LazyFrame      | Represents lazy evaluation in Dask and Polars. Use item() for     |
| (Dask/Polars)  | scalar access, avoid direct indexing, and handle lazy evaluation  |
|                | through proper Narwhals operations.                               |
+----------------+-------------------------------------------------------------------+
| PyArrow        | Handles scalar operations differently. Use cast("int64") for      |
|                | numeric operations, handle comparisons through Narwhals, and      |
|                | convert types before arithmetic operations.                       |
+----------------+-------------------------------------------------------------------+
| All Backends   | Let @nw.narwhalify handle conversions between backends. Use pure  |
|                | Narwhals operations and avoid any backend-specific code to ensure |
|                | consistent behavior across all supported types.                   |
+----------------+-------------------------------------------------------------------+

Example Usage
------------
.. code-block:: python

    import pandas as pd
    from temporalscope.target_shifters.single_step import SingleStepTargetShifter
    from temporalscope.core.temporal_data_loader import TimeFrame

    # With TimeFrame
    df = pd.DataFrame({"time": range(10), "target": range(10), "feature": range(10)})
    tf = TimeFrame(df=df, time_col="time", target_col="target")
    shifter = SingleStepTargetShifter(n_lags=1)
    transformed_tf = shifter.fit_transform(tf)

    # With DataFrame
    df = pd.DataFrame({"target": range(10), "feature": range(10)})
    shifter = SingleStepTargetShifter(target_col="target", n_lags=1)
    transformed_df = shifter.fit_transform(df)

.. note::
   - Uses a familiar fit/transform pattern for consistency, while implementing
     all operations through Narwhals' backend-agnostic API
   - Currently implements single-step prediction only. For multi-step sequence prediction,
     see the planned MultiStepTargetShifter in temporalscope.target_shifters.multi_step
   - When validating DataFrames, must get native format first since Narwhals wraps
     but does not implement actual DataFrame types
"""

from typing import Optional, Union

import narwhals as nw
import numpy as np
import pandas as pd

from temporalscope.core.core_utils import (
    MODE_SINGLE_STEP,
    SupportedTemporalDataFrame,
    is_valid_temporal_dataframe,
)
from temporalscope.core.temporal_data_loader import TimeFrame


class SingleStepTargetShifter:
    """A target shifter for time series data using Narwhals operations.

    This class provides target shifting functionality for single-step prediction tasks,
    working with both TimeFrame objects and raw DataFrames through Narwhals' backend-agnostic
    operations.

    Engineering Design Assumptions
    ----------------------------
    1. Separation of Concerns:
       - fit: Validates inputs and sets parameters
       - transform: Pure Narwhals operations for shifting
       - No mixing of validation and transformation

    2. Single-step Mode:
       - Each row represents one time step
       - Target variable is shifted by specified lag
       - Compatible with traditional ML frameworks
       - Supports scalar target prediction tasks

    3. Backend Agnostic:
       - Validation in fit() before any operations
       - Pure Narwhals operations in transform()
       - Clean separation of concerns

    4. Input Handling:
       - TimeFrame: Uses existing metadata
       - DataFrame: Validates in fit
       - numpy array: Converts in fit

    :param target_col: Column name to shift (optional, can be inferred from TimeFrame)
    :type target_col: str, optional
    :param n_lags: Number of steps to shift target, must be > 0
    :type n_lags: int
    :param drop_target: Whether to remove original target column
    :type drop_target: bool
    :param verbose: Enable progress/debug logging
    :type verbose: bool
    :param mode: Operation mode, defaults to single-step
    :type mode: str
    :raises ValueError: If n_lags ≤ 0

    Example:
    -------
    .. code-block:: python

        import pandas as pd
        from temporalscope.target_shifters.single_step import SingleStepTargetShifter

        # Create DataFrame
        df = pd.DataFrame({"target": range(5), "feature": range(5)})

        # Initialize and transform
        shifter = SingleStepTargetShifter(target_col="target")
        transformed = shifter.fit_transform(df)

    .. note::
        Backend-Specific Patterns:
        - Use item() for scalar access (LazyFrame)
        - Use cast("int64") for scalar operations (PyArrow)
        - Let @nw.narwhalify handle conversions

    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        n_lags: int = 1,
        drop_target: bool = True,
        verbose: bool = False,
        mode: str = MODE_SINGLE_STEP,
    ):
        """Initialize the shifter with target column and lag settings."""
        if n_lags <= 0:
            raise ValueError("`n_lags` must be greater than 0")

        self.target_col = target_col
        self.n_lags = n_lags
        self.drop_target = drop_target
        self.verbose = verbose
        self.mode = mode

        if verbose:
            print(f"Initialized SingleStepTargetShifter with target_col={target_col}, n_lags={n_lags}")

    @nw.narwhalify
    def _get_row_count(self, df: SupportedTemporalDataFrame) -> int:
        """Get row count using Narwhals operations.

        :param df: DataFrame to count rows for
        :type df: SupportedTemporalDataFrame
        :return: Number of rows in DataFrame
        :rtype: int

        .. note::
            Uses Narwhals operations for backend-agnostic row counting:
            - cast("int64") for scalar type conversion
            - item() for scalar access
            - Handles LazyFrame and PyArrow scalars
        """
        return df.select([nw.col(df.columns[0]).count().cast("int64").alias("count")]).item()

    @nw.narwhalify
    def _shift_target(self, df: SupportedTemporalDataFrame) -> SupportedTemporalDataFrame:
        """Shift target column using Narwhals operations.

        :param df: DataFrame to transform
        :type df: SupportedTemporalDataFrame
        :return: DataFrame with shifted target
        :rtype: SupportedTemporalDataFrame

        .. note::
            Uses Narwhals operations for backend-agnostic shifting:
            - with_columns() for adding shifted column
            - filter() for removing null values
            - drop() for removing original target
        """
        # Create shifted column
        shifted_df = df.with_columns(
            [nw.col(self.target_col).shift(-self.n_lags).alias(f"{self.target_col}_shift_{self.n_lags}")]
        )

        # Drop rows with null values from shifting
        shifted_df = shifted_df.filter(~nw.col(f"{self.target_col}_shift_{self.n_lags}").is_null())

        # Optionally drop original target column
        if self.drop_target:
            shifted_df = shifted_df.drop([self.target_col])

        return shifted_df

    def fit(self, X: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray], y=None) -> "SingleStepTargetShifter":
        """Validate inputs and prepare for transformation.

        This method handles all input validation before any Narwhals operations:
        - TimeFrame: Uses existing target_col
        - DataFrame: Validates using is_valid_temporal_dataframe
        - numpy array: Converts to DataFrame first

        :param X: Input data to validate
        :type X: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]
        :param y: Ignored, exists for scikit-learn compatibility
        :return: self
        :rtype: SingleStepTargetShifter
        :raises ValueError: If target_col not set and cannot be inferred
        :raises TypeError: If input type is not supported

        Example:
        -------
        .. code-block:: python

            import pandas as pd
            from temporalscope.target_shifters.single_step import SingleStepTargetShifter

            df = pd.DataFrame({"target": range(5)})
            shifter = SingleStepTargetShifter()
            shifter.fit(df)

        .. note::
            Input Validation:
            - No Narwhals operations in fit()
            - Validates before any transformations
            - Handles all input types consistently

        """
        if isinstance(X, TimeFrame):
            self.target_col = X._target_col
        elif isinstance(X, np.ndarray):
            # Convert numpy array to DataFrame first
            cols = [f"feature_{i}" for i in range(X.shape[1] - 1)]
            cols.append(self.target_col or f"feature_{X.shape[1] - 1}")
            X = pd.DataFrame(X, columns=cols)
            if self.target_col is None:
                self.target_col = cols[-1]
        else:
            # Get native DataFrame for validation
            native_df = X.to_native() if hasattr(X, "to_native") else X
            if not is_valid_temporal_dataframe(native_df):
                raise TypeError(f"Input type {type(X)} is not a supported DataFrame type")

        if self.target_col is None:
            raise ValueError("target_col must be set before transform (call fit first)")

        return self

    @nw.narwhalify
    def transform(
        self, X: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray], y=None
    ) -> Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]:
        """Transform input data using Narwhals operations.

        This method assumes inputs are already validated by fit() and uses pure
        Narwhals operations for all transformations.

        :param X: Input data to transform
        :type X: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]
        :param y: Ignored, exists for scikit-learn compatibility
        :return: Transformed data
        :rtype: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]

        .. note::
            Pure Narwhals implementation:
            - _get_row_count() for counting
            - _shift_target() for shifting
            - Backend-agnostic operations
            - Handles LazyFrame and PyArrow scalars

        Example:
        -------
        .. code-block:: python

            import pandas as pd
            from temporalscope.target_shifters.single_step import SingleStepTargetShifter

            df = pd.DataFrame({"target": range(5)})
            shifter = SingleStepTargetShifter(target_col="target")
            transformed = shifter.transform(df)

        """
        if isinstance(X, np.ndarray):
            # Convert numpy array to DataFrame
            cols = [f"feature_{i}" for i in range(X.shape[1] - 1)]
            cols.append(self.target_col)
            X = pd.DataFrame(X, columns=cols)

        # Get DataFrame to transform
        df = X.df if isinstance(X, TimeFrame) else X

        # Get row count before transformation
        rows_before = self._get_row_count(df)
        if rows_before == 0:
            raise ValueError("Cannot transform empty DataFrame")

        # Transform DataFrame
        transformed = self._shift_target(df)

        # Get row count after transformation
        rows_after = self._get_row_count(transformed)
        if rows_after == 0:
            raise ValueError("All rows were dropped during transformation")

        if self.verbose:
            print(f"Rows before: {rows_before}; Rows after: {rows_after}; Dropped: {rows_before - rows_after}")

        # Handle TimeFrame output
        if isinstance(X, TimeFrame):
            return TimeFrame(
                transformed,
                time_col=X._time_col,
                target_col=f"{self.target_col}_shift_{self.n_lags}",
                dataframe_backend=X.backend,
                mode=self.mode,
                ascending=X.ascending,
            )

        return transformed

    def fit_transform(
        self, X: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray], y=None
    ) -> Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]:
        """Fit the transformer and transform the input data.

        This method combines input validation (fit) with Narwhals transformations
        (transform) in a single operation.

        :param X: Input data to transform
        :type X: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]
        :param y: Ignored, exists for scikit-learn compatibility
        :return: Transformed data
        :rtype: Union[TimeFrame, SupportedTemporalDataFrame, np.ndarray]

        Example with TimeFrame:
        --------------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.target_shifters.single_step import SingleStepTargetShifter
            from temporalscope.core.temporal_data_loader import TimeFrame

            # Create TimeFrame
            df = pd.DataFrame({"time": range(5), "target": range(5), "feature": range(5)})
            tf = TimeFrame(df=df, time_col="time", target_col="target")

            # Initialize and transform
            shifter = SingleStepTargetShifter(n_lags=1)
            transformed_tf = shifter.fit_transform(tf)

        Example with DataFrame:
        -------------------
        .. code-block:: python

            import pandas as pd
            from temporalscope.target_shifters.single_step import SingleStepTargetShifter

            # Create DataFrame
            df = pd.DataFrame({"target": range(5), "feature": range(5)})

            # Initialize and transform
            shifter = SingleStepTargetShifter(target_col="target", n_lags=1)
            transformed_df = shifter.fit_transform(df)

        .. note::
            Operation Flow:
            1. fit(): Validates inputs
            2. transform(): Pure Narwhals operations
            3. Handles all backend types consistently
        """
        return self.fit(X).transform(X)
