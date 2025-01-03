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

"""TemporalScope/src/temporalscope/core/exceptions.py

This module defines custom exceptions and warnings used throughout the TemporalScope package.
These exceptions provide clear, actionable feedback for users during time-series forecasting workflows,
particularly for DataFrame validation and time column handling through Narwhals.
"""


class TimeFrameError(Exception):
    """Base class for exceptions in the TimeFrame module.

    This serves as the foundation for all `TimeFrame`-related errors.

    """

    pass


class TimeColumnError(TimeFrameError):
    """Exception raised for validation issues with `time_col`.

    Examples
    --------
    ```python
    # Validate time column type using Narwhals
    if not (nw.col(time_col).cast(nw.Float64()).is_valid() or nw.col(time_col).cast(nw.Datetime()).is_valid()):
        raise TimeColumnError("`time_col` must be numeric or timestamp-like.")
    ```
    """

    pass


class TargetColumnWarning(UserWarning):
    """Warning raised for potential issues with the target column.

    This warning is issued when the target column appears to contain sequential or vectorized data,
    which may require transformation depending on the selected mode (MODE_SINGLE_TARGET vs MODE_MULTI_TARGET).

    Examples
    --------
    ```python
    # Check if target column contains sequence data
    if mode == MODE_MULTI_TARGET and df.select([nw.col(target_col).is_list()]).item():
        warnings.warn(
            "`target_col` appears to contain sequential data. Ensure it is transformed appropriately for MODE_MULTI_TARGET.",
            TargetColumnWarning,
        )
    ```

    """

    pass


class ModeValidationError(TimeFrameError):
    """Exception raised when an invalid mode is specified.

    Parameters
    ----------
    mode : str
        The invalid mode that caused the error.
    message : str

    Examples
    --------
    ```python
    if mode not in VALID_MODES:
        raise ModeValidationError(mode, f"Invalid mode: {mode}. Must be one of {VALID_MODES}.")
    ```
    """

    def __init__(self, mode, message="Invalid mode specified"):
        """Initialize ModeValidationError.

        Parameters
        ----------
        mode : str
            The invalid mode that caused the error.
        message : str
            The error message to display.
        """
        self.mode = mode
        self.message = f"{message}: {mode}"
        super().__init__(self.message)


class DataFrameValidationError(TimeFrameError):
    """Exception raised for DataFrame validation issues.

    This error is raised when DataFrame operations fail due to invalid data,
    schema mismatches, or other validation issues.

    Examples
    --------
    ```python
    try:
        # Validate numeric columns using Narwhals
        for col in feature_cols:
            df = df.select([nw.col(col).cast(nw.Float64()).alias(col)])
    except Exception as e:
        raise DataFrameValidationError(f"Failed to validate numeric columns: {str(e)}")
    ```
    """

    pass
