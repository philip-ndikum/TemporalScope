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

This module defines custom exceptions and warnings used throughout the TemporalScope package,
specifically for handling errors and edge cases in the TimeFrame class. These custom error
types and warnings are designed to provide clear and actionable feedback for developers
when issues are encountered during time-series forecasting workflows.

Use Cases:
----------
- **TimeColumnError**: Raised when there are validation issues with the `time_col` such as unsupported types.
- **MixedTypesWarning**: Raised when mixed numeric and timestamp types are detected in `time_col`.
- **MixedTimezonesWarning**: Raised when `time_col` contains a mixture of timezone-aware and naive timestamps.

Classes:
--------
- `TimeFrameError`: The base class for all custom exceptions in the TimeFrame module.
- `TimeColumnError`: Raised when the time column has invalid values or types.
- `MixedTypesWarning`: Warning issued when the `time_col` contains mixed numeric and timestamp-like types.
- `MixedTimezonesWarning`: Warning issued when the `time_col` contains a mix of timezone-aware and naive timestamps.

Example Usage:
--------------
.. code-block:: python

    from temporalscope.core.exceptions import (
        TimeColumnError, MixedTypesWarning, MixedTimezonesWarning
    )

    def validate_time_column(df):
        if df['time'].dtype == object:
            raise TimeColumnError("Invalid time column data type.")
        elif contains_mixed_types(df['time']):
            warnings.warn("Mixed numeric and timestamp types.", MixedTypesWarning)

"""


class TimeFrameError(Exception):
    """Base class for exceptions in the TimeFrame module.

    This exception serves as the foundation for all errors related to the
    `TimeFrame` class. It should be subclassed to create more specific
    exceptions for different error conditions.
    """

    pass


class TimeColumnError(TimeFrameError):
    """ Exception raised for errors related to the `time_col`.

    This error is raised when the `time_col` in the TimeFrame is either 
    missing, contains unsupported types (non-numeric or non-timestamp), 
    or has invalid data like null values.

    Attributes:
        message (str): Explanation of the error.
    
    Example Usage:
    --------------
    .. code-block:: python

        if not pd.api.types.is_numeric_dtype(df[time_col]) and \
           not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            raise TimeColumnError("`time_col` must be numeric or timestamp-like.")
    """

    pass


class MixedTypesWarning(UserWarning):
    """Warning raised when mixed numeric and timestamp-like types are detected in `time_col`.

    This warning is issued when the time column contains both numeric and
    timestamp-like types, which could lead to unpredictable behavior in time
    series processing workflows.

    Example Usage:
    --------------
    .. code-block:: python

        if numeric_mask and timestamp_mask:
            warnings.warn("`time_col` contains mixed numeric and timestamp-like types.", MixedTypesWarning)
    """

    pass


class MixedTimezonesWarning(UserWarning):
    """Warning raised when mixed timezone-aware and naive timestamps are detected in `time_col`.

    This warning is issued when the time column contains a mix of timezone-aware
    and timezone-naive timestamps, which could cause errors in models that
    require consistent timestamp formats.

    Example Usage:
    --------------
    .. code-block:: python

        if df[time_col].dt.tz is not None and df[time_col].dt.tz.hasnans:
            warnings.warn("`time_col` contains mixed timezone-aware and naive timestamps.", MixedTimezonesWarning)
    """

    pass


class MixedFrequencyWarning(UserWarning):
    """Warning raised when mixed timestamp frequencies are detected in `time_col`.

    This warning is issued when the time column contains timestamps of mixed frequencies
    (e.g., daily, monthly, and yearly timestamps), which can lead to inconsistent behavior
    in time series operations that assume uniform frequency.

    Example Usage:
    --------------
    .. code-block:: python

        inferred_freq = pd.infer_freq(time_col.dropna())
        if inferred_freq is None:
            warnings.warn("`time_col` contains mixed timestamp frequencies.", MixedFrequencyWarning)
    """

    pass
