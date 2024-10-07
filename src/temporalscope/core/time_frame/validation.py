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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from temporalscope.core import TimeFrame

from temporalscope.core import exceptions


def _validate_time_column(tf: TimeFrame) -> None:
    """Validate the time column in the DataFrame."""
    if tf.time_col not in tf.df.columns:
        raise exceptions.TimeColumnError(f"Time column '{tf.time_col}' not found in the DataFrame.")

    if not tf.df[tf.time_col].dtype.is_numeric() and not tf.df[tf.time_col].dtype.is_temporal():
        raise exceptions.TimeColumnError("Time column must be numeric or timestamp-like.")


def _validate_target_column(tf: TimeFrame) -> None:
    """Validate the target column in the DataFrame."""
    if tf.target_col not in tf.df.columns:
        raise exceptions.TimeColumnError(f"Target column '{tf.target_col}' not found in the DataFrame.")

    if not tf.df[tf.target_col].dtype.is_numeric():
        raise exceptions.TimeColumnError("Target column must be numeric.")


def _validate_no_missing_values(tf: TimeFrame) -> None:
    """Check for missing values in the DataFrame."""
    total_missing = tf.df.null_count().sum_horizontal()[0]
    if total_missing > 0:
        raise exceptions.TimeFrameError(f"DataFrame contains {total_missing} missing values.")


def _validate_only_numeric_features(tf: TimeFrame) -> None:
    """Check if all columns are numeric except the time column."""
    non_numeric_cols = [col for col in tf.df.columns if col != tf.time_col and not tf.df[col].dtype.is_numeric()]
    if len(non_numeric_cols) > 0:
        raise exceptions.TimeFrameError(f"Non-numeric columns found in the DataFrame: {non_numeric_cols}")


def validate_time_frame(tf: TimeFrame, bypass_sort: bool) -> None:
    """Validate the TimeFrame object.

    Ensure that the time column, target column, and DataFrame meet the required constraints.
    """
    if tf.time_col == tf.target_col:
        raise exceptions.TimeFrameError("Time column and target column cannot have the same name.")

    _validate_time_column(tf)
    _validate_target_column(tf)
    _validate_no_missing_values(tf)
    _validate_only_numeric_features(tf)

    if not bypass_sort:
        tf.df.sort(tf.time_col)
    else:
        warnings.warn("Bypassing DataFrame sorting validation. Please ensure the DataFrame is sorted by `time_col`.")
