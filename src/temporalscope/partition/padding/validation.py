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

"""TemporalScope/src/temporalscope/partition/padding/validation.py

This module provides validation functions specifically for padding operations
in the TemporalScope padding module. These functions ensure data quality and
compatibility before padding operations are applied.

.. note::
   The order of null/NaN checks is important:
   1. Check for null values using is_null() - This catches both None and np.nan
   2. Check for NaN values using (x != x) - This only catches np.nan

   This ordering ensures consistent behavior across different DataFrame backends
   (pandas, polars, pyarrow) because:
   - Both None and np.nan are detected as null values by is_null()
   - Only np.nan values are detected by the NaN != NaN comparison
   - If we checked NaNs first, we would get inconsistent results since some backends
     treat None and np.nan differently

.. seealso::
   In the current narwhals implementation (particularly in _arrow/dataframe.py),
   all supported backends catch NaN values in their is_null() implementation.
   This means the NaN check is currently unreachable, but we keep it for
   compatibility with potential future backends that might distinguish between
   null and NaN values.
"""

import narwhals as nw

from temporalscope.core.core_utils import SupportedTemporalDataFrame, is_lazy_evaluation


@nw.narwhalify
def check_for_nulls_nans(df: SupportedTemporalDataFrame) -> None:
    """Check if DataFrame has null or NaN values that would prevent padding.

    This function checks numeric columns for both null and NaN values in a specific order:
    1. First checks for null values (catches both None and np.nan)
    2. Then checks for NaN values (catches only np.nan)

    This ordering ensures consistent behavior across different DataFrame backends
    since some backends treat None and np.nan differently.

    .. note::
        The PyArrow backend's is_null() implementation catches both None and np.nan values,
        meaning the NaN check and its PyArrow scalar conversion may not be executed.
        This is expected behavior and the code is kept for compatibility with other
        backends that might handle NaN values differently.

    :param df: DataFrame to check before padding
    :type df: SupportedTemporalDataFrame
    :raises ValueError: If null or NaN values are found
    """
    # Get all numeric columns in sorted order to ensure consistent checking
    numeric_cols = []
    for col in sorted(df.columns):
        # Check if column is numeric using a simple count operation
        try:
            df.select([nw.col(col).sum().alias("sum")])
            numeric_cols.append(col)
        except Exception:
            continue

    if not numeric_cols:
        raise ValueError("No numeric columns found in DataFrame")

    # Check each numeric column for nulls/nans
    for col in numeric_cols:
        # Check for nulls first (catches both None and np.nan)
        null_check = df.select([nw.col(col).is_null().any().cast(nw.Int64).alias("has_null")])

        # Handle both lazy and eager evaluation
        if is_lazy_evaluation(null_check):
            has_null = null_check.collect()["has_null"][0]
        else:
            has_null = null_check.to_pandas()["has_null"].iloc[0]

        if has_null:
            raise ValueError(f"Cannot process data containing null values in column {col}")
