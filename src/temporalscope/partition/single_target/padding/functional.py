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

"""TemporalScope/src/temporalscope/partition/padding/functional.py

This module provides backend-agnostic utility functions for padding time-series DataFrames.
All functions in this module follow a functional design pattern using Narwhals' API for
deferred and optimized execution. These utilities are designed to operate on `SupportedTemporalDataFrame`
objects, ensuring flexibility for end-users to integrate custom parallelization and optimization strategies.

These functions are suitable for use in partitioning workflows or standalone operations, with an emphasis
on numerical data compatibility. Users are expected to preprocess their DataFrames (e.g., ensuring numerical columns
and handling any specific column semantics like `time_col` or `target_col`) before applying these padding utilities.

Engineering Design:
-------------------
1. Functional Design:
    - Stateless functions that return padded DataFrames without modifying inputs.
    - Compatible with distributed or batch processing frameworks for scalability.
2. Validation:
    - All input DataFrames must conform to `SupportedTemporalDataFrame` standards, as defined in `core_utils.py`.
    - Explicit checks ensure all columns are numeric and free of null or NaN values.


Examples
--------
```python
import pandas as pd
import numpy as np
from temporalscope.partition.padding.functional import zero_pad

df = pd.DataFrame({"feature_1": [10, 20], "feature_2": [30, 40], "target": [50, 60]})
padded_df = zero_pad(df, target_len=5, pad_value=0, padding="post")
print(padded_df)
```

Notes
-----
This module draws inspiration from industry-standard patterns, including:
- TensorFlow's `TimeseriesGenerator` for its emphasis on preprocessing flexibility.
- PyTorch's `Dataset` API for its focus on functional design and data transformations.
- FastAI's modular `TSDataLoaders` for encouraging separation of concerns in time-series workflows.

Refer to the API documentation for further details on usage patterns and constraints.


DataFrame Evaluation Modes:
----------------------------

| Mode | Key Characteristics | Type Handling |
|------|---------------------|---------------|
| Eager | - Immediate execution <br>- Direct computation <br>- Memory-bound ops | - Use schema for types <br>- Get Narwhals types direct <br>- Narwhals ops supported |
| Lazy | - Deferred execution <br>- Optimized planning <br>- Large-scale data | - Must use native dtype <br>- Schema not supported <br>- Native type ops required |

Critical Rules:
---------------
- Never mix eager/lazy operations
- Use narwhals operations consistently, noting Dask requires special handling for concatenation
- Convert to native format only when required
- Maintain same mode in concatenations, using backend-specific methods when needed (e.g. dask.concat)

See Also
--------
1. Dwarampudi, M. and Reddy, N.V., 2019. Effects of padding on LSTMs and CNNs. arXiv preprint arXiv:1903.07288.
2. Lafabregue, B., Weber, J., et al., 2022. End-to-end deep representation learning for time
series clustering: a comparative study. Data Mining and Knowledge Discovery.

"""

import narwhals as nw
from narwhals.typing import FrameT

from temporalscope.core.core_utils import count_dataframe_column_nulls, is_dataframe_empty


@nw.narwhalify
def mean_fill_pad(
    df: FrameT,
    target_len: int,
    padding: str = "post",
) -> FrameT:
    """Pad a DataFrame to target length by filling with column means.

    A simple padding function that extends a DataFrame to a target length by adding
    rows filled with each column's mean value. Handles both eager and lazy evaluation.

    Parameters
    ----------
    df : FrameT
        DataFrame to pad
    target_len : int
        Desired length after padding
    padding : str
        Where to add padding ('pre' or 'post')

    Returns
    -------
    FrameT
        Padded DataFrame

    Raises
    ------
    ValueError
        If target_len <= current length or invalid padding direction
    """
    # Validate data quality first
    null_counts = count_dataframe_column_nulls(df, list(df.columns))
    if any(count > 0 for count in null_counts.values()):
        raise ValueError("Cannot process data containing null values")

    # Validate padding direction
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be 'pre' or 'post'")

    # Get current length safely for both eager/lazy evaluation
    count_expr = nw.col(df.columns[0]).count().cast(nw.Int64).alias("count")
    count_result = df.select([count_expr])

    if is_dataframe_empty(count_result):
        current_len = count_result.collect()["count"][0]
    else:
        current_len = count_result["count"][0]

    # Handle PyArrow scalar conversion safely
    if hasattr(current_len, "as_py"):
        current_len = current_len.as_py()  # Convert PyArrow scalar to Python int

    # Validate target length
    if target_len <= current_len:
        raise ValueError(f"target_len ({target_len}) must be greater than current length ({current_len})")

    # Calculate means for each column
    means = {}
    for col in df.columns:
        mean_expr = nw.col(col).mean().cast(nw.Float64).alias("mean")
        mean_result = df.select([mean_expr])

        mean_val = mean_result.collect()["mean"][0] if is_dataframe_empty(mean_result) else mean_result["mean"][0]

        # Handle PyArrow scalar for mean values too
        if hasattr(mean_val, "as_py"):
            mean_val = mean_val.as_py()

        means[col] = mean_val

    # Create padding DataFrame with means
    padding_rows = df.select([nw.lit(means[col]).alias(col) for col in df.columns])

    # Calculate number of padding rows needed
    padding_count = target_len - current_len

    if is_dataframe_empty(padding_rows):
        # For lazy evaluation backends (e.g. dask):
        # 1. Create a single-row DataFrame with scalar values
        # 2. Use dask's native concat for proper lazy evaluation
        # 3. Each column must be created independently to maintain proper lazy evaluation

        # Create a single row of padding values
        padding_df = df.select(
            [
                nw.lit(means[col]).alias(col)  # Create scalar expressions
                for col in df.columns
            ]
        ).head(1)  # Get single row using head() instead of take()

        # Create the padding DataFrames
        padding_dfs = [padding_df] * padding_count
    else:
        # For eager evaluation backends (pandas, polars, pyarrow):
        # 1. List multiplication efficiently creates multiple references
        # 2. This works because eager evaluation immediately computes the values
        # 3. The * operator properly handles the DataFrame copying
        padding_dfs = [padding_rows] * padding_count

    # Original data DataFrame
    original_df = df.select([nw.col(col) for col in df.columns])

    # Combine based on padding direction
    if padding == "pre":
        all_dfs = padding_dfs + [original_df]
    else:
        all_dfs = [original_df] + padding_dfs

    return nw.concat(all_dfs)
