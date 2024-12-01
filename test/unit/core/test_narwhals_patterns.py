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

"""Test file to understand correct Narwhals patterns.

This file contains minimal test cases to validate:
1. Basic DataFrame operations
2. Concatenation behavior
3. Eager vs Lazy evaluation
4. Type conversion patterns
"""

import narwhals as nw
import pandas as pd
import pytest


def test_basic_narwhals_operations():
    """Test basic DataFrame operations with Narwhals."""
    # Create sample data
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Convert to narwhals DataFrame
    nw_df = nw.from_native(df)

    # Test select operation
    result = nw_df.select([nw.col("a").alias("a"), nw.col("b").alias("b")])

    # Verify result is still a narwhals DataFrame
    assert hasattr(result, "to_native")


def test_narwhals_concat():
    """Test concatenation behavior with Narwhals."""
    # Create two sample DataFrames
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    # Convert both to narwhals
    nw_df1 = nw.from_native(df1)
    nw_df2 = nw.from_native(df2)

    # Test concat with select
    df1_selected = nw_df1.select([nw.col("a").alias("a"), nw.col("b").alias("b")])
    df2_selected = nw_df2.select([nw.col("a").alias("a"), nw.col("b").alias("b")])

    # Concat the selected DataFrames
    result = nw.concat([df1_selected, df2_selected])

    # Verify result is still a narwhals DataFrame
    assert hasattr(result, "to_native")

    # Convert back to pandas to verify
    result_pd = result.to_native()
    assert len(result_pd) == 4


def test_narwhals_repeat_rows():
    """Test repeating rows with Narwhals."""
    # Create sample data
    df = pd.DataFrame({"a": [1], "b": [2]})
    nw_df = nw.from_native(df)

    # Create repeated rows using select
    repeats = 3
    results = [nw_df.select([nw.col("a").alias("a"), nw.col("b").alias("b")]) for _ in range(repeats)]

    # Concat the repeated rows
    result = nw.concat(results)

    # Verify result is still a narwhals DataFrame
    assert hasattr(result, "to_native")

    # Convert to pandas to verify length
    result_pd = result.to_native()
    assert len(result_pd) == repeats


def test_narwhals_type_conversion():
    """Test type conversion patterns with Narwhals."""
    # Create sample data
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    # Convert to narwhals
    nw_df = nw.from_native(df)

    # Test type casting
    result = nw_df.select([nw.col("a").cast(nw.Float64).alias("a"), nw.col("b").cast(nw.Float64).alias("b")])

    # Verify result maintains types
    assert hasattr(result, "to_native")
    result_pd = result.to_native()
    assert result_pd["a"].dtype == "float64"
    assert result_pd["b"].dtype == "float64"


def test_narwhals_scalar_values():
    """Test scalar value handling with Narwhals."""
    # Create sample data
    df = pd.DataFrame({"a": [1.5, 2.5, 3.5]})
    nw_df = nw.from_native(df)

    # Get scalar value
    result = nw_df.select([nw.col("a").mean().cast(nw.Float64).alias("mean")])
    assert hasattr(result, "to_native")
    result_native = result.to_native()
    mean_value = float(result_native["mean"].iloc[0])

    # Verify mean value is a float
    assert isinstance(mean_value, float)


def test_narwhals_concat_mixed():
    """Test concatenation with mixed DataFrame types."""
    # Create sample data
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    # Convert first to narwhals
    nw_df1 = nw.from_native(df1)
    df1_selected = nw_df1.select([nw.col("a").alias("a"), nw.col("b").alias("b")])

    # Keep second as native
    df2_selected = df2

    # Try to concat mixed types
    with pytest.raises(NotImplementedError):
        nw.concat([df1_selected, df2_selected])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
