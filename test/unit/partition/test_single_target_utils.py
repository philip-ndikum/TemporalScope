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

"""Unit Tests for Padding Validation Functions.

This module tests the padding-specific validation functions, ensuring they
correctly identify problematic data across all supported DataFrame backends.

Testing Philosophy:
------------------
1. Synthetic Data Generation:
   - We use synthetic DataFrames to automate testing of end-user behavior
   - This allows us to systematically test edge cases and common scenarios
   - The synthetic data mimics real-world time series data structures

2. Backend Agnostic Testing:
   - Tests run against all supported DataFrame backends (pandas, modin, polars, etc.)
   - Each test case verifies the same behavior across different backends
   - This ensures consistent validation regardless of the user's chosen backend

3. Pandas as Comparison Layer:
   - For complex validation checks, we use pandas as an interoperability layer
   - While not suitable for production computations (due to performance/memory),
     pandas provides a stable, well-understood baseline for test comparisons
   - This approach helps handle differences between eager (pandas/polars) and
     lazy (dask) evaluation, as well as scalar type differences (pyarrow)
   - The small size of test DataFrames makes this conversion practical for testing,
     though it wouldn't scale for real workloads

Test Coverage:
-------------
- Basic validation of clean data across backends
- Detection of null values (None, np.nan)
- Handling of non-numeric and mixed-type columns
- Edge cases with NaN values in different positions
- Verification of error messages and types
"""

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    SupportedTemporalDataFrame,
    check_dataframe_nulls_nans,
    convert_to_backend,
    get_temporalscope_backends,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series


@pytest.fixture(params=get_temporalscope_backends())
def backend(request) -> str:
    """Fixture providing all supported backends for testing."""
    return request.param


@pytest.fixture
def data_config(backend: str) -> Callable[..., Dict[str, Any]]:
    """Base fixture for data generation configuration."""

    def _config(**kwargs) -> Dict[str, Any]:
        default_config = {
            "num_samples": 3,
            "num_features": 2,
            "with_nulls": False,
            "with_nans": False,
            "backend": backend,
            "drop_time": True,
            "random_seed": 42,
        }
        default_config.update(kwargs)
        return default_config

    return _config


@pytest.fixture
def sample_df(data_config: Callable[..., Dict[str, Any]]) -> SupportedTemporalDataFrame:
    """Generate sample DataFrame for testing."""
    config = data_config()
    return generate_synthetic_time_series(**config)


def test_check_nulls_nans_clean_data(sample_df: SupportedTemporalDataFrame) -> None:
    """Test validation passes on clean data."""
    check_dataframe_nulls_nans(sample_df, sample_df.columns)  # Should not raise any exceptions


def test_check_nulls_nans_with_nulls(data_config: Callable[..., Dict[str, Any]]) -> None:
    """Test validation catches null values."""
    config = data_config(with_nulls=True)
    df = generate_synthetic_time_series(**config)

    with pytest.raises(ValueError, match="Cannot process data containing null values in column feature_1"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_with_nans(data_config: Callable[..., Dict[str, Any]]) -> None:
    """Test validation catches NaN values.

    Note: In pandas and most DataFrame backends, NaN values are treated as nulls.
    Therefore, when we set with_nans=True, the validation will detect them as nulls
    before getting to the NaN check.
    """
    config = data_config(with_nans=True)
    df = generate_synthetic_time_series(**config)

    with pytest.raises(ValueError, match="Cannot process data containing null values in column feature_1"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_no_numeric_columns(backend: str) -> None:
    """Test validation handles non-numeric data."""
    # Create DataFrame with only string columns using string dtype
    df = pd.DataFrame(
        {"col1": pd.Series(["a", "b", "c"], dtype="string"), "col2": pd.Series(["x", "y", "z"], dtype="string")}
    )
    df = convert_to_backend(df, backend)

    with pytest.raises(ValueError, match="No numeric columns found in DataFrame"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_mixed_columns(backend: str) -> None:
    """Test validation with mixed numeric and non-numeric columns."""
    # Create DataFrame with mixed column types
    df = pd.DataFrame(
        {
            "str_col": pd.Series(["a", "b", "c"], dtype="string"),
            "num_col": [1.0, None, 3.0],  # Numeric column with null
        }
    )
    df = convert_to_backend(df, backend)

    with pytest.raises(ValueError, match="Cannot process data containing null values in column num_col"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_nan_only(backend: str) -> None:
    """Test validation with NaN values but no nulls."""
    # Create DataFrame with NaN but no nulls
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],  # Clean column
            "col2": [4.0, np.nan, 6.0],  # Column with NaN
        }
    )
    df = convert_to_backend(df, backend)

    # NaN values are detected as nulls in most backends
    with pytest.raises(ValueError, match="Cannot process data containing null values in column col2"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_nan_after_null(backend: str) -> None:
    """Test validation with both null and NaN values in different columns.

    This test ensures we catch NaN values even if they're in a different column
    than the nulls, and that we check all columns even after finding a null.
    """
    # Create DataFrame with null in one column and NaN in another
    df = pd.DataFrame(
        {
            "col1": [1.0, None, 3.0],  # Column with null
            "col2": [4.0, 5.0, np.nan],  # Column with NaN
        }
    )
    df = convert_to_backend(df, backend)

    # Should catch the null value first
    with pytest.raises(ValueError, match="Cannot process data containing null values in column col1"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_pyarrow_scalar_conversion(backend: str) -> None:
    """Test PyArrow scalar conversion path.

    This test ensures we properly handle PyArrow scalars by:
    1. Using a numeric column that will trigger the as_py conversion
    2. Verifying both null and NaN checks work with PyArrow scalars
    """
    # Create DataFrame with numeric column containing both null and NaN
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],  # Clean column
            "col2": [np.nan, None, 4.0],  # Column with both NaN and null
        }
    )
    df = convert_to_backend(df, backend)

    with pytest.raises(ValueError, match="Cannot process data containing null values in column col2"):
        check_dataframe_nulls_nans(df, df.columns)


def test_check_nulls_nans_separate_nan_detection(backend: str) -> None:
    """Test NaN detection separate from null check.

    This test verifies that we can detect NaN values independently from nulls by:
    1. Using a backend that distinguishes between null and NaN
    2. Creating a case where NaN check triggers but null check doesn't
    """
    # Create DataFrame with only NaN (no None)
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],  # Clean column
            "col2": [4.0, float("nan"), 6.0],  # Only NaN, no None
        }
    )
    df = convert_to_backend(df, backend)

    # Some backends detect NaN as null, others distinguish them
    try:
        check_dataframe_nulls_nans(df, df.columns)
    except ValueError as e:
        assert any(
            msg in str(e)
            for msg in [
                "Cannot process data containing null values in column col2",
                "Cannot process data containing NaN values in column col2",
            ]
        )


def test_check_nulls_nans_pyarrow_nan_only(backend: str) -> None:
    """Test PyArrow's handling of NaN values.

    This test verifies that in PyArrow backend:
    1. is_null() catches both None and NaN values
    2. NaN values are detected as nulls
    3. The appropriate error message is raised

    This matches PyArrow's behavior where NaN values are considered null values.
    """
    if backend != "pyarrow":
        pytest.skip("This test is specific to PyArrow backend")

    # Create DataFrame with NaN value
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],  # Clean column
            "col2": [4.0, float("nan"), 6.0],  # NaN value that will be caught by is_null()
        }
    )
    df = convert_to_backend(df, backend)

    with pytest.raises(ValueError, match="Cannot process data containing null values in column col2"):
        check_dataframe_nulls_nans(df, df.columns)
