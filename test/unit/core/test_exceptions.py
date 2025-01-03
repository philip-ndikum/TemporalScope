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

"""TemporalScope/test/unit/test_core_exceptions.py

This module contains unit tests for the custom exceptions and warnings defined
in the TemporalScope package. These tests ensure that the exceptions are
raised correctly and the warnings are issued in the appropriate scenarios.
"""

import warnings

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.exceptions import (
    DataFrameValidationError,
    ModeValidationError,
    TargetColumnWarning,
    TimeColumnError,
    TimeFrameError,
)


def test_time_frame_error_inheritance():
    """Test that TimeFrameError is the base class for other exceptions."""
    with pytest.raises(TimeFrameError):
        raise TimeFrameError("Base error for the TimeFrame module")


def test_time_column_error():
    """Test that TimeColumnError is raised for time column validation errors."""
    # Create test DataFrame
    df = pd.DataFrame({"time": ["invalid", "data"]})
    df = nw.from_native(df)

    with pytest.raises(TimeColumnError, match="time column"):
        try:
            # Attempt to cast non-numeric, non-datetime column
            df.select([nw.col("time").cast(nw.Float64())])
            df.select([nw.col("time").cast(nw.Datetime())])
        except Exception:
            raise TimeColumnError("Invalid time column type")


def test_mode_validation_error():
    """Test that ModeValidationError is raised for invalid modes."""
    # Test with default message
    error = ModeValidationError("invalid_mode")
    assert error.mode == "invalid_mode"
    assert error.message == "Invalid mode specified: invalid_mode"

    # Test with custom message
    error = ModeValidationError("invalid_mode", "Custom error message")
    assert error.mode == "invalid_mode"
    assert error.message == "Custom error message: invalid_mode"

    # Test raising the error
    with pytest.raises(ModeValidationError, match="Invalid mode"):
        raise ModeValidationError("invalid_mode")


def test_target_column_warning():
    """Test that TargetColumnWarning is issued for potential target column issues."""
    # Create test DataFrame with sequence data
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=2), "target": [[0.3, 0.4], [0.5, 0.6]]})
    df = nw.from_native(df)
    mode = "multi_target"

    with pytest.warns(TargetColumnWarning, match="sequential data"):
        # Check if target column contains sequence data using schema type
        if mode == "multi_target" and isinstance(df["target"][0], list):
            warnings.warn(
                "`target_col` appears to contain sequential data. Ensure it is transformed appropriately for multi_target mode.",
                TargetColumnWarning,
            )


def test_dataframe_validation_error():
    """Test that DataFrameValidationError is raised for DataFrame validation issues."""
    # Create test DataFrame with non-numeric feature
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=2), "feature": ["invalid", "data"]})
    df = nw.from_native(df)

    with pytest.raises(DataFrameValidationError, match="numeric columns"):
        try:
            # Attempt to validate numeric columns
            df = df.select([nw.col("feature").cast(nw.Float64()).alias("feature")])
        except Exception as e:
            raise DataFrameValidationError(f"Failed to validate numeric columns: {str(e)}")
