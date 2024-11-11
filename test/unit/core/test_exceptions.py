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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""TemporalScope/test/unit/test_core_exceptions.py

This module contains unit tests for the custom exceptions and warnings defined
in the TemporalScope package. These tests ensure that the exceptions are
raised correctly and the warnings are issued in the appropriate scenarios.
"""

import warnings
import pytest
from temporalscope.core.exceptions import (
    TimeColumnError,
    TimeFrameError,
    UnsupportedBackendError,
    ModeValidationError,
    TargetColumnWarning,
)


def test_unsupported_backend_error():
    """Test that UnsupportedBackendError is raised with the correct message."""
    with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
        raise UnsupportedBackendError("invalid_backend")


def test_time_frame_error_inheritance():
    """Test that TimeFrameError is the base class for other exceptions."""
    with pytest.raises(TimeFrameError):
        raise TimeFrameError("Base error for the TimeFrame module")


def test_time_column_error():
    """Test that TimeColumnError is raised for time column validation errors."""
    with pytest.raises(TimeColumnError, match="time column"):
        raise TimeColumnError("Error with the time column")


def test_mode_validation_error():
    """Test that ModeValidationError is raised for invalid modes."""
    with pytest.raises(ModeValidationError, match="Invalid mode"):
        raise ModeValidationError("invalid_mode")


def test_target_column_warning():
    """Test that TargetColumnWarning is issued for potential target column issues."""
    with pytest.warns(TargetColumnWarning, match="sequential data"):
        warnings.warn("`target_col` appears to contain sequential data", TargetColumnWarning)
