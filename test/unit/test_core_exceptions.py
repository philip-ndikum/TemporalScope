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

""" TemporalScope/test/unit/test_core_exceptions.py

This module contains unit tests for the custom exceptions and warnings defined 
in the TemporalScope package. These tests ensure that the exceptions are 
raised correctly and the warnings are issued in the appropriate scenarios.
"""

import pytest
import warnings

from temporalscope.core.exceptions import (
    TimeFrameError,
    TimeColumnError,
    MixedTypesWarning,
    MixedTimezonesWarning,
    MixedFrequencyWarning  
)

def test_time_frame_error_inheritance():
    """Test that TimeFrameError is the base class for other exceptions."""
    with pytest.raises(TimeFrameError):
        raise TimeFrameError("Base error for the TimeFrame module")


def test_time_column_error():
    """Test that TimeColumnError is raised for time column validation errors."""
    with pytest.raises(TimeColumnError):
        raise TimeColumnError("Error with the time column")


def test_mixed_types_warning():
    """Test that MixedTypesWarning is issued when mixed types are detected."""
    with pytest.warns(MixedTypesWarning, match="Mixed numeric and timestamp-like types"):
        warnings.warn("Mixed numeric and timestamp-like types", MixedTypesWarning)


def test_mixed_timezones_warning():
    """Test that MixedTimezonesWarning is issued for mixed timezone-aware and naive timestamps."""
    with pytest.warns(MixedTimezonesWarning, match="Mixed timezone-aware and naive timestamps"):
        warnings.warn("Mixed timezone-aware and naive timestamps", MixedTimezonesWarning)


def test_mixed_frequency_warning():
    """Test that MixedFrequencyWarning is issued when mixed timestamp frequencies are detected."""
    with pytest.warns(MixedFrequencyWarning, match="Mixed timestamp frequencies"):
        warnings.warn("Mixed timestamp frequencies", MixedFrequencyWarning)
