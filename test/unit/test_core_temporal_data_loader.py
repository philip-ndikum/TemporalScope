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

# TemporalScope/test/unit/test_core_temporal_data_loader.py

import warnings
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta, date, timezone
import modin.pandas as mpd
import numpy as np

import pandas as pd
import polars as pl
import pytest


from temporalscope.core.exceptions import (
    TimeColumnError, MixedTypesWarning, MixedTimezonesWarning, MixedFrequencyWarning
)


from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
)
from temporalscope.core.temporal_data_loader import TimeFrame


from datetime import datetime, timedelta, timezone
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import polars as pl
import modin.pandas as mpd


def create_sample_data(
    num_samples: int = 100, 
    num_features: int = 3, 
    empty: bool = False, 
    missing_values: bool = False, 
    mixed_types: bool = False,
    drop_columns: Optional[List[str]] = None,
    non_numeric_time: bool = False,
    empty_time: bool = False,
    mixed_numeric_and_timestamp: bool = False,
    date_like_string: bool = False,
    object_type_time_col: bool = False,
    mixed_timezones: bool = False,
    polars_specific: bool = False  
) -> Dict[str, Union[List[datetime], List[float], List[Optional[float]]]]:
    """ Create a sample dataset for scalable unit testing, supporting various edge cases.

    This function generates sample time-series data for different unit testing scenarios,
    including empty datasets, datasets with mixed data types, missing values, or different
    types of time columns. It is designed to be flexible, providing various ways to test
    data validation for time-series models.

    :param num_samples: Number of samples to generate.
    :param num_features: Number of feature columns to generate.
    :param empty: If True, generates an empty dataset.
    :param missing_values: If True, introduces missing values into the dataset.
    :param mixed_types: If True, mixes numeric and string data types in feature columns.
    :param drop_columns: List of columns to drop from the dataset.
    :param non_numeric_time: If True, replaces the `time_col` with non-numeric values.
    :param empty_time: If True, fills the `time_col` with empty values.
    :param mixed_numeric_and_timestamp: If True, mixes numeric and timestamp values in `time_col`.
    :param date_like_string: If True, fills the `time_col` with date-like string values.
    :param object_type_time_col: If True, inserts arrays or complex objects into the `time_col`.
    :param mixed_timezones: If True, mixes timestamps with and without timezone information in `time_col`.
    :param polars_specific: If True, handles edge cases specific to Polars.
    :return: A dictionary containing generated data with keys 'time', 'feature_1', ..., 'feature_n', and 'target'.
    """
    
    if empty:
        return {"time": [], "target": []}

    start_date = datetime(2021, 1, 1)

    if empty_time:
        data = {"time": [None for _ in range(num_samples)]}
    elif non_numeric_time:
        data = {"time": ["invalid_time" for _ in range(num_samples)]}
    elif mixed_numeric_and_timestamp:
        if polars_specific:
            data = {"time": [str(start_date + timedelta(days=i)) if i % 2 == 0 else float(i) for i in range(num_samples)]}
        else:
            data = {"time": [start_date + timedelta(days=i) if i % 2 == 0 else float(i) for i in range(num_samples)]}
    elif date_like_string:
        data = {"time": [f"2021-01-{i+1:02d}" for i in range(num_samples)]}
    elif object_type_time_col:
        data = {"time": [[start_date + timedelta(days=i)] for i in range(num_samples)]}
    elif mixed_timezones:
        data = {"time": [(start_date + timedelta(days=i)).replace(tzinfo=timezone.utc if i % 2 == 0 else None)
                for i in range(num_samples)]}
    else:
        data = {"time": [start_date + timedelta(days=i) for i in range(num_samples)]}

    for i in range(1, num_features + 1):
        if mixed_types:
            data[f"feature_{i}"] = [f"str_{i}" if j % 2 == 0 else j for j in range(num_samples)]
        else:
            data[f"feature_{i}"] = np.random.rand(num_samples).tolist()

    if missing_values:
        for i in range(num_samples):
            if i % 10 == 0:
                for j in range(1, num_features + 1):
                    data[f"feature_{j}"][i] = None

    data["target"] = [
        sum(data[f"feature_{j}"][i] for j in range(1, num_features + 1) if isinstance(data[f"feature_{j}"][i], float)) +
        np.random.normal(0, 0.1)
        for i in range(num_samples)
    ]

    if drop_columns:
        data = pd.DataFrame(data).drop(columns=drop_columns).to_dict(orient='list')

    return data



@pytest.mark.parametrize(
    "backend, case_type, expected_error, expected_warning, match_message",
    [
        (BACKEND_POLARS, "missing_time_col", TimeColumnError, None, r"Missing required column: time"),
        (BACKEND_PANDAS, "missing_time_col", TimeColumnError, None, r"Missing required column: time"),
        (BACKEND_MODIN, "missing_time_col", TimeColumnError, None, r"Missing required column: time"),
        (BACKEND_POLARS, "non_numeric_time_col", TimeColumnError, None, r"`time_col` must be numeric or timestamp-like"),
        (BACKEND_PANDAS, "non_numeric_time_col", TimeColumnError, None, r"`time_col` must be numeric or timestamp-like"),
        (BACKEND_MODIN, "non_numeric_time_col", TimeColumnError, None, r"`time_col` must be numeric or timestamp-like"),
        (BACKEND_PANDAS, "empty_time_col", TimeColumnError, None, r"Missing values found in `time_col`"),
        (BACKEND_POLARS, "mixed_frequencies", None, MixedFrequencyWarning, r"mixed timestamp frequencies"),
        (BACKEND_PANDAS, "mixed_frequencies", None, MixedFrequencyWarning, r"mixed timestamp frequencies"),
        (BACKEND_POLARS, "mixed_timezones", None, MixedTimezonesWarning, r"mixed timezone-aware and naive timestamps"),
        (BACKEND_PANDAS, "mixed_timezones", None, MixedTimezonesWarning, r"mixed timezone-aware and naive timestamps"),
        (BACKEND_POLARS, "date_like_string", TimeColumnError, None, r"`time_col` must be numeric or timestamp-like"),
        (BACKEND_PANDAS, "date_like_string", TimeColumnError, None, r"`time_col` must be numeric or timestamp-like"),
    ]
)
def test_validation_edge_cases(backend, case_type, expected_error, expected_warning, match_message):
    """Test validation logic under different edge cases and backends."""
    
    polars_specific = backend == BACKEND_POLARS

    if case_type == "missing_time_col":
        data = create_sample_data(drop_columns=["time"], polars_specific=polars_specific)
    elif case_type == "non_numeric_time_col":
        data = create_sample_data(non_numeric_time=True, polars_specific=polars_specific)
    elif case_type == "empty_time_col":
        data = create_sample_data(empty_time=True, polars_specific=polars_specific)
    elif case_type == "mixed_frequencies":
        data = create_sample_data(mixed_frequencies=True, polars_specific=polars_specific)
    elif case_type == "date_like_string":
        data = create_sample_data(date_like_string=True, polars_specific=polars_specific)
    elif case_type == "mixed_timezones":
        data = create_sample_data(mixed_timezones=True, polars_specific=polars_specific)

    if backend == BACKEND_POLARS:
        df = pl.DataFrame(data, strict=False)  # Allow mixed types for Polars
    elif backend == BACKEND_PANDAS:
        df = pd.DataFrame(data)
    elif backend == BACKEND_MODIN:
        df = mpd.DataFrame(data)

    if expected_error:
        with pytest.raises(expected_error, match=match_message):
            TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)
    elif expected_warning:
        with pytest.warns(expected_warning, match=match_message if match_message else None):
            TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)


        
        
# @pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
# def test_sort_data(backend):
#     """Test sorting method for various backends."""
#     data = create_sample_data(num_samples=100)
#     if backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend, sort=False)
#     # Shuffle and sort
#     if backend == BACKEND_POLARS:
#         shuffled_df = tf.get_data().sample(fraction=1.0)
#     else:
#         shuffled_df = tf.get_data().sample(frac=1).reset_index(drop=True)
#     tf.update_data(shuffled_df)
#     tf.sort_data(ascending=True)
#     sorted_df = tf.get_data()

#     # Verify sorting
#     times = sorted_df[tf.time_col].to_list() if backend == BACKEND_POLARS else sorted_df[tf.time_col].tolist()
#     assert times == sorted(times)


# @pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
# def test_update_target_col_invalid_length(backend):
#     """Test updating target column with mismatched length."""
#     data = create_sample_data(num_samples=100)
#     if backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#         new_target = pl.Series(np.random.rand(99))  # One less than expected
#     elif backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#         new_target = pd.Series(np.random.rand(99))
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)
#         new_target = mpd.Series(np.random.rand(99))

#     tf = TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)
#     with pytest.raises(ValueError):
#         tf.update_target_col(new_target)


# @pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
# def test_missing_columns(backend):
#     """Test initialization with missing required columns."""
#     data = create_sample_data(num_samples=100)
#     if backend == BACKEND_POLARS:
#         df = pl.DataFrame(data).drop(["target"])
#     elif backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data).drop(columns=["target"])
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data).drop(columns=["target"])

#     with pytest.raises(ValueError):
#         TimeFrame(df, time_col="time", target_col="target", dataframe_backend=backend)


# @pytest.mark.parametrize("backend", [BACKEND_POLARS, BACKEND_PANDAS, BACKEND_MODIN])
# def test_invalid_backend_initialization(backend):
#     """Test invalid backend during initialization."""
#     data = create_sample_data(num_samples=100)
#     if backend == BACKEND_POLARS:
#         df = pl.DataFrame(data)
#     elif backend == BACKEND_PANDAS:
#         df = pd.DataFrame(data)
#     elif backend == BACKEND_MODIN:
#         df = mpd.DataFrame(data)

#     invalid_backend = "invalid_backend"
#     with pytest.raises(ValueError):
#         TimeFrame(df, time_col="time", target_col="target", dataframe_backend=invalid_backend)

