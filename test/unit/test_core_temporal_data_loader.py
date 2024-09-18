""" TemporalScope/test/unit/test_core_temporal_data_loader.py

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import numpy as np
import polars as pl
import pandas as pd
import modin.pandas as mpd
from temporalscope.core.temporal_data_loader import TimeFrame
from typing import Union, Dict, Any, List
from datetime import date, timedelta

def create_sample_data(num_samples: int = 100, num_features: int = 3) -> Dict[str, Union[List[date], List[float], List[str]]]:
    """Create a sample data dictionary representative of a time series ML dataset."""
    start_date = date(2021, 1, 1)
    
    data = {
        "time": [start_date + timedelta(days=i) for i in range(num_samples)],
        "id": [f"ID_{i%3}" for i in range(num_samples)],  # 3 different IDs cycling
    }
    
    # Add feature columns
    for i in range(num_features):
        data[f"feature_{i+1}"] = np.random.rand(num_samples).tolist()
    
    # Add a target column (let's assume it's a function of the features plus some noise)
    data["target"] = [sum(data[f"feature_{j+1}"][i] for j in range(num_features)) + np.random.normal(0, 0.1) 
                      for i in range(num_samples)]
    
    return data

@pytest.fixture(params=["pd", "pl", "mpd"])
def sample_df(request):
    """Fixture for creating sample DataFrames for each backend."""
    data = create_sample_data()
    if request.param == "pd":
        return pd.DataFrame(data)
    elif request.param == "pl":
        return pl.DataFrame(data)
    elif request.param == "mpd":
        return mpd.DataFrame(data)

@pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
def test_initialize(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
    """Test TimeFrame initialization with various backends."""
    tf = TimeFrame(sample_df, time_col="time", target_col="target", backend=backend)
    
    assert tf.backend == backend
    assert tf.time_col == "time"
    assert tf.target_col == "target"
    assert tf.id_col is None

    if backend == "pd":
        assert isinstance(tf.get_data(), pd.DataFrame)
    elif backend == "pl":
        assert isinstance(tf.get_data(), pl.DataFrame)
    elif backend == "mpd":
        assert isinstance(tf.get_data(), mpd.DataFrame)

# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_initialize_with_id(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     """Test TimeFrame initialization with ID column."""
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", id_col="id", backend=backend)
    
#     assert tf.id_col == "id"

# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_validate_columns(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     """Test column validation."""
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", backend=backend)
#     tf.validate_columns()  # Should not raise an error

#     with pytest.raises(ValueError):
#         TimeFrame(sample_df, time_col="non_existent", target_col="target", backend=backend)


# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_get_data(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     """Test get_data method."""
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", backend=backend)
#     assert tf.get_data().shape == sample_df.shape



# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_update_data(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     """Test update_data method."""
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", backend=backend)
    
#     new_data = create_sample_data()
#     new_data["target"] = [x * 2 for x in new_data["target"]]  # Double the target values
    
#     if backend == "pd":
#         new_df = pd.DataFrame(new_data)
#     elif backend == "pl":
#         new_df = pl.DataFrame(new_data)
#     else:
#         new_df = mpd.DataFrame(new_data)

#     tf.update_data(new_df)
    
#     if backend == "pl":
#         assert tf.get_data()["target"].to_list() == new_df["target"].to_list()
#     else:
#         assert tf.get_data()["target"].tolist() == new_df["target"].tolist()

# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_update_target_col(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     """Test update_target_col method."""
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", backend=backend)
    
#     new_target = [x * 3 for x in range(100)]  # Triple the values
    
#     if backend == "pd":
#         new_target_series = pd.Series(new_target)
#     elif backend == "pl":
#         new_target_series = pl.Series(new_target)
#     else:
#         new_target_series = mpd.Series(new_target)

#     tf.update_target_col(new_target_series)
    
#     if backend == "pl":
#         assert tf.get_data()["target"].to_list() == new_target
#     else:
#         assert tf.get_data()["target"].tolist() == new_target
        

# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_sort_data(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", id_col="id", backend=backend, sort=True)
#     sorted_df = tf.get_data()
    
#     if backend == "pl":
#         time_values = sorted_df["time"].to_list()
#     else:
#         time_values = sorted_df["time"].tolist()
    
#     assert time_values == sorted(time_values)
        
# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_get_grouped_data(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", id_col="id", backend=backend)
#     grouped_data = tf.get_grouped_data()
    
#     if backend == "pl":
#         assert grouped_data.shape[0] == len(set(sample_df["id"].to_list()))
#     else:
#         assert grouped_data.shape[0] == len(set(sample_df["id"].tolist()))

#     with pytest.raises(ValueError):
#         tf_without_id = TimeFrame(sample_df, time_col="time", target_col="target", backend=backend)
#         tf_without_id.get_grouped_data()


# @pytest.mark.parametrize("backend", ["pd", "pl", "mpd"])
# def test_check_duplicates(sample_df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str):
#     tf = TimeFrame(sample_df, time_col="time", target_col="target", id_col="id", backend=backend)
#     tf.check_duplicates()  # Should not raise an error

#     # Create a DataFrame with duplicates
#     duplicate_data = sample_df.copy()
#     if backend == "pl":
#         duplicate_data = duplicate_data.with_columns(pl.col("time").shift(-1))
#     else:
#         duplicate_data.loc[1:, "time"] = duplicate_data.loc[:98, "time"].values
    
#     tf_with_duplicates = TimeFrame(duplicate_data, time_col="time", target_col="target", id_col="id", backend=backend)
    
#     with pytest.raises(ValueError):
#         tf_with_duplicates.check_duplicates()
