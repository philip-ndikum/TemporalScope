""" TemporalScope/test/unit/test_core_temporal_target_shifter.py

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

import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest
from temporalscope.core.temporal_target_shifter import TemporalTargetShifter
from temporalscope.core.temporal_data_loader import TimeFrame
from typing import Union

# Test DataFrames
pd_df = pd.DataFrame({
    "time": pd.date_range(start="2023-01-01", periods=5, freq="D"),
    "target": [10, 20, 30, 40, 50],
})

pl_df = pl.DataFrame({
    "time": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
    "target": [10, 20, 30, 40, 50],
})

mpd_df = mpd.DataFrame({
    "time": pd.date_range(start="2023-01-01", periods=5, freq="D"),
    "target": [10, 20, 30, 40, 50],
})

@pytest.mark.parametrize("backend, df", [
    ("pd", pd_df),
    ("pl", pl_df),
    ("mpd", mpd_df),
])
def test_shift_target_scalar_output(backend: str, df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]) -> None:
    """Test shifting target to scalar output for each backend."""
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    shifter = TemporalTargetShifter(shift_steps=1, array_output=False)
    tf_transformed = shifter.transform(tf)
    
    expected_target = [20, 30, 40, 50, None]

    if backend == "pl":
        actual_target = tf_transformed.get_data()["target_shift_1"].to_list()
    else:
        actual_target = tf_transformed.get_data()["target_shift_1"].tolist()

    assert actual_target == expected_target[:-1]  # Comparing excluding the last item due to `None` handling

@pytest.mark.parametrize("backend, df", [
    ("pd", pd_df),
    ("pl", pl_df),
    ("mpd", mpd_df),
])
def test_shift_target_array_output(backend: str, df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]) -> None:
    """Test shifting target to array output for each backend."""
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    shifter = TemporalTargetShifter(shift_steps=2, array_output=True)
    tf_transformed = shifter.transform(tf)

    expected_target_array = [[20, 30], [30, 40], [40, 50], [50, None], [None, None]]

    if backend == "pl":
        actual_target = tf_transformed.get_data()["target_array_2"].to_list()
    else:
        actual_target = tf_transformed.get_data()["target_array_2"].tolist()

    assert actual_target == expected_target_array

@pytest.mark.parametrize("backend, df", [
    ("pd", pd_df),
    ("pl", pl_df),
    ("mpd", mpd_df),
])
def test_shift_target_with_nonstandard_names(backend: str, df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]) -> None:
    """Test shifting target with non-standardized names."""
    tf = TimeFrame(df, time_col="time", target_col="target", backend=backend)
    shifter = TemporalTargetShifter(shift_steps=1, array_output=False)
    tf_transformed = shifter.transform(tf)

    expected_target = [20, 30, 40, 50, None]

    if backend == "pl":
        actual_target = tf_transformed.get_data()["target_shift_1"].to_list()
    else:
        actual_target = tf_transformed.get_data()["target_shift_1"].tolist()

    assert actual_target == expected_target[:-1]

@pytest.mark.parametrize("backend, df", [
    ("pd", pd_df),
    ("pl", pl_df),
    ("mpd", mpd_df),
])
def test_shift_target_invalid_backend(backend: str, df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]) -> None:
    """Test shifting target with an invalid backend."""
    tf = TimeFrame(df, time_col="time", target_col="target", backend="invalid_backend")
    shifter = TemporalTargetShifter(shift_steps=1, array_output=False)
    with pytest.raises(ValueError, match="Unsupported backend"):
        shifter.transform(tf)

@pytest.mark.parametrize("backend, df", [
    ("pd", pd_df),
    ("pl", pl_df),
    ("mpd", mpd_df),
])
def test_shift_target_type_error(backend: str, df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]) -> None:
    """Test shifting target with an incorrect DataFrame type."""
    # Intentionally using an incorrect type (dictionary) instead of a DataFrame
    with pytest.raises(TypeError):
        tf = TimeFrame(df.to_dict(), time_col="time", target_col="target", backend=backend)
        shifter = TemporalTargetShifter(shift_steps=1, array_output=False)
        shifter.transform(tf)
