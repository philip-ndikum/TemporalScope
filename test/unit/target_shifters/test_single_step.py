# # Licensed to the Apache Software Foundation (ASF) under one
# # or more contributor license agreements.  See the NOTICE file
# # distributed with this work for additional information
# # regarding copyright ownership.  The ASF licenses this file
# # to you under the Apache License, Version 2.0 (the
# # "License"); you may not use this file except in compliance
# # with the License.  You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing,
# # software distributed under the License is distributed on an
# # "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# # KIND, either express or implied.  See the License for the
# # specific language governing permissions and limitations
# # under the License.

# """Unit Test Design for TemporalScope's SingleStepTargetShifter.

# This module implements a systematic approach to testing the SingleStepTargetShifter class
# across multiple DataFrame backends while maintaining consistency and reliability.

# Testing Philosophy
# -----------------
# The testing strategy follows three core principles:

# 1. Backend-Agnostic Operations:
#    - All DataFrame manipulations use the Narwhals API (@nw.narwhalify)
#    - Operations are written once and work across all supported backends
#    - Backend-specific code is avoided to maintain test uniformity

# 2. Fine-Grained Data Generation:
#    - PyTest fixtures provide flexible, parameterized test data
#    - Base configuration fixture allows easy overrides
#    - Each test case specifies exact data characteristics needed

# 3. Consistent Validation Pattern:
#    - All validation steps convert to Pandas for reliable comparisons
#    - Complex validations use reusable helper functions
#    - Assertions focus on business logic rather than implementation

# .. note::
#    - The sample_df fixture handles backend conversion through data_config
#    - Tests should use DataFrames as-is without additional conversion
#    - Narwhals operations are used only for validation helpers
# """

# from typing import Any, Callable, Dict, Generator, Tuple

# import narwhals as nw
# import numpy as np
# import pandas as pd
# import pytest

# from temporalscope.core.core_utils import (
#     MODE_SINGLE_STEP,
#     SupportedTemporalDataFrame,
#     get_temporalscope_backends,
# )
# from temporalscope.core.temporal_data_loader import TimeFrame
# from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series
# from temporalscope.target_shifters.single_step import SingleStepTargetShifter

# # Test Configuration Types
# DataConfigType = Callable[..., Dict[str, Any]]


# @pytest.fixture(params=get_temporalscope_backends())
# def backend(request) -> str:
#     """Fixture providing all supported backends for testing."""
#     return request.param


# @pytest.fixture
# def data_config(backend: str) -> DataConfigType:
#     """Base fixture for data generation configuration."""

#     def _config(**kwargs) -> Dict[str, Any]:
#         default_config = {
#             "num_samples": 10,
#             "num_features": 2,
#             "with_nulls": False,
#             "with_nans": False,
#             "mode": MODE_SINGLE_STEP,
#             "time_col_numeric": True,
#             "backend": backend,
#         }
#         default_config.update(kwargs)
#         return default_config

#     return _config


# @pytest.fixture
# def sample_df(data_config: DataConfigType) -> Generator[Tuple[SupportedTemporalDataFrame, str], None, None]:
#     """Generate sample DataFrame for testing."""
#     config = data_config()
#     df = generate_synthetic_time_series(**config)
#     yield df, "target"


# @pytest.fixture
# def sample_timeframe(sample_df: Tuple[SupportedTemporalDataFrame, str]) -> TimeFrame:
#     """Create TimeFrame instance for testing."""
#     df, target_col = sample_df
#     return TimeFrame(df=df, time_col="time", target_col=target_col)


# # Assertion Helpers
# @nw.narwhalify
# def assert_shifted_columns(df: SupportedTemporalDataFrame, target_col: str, n_lags: int, drop_target: bool) -> None:
#     """Verify shifted columns using Narwhals operations."""
#     # Check shifted column exists
#     shifted_col = f"{target_col}_shift_{n_lags}"
#     assert shifted_col in df.columns, f"Shifted column {shifted_col} not found"

#     # Check original target column based on drop_target
#     if drop_target:
#         assert target_col not in df.columns, f"Target column {target_col} should be dropped"
#     else:
#         assert target_col in df.columns, f"Target column {target_col} should be kept"


# @nw.narwhalify
# def assert_row_reduction(df: SupportedTemporalDataFrame, original_df: SupportedTemporalDataFrame, n_lags: int) -> None:
#     """Verify row count reduction using Narwhals operations."""
#     # Get row counts using Narwhals
#     count_expr = nw.col(df.columns[0]).count().alias("count")
#     original_count = original_df.select([count_expr])["count"][0]
#     transformed_count = df.select([count_expr])["count"][0]

#     # Verify row reduction
#     assert (
#         transformed_count == original_count - n_lags
#     ), f"Expected {original_count - n_lags} rows, got {transformed_count}"


# def test_transform_dataframe(sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
#     """Test transformation of raw DataFrame."""
#     df, target_col = sample_df
#     shifter = SingleStepTargetShifter(target_col=target_col, n_lags=1)

#     # Transform DataFrame
#     transformed_df = shifter.fit_transform(df)

#     # Verify transformation
#     assert_shifted_columns(transformed_df, target_col, 1, True)
#     assert_row_reduction(transformed_df, df, 1)


# def test_transform_timeframe(sample_timeframe: TimeFrame) -> None:
#     """Test transformation of TimeFrame instance."""
#     shifter = SingleStepTargetShifter(target_col="target", n_lags=1)

#     # Transform TimeFrame
#     transformed_tf = shifter.fit_transform(sample_timeframe)

#     # Verify TimeFrame type preservation
#     assert isinstance(transformed_tf, TimeFrame)

#     # Verify metadata preservation
#     assert transformed_tf.mode == sample_timeframe.mode
#     assert transformed_tf.backend == sample_timeframe.backend
#     assert transformed_tf.ascending == sample_timeframe.ascending

#     # Verify transformation
#     assert_shifted_columns(transformed_tf.df, "target", 1, True)
#     assert_row_reduction(transformed_tf.df, sample_timeframe.df, 1)


# def test_verbose_output(sample_df: Tuple[SupportedTemporalDataFrame, str], capfd: Any) -> None:
#     """Test verbose mode output."""
#     df, target_col = sample_df
#     shifter = SingleStepTargetShifter(target_col=target_col, n_lags=1, verbose=True)

#     # Transform with verbose output
#     shifter.fit_transform(df)

#     # Check captured output
#     captured = capfd.readouterr()
#     assert "Rows before:" in captured.out
#     assert "Rows after:" in captured.out


# def test_multiple_lags(sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
#     """Test transformation with different lag values."""
#     df, target_col = sample_df
#     n_lags = 3
#     shifter = SingleStepTargetShifter(target_col=target_col, n_lags=n_lags)

#     # Transform DataFrame
#     transformed_df = shifter.fit_transform(df)

#     # Verify transformation
#     assert_shifted_columns(transformed_df, target_col, n_lags, True)
#     assert_row_reduction(transformed_df, df, n_lags)


# def test_empty_dataframe() -> None:
#     """Test handling of empty DataFrame."""
#     empty_df = pd.DataFrame(columns=["time", "target", "feature"])
#     shifter = SingleStepTargetShifter(target_col="target")

#     with pytest.raises(ValueError, match="Cannot transform empty DataFrame"):
#         shifter.fit_transform(empty_df)


# def test_target_column_inference(sample_timeframe: TimeFrame) -> None:
#     """Test target column inference from TimeFrame."""
#     # Initialize without target_col
#     shifter = SingleStepTargetShifter(n_lags=1)

#     # Fit should infer target_col from TimeFrame
#     shifter.fit(sample_timeframe)
#     assert shifter.target_col == sample_timeframe._target_col


# def test_drop_target_option(sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
#     """Test drop_target parameter behavior."""
#     df, target_col = sample_df

#     # Test with drop_target=True
#     shifter_drop = SingleStepTargetShifter(target_col=target_col, drop_target=True)
#     transformed_drop = shifter_drop.fit_transform(df)
#     assert_shifted_columns(transformed_drop, target_col, 1, True)

#     # Test with drop_target=False
#     shifter_keep = SingleStepTargetShifter(target_col=target_col, drop_target=False)
#     transformed_keep = shifter_keep.fit_transform(df)
#     assert_shifted_columns(transformed_keep, target_col, 1, False)


# def test_fit_transform_equivalence(sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
#     """Test fit_transform equivalence to separate fit and transform."""
#     df, target_col = sample_df
#     shifter1 = SingleStepTargetShifter(target_col=target_col)
#     shifter2 = SingleStepTargetShifter(target_col=target_col)

#     # Compare fit_transform vs separate fit/transform
#     result1 = shifter1.fit_transform(df)
#     result2 = shifter2.fit(df).transform(df)

#     # Convert both to pandas for comparison
#     result1_pd = nw.from_native(result1).to_pandas()
#     result2_pd = nw.from_native(result2).to_pandas()

#     pd.testing.assert_frame_equal(result1_pd, result2_pd)


# def test_numpy_array_handling() -> None:
#     """Test handling of numpy array inputs."""
#     # Create sample numpy array
#     X = np.random.rand(10, 4)  # 10 samples, 3 features + 1 target

#     # Initialize shifter
#     shifter = SingleStepTargetShifter(n_lags=1)

#     # Fit and transform
#     transformed = shifter.fit_transform(X)

#     # Verify output
#     assert isinstance(transformed, np.ndarray)
#     assert transformed.shape[0] == X.shape[0] - 1  # One row less due to shifting
#     assert transformed.shape[1] == X.shape[1]  # Same number of columns
