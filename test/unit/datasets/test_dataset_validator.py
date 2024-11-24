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

"""Unit Test Design for TemporalScope's DatasetValidator.

This module implements a systematic approach to testing the DatasetValidator class
across multiple DataFrame backends while maintaining consistency and reliability.

Testing Philosophy
-----------------
The testing strategy follows three core principles:

1. Backend-Agnostic Operations:
   - All DataFrame manipulations use the Narwhals API (@nw.narwhalify)
   - Operations are written once and work across all supported backends
   - Backend-specific code is avoided to maintain test uniformity

2. Fine-Grained Data Generation:
   - PyTest fixtures provide flexible, parameterized test data
   - Base configuration fixture allows easy overrides
   - Each test case specifies exact data characteristics needed

3. Consistent Validation Pattern:
   - All validation steps convert to Pandas for reliable comparisons
   - Complex validations use reusable helper functions
   - Assertions focus on business logic rather than implementation

.. note::
   - The sample_df fixture handles backend conversion through data_config
   - Tests should use DataFrames as-is without additional conversion
   - Narwhals operations are used only for validation helpers
"""

from typing import Any, Callable, Dict, Generator, Tuple

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.core_utils import SupportedTemporalDataFrame, get_temporalscope_backends
from temporalscope.datasets.dataset_validator import DatasetValidator, ValidationResult
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Test Configuration Types
DataConfigType = Callable[..., Dict[str, Any]]


@pytest.fixture(params=get_temporalscope_backends())
def backend(request) -> str:
    """Fixture providing all supported backends for testing."""
    return request.param


@pytest.fixture
def data_config(backend: str) -> DataConfigType:
    """Base fixture for data generation configuration."""

    def _config(**kwargs) -> Dict[str, Any]:
        default_config = {
            "num_samples": 100,
            "num_features": 4,
            "with_nulls": False,
            "with_nans": False,
            "backend": backend,
        }
        default_config.update(kwargs)
        return default_config

    return _config


@pytest.fixture
def sample_df(data_config: DataConfigType) -> Generator[Tuple[SupportedTemporalDataFrame, str], None, None]:
    """Generate sample DataFrame for testing."""
    config = data_config()
    df = generate_synthetic_time_series(**config)
    yield df, "target"


@pytest.fixture
def validator() -> DatasetValidator:
    """Create DatasetValidator with test thresholds."""
    return DatasetValidator(
        min_samples=50,
        max_samples=150,
        min_features=2,
        max_features=5,
        max_feature_ratio=0.1,
        min_unique_values=5,
        max_categorical_values=3,
        class_imbalance_threshold=1.5,
        enable_warnings=False,
    )


def _get_scalar_value(result, column: str) -> int:
    """Helper function to get scalar value from different DataFrame backends."""
    if hasattr(result, "collect"):  # For LazyFrame
        value = result.collect()[column][0]
    else:
        value = result[column][0]

    # Convert PyArrow scalar to Python int
    if hasattr(value, "as_py"):
        value = value.as_py()

    return int(value)


# Assertion Helpers
@nw.narwhalify
def assert_validation_result(result: ValidationResult, expected_pass: bool, expected_message: str = None) -> None:
    """Verify validation result using Narwhals operations."""
    assert result.passed == expected_pass
    if expected_message:
        assert expected_message in result.message


@nw.narwhalify
def assert_validation_details(result: ValidationResult, expected_keys: list) -> None:
    """Verify validation result details using Narwhals operations."""
    assert result.details is not None
    for key in expected_keys:
        assert key in result.details


def test_sample_size_validation(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test sample size validation across different scenarios."""
    # Test with too few samples
    small_df = generate_synthetic_time_series(**data_config(num_samples=40))
    result = validator._check_sample_size(small_df)
    assert_validation_result(result, False, "fewer than recommended minimum")
    assert_validation_details(result, ["num_samples"])
    assert result.details["num_samples"] == 40

    # Test with too many samples
    large_df = generate_synthetic_time_series(**data_config(num_samples=200))
    result = validator._check_sample_size(large_df)
    assert_validation_result(result, False, "more than recommended maximum")
    assert_validation_details(result, ["num_samples"])
    assert result.details["num_samples"] == 200

    # Test with acceptable sample size
    good_df = generate_synthetic_time_series(**data_config(num_samples=100))
    result = validator._check_sample_size(good_df)
    assert_validation_result(result, True)
    assert_validation_details(result, ["num_samples"])
    assert result.details["num_samples"] == 100


def test_feature_count_validation(
    validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]
) -> None:
    """Test feature count validation."""
    df, _ = sample_df
    result = validator._check_feature_count(df)
    assert_validation_result(result, True)
    assert_validation_details(result, ["num_features"])
    assert result.details["num_features"] == 4


def test_feature_ratio_validation(
    validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]
) -> None:
    """Test feature-to-sample ratio validation."""
    df, _ = sample_df
    result = validator._check_feature_ratio(df)
    assert_validation_result(result, True)
    assert_validation_details(result, ["ratio"])
    assert result.details["ratio"] == 4 / 100  # 4 features, 100 samples


def test_numerical_uniqueness_validation(
    validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]
) -> None:
    """Test numerical feature uniqueness validation."""
    df, _ = sample_df
    result = validator._check_feature_variability(df)
    assert_validation_result(result, True)
    assert "numeric_feature" in result.details


def test_class_balance_validation(
    validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]
) -> None:
    """Test class balance validation."""
    df, target_col = sample_df
    result = validator._check_class_balance(df, target_col)
    assert_validation_result(result, True)
    assert_validation_details(result, ["class_counts"])


def test_validate_all(validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
    """Test running all validation checks."""
    df, target_col = sample_df
    results = validator.transform(df, target_col=target_col)
    assert all(result.passed for result in results.values())
    assert set(results.keys()) >= {"sample_size", "feature_count", "feature_ratio"}


def test_selective_validation(validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
    """Test running only selected validation checks."""
    df, _ = sample_df
    validator.checks_to_run = {"sample_size", "feature_count"}
    results = validator.transform(df)
    assert set(results.keys()) == {"sample_size", "feature_count"}


def test_lazy_evaluation(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test validation with lazy evaluation."""
    config = data_config()
    config["backend"] = "polars"  # Force Polars backend for lazy evaluation
    df = generate_synthetic_time_series(**config).lazy()
    results = validator.transform(df, target_col="target")
    assert all(result.passed for result in results.values())


def test_invalid_checks() -> None:
    """Test validator creation with invalid checks."""
    with pytest.raises(ValueError, match="Invalid checks"):
        DatasetValidator(checks_to_run=["invalid_check"])


def test_print_report(
    validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str], capsys: Any
) -> None:
    """Test validation report printing."""
    df, target_col = sample_df
    results = validator.transform(df, target_col=target_col)
    validator.print_report(results)
    captured = capsys.readouterr()
    assert "Dataset Validation Report" in captured.out
    assert "recommendations" in captured.out


def test_edge_cases(validator: DatasetValidator) -> None:
    """Test validation with edge cases."""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    results = validator.transform(empty_df)
    assert not results["sample_size"].passed

    # Single column DataFrame
    single_col_df = pd.DataFrame({"col": range(100)})
    results = validator.transform(single_col_df)
    assert not results["feature_count"].passed

    # All null values
    null_df = pd.DataFrame({"col": [None] * 100})
    results = validator.transform(null_df)
    assert not results["feature_variability"].passed


def test_fit_transform_equivalence(
    validator: DatasetValidator, sample_df: Tuple[SupportedTemporalDataFrame, str]
) -> None:
    """Test fit_transform equivalence to separate fit and transform."""
    df, target_col = sample_df

    # Compare fit_transform vs separate fit/transform
    result1 = validator.fit_transform(df, target_col=target_col)
    result2 = validator.fit(df).transform(df, target_col=target_col)

    # Compare results
    assert result1.keys() == result2.keys()
    for key in result1:
        assert result1[key].passed == result2[key].passed
        assert result1[key].message == result2[key].message
        assert result1[key].details == result2[key].details


def test_narwhals_integration(sample_df: Tuple[SupportedTemporalDataFrame, str]) -> None:
    """Test integration with Narwhals operations.

    This test verifies that:
    1. We can use narwhals operations (select, col) on the input DataFrame
    2. The validator can process a narwhalified DataFrame
    3. The validator returns results in the expected format
    """
    df, target_col = sample_df

    # Test with narwhalified DataFrame
    @nw.narwhalify
    def process_df(df):
        # Select all columns explicitly instead of using "*"
        return df.select([nw.col(col) for col in df.columns])

    narwhals_df = process_df(df)
    validator = DatasetValidator(enable_warnings=False)
    results = validator.transform(narwhals_df, target_col=target_col)

    # Verify we got results in the expected format
    assert isinstance(results, dict)
    assert all(isinstance(result, ValidationResult) for result in results.values())
    assert set(results.keys()) >= {"sample_size", "feature_count", "feature_ratio"}
