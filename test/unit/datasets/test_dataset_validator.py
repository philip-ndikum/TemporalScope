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
    # Remove time_col and target_col from config
    df = generate_synthetic_time_series(**config)
    yield df, "target"  # target column is always "target" in synthetic data


@pytest.fixture
def validator() -> DatasetValidator:
    """Create DatasetValidator with test thresholds."""
    return DatasetValidator(
        time_col="time",
        target_col="target",
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
        DatasetValidator(time_col="time", target_col="target", checks_to_run=["invalid_check"])


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

    # Compare fit_trƒansform vs separate fit/transform
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
        return df.select([nw.col(col) for col in df.columns])

    narwhals_df = process_df(df)
    validator = DatasetValidator(time_col="time", target_col=target_col, enable_warnings=False)
    results = validator.transform(narwhals_df, target_col=target_col)

    # Verify we got results in the expected format
    assert isinstance(results, dict)
    assert all(isinstance(result, ValidationResult) for result in results.values())
    assert set(results.keys()) >= {"sample_size", "feature_count", "feature_ratio"}


def test_validation_error_paths(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test validation error paths across all checks."""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    results = validator.transform(empty_df)
    assert not results["sample_size"].passed
    assert not results["feature_count"].passed
    assert "empty" in results["sample_size"].message.lower()
    assert results["sample_size"].details["num_samples"] == 0

    # No feature columns
    config = data_config(num_features=0)
    no_features_df = generate_synthetic_time_series(**config)
    results = validator.transform(no_features_df)
    assert not results["feature_count"].passed
    assert f"fewer than recommended minimum ({validator.min_features})" in results["feature_count"].message
    assert results["feature_count"].details["num_features"] == 0

    # Too many features
    config = data_config(num_features=validator.max_features + 1)
    many_features_df = generate_synthetic_time_series(**config)
    results = validator.transform(many_features_df)
    assert not results["feature_count"].passed
    assert f"more than recommended maximum ({validator.max_features})" in results["feature_count"].message
    assert results["feature_count"].details["num_features"] == validator.max_features + 1

    # Too many samples
    config = data_config(num_samples=validator.max_samples + 1)
    large_df = generate_synthetic_time_series(**config)
    results = validator.transform(large_df)
    assert not results["sample_size"].passed
    assert f"more than recommended maximum ({validator.max_samples})" in results["sample_size"].message
    assert results["sample_size"].details["num_samples"] == validator.max_samples + 1


def test_validation_edge_cases_extended(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test additional edge cases in validation.

    Tests edge cases around:
    - Feature variability with null/missing values
    - Class balance validation with/without target column
    - Feature validation with invalid column types
    """
    # Test feature variability with null values
    config = data_config(with_nulls=True)
    null_df = generate_synthetic_time_series(**config)
    results = validator.transform(null_df)
    assert not results["feature_variability"].passed
    assert "quality issues" in results["feature_variability"].message.lower()

    # Test class balance behavior with/without target column
    config = data_config()
    df = generate_synthetic_time_series(**config)

    # Without target_col - class_balance should not be in results
    results_no_target = validator.transform(df)  # No target_col
    assert "class_balance" not in results_no_target

    # With target_col - class_balance should be in results
    results_with_target = validator.transform(df, target_col="target")
    assert "class_balance" in results_with_target

    # Test feature validation with invalid columns
    config = data_config(num_features=0)
    invalid_df = generate_synthetic_time_series(**config)
    results = validator.transform(invalid_df)
    assert not results["feature_count"].passed
    assert "Dataset has 0 features" in results["feature_count"].message.lower()  # Match exact message


def test_validation_report_formatting_andf_null_handling(
    validator: DatasetValidator, data_config: DataConfigType, capsys: Any
) -> None:
    """Test validation report formatting with null/empty values and edge cases.

    Tests:
    1. Report formatting with empty details
    2. Report formatting with null messages
    3. Report formatting with completely null results
    4. Report formatting with failed checks and no details
    5. Class balance handling with null/empty target columns
    6. Feature variability handling with insufficient unique values
    """
    # Create validator with stricter thresholds to force failures
    strict_validator = DatasetValidator(
        time_col="time",
        target_col="target",
        min_unique_values=200,  # More than synthetic data provides
        enable_warnings=False,
    )

    # Test feature variability with insufficient unique values
    df = generate_synthetic_time_series(**data_config(num_features=2))
    results = strict_validator.transform(df)
    assert not results["feature_variability"].passed
    assert_validation_details(results["feature_variability"], ["numeric_feature"])
    assert "insufficient variability" in results["feature_variability"].message.lower()

    # Test class balance with empty target column
    results = validator.transform(df, target_col="")
    assert "class_balance" not in results

    # Test class balance with null target column
    results = validator.transform(df, target_col=None)
    assert "class_balance" not in results

    # Test print report with various null/empty cases
    validation_results = {
        "empty_details": ValidationResult(True, "No details", None),
        "null_message": ValidationResult(True, None, {"key": "value"}),
        "null_everything": ValidationResult(True),
        "empty_details_failed": ValidationResult(False, "Failed, no details", None),
    }

    validator.print_report(validation_results)
    captured = capsys.readouterr()

    # Verify report formatting
    assert "Dataset Validation Report" in captured.out
    assert "✓" in captured.out  # Pass symbol
    assert "✗" in captured.out  # Fail symbol
    assert "No details" in captured.out
    assert "Check passed" in captured.out  # Default message
    assert "Failed, no details" in captured.out
    assert "recommendations" in captured.out


def test_invalid_dataframe_validation(validator: DatasetValidator) -> None:
    """Test validation of invalid DataFrames through fit().

    Tests that fit() properly validates DataFrame type before transform()
    can use narwhals operations on it.
    """

    class InvalidDF:
        def __init__(self):
            self.columns = ["col1"]

    with pytest.raises(TypeError, match="must be a valid temporal DataFrame"):
        validator.fit(InvalidDF())


def test_feature_ratio_edge_cases(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test feature ratio validation edge cases.

    Tests:
    - Zero samples
    - No features
    - High feature-to-sample ratio
    """
    # Test with zero samples
    zero_samples_df = generate_synthetic_time_series(**data_config(num_samples=1))
    results = validator.transform(zero_samples_df)
    assert not results["feature_ratio"].passed
    assert_validation_details(results["feature_ratio"], ["ratio"])
    assert results["feature_ratio"].details["ratio"] > validator.max_feature_ratio

    # Test with no features
    no_features_df = generate_synthetic_time_series(**data_config(num_features=0))
    results = validator.transform(no_features_df)
    assert not results["feature_ratio"].passed
    assert_validation_result(results["feature_ratio"], False, "No features found")
    assert_validation_details(results["feature_ratio"], ["ratio"])
    assert results["feature_ratio"].details["ratio"] == 0

    # Test with high ratio
    high_ratio_df = generate_synthetic_time_series(**data_config(num_samples=10, num_features=5))
    results = validator.transform(high_ratio_df)
    assert not results["feature_ratio"].passed
    assert_validation_result(results["feature_ratio"], False, "exceeds recommended maximum")
    assert_validation_details(results["feature_ratio"], ["ratio"])
    assert results["feature_ratio"].details["ratio"] > validator.max_feature_ratio


def test_class_balance_without_target(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test class balance validation without target column."""
    df = generate_synthetic_time_series(**data_config())
    results = validator.transform(df)  # No target_col provided
    assert "class_balance" not in results


def test_validation_report_formatting(validator: DatasetValidator) -> None:
    """Test validation report formatting with different result types."""
    results = {
        "passed": ValidationResult(True, None, {"detail": "value"}),
        "warning": ValidationResult(False, "Warning message", {"warn": "details"}, "WARNING"),
        "error": ValidationResult(False, "Error message", {"err": "critical"}, "ERROR"),
        "no_message": ValidationResult(True),
        "no_details": ValidationResult(True, "Just message"),
    }

    validator.print_report(results)


def test_check_method_error_paths(validator: DatasetValidator, data_config: DataConfigType) -> None:
    """Test error paths in check methods.

    Tests:
    1. Empty DataFrame handling in _check_sample_size
    2. PyArrow scalar conversion in _check_sample_size
    3. collect() and to_list() paths in _check_feature_count
    """

    # Test empty DataFrame in _check_sample_size
    @nw.narwhalify
    def create_empty_df():
        return pd.DataFrame()

    empty_df = create_empty_df()
    result = validator._check_sample_size(empty_df)
    assert not result.passed
    assert result.details["num_samples"] == 0
    assert "insufficient for any modeling" in result.message.lower()

    # Test PyArrow scalar conversion
    config = data_config()
    config["backend"] = "pyarrow"
    df = generate_synthetic_time_series(**config)
    result = validator._check_sample_size(df)
    assert result.details["num_samples"] == config["num_samples"]

    # Test collect() and to_list() paths
    config["backend"] = "polars"
    df = generate_synthetic_time_series(**config)
    result = validator._check_feature_count(df)
    assert result.details["num_features"] == config["num_features"]

    # Test LazyFrame collect()
    df = generate_synthetic_time_series(**config).lazy()
    result = validator._check_feature_count(df)
    assert result.details["num_features"] == config["num_features"]


def test_ensure_narwhals_df_error_path(validator: DatasetValidator) -> None:
    """Test error path in _ensure_narwhals_df."""

    class InvalidDF:
        def __init__(self):
            self.columns = ["col1"]
            self.select = None  # Has select but it's None

    with pytest.raises(TypeError, match="must be a valid temporal DataFrame"):
        validator.fit(InvalidDF())


def test_check_sample_size_error_paths(validator: DatasetValidator) -> None:
    """Test error paths in _check_sample_size."""

    @nw.narwhalify
    def create_df_with_nulls():
        return pd.DataFrame(
            {
                "col1": range(10),  # Non-null values in first column
                "col2": [None] * 10,  # Nulls in second column
            }
        )

    df_with_nulls = create_df_with_nulls()
    result = validator._check_sample_size(df_with_nulls)
    assert result.details["num_samples"] == 10


def test_check_feature_count_error_paths(validator: DatasetValidator) -> None:
    """Test error paths in _check_feature_count."""

    @nw.narwhalify
    def create_df_no_features():
        return pd.DataFrame({"time": range(10), "target": range(10)})

    df_no_features = create_df_no_features()
    result = validator._check_feature_count(df_no_features)
    assert not result.passed
    assert result.details["num_features"] == 0
    assert "Dataset has 0 features" in result.message.lower()


def test_check_feature_ratio_error_paths(validator: DatasetValidator) -> None:
    """Test error paths in _check_feature_ratio."""

    @nw.narwhalify
    def create_df_high_ratio():
        return pd.DataFrame({f"feature_{i}": range(10) for i in range(10)})

    df_high_ratio = create_df_high_ratio()
    result = validator._check_feature_ratio(df_high_ratio)
    assert not result.passed
    assert result.details["ratio"] > validator.max_feature_ratio
    assert "exceeds recommended maximum" in result.message.lower()


def test_check_class_balance_error_paths(validator: DatasetValidator) -> None:
    """Test error paths in _check_class_balance."""

    @nw.narwhalify
    def create_df_imbalanced():
        return pd.DataFrame(
            {
                "feature_1": range(10),
                "target": [0] * 9 + [1],  # Highly imbalanced
            }
        )

    df_imbalanced = create_df_imbalanced()

    # Test with imbalanced data
    result = validator._check_class_balance(df_imbalanced, target_col="target")
    assert result.details["class_counts"]["total"] == 10

    # Test empty target column
    result = validator._check_class_balance(df_imbalanced, target_col="")
    assert result.message == "No target column specified"


def test_transform_warning_paths(validator: DatasetValidator) -> None:
    """Test transform() warning paths.

    Tests both warning paths in transform():
    1. Non-critical failures (UserWarning)
    2. Critical failures (RuntimeWarning)
    """

    @nw.narwhalify
    def create_df():
        return pd.DataFrame({"feature_1": range(10), "feature_2": range(10)})

    df = create_df()
    validator.enable_warnings = True

    # Test non-critical warning (UserWarning)
    with pytest.warns(UserWarning, match="Some validation checks failed"):
        results = validator.transform(df)
        assert not all(r.passed for r in results.values())
        # Verify we got WARNING severity
        assert any(r.severity == "WARNING" for r in results.values() if not r.passed)


def test_comprehensive_error_paths(validator: DatasetValidator) -> None:
    """Test all error paths systematically.

    Creates a DataFrame that triggers multiple validation failures to test:
    1. No features case
    2. Sample size issues
    3. Invalid ratios
    4. Missing target
    5. Warning paths
    """

    @nw.narwhalify
    def create_problematic_df():
        return pd.DataFrame(
            {
                "time": [1],  # Single row to avoid index issues
                "not_a_feature": [None],  # No feature columns
                "target": [None],  # Null target
            }
        )

    # Enable warnings to test warning paths
    validator.enable_warnings = True
    df = create_problematic_df()

    # Should trigger multiple warnings
    with pytest.warns(UserWarning) as record:
        results = validator.transform(df)

        # Verify all error paths were hit
        assert not results["sample_size"].passed
        assert results["sample_size"].details["num_samples"] == 1

        assert not results["feature_count"].passed
        assert results["feature_count"].details["num_features"] == 0

        assert not results["feature_ratio"].passed
        assert results["feature_ratio"].details["ratio"] == 0

        assert not results["feature_variability"].passed
        assert results["feature_variability"].details["numeric_feature"] == True

        # Verify warning messages
        messages = [str(w.message) for w in record]
        assert any("fewer than recommended minimum" in msg.lower() for msg in messages)
        assert any("no feature columns found" in msg.lower() for msg in messages)
        assert any("cannot calculate feature ratio" in msg.lower() for msg in messages)
        assert any("validation checks failed" in msg.lower() for msg in messages)


def test_dataframe_method_failures(validator: DatasetValidator) -> None:
    """Test validation when DataFrame methods fail.

    Tests handling of DataFrame method failures:
    1. select() returns None
    2. collect() returns None
    3. to_list() returns None
    4. as_py() returns None

    These represent error conditions that could occur when DataFrame methods
    fail to return expected values.
    """

    class SelectFailureDF:
        def __init__(self):
            self.columns = ["col1"]

        def select(self, *args, **kwargs):
            return None

    class CollectFailureDF:
        def __init__(self):
            self.columns = ["col1"]

        def select(self, *args, **kwargs):
            return CollectFailureDF()

        def collect(self):
            return None

    class ToListFailureDF:
        def __init__(self):
            self.columns = ["col1"]

        def select(self, *args, **kwargs):
            return ToListFailureDF()

        def to_list(self):
            return None

    class AsPyFailureDF:
        def __init__(self):
            self.columns = ["col1"]
            self.count = [self]  # Mock column access

        def select(self, *args, **kwargs):
            return AsPyFailureDF()

        def as_py(self):
            return None

    # Test select() failure
    with pytest.raises(TypeError):
        validator._check_sample_size(SelectFailureDF())

    # Test collect() failure
    with pytest.raises(TypeError):
        validator._check_sample_size(CollectFailureDF())

    # Test to_list() failure
    with pytest.raises(TypeError):
        validator._check_feature_count(ToListFailureDF())

    # Test as_py() failure
    with pytest.raises(TypeError):
        validator._check_sample_size(AsPyFailureDF())
