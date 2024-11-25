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

"""TemporalScope/src/temporalscope/datasets/dataset_validator.py.

This module provides backend-agnostic dataset validation utilities based on research-backed
heuristics. Using Narwhals operations, it enables consistent validation across different
DataFrame backends while supporting domain-specific requirements through customizable
thresholds.

For academic illustration only (see LICENSE for terms), consider domains like quantitative
finance, where these validators could check high-frequency trading data windows for
sufficient samples and feature quality, or healthcare analytics, where validators might
be configured with different thresholds for smaller datasets. While these examples
demonstrate the validator's flexibility, users must determine appropriate validation
criteria for their specific use cases, as all functionality is provided AS-IS without
warranty of any kind.

Engineering Design
-----------------
The validator follows a clear separation between validation configuration and execution,
designed to work seamlessly with both TimeFrame and raw DataFrame inputs.

+----------------+-------------------------------------------------------------------+
| Component      | Description                                                       |
+----------------+-------------------------------------------------------------------+
| fit()          | Input validation phase that ensures:                              |
|                | - Valid DataFrame type                                            |
|                | - Required columns present                                        |
|                | - Validation thresholds configured                                |
+----------------+-------------------------------------------------------------------+
| transform()    | Pure Narwhals validation phase that:                              |
|                | - Uses backend-agnostic operations only                           |
|                | - Performs configured validation checks                           |
|                | - Returns detailed validation results                             |
+----------------+-------------------------------------------------------------------+

Backend-Specific Patterns
------------------------
The following table outlines key patterns for working with different DataFrame backends
through Narwhals operations:

+----------------+-------------------------------------------------------------------+
| Backend        | Implementation Pattern                                            |
+----------------+-------------------------------------------------------------------+
| LazyFrame      | Uses collect() for scalar access, handles lazy evaluation through |
| (Dask/Polars)  | proper Narwhals operations, avoids direct indexing.               |
+----------------+-------------------------------------------------------------------+
| PyArrow        | Uses nw.Int64 for numeric operations, handles comparisons through |
|                | Narwhals, converts types before arithmetic operations.            |
+----------------+-------------------------------------------------------------------+
| All Backends   | Uses pure Narwhals operations for validation checks, avoids any   |
|                | backend-specific code to ensure consistent behavior.              |
+----------------+-------------------------------------------------------------------+

Research-Backed Thresholds
-------------------------
The following table summarizes validation thresholds derived from key research:

+------------------------+----------------------+---------------------------+--------------------------------+
| Validation Check       | Default Threshold    | Source                    | Reasoning                      |
+------------------------+----------------------+---------------------------+--------------------------------+
| Minimum Samples        | ≥ 3,000              | Grinsztajn et al. (2022)  | Ensures sufficient data for    |
|                        |                      |                           | complex model training         |
+------------------------+----------------------+---------------------------+--------------------------------+
| Maximum Samples        | ≤ 50,000             | Shwartz-Ziv et al. (2021) | Defines medium-sized dataset   |
|                        |                      |                           | upper bound                    |
+------------------------+----------------------+---------------------------+--------------------------------+
| Minimum Features       | ≥ 4                  | Shwartz-Ziv et al. (2021) | Ensures meaningful complexity  |
|                        |                      |                           | for model learning             |
+------------------------+----------------------+---------------------------+--------------------------------+
| Maximum Features       | < 500                | Gorishniy et al. (2021)   | Avoids high-dimensional data   |
|                        |                      |                           | challenges                     |
+------------------------+----------------------+---------------------------+--------------------------------+
| Feature/Sample Ratio   | d/n < 1/10           | Grinsztajn et al. (2022)  | Prevents overfitting risk      |
+------------------------+----------------------+---------------------------+--------------------------------+
| Categorical Cardinality| ≤ 20 unique values   | Grinsztajn et al. (2022)  | Manages categorical feature    |
|                        |                      |                           | complexity                     |
+------------------------+----------------------+---------------------------+--------------------------------+
| Numerical Uniqueness   | ≥ 10 unique values   | Gorishniy et al. (2021)   | Ensures sufficient feature     |
|                        |                      |                           | variability                    |
+------------------------+----------------------+---------------------------+--------------------------------+

Example Usage
------------
.. code-block:: python

    import pandas as pd
    from temporalscope.datasets.dataset_validator import DatasetValidator

    # Create sample data
    df = pd.DataFrame({"numeric_feature": range(100), "categorical_feature": ["A", "B"] * 50, "target": range(100)})

    # Create validator with custom thresholds
    validator = DatasetValidator(
        min_samples=1000, max_samples=10000, checks_to_run=["sample_size", "feature_count"], enable_warnings=True
    )

    # Run validation checks
    results = validator.validate(df, target_col="target")

    # Print detailed report
    validator.print_report(results)

.. note::
   - Uses the scikit-learn-style fit/transform pattern but adapted for TemporalScope:
     * fit() validates input DataFrame compatibility
     * transform() is @nw.narwhalify'd for backend-agnostic operations
   - This pattern is used throughout TemporalScope to ensure:
     * Input validation happens in fit()
     * All operations use Narwhals' backend-agnostic API in transform()
   - Supports customizable thresholds for different domain requirements
   - Integrates with data pipelines through scikit-learn compatible API
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import narwhals as nw
from narwhals.typing import FrameT
from tabulate import tabulate

from temporalscope.core.core_utils import SupportedTemporalDataFrame, is_valid_temporal_dataframe


@dataclass
class ValidationResult:
    """Container for dataset validation results, designed for integration with data pipelines and monitoring systems.

    This class provides structured validation results that can be easily integrated into
    data pipelines, logging systems, and monitoring dashboards. It includes methods for
    serialization and log formatting to support automated decision making in pipelines.

    :param passed: Whether the check passed
    :type passed: bool
    :param message: Optional message explaining the result
    :type message: Optional[str]
    :param details: Optional dictionary with detailed results
    :type details: Optional[Dict[str, Any]]
    :param severity: Log level for the validation result (e.g., 'WARNING', 'ERROR')
    :type severity: Optional[str]

    Example:
    -------
    .. code-block:: python

        # In an Airflow DAG
        def validate_dataset(**context):
            validator = DatasetValidator()
            results = validator.fit_transform(df)

            # Get structured results for logging
            for check_name, result in results.items():
                log_entry = result.to_log_entry()
                if not result.passed:
                    context["task_instance"].xcom_push(key=f"validation_failure_{check_name}", value=result.to_dict())

                    # Log to monitoring system
                    logger.log(level=log_entry["log_level"], msg=f"Validation check '{check_name}' failed", extra=log_entry)

    """

    passed: bool
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization in data pipelines."""
        return {"passed": self.passed, "message": self.message, "details": self.details, "severity": self.severity}

    def to_log_entry(self) -> Dict[str, Any]:
        """Format result as a structured log entry for monitoring systems."""
        return {
            "validation_passed": self.passed,
            "validation_message": self.message,
            "validation_details": self.details,
            "log_level": self.severity or ("INFO" if self.passed else "WARNING"),
        }

    @classmethod
    def get_failed_checks(cls, results: Dict[str, "ValidationResult"]) -> Dict[str, "ValidationResult"]:
        """Get all failed validation checks for pipeline decision making.

        :param results: Dictionary of validation results
        :type results: Dict[str, ValidationResult]
        :return: Dictionary of failed checks
        :rtype: Dict[str, ValidationResult]
        """
        return {name: result for name, result in results.items() if not result.passed}

    @classmethod
    def get_validation_summary(cls, results: Dict[str, "ValidationResult"]) -> Dict[str, Any]:
        """Get summary statistics for monitoring dashboards.

        :param results: Dictionary of validation results
        :type results: Dict[str, ValidationResult]
        :return: Summary statistics
        :rtype: Dict[str, Any]
        """
        return {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results.values() if r.passed),
            "failed_checks": sum(1 for r in results.values() if not r.passed),
            "check_details": {name: result.to_dict() for name, result in results.items()},
        }


class DatasetValidator:
    """A validator for ensuring dataset quality using research-backed heuristics.

    This class provides comprehensive dataset validation functionality through
    Narwhals' backend-agnostic operations, supporting both TimeFrame objects
    and raw DataFrames. Designed for integration into data pipelines and
    temporal workflows, it enables automated quality checks and monitoring.

    Engineering Design Assumptions
    ----------------------------
    1. Input Validation:
    - Supports all Narwhals-compatible DataFrame types
    - Handles both eager and lazy evaluation patterns
    - Validates column presence and types

    2. Validation Checks:
    - Each check is independent and configurable
    - Uses pure Narwhals operations for backend compatibility
    - Returns detailed results with messages and metrics

    3. Backend Compatibility:
    - No direct DataFrame indexing or operations
    - Handles LazyFrame evaluation properly
    - Uses type-safe numeric operations

    Pipeline Integration Features
    ---------------------------
    - Automated quality gates for pipeline decision making
    - Structured results for monitoring and alerting systems
    - Support for temporal workflow validation

    :param min_samples: Minimum number of samples required, based on Grinsztajn et al. (2022)
    :type min_samples: int
    :param max_samples: Maximum number of samples allowed, based on Shwartz-Ziv et al. (2021)
    :type max_samples: int
    :param min_features: Minimum number of features required, based on Shwartz-Ziv et al. (2021)
    :type min_features: int
    :param max_features: Maximum number of features allowed, based on Gorishniy et al. (2021)
    :type max_features: int
    :param max_feature_ratio: Maximum feature-to-sample ratio, based on Grinsztajn et al. (2022)
    :type max_feature_ratio: float
    :param min_unique_values: Minimum unique values for numerical features
    :type min_unique_values: int
    :param max_categorical_values: Maximum unique values for categorical features
    :type max_categorical_values: int
    :param class_imbalance_threshold: Maximum ratio between largest and smallest classes
    :type class_imbalance_threshold: float
    :param checks_to_run: List of validation checks to run. If None, runs all checks.
    :type checks_to_run: Optional[List[str]]
    :param enable_warnings: Whether to show warning messages for failed checks
    :type enable_warnings: bool
    :raises ValueError: If invalid checks are specified

    Example with default thresholds:
    ----------------------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.datasets import DatasetValidator

        # Create sample data
        df = pd.DataFrame({"feature1": range(5000), "target": range(5000)})

        # Initialize and run validator
        validator = DatasetValidator()
        results = validator.fit_transform(df)
        print(f"All checks passed: {all(r.passed for r in results.values())}")

    Example Pipeline Integration:
    -------------------------
    .. code-block:: python

        # In an Airflow DAG
        def validate_dataset_task(**context):
            validator = DatasetValidator(min_samples=1000, checks_to_run=["sample_size", "feature_count"])

            results = validator.fit_transform(df)
            failed = ValidationResult.get_failed_checks(results)

            if failed:
                # Log failures and push metrics
                metrics = ValidationResult.get_validation_summary(results)
                monitoring.push_metrics("data_validation", metrics)

                # Fail pipeline if critical checks failed
                if any(r.severity == "ERROR" for r in failed.values()):
                    raise AirflowException("Critical validation checks failed")

    .. note::
        Backend-Specific Patterns:
        - Use collect() for scalar access (LazyFrame)
        - Use nw.Int64 for numeric operations (PyArrow)
        - Let @nw.narwhalify handle conversions
        - Supports integration with workflow systems (Airflow, Prefect)
    """

    # Available validation checks
    AVAILABLE_CHECKS = {
        "sample_size",
        "feature_count",
        "feature_ratio",
        "feature_variability",
        "categorical_cardinality",
        "class_balance",
        "binary_features",
    }

    def __init__(
        self,
        time_col: str,
        target_col: str,
        min_samples: int = 3000,
        max_samples: int = 50000,
        min_features: int = 4,
        max_features: int = 500,
        max_feature_ratio: float = 0.1,
        min_unique_values: int = 10,
        max_categorical_values: int = 20,
        class_imbalance_threshold: float = 1.5,
        checks_to_run: Optional[List[str]] = None,
        enable_warnings: bool = True,
    ):
        """Initialize validator with column configuration and thresholds.

        This validator performs quality checks on single DataFrames, designed for integration
        into automated pipelines (e.g., Airflow). It validates data quality using research-backed
        thresholds while letting end users handle partitioning and parallelization.

        Engineering Design Assumptions:
        1. Single DataFrame Focus:
        - Works on individual DataFrames
        - End users handle partitioning/parallelization
        - Suitable for pipeline integration

        2. Basic Validation:
        - Ensures time_col and target_col exist
        - Validates numeric columns (except time_col)
        - Checks for null values

        3. Research-Backed Thresholds:
        - Sample size (Grinsztajn et al. 2022)
        - Feature counts (Shwartz-Ziv et al. 2021)
        - Feature ratios (Gorishniy et al. 2021)

        :param time_col: Column representing time values
        :type time_col: str
        :param target_col: Column representing target variable
        :type target_col: str
        :param min_samples: Minimum samples required (Grinsztajn et al. 2022)
        :type min_samples: int
        :param max_samples: Maximum samples allowed (Shwartz-Ziv et al. 2021)
        :type max_samples: int
        :param min_features: Minimum features required (Shwartz-Ziv et al. 2021)
        :type min_features: int
        :param max_features: Maximum features allowed (Gorishniy et al. 2021)
        :type max_features: int
        :param max_feature_ratio: Maximum feature-to-sample ratio (Grinsztajn et al. 2022)
        :type max_feature_ratio: float
        :param min_unique_values: Minimum unique values for numerical features
        :type min_unique_values: int
        :param max_categorical_values: Maximum unique values for categorical features
        :type max_categorical_values: int
        :param class_imbalance_threshold: Maximum ratio between largest and smallest classes
        :type class_imbalance_threshold: float
        :param checks_to_run: List of validation checks to run
        :type checks_to_run: Optional[List[str]]
        :param enable_warnings: Whether to show warning messages
        :type enable_warnings: bool
        :raises ValueError: If invalid checks are specified
        """
        self.time_col = time_col
        self.target_col = target_col
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_features = min_features
        self.max_features = max_features
        self.max_feature_ratio = max_feature_ratio
        self.min_unique_values = min_unique_values
        self.max_categorical_values = max_categorical_values
        self.class_imbalance_threshold = class_imbalance_threshold
        self.enable_warnings = enable_warnings

        # Validate and store checks to run
        if checks_to_run:
            invalid_checks = set(checks_to_run) - self.AVAILABLE_CHECKS
            if invalid_checks:
                raise ValueError(f"Invalid checks: {invalid_checks}")
            self.checks_to_run = set(checks_to_run)
        else:
            self.checks_to_run = self.AVAILABLE_CHECKS

    def _ensure_narwhals_df(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> FrameT:
        """Ensure DataFrame is Narwhals-compatible.

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: Narwhals-compatible DataFrame
        :rtype: FrameT
        :raises TypeError: If input is not a valid temporal DataFrame
        """
        is_valid, _ = is_valid_temporal_dataframe(df)
        if not is_valid:
            raise TypeError("Input must be a valid temporal DataFrame")
        return df

    @nw.narwhalify
    def _check_feature_variability(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> ValidationResult:
        """Check feature value variability and quality.

        This method evaluates feature variability through a simple process:
        1. Identifies feature columns in the dataset
        2. For each feature:
        - Counts unique values to ensure sufficient variability
        - Checks for null values to ensure data quality
        3. Validates against minimum uniqueness threshold

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: ValidationResult with:
                - passed: Whether all features meet variability requirements
                - message: Description of any issues found
                - details: Dictionary containing:
                    * numeric_feature: Whether features are numeric
                    * {column_name}: Number of unique values for each feature
        :rtype: ValidationResult
        """
        details = {"numeric_feature": True}

        # Get feature columns
        feature_cols = self._get_feature_columns(df)
        if not feature_cols:
            msg = "No feature columns found. Cannot check feature variability."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Check each feature's variability
        failed_columns = []
        for col in feature_cols:
            # Get unique and null counts
            counts = df.select(
                [
                    nw.col(col).n_unique().cast(nw.Int64).alias("unique"),
                    nw.col(col).is_null().sum().cast(nw.Int64).alias("nulls"),
                ]
            )
            if hasattr(counts, "collect"):
                counts = counts.collect()

            unique_count = int(
                counts["unique"][0].as_py() if hasattr(counts["unique"][0], "as_py") else counts["unique"][0]
            )
            null_count = int(counts["nulls"][0].as_py() if hasattr(counts["nulls"][0], "as_py") else counts["nulls"][0])

            details[col] = unique_count
            if unique_count < self.min_unique_values or null_count > 0:
                failed_columns.append(col)

        # Return validation result
        if failed_columns:
            msg = (
                f"Features with insufficient variability or quality issues: {failed_columns}. "
                f"Minimum unique values: {self.min_unique_values}, no null values allowed."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        return ValidationResult(True, None, details)

    @nw.narwhalify
    def _check_class_balance(
        self,
        df: Union[SupportedTemporalDataFrame, FrameT],
        target_col: str,
    ) -> ValidationResult:
        """Check class balance in target column.

        This method provides a simple class balance check by:
        1. Counting total samples
        2. Adding class count information to details

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :param target_col: Target column name
        :type target_col: str
        :return: ValidationResult with:
                - passed: Always True (basic check)
                - details: Dictionary containing:
                    * class_counts: Basic count information
        :rtype: ValidationResult

        .. note::
            Implementation Details:
            - Uses count() for backend-agnostic counting
            - Handles LazyFrame evaluation through collect()
            - Converts PyArrow scalars using as_py()
        """
        if not target_col:
            return ValidationResult(True, "No target column specified")

        # Get total count for basic class balance info
        total_count = df.select([nw.col(target_col).count().cast(nw.Int64).alias("count")])
        if hasattr(total_count, "collect"):
            total_count = total_count.collect()

        value = total_count["count"][0]
        if hasattr(value, "as_py"):
            value = value.as_py()
        count = int(value)

        # Return basic class balance info
        details = {"class_counts": {"total": count}}
        return ValidationResult(True, None, details)

    def _execute_check(
        self,
        check_name: str,
        df: Union[SupportedTemporalDataFrame, FrameT],
        target_col: Optional[str] = None,
    ) -> Optional[ValidationResult]:
        """Execute a single validation check.

        :param check_name: Name of the check to execute
        :type check_name: str
        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :param target_col: Column name for target-specific checks
        :type target_col: Optional[str]
        :return: Result of the validation check if enabled, None otherwise
        :rtype: Optional[ValidationResult]
        :raises ValueError: If check_name is not a valid check name

        .. note::
            - Executes a single validation check based on check_name
            - Returns None if check is not enabled
            - Handles target-specific checks appropriately
        """
        if check_name not in self.checks_to_run:
            return None

        match check_name:
            case "sample_size":
                result = self._check_sample_size(df)
            case "feature_count":
                result = self._check_feature_count(df)
            case "feature_ratio":
                result = self._check_feature_ratio(df)
            case "feature_variability":
                result = self._check_feature_variability(df)
            case "class_balance" if target_col:
                result = self._check_class_balance(df, target_col)
            case _:
                result = None

        return result

    @nw.narwhalify
    def _check_sample_size(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> ValidationResult:
        """Check if dataset meets sample size requirements.

        This method evaluates sample size through a simple process:
        1. Counts total samples using backend-agnostic operations
        2. Validates against configured minimum and maximum thresholds

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: ValidationResult with:
                - passed: Whether sample size is within acceptable range
                - message: Description of any issues found
                - details: Dictionary containing:
                    * num_samples: Total number of samples in dataset
        :rtype: ValidationResult

        .. note::
            Implementation Details:
            - Uses count() for backend-agnostic sample counting
            - Handles LazyFrame evaluation through collect()
            - Converts PyArrow scalars using as_py()
            - Handles empty DataFrames gracefully
        """
        # Handle empty DataFrame
        if not df.columns:
            details = {"num_samples": 0}
            msg = "Dataset is empty. This is insufficient for any modeling."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Step 1: Count total samples using first column
        num_samples_df = df.select([nw.col(df.columns[0]).count().cast(nw.Int64).alias("count")])
        if hasattr(num_samples_df, "collect"):
            num_samples_df = num_samples_df.collect()

        # Handle PyArrow scalar conversion
        value = num_samples_df["count"][0]
        if hasattr(value, "as_py"):
            value = value.as_py()
        num_samples = int(value)

        details = {"num_samples": num_samples}

        # Step 2: Validate against thresholds
        if num_samples < self.min_samples:
            msg = (
                f"Dataset has {num_samples} samples, fewer than recommended minimum ({self.min_samples}). "
                "This may be insufficient for complex models."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        if num_samples > self.max_samples:
            msg = (
                f"Dataset has {num_samples} samples, more than recommended maximum ({self.max_samples}). "
                "Consider using scalable implementations."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        return ValidationResult(True, None, details)

    @nw.narwhalify
    def _check_feature_count(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> ValidationResult:
        """Check if dataset meets feature count requirements.

        This method evaluates feature count through a simple process:
        1. Counts total features excluding time and target columns
        2. Validates against configured minimum and maximum thresholds

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: ValidationResult with:
                - passed: Whether feature count is within acceptable range
                - message: Description of any issues found
                - details: Dictionary containing:
                    * num_features: Total number of features in dataset
        :rtype: ValidationResult
        """
        df = self._ensure_narwhals_df(df)

        # Get feature columns using _get_feature_columns
        feature_cols = self._get_feature_columns(df)
        num_features = len(feature_cols)
        details = {"num_features": num_features}

        # Validate
        if num_features < self.min_features:
            msg = (
                f"Dataset has {num_features} features, "
                f"fewer than recommended minimum ({self.min_features}). "
                "This may result in an oversimplified model."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(
                passed=False,
                message=msg,
                details=details,
                severity="WARNING",
            )

        if num_features > self.max_features:
            msg = (
                f"Dataset has {num_features} features, more than recommended maximum ({self.max_features}). "
                "Consider dimensionality reduction."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(passed=False, message=msg, details=details, severity="WARNING")

        return ValidationResult(passed=True, details=details, severity="INFO")

    @nw.narwhalify
    def _check_feature_ratio(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> ValidationResult:
        """Check feature-to-sample ratio.

        This method evaluates feature-to-sample ratio through a simple process:
        1. Counts total samples using backend-agnostic operations
        2. Counts feature columns (excluding time and target)
        3. Calculates ratio and validates against threshold

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: ValidationResult with:
                - passed: Whether ratio is within acceptable range
                - message: Description of any issues found
                - details: Dictionary containing:
                    * ratio: Feature-to-sample ratio (num_features/num_samples)
        :rtype: ValidationResult

        .. note::
            Implementation Details:
            - Uses count() for backend-agnostic sample counting
            - Handles LazyFrame evaluation through collect()
            - Converts PyArrow scalars using as_py()
            - Only counts feature columns in ratio calculation
        """
        # Handle empty DataFrame
        if not df.columns:
            details = {"ratio": 0}
            msg = "Dataset is empty. Cannot calculate feature ratio."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Step 1: Count total samples
        num_samples_df = df.select([nw.col(df.columns[0]).count().cast(nw.Int64).alias("count")])
        if hasattr(num_samples_df, "collect"):
            num_samples_df = num_samples_df.collect()

        value = num_samples_df["count"][0]
        if hasattr(value, "as_py"):
            value = value.as_py()
        num_samples = int(value)

        # Handle zero samples case
        if num_samples == 0:
            details = {"ratio": float("inf")}
            msg = "Dataset has zero samples. Cannot calculate feature ratio."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Step 2: Get feature columns using _get_feature_columns
        feature_cols = self._get_feature_columns(df)
        num_features = len(feature_cols)

        # Handle no features case
        if num_features == 0:
            details = {"ratio": 0}
            msg = "No features found. Cannot calculate feature ratio."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Step 3: Calculate and validate ratio
        ratio = num_features / num_samples
        details = {"ratio": ratio}

        if ratio > self.max_feature_ratio:
            msg = (
                f"Feature-to-sample ratio ({ratio:.3f}) exceeds recommended maximum ({self.max_feature_ratio}). "
                "This may increase risk of overfitting."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(passed=False, message=msg, details=details, severity="WARNING")

        return ValidationResult(True, None, details)

    def _get_feature_columns(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> List[str]:
        """Get feature columns (all except time and target).

        Features are defined as all columns except time_col and target_col.

        :param df: DataFrame to get columns from
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: List of feature column names
        :rtype: List[str]
        """
        cols = df.columns
        if hasattr(cols, "collect"):
            cols = cols.collect()
        if hasattr(cols, "to_list"):
            cols = cols.to_list()
        return [col for col in cols if col not in {self.time_col, self.target_col}]

    def fit(self, df: Union[SupportedTemporalDataFrame, FrameT]) -> "DatasetValidator":
        """Validate input DataFrame and prepare for validation checks.

        This method ensures the input DataFrame meets basic validation requirements:

        1. Validates DataFrame type and required columns
        2. Ensures numeric columns and checks for null values

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :return: DatasetValidator instance for method chaining
        :rtype: DatasetValidator
        :raises TypeError: If input is not a valid temporal DataFrame
        :raises ValueError: If columns are missing or invalid

        Example:
        -------
        .. code-block:: python
            validator = DatasetValidator(time_col="time", target_col="target")
            validator.fit(df)

        """

        @nw.narwhalify
        def validate_numeric(df: Union[SupportedTemporalDataFrame, FrameT]) -> FrameT:
            """Validate that all columns except time are numeric."""
            for col in df.columns:
                if col != self.time_col:
                    try:
                        df.select([nw.col(col).cast(nw.Float64)])
                    except Exception as e:
                        raise ValueError(f"Column {col} must be numeric. Error: {str(e)}")
            return df

        @nw.narwhalify
        def check_nulls(df: Union[SupportedTemporalDataFrame, FrameT], columns: List[str]) -> Dict[str, int]:
            """Check for null values in specified columns."""
            null_counts = {}
            for col in columns:
                null_count = df.select([nw.col(col).is_null().sum().cast(nw.Int64).alias("nulls")])
                if hasattr(null_count, "collect"):
                    null_count = null_count.collect()
                value = null_count["nulls"][0]
                if hasattr(value, "as_py"):
                    value = value.as_py()
                null_counts[col] = int(value)
            return null_counts

        # Step 1: Validate DataFrame type using core_utils
        is_valid, df_type = is_valid_temporal_dataframe(df)
        if not is_valid:
            raise TypeError("Input must be a valid temporal DataFrame")

        # Step 2: Get column names based on DataFrame type
        cols = df.column_names if hasattr(df, "column_names") else df.columns
        if hasattr(cols, "collect"):
            cols = cols.collect()
        if hasattr(cols, "to_list"):
            cols = cols.to_list()

        # Step 3: Validate required columns exist
        if self.time_col not in cols or self.target_col not in cols:
            raise ValueError(f"Columns {self.time_col} and {self.target_col} must exist")

        # Step 4: Validate numeric columns
        validate_numeric(df)

        # Step 5: Check nulls
        null_counts = check_nulls(df, cols)
        null_columns = [col for col, count in null_counts.items() if count > 0]
        if null_columns:
            raise ValueError(f"Missing values detected in columns: {', '.join(null_columns)}")

        return self

    @nw.narwhalify
    def transform(
        self,
        df: Union[SupportedTemporalDataFrame, FrameT],
        target_col: Optional[str] = None,
    ) -> Dict[str, ValidationResult]:
        """Run configured validation checks on the DataFrame.

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :param target_col: Column name for target-specific checks
        :type target_col: Optional[str]
        :return: Dictionary of validation results for each check
        :rtype: Dict[str, ValidationResult]

        Example:
        -------
        .. code-block:: python

            import pandas as pd
            from temporalscope.datasets import DatasetValidator

            # Create sample data
            df = pd.DataFrame({"feature1": range(5000), "target": range(5000)})

            # Initialize and run validator
            validator = DatasetValidator()
            validator.fit(df)
            results = validator.transform(df, target_col="target")

            # Check results
            for check, result in results.items():
                print(f"{check}: {'Passed' if result.passed else 'Failed'}")

        .. note::
            - Uses pure Narwhals operations
            - Handles LazyFrame evaluation
            - Returns detailed results

        """
        # Execute validation checks
        results = {}
        check_names = ["sample_size", "feature_count", "feature_ratio", "feature_variability", "class_balance"]

        for check_name in check_names:
            result = self._execute_check(check_name, df, target_col)
            if result is not None:
                results[check_name] = result

        # Summarize results
        all_passed = all(result.passed for result in results.values())
        if not all_passed:
            critical_failures = any(result.severity == "ERROR" for result in results.values() if not result.passed)
            if critical_failures and self.enable_warnings:
                warnings.warn(
                    "Critical validation checks failed. These failures may significantly impact model performance.",
                    RuntimeWarning,
                )
            elif self.enable_warnings:
                warnings.warn(
                    "Some validation checks failed. These are research-backed recommendations "
                    "and may not apply to all use cases. Adjust thresholds as needed.",
                    UserWarning,
                )

        return results

    def fit_transform(
        self,
        df: Union[SupportedTemporalDataFrame, FrameT],
        target_col: Optional[str] = None,
    ) -> Dict[str, ValidationResult]:
        """Fit the validator and run validation checks in one step.

        :param df: DataFrame to validate
        :type df: Union[SupportedTemporalDataFrame, FrameT]
        :param target_col: Column name for target-specific checks
        :type target_col: Optional[str]
        :return: Dictionary of validation results for each check
        :rtype: Dict[str, ValidationResult]
        :raises TypeError: If input is not convertible to a Narwhals DataFrame

        Example:
        -------
        .. code-block:: python

            import pandas as pd
            from temporalscope.datasets import DatasetValidator

            # Create sample data
            df = pd.DataFrame({"feature1": range(5000), "target": range(5000)})

            # Initialize and run validator
            validator = DatasetValidator()
            results = validator.fit_transform(df, target_col="target")

            # Print report
            validator.print_report(results)

        .. note::
            - Combines fit() and transform()
            - Validates input then runs checks
            - Returns detailed results

        """
        return self.fit(df).transform(df, target_col)

    def print_report(self, results: Dict[str, ValidationResult]) -> None:
        """Print validation results in a tabular format.

        A simple utility function to display validation results in a readable format.
        For production use cases, use the structured results directly from the
        validation methods.

        :param results: Dictionary of validation results to report
        :type results: Dict[str, ValidationResult]
        """
        rows = []
        for check_name, result in results.items():
            status = "✓" if result.passed else "✗"
            message = result.message or "Check passed"
            details = ", ".join(f"{k}: {v}" for k, v in (result.details or {}).items())
            rows.append([check_name, status, message, details])

        print("\nDataset Validation Report")
        print(tabulate(rows, headers=["Check", "Status", "Message", "Details"], tablefmt="grid"))
        print("\nNote: These are research-backed recommendations and may not apply to all use cases.")
