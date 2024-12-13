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
------------------
The validator follows a clear separation between validation configuration and execution,
designed to work seamlessly with both TimeFrame and raw DataFrame inputs.

| Component | Description |
|-----------|-------------|
| `fit()` | Input validation phase that ensures: <br>- Valid DataFrame type <br>- Required columns present <br>- Validation thresholds configured |
| `transform()` | Pure Narwhals validation phase that: <br>- Uses backend-agnostic operations only <br>- Performs configured validation checks <br>- Returns detailed validation results |
Backend-Specific Patterns
-------------------------
The following table outlines key patterns for working with different DataFrame backends
through Narwhals operations:

| Backend | Implementation Pattern |
|---------|------------------------|
| LazyFrame (Dask/Polars) | Uses `collect()` for scalar access, handles lazy evaluation through proper Narwhals operations, avoids direct indexing. |
| PyArrow | Uses `nw.Int64` for numeric operations, handles comparisons through Narwhals, converts types before arithmetic operations. |
| All Backends | Uses pure Narwhals operations for validation checks, avoids any backend-specific code to ensure consistent behavior. |

Research-Backed Thresholds
--------------------------
The following table summarizes validation thresholds derived from key research:

| Validation Check | Default Threshold | Source | Reasoning |
|-----------------|-------------------|--------|-----------|
| Minimum Samples | ≥ 3,000 | Grinsztajn et al. (2022) | Ensures sufficient data for complex model training |
| Maximum Samples | ≤ 50,000 | Shwartz-Ziv et al. (2021) | Defines medium-sized dataset upper bound |
| Minimum Features | ≥ 4 | Shwartz-Ziv et al. (2021) | Ensures meaningful complexity for model learning |
| Maximum Features | < 500 | Gorishniy et al. (2021) | Avoids high-dimensional data challenges |
| Feature/Sample Ratio | d/n < 1/10 | Grinsztajn et al. (2022) | Prevents overfitting risk |
| Categorical Cardinality | ≤ 20 unique values | Grinsztajn et al. (2022) | Manages categorical feature complexity |
| Numerical Uniqueness | ≥ 10 unique values | Gorishniy et al. (2021) | Ensures sufficient feature variability |

Examples
--------
```python
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
```

Notes
-----
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
from tabulate import tabulate  # type: ignore

from temporalscope.core.core_utils import SupportedTemporalDataFrame, is_valid_temporal_dataframe


@dataclass
class ValidationResult:
    """Container for dataset validation results, designed for integration with data pipelines and monitoring systems.

    This class provides structured validation results that can be easily integrated into
    data pipelines, logging systems, and monitoring dashboards. It includes methods for
    serialization and log formatting to support automated decision making in pipelines.

    Parameters
    ----------
    passed : bool
        Whether the check passed
    message : Optional[str]
        Optional message explaining the result
    details : Optional[Dict[str, Any]]
        Optional dictionary with detailed results
    severity : Optional[str]

    Examples
    --------
    ```python
    # In an Airflow DAG
    def validate_dataframeset(**context):
       validator = DatasetValidator()
       results = validator.fit_transform(df)

       # Get structured results for logging
       for check_name, result in results.items():
           log_entry = result.to_log_entry()
           if not result.passed:
               context["task_instance"].xcom_push(key=f"validation_failure_{check_name}", value=result.to_dict())

               # Log to monitoring system
               logger.log(level=log_entry["log_level"], msg=f"Validation check '{check_name}' failed", extra=log_entry)
    Log level for the validation result (e.g., 'WARNING', 'ERROR')
    ```

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

        Parameters
        ----------
        results : Dict[str, ValidationResult]
            Dictionary of validation results

        Returns
        -------
        Dict[str, ValidationResult]
            Dictionary of failed checks

        """
        return {name: result for name, result in results.items() if not result.passed}

    @classmethod
    def get_validation_summary(cls, results: Dict[str, "ValidationResult"]) -> Dict[str, Any]:
        """Get summary statistics for monitoring dashboards.

        Parameters
        ----------
        results : Dict[str, ValidationResult]
            Dictionary of validation results

        Returns
        -------
        Dict[str, Any]
            Summary statistics

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

    Engineering Design Assumptions:
    -------------------------------
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

    Pipeline Integration Features:
    ------------------------------
    - Automated quality gates for pipeline decision making
    - Structured results for monitoring and alerting systems
    - Support for temporal workflow validation

    Attributes
    ----------
    min_samples : int
        Minimum number of samples required, based on Grinsztajn et al. (2022)
    max_samples : int
        Maximum number of samples allowed, based on Shwartz-Ziv et al. (2021)
    min_features : int
        Minimum number of features required, based on Shwartz-Ziv et al. (2021)
    max_features : int
        Maximum number of features allowed, based on Gorishniy et al. (2021)
    max_feature_ratio : float
        Maximum feature-to-sample ratio, based on Grinsztajn et al. (2022)
    min_unique_values : int
        Minimum unique values for numerical features
    max_categorical_values : int
        Maximum unique values for categorical features
    class_imbalance_threshold : float
        Maximum ratio between largest and smallest classes
    checks_to_run : Optional[List[str]]
        List of validation checks to run. If None, runs all checks.
    enable_warnings : bool
        Whether to show warning messages for failed checks


    Raises
    ------
    ValueError
        If invalid checks are specified

    Examples
    --------
    ```python
    import pandas as pd
    from temporalscope.datasets import DatasetValidator

    # Create sample data
    df = pd.DataFrame({"feature1": range(5000), "target": range(5000)})

    # Initialize and run validator
    validator = DatasetValidator()
    results = validator.fit_transform(df)
    print(f"All checks passed: {all(r.passed for r in results.values())}")
    ```

    ```python
    # In an Airflow DAG
    def validate_dataframeset_task(**context):
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
    ```

    Notes
    -----
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
        """
        Initialize the validator with column configuration and thresholds.

        This validator performs quality checks on single DataFrames, designed for
        integration into automated pipelines (e.g., Airflow). It validates data quality
        using research-backed thresholds while leaving partitioning and parallelization
        to end users.

        Engineering Design Assumptions
        -------------------------------
        1. **Single DataFrame Focus**:
           - Operates on individual DataFrames.
           - Assumes end-users handle partitioning and parallelization.
           - Designed for pipeline integration.

        2. **Basic Validation**:
           - Verifies the existence of `time_col` and `target_col`.
           - Validates numeric columns (excluding `time_col`).
           - Checks for null values.

        3. **Research-Backed Thresholds**:
           - Sample size thresholds (Grinsztajn et al., 2022).
           - Feature counts (Shwartz-Ziv et al., 2021).
           - Feature ratios (Gorishniy et al., 2021).

        Parameters
        ----------
        time_col : str
            Column representing time values.
        target_col : str
            Column representing the target variable.
        min_samples : int
            Minimum samples required (Grinsztajn et al., 2022).
        max_samples : int
            Maximum samples allowed (Shwartz-Ziv et al., 2021).
        min_features : int
            Minimum features required (Shwartz-Ziv et al., 2021).
        max_features : int
            Maximum features allowed (Gorishniy et al., 2021).
        max_feature_ratio : float
            Maximum feature-to-sample ratio (Grinsztajn et al., 2022).
        min_unique_values : int
            Minimum unique values required for numerical features.
        max_categorical_values : int
            Maximum unique values allowed for categorical features.
        class_imbalance_threshold : float
            Maximum ratio between the largest and smallest class sizes.
        checks_to_run : Optional[List[str]]
            List of validation checks to execute.
        enable_warnings : bool
            Whether to display warning messages.

        Raises
        ------
        ValueError
            If invalid checks are specified.

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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate

        Returns
        -------
        FrameT
            Narwhals-compatible DataFrame

        Raises
        ------
        TypeError
            If input is not a valid temporal DataFrame

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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate

        Returns
        -------
        ValidationResult
            ValidationResult with:
            - passed: Whether all features meet variability requirements
            - message: Description of any issues found
            - details: Dictionary containing:
            - numeric_feature: Whether features are numeric
            - column_name: Number of unique values for each feature

        """
        details: Dict[str, Any] = {"numeric_feature": True}

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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate
        target_col : str
            Target column name
        df: Union[SupportedTemporalDataFrame :

        FrameT] :

        target_col: str :


        Returns
        -------
        ValidationResult

        Notes
        -----
        Implementation Details:
        - Uses count() for backend-agnostic counting
        - Handles LazyFrame evaluation through collect()
        - Converts PyArrow scalars using as_py()
            ValidationResult with:
            - passed: Always True (basic check)
            - details: Dictionary containing:
            - class_counts: Basic count information

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

        Parameters
        ----------
        check_name : str
            Name of the check to execute
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate
        target_col : Optional[str]
            Column name for target-specific checks

        Returns
        -------
        Optional[ValidationResult]
            Result of the validation check if enabled, None otherwise

        Raises
        ------
        ValueError
            If check_name is not a valid check name

        Notes
        -----
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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate

        Returns
        -------
        ValidationResult

        Notes
        -----
        Implementation Details:
        - Uses count() for backend-agnostic sample counting
        - Handles LazyFrame evaluation through collect()
        - Converts PyArrow scalars using as_py()
        - Handles empty DataFrames gracefully
            ValidationResult with:
            - passed: Whether sample size is within acceptable range
            - message: Description of any issues found
            - details: Dictionary containing:
            - num_samples: Total number of samples in dataset

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
        """
        Validate if the dataset meets feature count requirements.

        This method performs feature count validation using the following steps:
        1. Counts the total number of features, excluding the time and target columns.
        2. Verifies the count against the configured minimum and maximum thresholds.

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            The DataFrame to validate.

        Returns
        -------
        ValidationResult
            An object containing the validation outcome, with the following attributes:
            - passed (bool): Indicates whether the feature count is within the acceptable range.
            - message (str): Describes any issues identified during validation.
            - details (dict): Provides additional context with the following key:
                - num_features (int): The total number of features in the dataset.
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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate

        Returns
        -------
        ValidationResult

        Notes
        -----
        Implementation Details:
        - Uses count() for backend-agnostic sample counting
        - Handles LazyFrame evaluation through collect()
        - Converts PyArrow scalars using as_py()
        - Only counts feature columns in ratio calculation
            ValidationResult with:
            - passed: Whether ratio is within acceptable range
            - message: Description of any issues found
            - details: Dictionary containing:
            - ratio: Feature-to-sample ratio (num_features/num_samples)

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
            details = {"ratio": 0}  # Zero samples means zero ratio# Convert to percentage as int
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
        details = {"ratio": ratio}  # type: ignore  # Store actual ratio for validation

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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to get columns from

        Returns
        -------
        List[str]
            List of feature column names

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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate

        Returns
        -------
        DatasetValidator
            DatasetValidator instance for method chaining

        Raises
        ------
        TypeError
            If input is not a valid temporal DataFrame
        ValueError
            If columns are missing or invalid

        Examples
        --------
        ```python
        validator = DatasetValidator(time_col="time", target_col="target")
        validator.fit(df)
        ```

        """

        @nw.narwhalify
        def validate_numeric(df: Union[SupportedTemporalDataFrame, FrameT]) -> FrameT:
            """Validate that all columns except time are numeric.

            Parameters
            ----------
            df: Union[SupportedTemporalDataFrame, FrameT] :

            Returns
            -------
            FrameT

            """
            for col in df.columns:
                if col != self.time_col:
                    try:
                        df.select([nw.col(col).cast(nw.Float64)])
                    except Exception as e:
                        raise ValueError(f"Column {col} must be numeric. Error: {str(e)}")
            return df

        @nw.narwhalify
        def check_nulls(df: Union[SupportedTemporalDataFrame, FrameT], columns: List[str]) -> Dict[str, int]:
            """Check for null values in specified columns.

            Parameters
            ----------
            df: Union[SupportedTemporalDataFrame, FrameT] :
            columns: List[str]

            Returns
            -------
            Dict[str, int]

            """
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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate
        target_col : Optional[str]
            Column name for target-specific checks

        target_col: Optional[str] :
             (Default value = None)

        Returns
        -------
        Dict[str, ValidationResult]

        Examples
        --------
        ```python
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
        ```

        Notes
        -----
        - Uses pure Narwhals operations
        - Handles LazyFrame evaluation
        - Returns detailed results (Dictionary of validation results for each check)

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

        Parameters
        ----------
        df : Union[SupportedTemporalDataFrame, FrameT]
            DataFrame to validate
        target_col : Optional[str]
            Column name for target-specific checks

        target_col: Optional[str] :
             (Default value = None)

        Returns
        -------
        Dict[str, ValidationResult]
            Dictionary of validation results for each check

        Raises
        ------
        TypeError
            If input is not convertible to a Narwhals DataFrame

        Examples
        --------
        ```python
        import pandas as pd
        from temporalscope.datasets import DatasetValidator

        # Create sample data
        df = pd.DataFrame({"feature1": range(5000), "target": range(5000)})

        # Initialize and run validator
        validator = DatasetValidator()
        results = validator.fit_transform(df, target_col="target")

        # Print report
        validator.print_report(results)
        ```

        Notes
        -----
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

        Parameters
        ----------
        results : Dict[str, ValidationResult]
            Dictionary of validation results to report

        Returns
        -------
        None

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
