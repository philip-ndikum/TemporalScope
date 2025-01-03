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
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import narwhals as nw
from narwhals.typing import FrameT
from tabulate import tabulate  # type: ignore


@dataclass
class ValidationResult:
    """Container for dataset validation results."""

    passed: bool
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {"passed": self.passed, "message": self.message, "details": self.details, "severity": self.severity}

    def to_log_entry(self) -> Dict[str, Any]:
        """Format result as a structured log entry."""
        return {
            "validation_passed": self.passed,
            "validation_message": self.message,
            "validation_details": self.details,
            "log_level": self.severity or ("INFO" if self.passed else "WARNING"),
        }

    @classmethod
    def get_failed_checks(cls, results: Dict[str, "ValidationResult"]) -> Dict[str, "ValidationResult"]:
        """Get all failed validation checks."""
        return {name: result for name, result in results.items() if not result.passed}

    @classmethod
    def get_validation_summary(cls, results: Dict[str, "ValidationResult"]) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results.values() if r.passed),
            "failed_checks": sum(1 for r in results.values() if not r.passed),
            "check_details": {name: result.to_dict() for name, result in results.items()},
        }


class DatasetValidator:
    """A validator for ensuring dataset quality using research-backed heuristics."""

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
        """Initialize the validator with column configuration and thresholds."""
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

    def _ensure_narwhals_df(self, df: Union[Any, FrameT]) -> FrameT:
        """Ensure DataFrame is Narwhals-compatible."""
        try:
            return nw.from_native(df)
        except Exception as e:
            raise TypeError(f"Input must be convertible to a Narwhals DataFrame. Error: {str(e)}")

    @nw.narwhalify
    def _check_feature_variability(self, df: FrameT) -> ValidationResult:
        """Check feature value variability and quality."""
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
    def _check_class_balance(self, df: FrameT, target_col: str) -> ValidationResult:
        """Check class balance in target column."""
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
        self, check_name: str, df: FrameT, target_col: Optional[str] = None
    ) -> Optional[ValidationResult]:
        """Execute a single validation check."""
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
    def _check_sample_size(self, df: FrameT) -> ValidationResult:
        """Check if dataset meets sample size requirements."""
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
    def _check_feature_count(self, df: FrameT) -> ValidationResult:
        """Validate if the dataset meets feature count requirements."""
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
    def _check_feature_ratio(self, df: FrameT) -> ValidationResult:
        """Check feature-to-sample ratio."""
        # Handle empty DataFrame
        if not df.columns:
            details = {"ratio": 0.0}
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
            details = {"ratio": 0.0}  # Zero samples means zero ratio
            msg = "Dataset has zero samples. Cannot calculate feature ratio."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Step 2: Get feature columns using _get_feature_columns
        feature_cols = self._get_feature_columns(df)
        num_features = len(feature_cols)

        # Handle no features case
        if num_features == 0:
            details = {"ratio": 0.0}
            msg = "No features found. Cannot calculate feature ratio."
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(False, msg, details)

        # Step 3: Calculate and validate ratio
        ratio = num_features / num_samples
        details = {"ratio": float(f"{ratio:.3f}")}  # Store actual ratio for validation, rounded to 3 decimal places

        if ratio > self.max_feature_ratio:
            msg = (
                f"Feature-to-sample ratio ({ratio:.3f}) exceeds recommended maximum ({self.max_feature_ratio}). "
                "This may increase risk of overfitting."
            )
            if self.enable_warnings:
                warnings.warn(msg)
            return ValidationResult(passed=False, message=msg, details=details, severity="WARNING")

        return ValidationResult(True, None, details)

    def _get_feature_columns(self, df: FrameT) -> List[str]:
        """Get feature columns (all except time and target)."""
        cols = df.columns
        if hasattr(cols, "collect"):
            cols = cols.collect()
        if hasattr(cols, "to_list"):
            cols = cols.to_list()
        return [col for col in cols if col not in {self.time_col, self.target_col}]

    def fit(self, df: Union[Any, FrameT]) -> "DatasetValidator":
        """Validate input DataFrame and prepare for validation checks."""
        # Convert to Narwhals DataFrame
        df = self._ensure_narwhals_df(df)

        # Validate required columns exist
        if self.time_col not in df.columns or self.target_col not in df.columns:
            raise ValueError(f"Columns {self.time_col} and {self.target_col} must exist")

        # Validate numeric columns
        for col in df.columns:
            if col != self.time_col:
                try:
                    df.select([nw.col(col).cast(nw.Float64)])
                except Exception as e:
                    raise ValueError(f"Column {col} must be numeric. Error: {str(e)}")

        # Check nulls
        null_counts = {}
        for col in df.columns:
            null_count = df.select([nw.col(col).is_null().sum().cast(nw.Int64).alias("nulls")])
            if hasattr(null_count, "collect"):
                null_count = null_count.collect()
            value = null_count["nulls"][0]
            if hasattr(value, "as_py"):
                value = value.as_py()
            null_counts[col] = int(value)

        # Raise error if any nulls found
        null_columns = [col for col, count in null_counts.items() if count > 0]
        if null_columns:
            raise ValueError(f"Missing values detected in columns: {', '.join(null_columns)}")

        return self

    @nw.narwhalify
    def transform(self, df: FrameT, target_col: Optional[str] = None) -> Dict[str, ValidationResult]:
        """Run configured validation checks on the DataFrame."""
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

    def fit_transform(self, df: Union[Any, FrameT], target_col: Optional[str] = None) -> Dict[str, ValidationResult]:
        """Fit the validator and run validation checks in one step."""
        return self.fit(df).transform(df, target_col)

    def print_report(self, results: Dict[str, ValidationResult]) -> None:
        """Print validation results in a tabular format."""
        rows = []
        for check_name, result in results.items():
            status = "✓" if result.passed else "✗"
            message = result.message or "Check passed"
            details = ", ".join(f"{k}: {v}" for k, v in (result.details or {}).items())
            rows.append([check_name, status, message, details])

        print("\nDataset Validation Report")
        print(tabulate(rows, headers=["Check", "Status", "Message", "Details"], tablefmt="grid"))
        print("\nNote: These are research-backed recommendations and may not apply to all use cases.")
