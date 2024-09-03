"""temporalscope/partitioning/partitioning_guidelines.py

This module provides functions to validate dataset partitions against
a set of heuristics derived from key literature in the field.
"""

from typing import Union
import pandas as pd
import polars as pl
import modin.pandas as mpd
import warnings
from temporalscope.config import validate_backend


def check_sample_size(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    min_samples: int = 3000,
    max_samples: int = 50000,
):
    """Check if the dataset meets the minimum and maximum sample size requirements.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :param min_samples: Minimum number of samples required. Default is 3000.
    :param max_samples: Maximum number of samples allowed. Default is 50000.
    :raises Warning: If the dataset does not meet the sample size requirements.
    """
    validate_backend(backend)

    num_samples = data.shape[0]
    if num_samples < min_samples:
        warnings.warn(f"Dataset has fewer than {min_samples} samples.")
    if num_samples > max_samples:
        warnings.warn(f"Dataset has more than {max_samples} samples.")


def check_feature_count(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    min_features: int = 4,
    max_features: int = 500,
):
    """Check if the dataset meets the minimum and maximum feature count requirements.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :param min_features: Minimum number of features required. Default is 4.
    :param max_features: Maximum number of features allowed. Default is 500.
    :raises Warning: If the dataset does not meet the feature count requirements.
    """
    validate_backend(backend)

    num_features = data.shape[1]
    if num_features < min_features:
        warnings.warn(f"Dataset has fewer than {min_features} features.")
    if num_features > max_features:
        warnings.warn(f"Dataset has more than {max_features} features.")


def check_feature_to_sample_ratio(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    max_ratio: float = 0.1,
):
    """Check if the feature-to-sample ratio is within acceptable limits.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :param max_ratio: Maximum allowable ratio of features to samples. Default is 0.1.
    :raises Warning: If the feature-to-sample ratio exceeds the maximum allowable ratio.
    """
    validate_backend(backend)

    num_samples = data.shape[0]
    num_features = data.shape[1]
    ratio = num_features / num_samples
    if ratio > max_ratio:
        warnings.warn(
            f"Feature-to-sample ratio exceeds the maximum allowable ratio of {max_ratio}."
        )


def check_categorical_feature_cardinality(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    max_unique_values: int = 20,
):
    """Check that categorical features do not have too many unique values.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :param max_unique_values: Maximum allowable unique values in categorical features. Default is 20.
    :raises Warning: If any categorical feature has more than the allowed number of unique values.
    """
    validate_backend(backend)

    if backend == "pd" or backend == "mpd":
        for col in data.select_dtypes(include=["category", "object"]).columns:
            if data[col].nunique() > max_unique_values:
                warnings.warn(
                    f"Categorical feature '{col}' has more than {max_unique_values} unique values."
                )
    elif backend == "pl":
        for col in data.columns:
            if data[col].dtype in [pl.Categorical, pl.Utf8]:
                if data[col].n_unique() > max_unique_values:
                    warnings.warn(
                        f"Categorical feature '{col}' has more than {max_unique_values} unique values."
                    )


def check_numerical_feature_uniqueness(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    min_unique_values: int = 10,
):
    """Check that numerical features have a sufficient number of unique values.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :param min_unique_values: Minimum number of unique values required in numerical features. Default is 10.
    :raises Warning: If any numerical feature has fewer than the required number of unique values.
    """
    validate_backend(backend)

    if backend == "pd" or backend == "mpd":
        for col in data.select_dtypes(include=["number"]).columns:
            if data[col].nunique() < min_unique_values:
                warnings.warn(
                    f"Numerical feature '{col}' has fewer than {min_unique_values} unique values."
                )
    elif backend == "pl":
        for col in data.columns:
            if data[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                if data[col].n_unique() < min_unique_values:
                    warnings.warn(
                        f"Numerical feature '{col}' has fewer than {min_unique_values} unique values."
                    )


def check_binary_numerical_features(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str = "pl"
):
    """Check if any numerical features are binary and suggest converting them to categorical.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :raises Warning: If any numerical feature has exactly 2 unique values, suggesting it should be categorical.
    """
    validate_backend(backend)

    if backend == "pd" or backend == "mpd":
        for col in data.select_dtypes(include=["number"]).columns:
            if data[col].nunique() == 2:
                warnings.warn(
                    f"Numerical feature '{col}' has only 2 unique values. Consider converting it to categorical."
                )
    elif backend == "pl":
        for col in data.columns:
            if data[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                if data[col].n_unique() == 2:
                    warnings.warn(
                        f"Numerical feature '{col}' has only 2 unique values. Consider converting it to categorical."
                    )


def check_class_balance(
    data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    target_col: str,
    backend: str = "pl",
):
    """Check that classes in a classification dataset are balanced.

    :param data: The dataset to check.
    :type data: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param target_col: The name of the target column in the dataset.
    :type target_col: str
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin).
    :type backend: str
    :raises Warning: If the classes in the target column are imbalanced.
    """
    validate_backend(backend)

    if backend == "pd" or backend == "mpd":
        class_counts = data[target_col].value_counts()
    elif backend == "pl":
        class_counts = data[target_col].value_counts().to_dict()

    if class_counts[max(class_counts)] / class_counts[min(class_counts)] > 1.5:
        warnings.warn("Classes are imbalanced. Consider balancing the dataset.")
