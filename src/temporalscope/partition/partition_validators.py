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

"""TemporalScope/temporalscope/partition/partition_validators.py.

This module provides functions to validate dataset partitions against
a set of heuristics derived from key literature in the field.

.. seealso::
    1. Shwartz-Ziv, R. and Armon, A., 2022. Tabular data: Deep learning is not all you need. Information Fusion, 81, pp.84-90.
    2. Grinsztajn, L., Oyallon, E. and Varoquaux, G., 2022. Why do tree-based models still outperform deep learning on typical tabular data?
    3. Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A., 2021. Revisiting deep learning models for tabular data.
"""

import warnings
from typing import Any, Dict, TypeVar, cast

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.core_utils import validate_backend
from temporalscope.core.core_utils import SupportedBackendDataFrame

PandasLike = TypeVar("PandasLike", pd.DataFrame, mpd.DataFrame)


def check_sample_size(
    df: SupportedBackendDataFrame,
    backend: str = "pl",
    min_samples: int = 3000,
    max_samples: int = 50000,
    enable_warnings: bool = False,
) -> bool:
    """Check if the dataset meets the minimum and maximum sample size requirements.

    This function checks if the dataset contains an appropriate number of samples
    for training machine learning models. If the dataset has too few or too many samples,
    warnings can be triggered depending on the `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param min_samples: Minimum number of samples required.
    :type min_samples: int
    :param max_samples: Maximum number of samples allowed.
    :type max_samples: int
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :return: True if the dataset meets the sample size requirements, otherwise False.
    :rtype: bool
    """
    validate_backend(backend)

    num_samples = df.shape[0]

    if num_samples < min_samples:
        if enable_warnings:
            warnings.warn(
                f"Dataset has fewer than {min_samples} samples. "
                "Based on heuristics from the literature, this dataset size may not be suitable for complex machine learning models. "
                "Consider alternative approaches such as Linear, Bayesian, or other models that work well with smaller datasets."
            )
        return False

    if num_samples > max_samples:
        if enable_warnings:
            warnings.warn(
                f"Dataset has more than {max_samples} samples. "
                "Larger datasets like this might benefit from scalable implementations of classical models or deep learning techniques."
            )
        return False

    return True


def check_feature_count(
    df: SupportedBackendDataFrame,
    backend: str = "pl",
    min_features: int = 4,
    max_features: int = 500,
    enable_warnings: bool = False,
) -> bool:
    """Check if the dataset meets the minimum and maximum feature count requirements.

    This function ensures the dataset has an appropriate number of features for modeling.
    If the feature count is too low or too high, warnings can be triggered depending on the
    `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param min_features: Minimum number of features required.
    :type min_features: int
    :param max_features: Maximum number of features allowed.
    :type max_features: int
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :return: True if the dataset meets the feature count requirements, otherwise False.
    :rtype: bool
    """
    validate_backend(backend)

    num_features = df.shape[1]

    if num_features < min_features:
        if enable_warnings:
            warnings.warn(
                f"Dataset has fewer than {min_features} features. "
                "According to best practices, having too few features may result in an oversimplified model, "
                "which may not capture the complexity of the data. Consider adding more informative features."
            )
        return False

    if num_features > max_features:
        if enable_warnings:
            warnings.warn(
                f"Dataset has more than {max_features} features. "
                "High-dimensional data can cause issues like overfitting. Consider dimensionality reduction techniques such as PCA or feature selection."
            )
        return False

    return True


def check_feature_to_sample_ratio(
    df: SupportedBackendDataFrame,
    backend: str = "pl",
    max_ratio: float = 0.1,
    enable_warnings: bool = False,
) -> bool:
    """Check if the feature-to-sample ratio is within acceptable limits.

    This function verifies if the dataset's feature-to-sample ratio exceeds the maximum allowable ratio,
    which may increase the risk of overfitting. Warnings can be triggered depending on the `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param max_ratio: Maximum allowable feature-to-sample ratio.
    :type max_ratio: float
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :return: True if the feature-to-sample ratio is within limits, otherwise False.
    :rtype: bool
    """
    validate_backend(backend)

    num_samples = df.shape[0]
    num_features = df.shape[1]

    ratio = num_features / num_samples
    if ratio > max_ratio:
        if enable_warnings:
            warnings.warn(
                f"Feature-to-sample ratio exceeds {max_ratio}. "
                "This can increase the risk of overfitting. Consider using regularization techniques such as L2 regularization, "
                "or applying feature selection methods to reduce the dimensionality of the dataset."
            )
        return False

    return True


def check_categorical_feature_cardinality(
    df: SupportedBackendDataFrame,
    backend: str = "pl",
    max_unique_values: int = 20,
    enable_warnings: bool = False,
) -> bool:
    """Check that categorical features do not have too many unique values.

    This function ensures that categorical features have an acceptable number of unique values.
    High-cardinality categorical features can complicate model training and increase the risk of overfitting.
    Warnings can be triggered depending on the `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param max_unique_values: Maximum number of unique values allowed for categorical features.
    :type max_unique_values: int
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :return: True if the categorical features meet the cardinality limits, otherwise False.
    :rtype: bool
    :raises: ValueError if backend is not supported.
    """
    validate_backend(backend)

    if backend == "pl":
        # Explicitly cast to Polars DataFrame
        polars_df = cast(pl.DataFrame, df)
        for col in polars_df.columns:
            # Check categorical or string columns
            if polars_df[col].dtype in [pl.Categorical, pl.Utf8]:
                if polars_df[col].n_unique() > max_unique_values:
                    if enable_warnings:
                        warnings.warn(
                            f"Categorical feature '{col}' has more than {max_unique_values} unique values. "
                            "Consider using encoding techniques such as target encoding, one-hot encoding, or embeddings to handle high-cardinality features."
                        )
                    return False

    elif backend in ["pd", "mpd"]:
        # Explicitly cast to Pandas/Modin DataFrame
        pandas_df = cast(pd.DataFrame, df) if backend == "pd" else cast(mpd.DataFrame, df)
        categorical_columns = pandas_df.select_dtypes(include=["category", "object"]).columns
        for col in categorical_columns:
            if pandas_df[col].nunique() > max_unique_values:
                if enable_warnings:
                    warnings.warn(
                        f"Categorical feature '{col}' has more than {max_unique_values} unique values. "
                        "Consider using encoding techniques such as target encoding, one-hot encoding, or embeddings to handle high-cardinality features."
                    )
                return False

    return True


def check_numerical_feature_uniqueness(
    df: SupportedBackendDataFrame,
    backend: str = "pl",
    min_unique_values: int = 10,
    enable_warnings: bool = False,
) -> bool:
    """Check that numerical features have a sufficient number of unique values.

    This function ensures that numerical features contain a minimum number of unique values.
    Features with too few unique values may lack variability, reducing model expressiveness.
    Warnings can be triggered depending on the `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param min_unique_values: Minimum number of unique values required for numerical features.
    :type min_unique_values: int
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :return: True if all numerical features have at least `min_unique_values` unique values, otherwise False.
    :rtype: bool
    """
    validate_backend(backend)

    if backend in ["pd", "mpd"]:
        pandas_df = df  # Type narrowing for mypy
        if isinstance(pandas_df, (pd.DataFrame, mpd.DataFrame)):
            numerical_columns = pandas_df.select_dtypes(include=["number"]).columns
            for col in numerical_columns:
                if pandas_df[col].nunique() < min_unique_values:
                    if enable_warnings:
                        warnings.warn(
                            f"Numerical feature '{col}' has fewer than {min_unique_values} unique values. "
                            "Low feature variability can limit model expressiveness and accuracy. "
                            "Consider feature engineering or transformations (e.g., log transformation, interaction terms) to increase variability."
                        )
                    return False
    elif backend == "pl":
        polars_df = df  # Type narrowing for mypy
        if isinstance(polars_df, pl.DataFrame):
            for col in polars_df.columns:
                if polars_df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    if polars_df[col].n_unique() < min_unique_values:
                        if enable_warnings:
                            warnings.warn(
                                f"Numerical feature '{col}' has fewer than {min_unique_values} unique values. "
                                "Low feature variability can limit model expressiveness and accuracy. "
                                "Consider feature engineering or transformations (e.g., log transformation, interaction terms) to increase variability."
                            )
                        return False

    return True


def check_binary_numerical_features(
    df: SupportedBackendDataFrame,
    backend: str = "pl",
    enable_warnings: bool = False,
) -> bool:
    """Detect binary numerical features and suggest converting them to categorical.

    Binary numerical features (i.e., features with only two unique values) are
    often better represented as categorical features. This function detects
    such features and suggests conversion. Warnings can be triggered depending
    on the `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :return: True if no binary numerical features are found, otherwise False.
    :rtype: bool
    """
    BINARY_UNIQUE_VALUES = 2  # Constant for binary unique values

    validate_backend(backend)

    if backend in ["pd", "mpd"]:
        pandas_df = df  # Type narrowing for mypy
        if isinstance(pandas_df, (pd.DataFrame, mpd.DataFrame)):
            numerical_columns = pandas_df.select_dtypes(include=["number"]).columns
            for col in numerical_columns:
                if pandas_df[col].nunique() == BINARY_UNIQUE_VALUES:
                    if enable_warnings:
                        warnings.warn(
                            f"Numerical feature '{col}' has only {BINARY_UNIQUE_VALUES} "
                            "unique values. Binary numerical features should typically "
                            "be converted to categorical for better model performance and "
                            "interpretability."
                        )
                    return False

    elif backend == "pl":
        polars_df = df  # Type narrowing for mypy
        if isinstance(polars_df, pl.DataFrame):
            for col in polars_df.columns:
                if polars_df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    if polars_df[col].n_unique() == BINARY_UNIQUE_VALUES:
                        if enable_warnings:
                            warnings.warn(
                                f"Numerical feature '{col}' has only {BINARY_UNIQUE_VALUES} "
                                "unique values. Binary numerical features should typically "
                                "be converted to categorical for better model performance and "
                                "interpretability."
                            )
                        return False

    return True


def check_class_balance(
    df: SupportedBackendDataFrame,
    target_col: str,
    backend: str = "pl",
    enable_warnings: bool = False,
    imbalance_threshold: float = 1.5,  # Default threshold for class imbalance
) -> bool:
    """Check that classes in a classification dataset are balanced.

    This function checks the class distribution in the target column of a classification dataset.
    If the ratio between the largest and smallest classes exceeds the defined threshold, the dataset is considered imbalanced.
    Warnings can be triggered depending on the `enable_warnings` flag.

    :param df: The dataset to check.
    :type df: SupportedBackendDataFrame
    :param target_col: The column containing the target labels.
    :type target_col: str
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param enable_warnings: Flag to enable warnings, defaults to False.
    :type enable_warnings: bool
    :param imbalance_threshold: The threshold for determining class imbalance, defaults to 1.5.
    :type imbalance_threshold: float
    :return: True if classes are balanced (ratio <= threshold), otherwise False.
    :rtype: bool
    :raises: ValueError if backend is not supported.
    """
    validate_backend(backend)

    class_counts: Dict[Any, int] = {}

    if backend in ["pd", "mpd"]:
        # Explicitly cast to Pandas/Modin DataFrame
        pandas_df = cast(pd.DataFrame, df) if backend == "pd" else cast(mpd.DataFrame, df)
        value_counts = pandas_df[target_col].value_counts()
        class_counts = {k: int(v) for k, v in value_counts.items()}

    elif backend == "pl":
        # Explicitly cast to Polars DataFrame
        polars_df = cast(pl.DataFrame, df)
        value_counts = polars_df[target_col].value_counts()
        class_counts = {str(row[target_col]): int(row["count"]) for row in value_counts.to_dicts()}

    if class_counts:
        count_values = list(class_counts.values())
        max_count = max(count_values)
        min_count = min(count_values)

        if max_count / min_count > imbalance_threshold:
            if enable_warnings:
                warnings.warn(
                    "Classes are imbalanced. Consider using techniques like class weighting, SMOTE, or resampling to address class imbalance."
                )
            return False

    return True
