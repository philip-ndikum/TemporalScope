""" temporalscope/partitioning/partitioning_guidelines.py

This module provides functions to validate dataset partitions against
a set of heuristics derived from key literature in the field.
<<<<<<< HEAD

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======
>>>>>>> 6ecf0623f3d8d3c1f7607c1dd06e9c824d0dab98
"""

from typing import Union, TypeVar, Any, Dict
import warnings
import pandas as pd
import polars as pl
import modin.pandas as mpd
from temporalscope.conf import validate_backend

PandasLike = TypeVar("PandasLike", pd.DataFrame, mpd.DataFrame)


def check_sample_size(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    min_samples: int = 3000,
    max_samples: int = 50000,
) -> None:
    """Check if the dataset meets the minimum and maximum sample size requirements.

    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param min_samples: Minimum number of samples required.
    :type min_samples: int
    :param max_samples: Maximum number of samples allowed.
    :type max_samples: int
    :return: None
    :rtype: None
    """
    validate_backend(backend)

    num_samples = df.shape[0]

    if num_samples < min_samples:
        warnings.warn(f"Dataset has fewer than {min_samples} samples.")
    if num_samples > max_samples:
        warnings.warn(f"Dataset has more than {max_samples} samples.")


def check_feature_count(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    min_features: int = 4,
    max_features: int = 500,
) -> None:
    """Check if the dataset meets the minimum and maximum feature count requirements.

    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param min_features: Minimum number of features required.
    :type min_features: int
    :param max_features: Maximum number of features allowed.
    :type max_features: int
    :return: None
    :rtype: None
    """
    validate_backend(backend)

    num_features = df.shape[1]

    if num_features < min_features:
        warnings.warn(f"Dataset has fewer than {min_features} features.")
    if num_features > max_features:
        warnings.warn(f"Dataset has more than {max_features} features.")


def check_feature_to_sample_ratio(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    max_ratio: float = 0.1,
) -> None:
    """Check if the feature-to-sample ratio is within acceptable limits.

    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param max_ratio: Maximum allowable feature-to-sample ratio.
    :type max_ratio: float
    :return: None
    :rtype: None
    """
    validate_backend(backend)

    num_samples = df.shape[0]
    num_features = df.shape[1]

    ratio = num_features / num_samples
    if ratio > max_ratio:
        warnings.warn(
            f"Feature-to-sample ratio exceeds the maximum allowable ratio of {max_ratio}."
        )


def check_categorical_feature_cardinality(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    max_unique_values: int = 20,
) -> None:
    """Check that categorical features do not have too many unique values.

    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param max_unique_values: Maximum number of unique values allowed for categorical features.
    :type max_unique_values: int
    :return: None
    :rtype: None
    """
    validate_backend(backend)
    if backend == "pl":
        for col in df.columns:
            if isinstance(df[col].dtype, (pl.Categorical, pl.Utf8)):
                if df[col].n_unique() > max_unique_values:
                    warnings.warn(
                        f"Categorical feature '{col}' has more than {max_unique_values} unique values."
                    )
    elif backend in ["pd", "mpd"]:
        pandas_df = df  # Type narrowing for mypy
        if isinstance(pandas_df, (pd.DataFrame, mpd.DataFrame)):
            categorical_columns = pandas_df.select_dtypes(
                include=["category", "object"]
            ).columns
            for col in categorical_columns:
                if pandas_df[col].nunique() > max_unique_values:
                    warnings.warn(
                        f"Categorical feature '{col}' has more than {max_unique_values} unique values."
                    )


def check_numerical_feature_uniqueness(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    backend: str = "pl",
    min_unique_values: int = 10,
) -> None:
    """Check that numerical features have a sufficient number of unique values.

    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :param min_unique_values: Minimum number of unique values required for numerical features.
    :type min_unique_values: int
    :return: None
    :rtype: None
    """
    validate_backend(backend)
    if backend in ["pd", "mpd"]:
        pandas_df = df  # Type narrowing for mypy
        if isinstance(pandas_df, (pd.DataFrame, mpd.DataFrame)):
            numerical_columns = pandas_df.select_dtypes(include=["number"]).columns
            for col in numerical_columns:
                if pandas_df[col].nunique() < min_unique_values:
                    warnings.warn(
                        f"Numerical feature '{col}' has fewer than {min_unique_values} unique values."
                    )
    elif backend == "pl":
        polars_df = df  # Type narrowing for mypy
        if isinstance(polars_df, pl.DataFrame):
            for col in polars_df.columns:
                if polars_df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    if polars_df[col].n_unique() < min_unique_values:
                        warnings.warn(
                            f"Numerical feature '{col}' has fewer than {min_unique_values} unique values."
                        )


def check_binary_numerical_features(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame], backend: str = "pl"
) -> None:
    """Check if any numerical features are binary and suggest converting them to categorical.
    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :return: None
    :rtype: None
    """
    validate_backend(backend)
    if backend in ["pd", "mpd"]:
        pandas_df = df  # Type narrowing for mypy
        if isinstance(pandas_df, (pd.DataFrame, mpd.DataFrame)):
            numerical_columns = pandas_df.select_dtypes(include=["number"]).columns
            for col in numerical_columns:
                if pandas_df[col].nunique() == 2:
                    warnings.warn(
                        f"Numerical feature '{col}' has only 2 unique values. Consider converting it to categorical."
                    )
    elif backend == "pl":
        polars_df = df  # Type narrowing for mypy
        if isinstance(polars_df, pl.DataFrame):
            for col in polars_df.columns:
                if polars_df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    if polars_df[col].n_unique() == 2:
                        warnings.warn(
                            f"Numerical feature '{col}' has only 2 unique values. Consider converting it to categorical."
                        )


def check_class_balance(
    df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
    target_col: str,
    backend: str = "pl",
) -> None:
    """Check that classes in a classification dataset are balanced.
    :param df: The dataset to check.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param target_col: The column containing the target labels.
    :type target_col: str
    :param backend: The backend used for processing ('pd', 'pl', 'mpd').
    :type backend: str
    :return: None
    :rtype: None
    """
    validate_backend(backend)

    class_counts: Dict[Any, int] = {}

    if backend in ["pd", "mpd"]:
        pandas_df = df
        if isinstance(pandas_df, (pd.DataFrame, mpd.DataFrame)):
            value_counts = pandas_df[target_col].value_counts()
            class_counts = {k: int(v) for k, v in value_counts.items()}
    elif backend == "pl":
        polars_df = df
        if isinstance(polars_df, pl.DataFrame):
            value_counts = polars_df[target_col].value_counts()
            class_counts = {
                str(k): int(v)
                for k, v in zip(value_counts["values"], value_counts["counts"])
            }

    if class_counts:
        count_values = list(class_counts.values())
        max_count = max(count_values)
        min_count = min(count_values)
        if max_count / min_count > 1.5:
            warnings.warn("Classes are imbalanced. Consider balancing the dataset.")
