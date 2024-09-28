# """
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# """
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest

from temporalscope.partition.partition_validators import (
    check_binary_numerical_features,
    check_categorical_feature_cardinality,
    check_class_balance,
    check_feature_count,
    check_feature_to_sample_ratio,
    check_numerical_feature_uniqueness,
    check_sample_size,
)


@pytest.mark.parametrize(
    "dataframe,backend,min_samples,max_samples,expected_result",
    [
        (pd.DataFrame({"feature1": range(100)}), "pd", 3000, 50000, False),
        (
            pl.DataFrame({"feature1": pl.Series(range(100))}),
            "pl",
            3000,
            50000,
            False,
        ),
        (
            mpd.DataFrame({"feature1": range(100000)}),
            "mpd",
            3000,
            50000,
            False,
        ),
    ],
)
def test_check_sample_size(
    dataframe, backend, min_samples, max_samples, expected_result
):
    """Test sample size check for various dataframes and backends."""
    assert (
        check_sample_size(
            dataframe,
            backend=backend,
            min_samples=min_samples,
            max_samples=max_samples,
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "dataframe,backend,min_features,expected_result",
    [
        # Pandas DataFrame
        (
            pd.DataFrame({"feature1": range(100)}),
            "pd",
            4,
            False,
        ),  # Too few features - Pandas
        # Polars DataFrame
        (
            pl.DataFrame({f"feature{i}": pl.Series(range(100000)) for i in range(10)}),
            "pl",
            4,
            True,
        ),  # Enough features - Polars
        # Modin DataFrame
        (
            mpd.DataFrame({f"feature{i}": range(100000) for i in range(10)}),
            "mpd",
            4,
            True,
        ),  # Enough features - Modin
    ],
)
def test_check_feature_count(dataframe, backend, min_features, expected_result):
    """Tests check_feature_count for various dataframes and backends."""
    assert (
        check_feature_count(dataframe, backend=backend, min_features=min_features)
        == expected_result
    )


@pytest.mark.parametrize(
    "dataframe,backend,max_ratio,expected_result",
    [
        (
            pl.DataFrame({f"feature{i}": pl.Series(range(100000)) for i in range(10)}),
            "pl",
            0.1,
            True,
        ),
        (
            mpd.DataFrame({f"feature{i}": range(100000) for i in range(10)}),
            "mpd",
            0.1,
            True,
        ),
        (
            pd.DataFrame({f"feature{i}": range(100000) for i in range(10)}),
            "pd",
            0.1,
            True,
        ),
    ],
)
def test_check_feature_to_sample_ratio(dataframe, backend, max_ratio, expected_result):
    """Tests check_feature_to_sample_ratio for various dataframes and backends."""
    assert (
        check_feature_to_sample_ratio(dataframe, backend=backend, max_ratio=max_ratio)
        == expected_result
    )


@pytest.mark.parametrize(
    "dataframe,backend,max_unique_values,expected_result",
    [
        # Pandas DataFrames
        (
            pd.DataFrame({"category1": [str(i) for i in range(25)]}),
            "pd",
            20,
            False,
        ),  # Too many unique values - Pandas
        (
            pd.DataFrame({"category1": ["A", "B", "C"] * 100}),
            "pd",
            20,
            True,
        ),  # Normal unique values - Pandas
        # Polars DataFrames
        (
            pl.DataFrame({"category1": pl.Series([str(i) for i in range(25)])}),
            "pl",
            20,
            False,
        ),  # Too many unique values - Polars
        (
            pl.DataFrame({"category1": pl.Series(["A", "B", "C"] * 100)}),
            "pl",
            20,
            True,
        ),  # Normal unique values - Polars
        # Modin DataFrames
        (
            mpd.DataFrame({"category1": [str(i) for i in range(25)]}),
            "mpd",
            20,
            False,
        ),  # Too many unique values - Modin
        (
            mpd.DataFrame({"category1": ["A", "B", "C"] * 100}),
            "mpd",
            20,
            True,
        ),  # Normal unique values - Modin
    ],
)
def test_check_categorical_feature_cardinality(
    dataframe, backend, max_unique_values, expected_result
):
    """Tests check_categorical_feature_cardinality for various dataframe backends."""
    assert (
        check_categorical_feature_cardinality(
            dataframe, backend=backend, max_unique_values=max_unique_values
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "dataframe,backend,min_unique_values,expected_result",
    [
        # Pandas DataFrame
        (
            pd.DataFrame({"feature1": range(100)}),
            "pd",
            10,
            True,
        ),  # Enough unique values - Pandas
        # Polars DataFrame
        (
            pl.DataFrame({"feature1": pl.Series(range(100))}),
            "pl",
            10,
            True,
        ),  # Enough unique values - Polars
        # Modin DataFrame
        (
            mpd.DataFrame({"feature1": [1, 1, 1, 2, 2, 2, 3, 3]}),
            "mpd",
            10,
            False,
        ),  # Too few unique values - Modin
        (
            mpd.DataFrame({"feature1": range(100)}),
            "mpd",
            10,
            True,
        ),  # Enough unique values - Modin
    ],
)
def test_check_numerical_feature_uniqueness(
    dataframe, backend, min_unique_values, expected_result
):
    """Tests check_numerical_feature_uniqueness for various dataframes and backends."""
    assert (
        check_numerical_feature_uniqueness(
            dataframe, backend=backend, min_unique_values=min_unique_values
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "dataframe,backend,expected_result",
    [
        # Pandas DataFrame
        (
            pd.DataFrame({"binary_feature": [0, 1] * 50}),
            "pd",
            False,
        ),  # Binary numerical feature - Pandas
        (
            pd.DataFrame({"feature1": range(100)}),
            "pd",
            True,
        ),  # No binary feature - Pandas
        # Polars DataFrame
        (
            pl.DataFrame({"binary_feature": pl.Series([0, 1] * 50)}),
            "pl",
            False,
        ),  # Binary numerical feature - Polars
        (
            pl.DataFrame({"feature1": pl.Series(range(100))}),
            "pl",
            True,
        ),  # No binary feature - Polars
        # Modin DataFrame
        (
            mpd.DataFrame({"binary_feature": [0, 1] * 50}),
            "mpd",
            False,
        ),  # Binary numerical feature - Modin
        (
            mpd.DataFrame({"feature1": range(100)}),
            "mpd",
            True,
        ),  # No binary feature - Modin
    ],
)
def test_check_binary_numerical_features(dataframe, backend, expected_result):
    """Tests check_binary_numerical_features for various dataframes and backends."""
    assert (
        check_binary_numerical_features(dataframe, backend=backend) == expected_result
    )


@pytest.mark.parametrize(
    "dataframe,target_col,backend,expected_result",
    [
        (
            pd.DataFrame({"feature1": range(100), "target": [1] * 90 + [0] * 10}),
            "target",
            "pd",
            False,
        ),
        (
            pd.DataFrame({"feature1": range(100), "target": [0, 1] * 50}),
            "target",
            "pd",
            True,
        ),
        (
            pl.DataFrame(
                {
                    "feature1": pl.Series(range(100)),
                    "target": pl.Series([1] * 90 + [0] * 10),
                }
            ),
            "target",
            "pl",
            False,
        ),
        (
            pl.DataFrame(
                {
                    "feature1": pl.Series(range(100)),
                    "target": pl.Series([0, 1] * 50),
                }
            ),
            "target",
            "pl",
            True,
        ),
        (
            mpd.DataFrame({"feature1": range(100), "target": [1] * 90 + [0] * 10}),
            "target",
            "mpd",
            False,
        ),
        (
            mpd.DataFrame({"feature1": range(100), "target": [0, 1] * 50}),
            "target",
            "mpd",
            True,
        ),
    ],
)
def test_check_class_balance(dataframe, target_col, backend, expected_result):
    """Tests check_class_balance for various dataframes and backends."""
    result = check_class_balance(dataframe, target_col=target_col, backend=backend)
    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result} for backend {backend}"
