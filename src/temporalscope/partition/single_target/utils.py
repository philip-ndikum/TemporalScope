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

"""TemporalScope/src/temporalscope/partition/single_target/utils.py

This module provides utility functions for single-target partitioning operations,
including validation and computation of train/test/validation split percentages.
"""

from typing import Optional, Tuple


def validate_percentages(
    train_pct: float, test_pct: Optional[float], val_pct: Optional[float], precision: float = 1e-6
) -> Tuple[float, float, float]:
    """Validate and compute train, test, and validation percentages.

    This function ensures percentages are within the range [0, 1], computes missing values,
    and validates that their sum equals 1.0.

    :param train_pct: Percentage of data allocated for training.
    :type train_pct: float
    :param test_pct: Percentage of data allocated for testing.
    :type test_pct: Optional[float]
    :param val_pct: Percentage of data allocated for validation.
    :type val_pct: Optional[float]
    :param precision: Tolerance for floating-point imprecision. Default is 1e-6.
    :type precision: float
    :return: Tuple of validated percentages (train_pct, test_pct, val_pct).
    :rtype: Tuple[float, float, float]
    :raises ValueError: If percentages are invalid or do not sum to 1.0.
    """
    if not (0 <= train_pct <= 1):
        raise ValueError("`train_pct` must be between 0 and 1.")
    if test_pct is not None and not (0 <= test_pct <= 1):
        raise ValueError("`test_pct` must be between 0 and 1.")
    if val_pct is not None and not (0 <= val_pct <= 1):
        raise ValueError("`val_pct` must be between 0 and 1.")

    # Compute missing percentages
    if test_pct is None and val_pct is None:
        test_pct = 1.0 - train_pct
        val_pct = 0.0
    elif test_pct is not None and val_pct is None:
        val_pct = 1.0 - train_pct - test_pct
    elif test_pct is None and val_pct is not None:
        test_pct = 1.0 - train_pct - val_pct

    # Ensure percentages sum to 1.0
    total_pct = train_pct + test_pct + val_pct
    if not abs(total_pct - 1.0) < precision:
        raise ValueError("Train, test, and validation percentages must sum to 1.0.")

    return float(train_pct), float(test_pct), float(val_pct)


def determine_partition_scheme(
    num_partitions: Optional[int], window_size: Optional[int], total_rows: int, stride: Optional[int]
) -> Tuple[str, int, int]:
    """Determine partition scheme based on user inputs.

    This function calculates `num_partitions` or `window_size` based on the dataset size.

    :param num_partitions: Number of partitions, optional.
    :type num_partitions: Optional[int]
    :param window_size: Size of each partition, optional.
    :type window_size: Optional[int]
    :param total_rows: Total number of rows in the dataset.
    :type total_rows: int
    :param stride: Number of rows to skip between partitions. Defaults to `window_size`.
    :type stride: Optional[int]
    :return: Tuple containing the partition scheme ("num_partitions" or "window_size"),
             the determined number of partitions, and window size.
    :rtype: Tuple[str, int, int]
    :raises ValueError: If both `num_partitions` and `window_size` are invalid.
    """
    if num_partitions is None and window_size is None:
        raise ValueError("Either `num_partitions` or `window_size` must be specified.")

    if num_partitions is not None:
        if num_partitions <= 0:
            raise ValueError("`num_partitions` must be a positive integer.")
        window_size = total_rows // num_partitions
        return "num_partitions", num_partitions, window_size

    if window_size is not None:
        if window_size <= 0:
            raise ValueError("`window_size` must be a positive integer.")
        num_partitions = (total_rows - window_size) // (stride or window_size) + 1
        return "window_size", num_partitions, window_size


def validate_cardinality(num_partitions: int, window_size: int, total_rows: int) -> None:
    """Validate dataset cardinality for the partitioning configuration.

    :param num_partitions: Number of partitions.
    :type num_partitions: int
    :param window_size: Size of each partition.
    :type window_size: int
    :param total_rows: Total number of rows in the dataset.
    :type total_rows: int
    :raises ValueError: If dataset cardinality is insufficient for the configuration.
    """
    if num_partitions > total_rows:
        raise ValueError(f"Insufficient rows ({total_rows}) for `num_partitions={num_partitions}`.")
    if window_size > total_rows:
        raise ValueError(f"Insufficient rows ({total_rows}) for `window_size={window_size}`.")


def print_config(config: dict) -> None:
    """Print a configuration as a table with validation for allowed data types.

    This function ensures that all values in the configuration are of allowed types
    (`int`, `float`, `bool`, `str`). It raises an error for any invalid types and then
    prints the configuration as a table.

    :param config: Configuration dictionary with parameter names as keys and their values.
    :type config: dict
    :raises TypeError: If any value in the config dictionary is not an allowed type.
    """
    from tabulate import tabulate

    # Allowed data types for config values
    allowed_types = (int, float, bool, str)

    # Validate data types in config
    invalid_entries = [
        (key, type(value).__name__) for key, value in config.items() if not isinstance(value, allowed_types)
    ]
    if invalid_entries:
        error_message = "\n".join([f"{key}: {dtype}" for key, dtype in invalid_entries])
        raise TypeError(f"Invalid data types in config:\n{error_message}")

    # Prepare table data
    table_data = [[key, value] for key, value in config.items()]

    # Print table
    print("Configuration Details:\n")
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
