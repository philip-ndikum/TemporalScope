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

from tabulate import tabulate  # type: ignore[import-untyped]


def validate_percentages(
    train_pct: float, test_pct: Optional[float], val_pct: Optional[float], precision: float = 1e-6
) -> Tuple[float, float, float]:
    """Validate and compute train, test, and validation percentages.

    This function ensures percentages are within the range [0, 1], computes missing values,
    and validates that their sum equals 1.0.

    Parameters
    ----------
    train_pct : float
        Percentage of data allocated for training.
    test_pct : Optional[float]
        Percentage of data allocated for testing.
    val_pct : Optional[float]
        Percentage of data allocated for validation.
    precision : float
        Tolerance for floating-point imprecision. Default is 1e-6.
    train_pct: float :

    test_pct: Optional[float] :

    val_pct: Optional[float] :

    precision: float :
         (Default value = 1e-6)

    Returns
    -------
    Tuple[float, float, float]
        Tuple of validated percentages (train_pct, test_pct, val_pct).

    Raises
    ------
    ValueError
        If percentages are invalid or do not sum to 1.0.

    """
    if not (0 <= train_pct <= 1):
        raise ValueError("`train_pct` must be between 0 and 1.")
    if test_pct is not None and not (0 <= test_pct <= 1):
        raise ValueError("`test_pct` must be between 0 and 1.")
    if val_pct is not None and not (0 <= val_pct <= 1):
        raise ValueError("`val_pct` must be between 0 and 1.")

    # Compute missing percentages
    test_pct_val: float = 0.0
    val_pct_val: float = 0.0

    if test_pct is None and val_pct is None:
        test_pct_val = 1.0 - train_pct
        val_pct_val = 0.0
    elif test_pct is not None and val_pct is None:
        test_pct_val = test_pct
        val_pct_val = 1.0 - train_pct - test_pct
    elif test_pct is None and val_pct is not None:
        val_pct_val = val_pct
        test_pct_val = 1.0 - train_pct - val_pct
    else:
        # Both are not None
        test_pct_val = test_pct if test_pct is not None else 0.0
        val_pct_val = val_pct if val_pct is not None else 0.0

    # Ensure percentages sum to 1.0
    total_pct = train_pct + test_pct_val + val_pct_val
    if not abs(total_pct - 1.0) < precision:
        raise ValueError("Train, test, and validation percentages must sum to 1.0.")

    return train_pct, test_pct_val, val_pct_val


def determine_partition_scheme(
    num_partitions: Optional[int], window_size: Optional[int], total_rows: int, stride: Optional[int]
) -> Tuple[str, int, int]:
    """Determine partition scheme based on user inputs.

    This function calculates `num_partitions` or `window_size` based on the dataset size.

    Parameters
    ----------
    num_partitions : Optional[int]
        Number of partitions, optional.
    window_size : Optional[int]
        Size of each partition, optional.
    total_rows : int
        Total number of rows in the dataset.
    stride : Optional[int]
        Number of rows to skip between partitions. Defaults to `window_size`.
    num_partitions: Optional[int] :

    window_size: Optional[int] :

    total_rows: int :

    stride: Optional[int] :


    Returns
    -------
    Tuple[str, int, int]
        Tuple containing the partition scheme ("num_partitions" or "window_size"),
        the determined number of partitions, and window size.

    Raises
    ------
    ValueError
        If both `num_partitions` and `window_size` are invalid.

    """
    if num_partitions is None and window_size is None:
        raise ValueError("Either `num_partitions` or `window_size` must be specified.")

    if num_partitions is not None:
        if num_partitions <= 0:
            raise ValueError("`num_partitions` must be a positive integer.")
        window_size_val = total_rows // num_partitions
        return "num_partitions", num_partitions, window_size_val

    if window_size is not None:
        if window_size <= 0:
            raise ValueError("`window_size` must be a positive integer.")
        stride_val = stride if stride is not None else window_size
        num_partitions_val = (total_rows - window_size) // stride_val + 1
        return "window_size", num_partitions_val, window_size

    # This should never happen due to the first check
    raise ValueError("Either `num_partitions` or `window_size` must be specified.")


def validate_cardinality(num_partitions: int, window_size: int, total_rows: int) -> None:
    """Validate dataset cardinality for the partitioning configuration.

    Parameters
    ----------
    num_partitions : int
        Number of partitions.
    window_size : int
        Size of each partition.
    total_rows : int
        Total number of rows in the dataset.
    num_partitions: int :

    window_size: int :

    total_rows: int :


    Returns
    -------
    None

    Raises
    ------
    ValueError
        If dataset cardinality is insufficient for the configuration.

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

    Parameters
    ----------
    config : dict
        Configuration dictionary with parameter names as keys and their values.
    config: dict :


    Returns
    -------
    None

    Raises
    ------
    TypeError
        If any value in the config dictionary is not an allowed type.

    """
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
