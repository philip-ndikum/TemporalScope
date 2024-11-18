# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""TemporalScope/src/temporalscope/datasets/datasets.py

Utility for loading datasets with multi-backend support. This class simplifies dataset loading, enabling
compatibility with multiple DataFrame backends (such as Pandas, Modin, Polars) for TemporalScope tutorials and examples.

Example Usage:
--------------
.. code-block:: python

    from temporalscope.datasets.datasets import DatasetLoader

    # Initialize with 'macrodata' dataset
    dataset_loader = DatasetLoader("macrodata")

    # Load dataset with specified backend
    data = dataset_loader.load_data(backend="polars")
    print(data.head())  # Example access
"""

from typing import Any, Tuple

import pandas as pd
from statsmodels.datasets import macrodata

from temporalscope.core.core_utils import (
    convert_to_backend,
    is_valid_temporal_backend,
    print_divider,
)

# Dictionary of available datasets and their loaders
AVAILABLE_DATASETS = {
    "macrodata": lambda: _load_macrodata(),  # Extend here with more datasets
}


def _load_macrodata() -> Tuple[pd.DataFrame, str]:
    """Load and preprocess the macrodata dataset.

    :return: Preprocessed DataFrame and default target column 'realgdp'.
    :rtype: Tuple[pd.DataFrame, str]
    """
    loaded_data = macrodata.load_pandas().data
    if loaded_data is None:
        raise ValueError("Failed to load macrodata dataset")
    dataset_df = loaded_data.copy()
    dataset_df["year"] = dataset_df["year"].astype(int)
    dataset_df["quarter"] = dataset_df["quarter"].astype(int)
    dataset_df["ds"] = pd.to_datetime(
        dataset_df["year"].astype(str) + "-" + ((dataset_df["quarter"] - 1) * 3 + 1).astype(str) + "-01"
    )
    dataset_df.drop(columns=["year", "quarter"], inplace=True)
    return dataset_df, "realgdp"


class DatasetLoader:
    """Class for loading datasets with multi-backend support.

    This class enables loading and conversion of datasets into various backend formats (e.g., pandas, modin, polars).

    Attributes
    ----------
    dataset_name : str
        Name of the dataset to load, as defined in AVAILABLE_DATASETS.

    Methods
    -------
    load_data(backend: str = "pandas") -> Any
        Load the dataset and convert it to the specified backend format.

    """

    def __init__(self, dataset_name: str = "macrodata") -> None:
        """Initialize DatasetLoader with a specified dataset.

        :param dataset_name: Name of the dataset to load. Must be available in AVAILABLE_DATASETS.
        :raises ValueError: If the specified dataset is not available.
        """
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' is not supported. Available datasets: {list(AVAILABLE_DATASETS.keys())}"
            )
        self.dataset_name = dataset_name

    def _load_dataset_and_target(self) -> Tuple[pd.DataFrame, str]:
        """Load the dataset and its target column.

        :return: DataFrame and associated target column name.
        :rtype: Tuple[pd.DataFrame, str]
        """
        print_divider()
        print(f"Loading dataset: '{self.dataset_name}'")
        print_divider()
        dataset_df, target_col = AVAILABLE_DATASETS[self.dataset_name]()
        print(f"DataFrame shape: {dataset_df.shape}")
        print(f"Target column: {target_col}")
        print_divider()
        return dataset_df, target_col

    def load_data(self, backend: str = "pandas") -> Any:
        """Load the dataset and convert it to the specified backend format.

        :param backend: Backend to convert the dataset to. Default is 'pandas'.
        :type backend: str
        :return: Dataset in the specified backend format.
        :rtype: Backend-specific DataFrame type (e.g., pandas.DataFrame, modin.DataFrame, polars.DataFrame)
        :raises ValueError: If the backend is unsupported.
        """
        # Validate and load the dataset in pandas format
        is_valid_temporal_backend(backend)
        df, _ = self._load_dataset_and_target()

        # Convert to specified backend
        return convert_to_backend(df, backend)
