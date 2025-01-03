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

"""TemporalScope/src/temporalscope/datasets/datasets.py

Utility for loading datasets with multi-backend support. This class simplifies dataset loading, enabling
compatibility with multiple DataFrame backends (such as Pandas, Modin, Polars) for TemporalScope tutorials and examples.

Examples
--------
```python
from temporalscope.datasets.datasets import DatasetLoader

# Initialize with 'macrodata' dataset
dataset_loader = DatasetLoader("macrodata")

# Load dataset with specified backend
data = dataset_loader.load_data(backend="polars")
print(data.head())  # Example access
```
"""

from typing import Tuple

import narwhals as nw
import pandas as pd
from narwhals.typing import FrameT
from statsmodels.datasets import macrodata

from temporalscope.core.core_utils import NARWHALS_BACKENDS, print_divider

# Dictionary of available datasets and their loaders
AVAILABLE_DATASETS = {
    "macrodata": lambda: _load_macrodata(),  # Extend here with more datasets
}


def _load_macrodata() -> Tuple[pd.DataFrame, str]:
    """Load and preprocess the macrodata dataset.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        Preprocessed DataFrame and default target column 'realgdp'.
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
    """

    def __init__(self, dataset_name: str = "macrodata") -> None:
        """Initialize DatasetLoader with a specified dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load. Default is 'macrodata'.

        Raises
        ------
        ValueError
            if the specified dataset is not available.
        """
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' is not supported. Available datasets: {list(AVAILABLE_DATASETS.keys())}"
            )
        self.dataset_name = dataset_name

    def _load_dataset_and_target(self) -> Tuple[pd.DataFrame, str]:
        """Load the dataset and its target column.

        Returns
        -------
        Tuple[pd.DataFrame, str]
            DataFrame and associated target column name.
        """
        print_divider()
        print(f"Loading dataset: '{self.dataset_name}'")
        print_divider()
        dataset_df, target_col = AVAILABLE_DATASETS[self.dataset_name]()
        print(f"DataFrame shape: {dataset_df.shape}")
        print(f"Target column: {target_col}")
        print_divider()
        return dataset_df, target_col

    @nw.narwhalify
    def load_data(self, backend: str = "pandas") -> FrameT:
        """Load the dataset and convert it to the specified backend format.

        Parameters
        ----------
        backend : str
            Backend to convert the dataset to. Default is 'pandas'.

        Returns
        -------
        FrameT
            Dataset in the specified backend format.

        Raises
        ------
        ValueError
            If the backend is unsupported.
        """
        # Validate backend
        backend = backend.lower()
        if backend not in NARWHALS_BACKENDS:
            raise ValueError(f"Backend '{backend}' is not supported. Available backends: {NARWHALS_BACKENDS}")

        # Load dataset in pandas format
        df, _ = self._load_dataset_and_target()

        # Convert to Narwhals DataFrame
        return nw.from_native(df)
