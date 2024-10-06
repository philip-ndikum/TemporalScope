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

Utility class for loading datasets and initializing TimeFrame objects with multi-backend support.
Supports Pandas, Modin, and Polars as backends for time series forecasting and analysis.

This class is intended to be used for tutorials and examples that involve open-source datasets
licensed under Apache, MIT, or similar valid open-source licenses. It simplifies dataset loading
and preprocessing while providing compatibility with multiple DataFrame backends, including Pandas,
Modin, and Polars. The class can be easily extended to include additional datasets in the future.

Example:
---------
.. code-block:: python

    from temporalscope.datasets.datasets import DatasetLoader

    # Initialize the dataset loader with the 'macrodata' dataset
    dataset_loader = DatasetLoader(dataset_name="macrodata")

    # Load and initialize TimeFrames for the specified backends (Pandas, Modin, Polars)
    timeframes = dataset_loader.load_and_init_timeframes()

    # Access the Modin TimeFrame and perform operations
    modin_tf = timeframes["modin"]
    print(modin_tf.get_data().head())

    # Access metadata of the Modin TimeFrame object
    print(modin_tf.__dict__)

"""

import pandas as pd
import modin.pandas as mpd
import polars as pl
from statsmodels.datasets import macrodata
from typing import Tuple, Dict, Callable, Union
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.core.core_utils import (
    BACKEND_PANDAS,
    BACKEND_MODIN,
    BACKEND_POLARS,
    SupportedBackendDataFrame,
    print_divider,
)


def _load_macrodata() -> Tuple[pd.DataFrame, str]:
    """Load and preprocess the macrodata dataset.

    Combines the 'year' and 'quarter' columns to create a datetime 'ds' column.
    The dataset is then returned with the 'realgdp' column as the default target.

    :return: A tuple containing the preprocessed DataFrame and the default target column 'realgdp'.
    :rtype: Tuple[pd.DataFrame, str]
    """
    dataset_df = macrodata.load_pandas().data.copy()

    # Ensure 'year' and 'quarter' are integers
    dataset_df["year"] = dataset_df["year"].astype(int)
    dataset_df["quarter"] = dataset_df["quarter"].astype(int)

    # Combine 'year' and 'quarter' to create a datetime 'ds' column
    dataset_df["ds"] = pd.to_datetime(
        dataset_df["year"].astype(str) + "-" + ((dataset_df["quarter"] - 1) * 3 + 1).astype(str) + "-01"
    )

    # Drop the 'year' and 'quarter' columns
    dataset_df = dataset_df.drop(columns=["year", "quarter"])  # Remove redundant columns

    # Return the dataset and the default target column
    return dataset_df, "realgdp"


# Map of available datasets to their respective loader functions
AVAILABLE_DATASETS = {
    "macrodata": _load_macrodata,
    # Future datasets can be added here with their corresponding loading functions
}

SupportedBackends = Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]


class DatasetLoader:
    """A utility class for loading datasets and initializing TimeFrame objects for multiple backends.

    This class supports datasets that are licensed under valid open-source licenses (such as Apache and MIT).
    It simplifies loading and preprocessing of datasets and enables compatibility with Pandas, Modin, and Polars
    DataFrame backends. Designed for tutorials and practical examples, this class is ideal for educational purposes
    and demonstration of time series forecasting workflows.

    Attributes:
    ------------
    dataset_name : str
        The name of the dataset to be loaded. It must be available in the `AVAILABLE_DATASETS` dictionary.

    Methods:
    ---------
    load_and_init_timeframes:
        Load the specified dataset and initialize TimeFrame objects for multiple backends.

    Example:
    ---------
    .. code-block:: python

        # Initialize the loader with the 'macrodata' dataset
        dataset_loader = DatasetLoader(dataset_name="macrodata")

        # Load and initialize TimeFrames for Pandas, Modin, and Polars
        timeframes = dataset_loader.load_and_init_timeframes()

        # Access the Modin TimeFrame object
        modin_tf = timeframes["modin"]
        print(modin_tf.get_data().head())

    """

    def __init__(self, dataset_name: str = "macrodata") -> None:
        """
        Initialize DatasetLoader with a specified dataset.

        :param dataset_name: The name of the dataset to load. Must be available in AVAILABLE_DATASETS.
        :raises ValueError: If the specified dataset is not available.
        """
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' is not supported. Available datasets: {list(AVAILABLE_DATASETS.keys())}"
            )
        self.dataset_name = dataset_name

    def _load_dataset_and_target(self) -> Tuple[pd.DataFrame, str]:
        """
        Internal method to load the dataset and its associated target column.

        :return: A tuple containing the preprocessed DataFrame and the associated target column name.
        :rtype: Tuple[pd.DataFrame, str]
        """
        print_divider()
        print(f"Loading the '{self.dataset_name}' dataset.")
        print_divider()

        # Fetch the dataset loader function and load the dataset with its target column
        loader_func: Callable[[], Tuple[pd.DataFrame, str]] = AVAILABLE_DATASETS[self.dataset_name]
        dataset_df, target_col = loader_func()

        print(f"Loaded DataFrame shape: {dataset_df.shape}")
        print(f"Target column: {target_col}")
        print_divider()

        return dataset_df, target_col

    def init_timeframes_for_backends(
        self,
        df: pd.DataFrame,
        target_col: str,
        backends: Tuple[str, ...] = (BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS),
    ) -> Dict[str, TimeFrame]:
        """Initialize TimeFrame objects for the specified backends using the provided DataFrame.

        :param df: The preprocessed DataFrame to initialize TimeFrames with.
        :param target_col: The target column to use for TimeFrame initialization.
        :param backends: A tuple of supported backends to initialize. Defaults to Pandas, Modin, and Polars.
        :return: A dictionary containing TimeFrame objects for each requested backend.
        :rtype: Dict[str, TimeFrame]
        :raises ValueError: If an unsupported backend is specified.

        Example:
        ---------
        .. code-block:: python

            from temporalscope.datasets.datasets import DatasetLoader

            dataset_loader = DatasetLoader(dataset_name="macrodata")
            timeframes = dataset_loader.init_timeframes_for_backends(df, "realgdp")

        """
        print_divider()
        print("Initializing TimeFrame objects for specified backends.")
        print_divider()

        timeframes: Dict[str, TimeFrame] = {}

        # Loop through the specified backends and create TimeFrame objects
        for backend in backends:
            if backend == BACKEND_PANDAS:
                timeframes[BACKEND_PANDAS] = TimeFrame(
                    df=pd.DataFrame(df), time_col="ds", target_col=target_col, dataframe_backend=BACKEND_PANDAS
                )
            elif backend == BACKEND_MODIN:
                timeframes[BACKEND_MODIN] = TimeFrame(
                    df=mpd.DataFrame(df), time_col="ds", target_col=target_col, dataframe_backend=BACKEND_MODIN
                )
            elif backend == BACKEND_POLARS:
                timeframes[BACKEND_POLARS] = TimeFrame(
                    df=pl.DataFrame(df), time_col="ds", target_col=target_col, dataframe_backend=BACKEND_POLARS
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        return timeframes

    def load_and_init_timeframes(
        self, backends: Tuple[str, ...] = (BACKEND_PANDAS, BACKEND_MODIN, BACKEND_POLARS)
    ) -> Dict[str, TimeFrame]:
        """Load the dataset and initialize TimeFrames for the specified backends.

        :param backends: A tuple of supported backends to initialize. Defaults to Pandas, Modin, and Polars.
        :return: A dictionary containing TimeFrame objects for each backend.
        :rtype: Dict[str, TimeFrame]

        Example:
        ---------
        .. code-block:: python

            dataset_loader = DatasetLoader(dataset_name="macrodata")
            timeframes = dataset_loader.load_and_init_timeframes()

        """
        # Load and preprocess the dataset, including determining the target column
        df, target_col = self._load_dataset_and_target()

        # Initialize TimeFrames for the specified backends
        return self.init_timeframes_for_backends(df, target_col, backends)
