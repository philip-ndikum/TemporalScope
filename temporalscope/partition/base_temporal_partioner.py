"""temporalscope/partitioning/base_temporal_partitioner.py

This module defines the BaseTemporalPartitioner class, an abstract base class for all temporal partitioning methods.
Each partitioning method must inherit from this class and implement the required methods.
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

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Tuple, Any
import pandas as pd
import polars as pl
import modin.pandas as mpd
from temporalscope.config import validate_backend


class BaseTemporalPartitioner(ABC):
    """Abstract base class for temporal partitioning methods. This class enforces a
    consistent API for all partitioning methods.

    :param df: The dataset to be partitioned.
    :type df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
    :param time_col: The time column name, which will be used for sorting.
    :type time_col: str
    :param id_col: Optional. The column used for grouping (e.g., stock ticker, item ID).
    :type id_col: Optional[str]
    :param backend: The backend to use ('pl' for Polars, 'pd' for Pandas, 'mpd' for Modin). Default is 'pl'.
    :type backend: str

    .. note::
        The backend parameter controls whether the partitioning is performed using Polars, Pandas, or Modin. Ensure that the
        df passed to the partitioner is compatible with the specified backend.

    .. rubric:: Examples

    .. code-block:: python

        # Example usage with Pandas DataFrame
        df = pd.DataFrame({'time': pd.date_range(start='2021-01-01', periods=20, freq='D'), 'value': range(20)})
        partitioner = SlidingWindowPartitioner(df=df, time_col='time', backend='pd')

        # Example usage with Polars DataFrame
        df = pl.DataFrame({'time': pl.date_range(start='2021-01-01', periods=20, interval='1d'), 'value': range(20)})
        partitioner = SlidingWindowPartitioner(df=df, time_col='time', backend='pl')
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame],
        time_col: str,
        id_col: Optional[str] = None,
        backend: str = "pl",
    ):
        self.df = df
        self.time_col = time_col
        self.id_col = id_col

        validate_backend(backend)
        self.backend = backend

        # Sort df by time_col and id_col (if provided)
        self.df = self._sort_data()

    def _sort_data(self) -> Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]:
        """Sorts the df by time_col and id_col (if provided), according to the backend.

        :return: The sorted DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame, mpd.DataFrame]
        :raises ValueError: If the backend is unsupported.
        """
        sort_columns = [self.id_col, self.time_col] if self.id_col else [self.time_col]

        if isinstance(self.df, (pd.DataFrame, mpd.DataFrame)):
            # Pandas and Modin use 'sort_values' and 'ascending' instead of 'reverse'
            return self.df.sort_values(by=sort_columns, ascending=True).reset_index(
                drop=True
            )

        elif isinstance(self.df, pl.DataFrame):
            # Polars uses 'sort' and 'reverse' to control ascending/descending order
            if self.id_col:
                return self.df.sort(
                    [self.id_col, self.time_col], reverse=[False, False]
                )
            else:
                return self.df.sort(self.time_col, reverse=False)

        else:
            raise ValueError(f"Unsupported backend configuration: {self.backend}")

    def _check_data_type(self, data: Any, expected_type: str) -> None:
        """Check the type of the data against the expected type (Pandas or Polars).

        :param data: The data to check.
        :type data: Any
        :param expected_type: The expected type ('pandas' or 'polars').
        :type expected_type: str
        :raises TypeError: If the data type does not match the expected type.
        """
        if expected_type == "pandas" and not isinstance(data, pd.DataFrame):
            raise TypeError("Expected data to be a Pandas DataFrame.")
        elif expected_type == "polars" and not isinstance(data, pl.DataFrame):
            raise TypeError("Expected data to be a Polars DataFrame.")

    def get_partitioned_data(self) -> List[Union[pd.DataFrame, pl.DataFrame]]:
        """Helper method that returns the data for each partition as a list of DataFrames.

        :return: List of partitioned DataFrames.
        :rtype: List[Union[pd.DataFrame, pl.DataFrame]]
        """
        partitions = self.get_partitions()
        partitioned_data = [self.apply_partition(partition) for partition in partitions]
        self._check_data_type(
            partitioned_data[0],
            "pandas" if self.backend == "pandas" else "polars",
        )
        return partitioned_data

    def get_partition_indices(self) -> List[Tuple[int, int]]:
        """Helper method that returns the indices for each partition.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        return self.get_partitions()

    @abstractmethod
    def get_partitions(self) -> List[Tuple[int, int]]:
        """Abstract method that must be implemented by subclasses to generate partitions.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        pass

    @abstractmethod
    def apply_partition(
        self, partition: Tuple[int, int]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Abstract method that must be implemented by subclasses to apply a partition to the data.

        :param partition: A tuple representing the start and end indices of the partition.
        :type partition: Tuple[int, int]
        :return: The partitioned DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        pass
