"""temporalscope/partitioning/base_temporal_partitioner.py

This module defines the BaseTemporalPartitioner class, an abstract base class for all temporal partitioning methods.
Each partitioning method must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union
import pandas as pd
import polars as pl


class BaseTemporalPartitioner(ABC):
    """Abstract base class for temporal partitioning methods. This class enforces a
    consistent API for all partitioning methods.

    :param data: The dataset to be partitioned.
    :type data: Union[pd.DataFrame, pl.DataFrame]
    :param target: The target column name.
    :type target: str
    :param id_col: Optional. The column used for grouping (e.g., stock ticker, item ID).
    :type id_col: Optional[str]
    """

    def __init__(
        self, data: Union[pd.DataFrame, pl.DataFrame], target: str, id_col: str = None
    ):
        self.data = data
        self.target = target
        self.id_col = id_col

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

    def get_partitioned_data(self) -> List[Union[pd.DataFrame, pl.DataFrame]]:
        """Helper method that returns the data for each partition as a list of DataFrames.

        :return: List of partitioned DataFrames.
        :rtype: List[Union[pd.DataFrame, pl.DataFrame]]
        """
        partitions = self.get_partitions()
        partitioned_data = [self.apply_partition(partition) for partition in partitions]
        self._check_data_type(
            partitioned_data[0],
            "pandas" if isinstance(self.data, pd.DataFrame) else "polars",
        )
        return partitioned_data

    def get_partition_indices(self) -> List[Tuple[int, int]]:
        """Helper method that returns the indices for each partition.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        return self.get_partitions()
