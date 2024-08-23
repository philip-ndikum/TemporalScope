"""
temporalscope/methods/base_temporal_partitioner.py

This module defines the BaseTemporalPartitioner class, an abstract base class for all temporal partitioning methods.
Each partitioning method must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import pandas as pd

class BaseTemporalPartitioner(ABC):
    """
    Abstract base class for temporal partitioning methods. This class enforces a consistent API for all
    partitioning methods.

    :param data: The dataset to be partitioned.
    :type data: pd.DataFrame
    :param target: The target column name.
    :type target: str
    :param id_col: Optional. The column used for grouping (e.g., stock ticker, item ID).
    :type id_col: Optional[str]
    """
    def __init__(self, data: pd.DataFrame, target: str, id_col: str = None):
        self.data = data
        self.target = target
        self.id_col = id_col

    @abstractmethod
    def get_partitions(self) -> List[Tuple[int, int]]:
        """
        Abstract method that must be implemented by subclasses to generate partitions.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        pass

    @abstractmethod
    def apply_partition(self, partition: Tuple[int, int]) -> pd.DataFrame:
        """
        Abstract method that must be implemented by subclasses to apply a partition to the data.

        :param partition: A tuple representing the start and end indices of the partition.
        :type partition: Tuple[int, int]
        :return: The partitioned DataFrame.
        :rtype: pd.DataFrame
        """
        pass

    def get_partitioned_data(self) -> List[pd.DataFrame]:
        """
        Helper method that returns the data for each partition as a list of DataFrames.

        :return: List of partitioned DataFrames.
        :rtype: List[pd.DataFrame]
        """
        partitions = self.get_partitions()
        return [self.apply_partition(partition) for partition in partitions]
