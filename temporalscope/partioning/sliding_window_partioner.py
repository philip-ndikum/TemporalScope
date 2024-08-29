""" temporalscope/partitioning/sliding_window_partitioner.py

This module defines the SlidingWindowPartitioner class, a specific implementation of the
BaseTemporalPartitioner for creating overlapping windows of data.
"""

from typing import List, Tuple, Union
import pandas as pd
import polars as pl
from temporalscope.partitioning.base_temporal_partitioner import BaseTemporalPartitioner

class SlidingWindowPartitioner(BaseTemporalPartitioner):
    """Sliding Window Partitioner class that divides the data into overlapping windows.

    :param data: The dataset to be partitioned.
    :type data: Union[pd.DataFrame, pl.DataFrame]
    :param time_col: The time column name, which will be used for sorting.
    :type time_col: str
    :param id_col: Optional. The column used for grouping (e.g., stock ticker, item ID).
    :type id_col: Optional[str]
    :param window_size: The size of each window.
    :type window_size: int
    :param stride: The stride (step size) between consecutive windows.
    :type stride: int
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        time_col: str,
        id_col: str = None,
        window_size: int = 10,
        stride: int = 1,
    ):
        super().__init__(data, time_col, id_col)
        self.window_size = window_size
        self.stride = stride

        # Sort data by time_col and id_col (if provided)
        self.data = self._sort_data()

    def get_partitions(self) -> List[Tuple[int, int]]:
        """Generates partitions using a sliding window approach.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        num_rows = self.data.shape[0]
        partitions = []

        for start in range(0, num_rows - self.window_size + 1, self.stride):
            end = start + self.window_size
            partitions.append((start, end))

        return partitions

    def apply_partition(
        self, partition: Tuple[int, int]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Applies a partition to the data and returns the corresponding subset.

        :param partition: A tuple representing the start and end indices of the partition.
        :type partition: Tuple[int, int]
        :return: The partitioned DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        start, end = partition

        if isinstance(self.data, pd.DataFrame):
            return self.data.iloc[start:end].copy()
        elif isinstance(self.data, pl.DataFrame):
            return self.data[start:end].clone()

        raise TypeError("Unsupported data type. Data must be a Pandas or Polars DataFrame.")

    def _sort_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        """Sorts the data by time_col and id_col (if provided).

        :return: The sorted DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        if isinstance(self.data, pd.DataFrame):
            if self.id_col:
                return self.data.sort_values(by=[self.id_col, self.time_col]).reset_index(drop=True)
            return self.data.sort_values(by=self.time_col).reset_index(drop=True)
        elif isinstance(self.data, pl.DataFrame):
            if self.id_col:
                return self.data.sort([self.id_col, self.time_col])
            return self.data.sort(self.time_col)

        raise TypeError("Unsupported data type. Data must be a Pandas or Polars DataFrame.")
