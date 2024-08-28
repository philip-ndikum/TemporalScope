"""temporalscope/partioning/naive_partitioner.py

This module implements the NaivePartitioner class, which is designed to partition the dataset into equal-sized
partitions. The partitions are driven by index values and can be customized by specifying the number of partitions.
"""

from temporalscope.partioning.base_temporal_partitioner import BaseTemporalPartitioner
import pandas as pd
from typing import List, Tuple

class NaivePartitioner(BaseTemporalPartitioner):
    """
    Naive partitioning method that divides the dataset into equal-sized partitions.

    :param data: The dataset to be partitioned.
    :type data: pd.DataFrame
    :param target: The target column name.
    :type target: str
    :param id_col: Optional. The column used for grouping (e.g., stock ticker, item ID).
    :type id_col: Optional[str]
    :param n_partitions: Number of partitions to create.
    :type n_partitions: int
    """
    
    def __init__(self, data: pd.DataFrame, target: str, id_col: str = None, n_partitions: int = 3):
        super().__init__(data, target, id_col)
        self.n_partitions = n_partitions

    def get_partitions(self) -> List[Tuple[int, int]]:
        """Generate naive partitions of equal size."""
        total_rows = len(self.data)
        partition_size = total_rows // self.n_partitions
        partitions = [(i * partition_size, (i + 1) * partition_size) for i in range(self.n_partitions)]
        partitions[-1] = (partitions[-1][0], total_rows)  # Ensure the last partition includes all remaining data
        return partitions

    def apply_partition(self, partition: Tuple[int, int]) -> pd.DataFrame:
        """Apply a naive partition to the data."""
        start, end = partition
        return self.data.iloc[start:end]
