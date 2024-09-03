""" temporalscope/partitioning/sliding_window_partitioner.py

This module defines the SlidingWindowPartitioner class, a specific implementation of the
BaseTemporalPartitioner for creating overlapping windows of data.
"""

from typing import List, Tuple, Union, Optional
import pandas as pd
import polars as pl
from temporalscope.partition.base_temporal_partioner import BaseTemporalPartitioner


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
    :param truncate: Whether to truncate the last partition if it is smaller than the window size.
    :type truncate: bool
    :param expand_last: Whether to expand the last partition to match the window size (if it is smaller).
    :type expand_last: bool
    :param fill_value: The value used to fill the expanded partition if `expand_last` is True.
    :type fill_value: Optional[Union[int, float, str, None]]

    .. note::
        This partitioner divides the data into overlapping windows using the specified stride.
        If the last partition is smaller than the window size, it can be truncated or expanded based on the settings.
        Expanding partitions will fill missing values using the specified `fill_value`.
        The sliding direction (forward or reverse) does not affect the partitioning process.

    .. warning::
        Ensure that the stride and window size are chosen such that they do not result in out-of-bounds access,
        especially when not truncating the last partition.

    .. rubric:: Examples

    .. code-block:: python

        # Example with Pandas DataFrame
        data = pd.DataFrame({'time': pd.date_range(start='2021-01-01', periods=20, freq='D'), 'value': range(20)})
        partitioner = SlidingWindowPartitioner(data=data, time_col='time', window_size=5, stride=3)
        partitions = partitioner.get_partitions()
        print(partitions)
        # Output: [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17)]

        # Example with handling the last partition by expanding
        partitioner = SlidingWindowPartitioner(data=data, time_col='time', window_size=5, stride=3, expand_last=True, fill_value=0)
        partitions = partitioner.get_partitions()
        print(partitions)
        # Output: [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]

        # Example with Polars DataFrame
        data = pl.DataFrame({'time': pl.date_range(start='2021-01-01', periods=20, interval='1d'), 'value': range(20)})
        partitioner = SlidingWindowPartitioner(data=data, time_col='time', window_size=5, stride=3)
        partitions = partitioner.get_partitions()
        print(partitions)
        # Output: [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17)]
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        time_col: str,
        id_col: Optional[str] = None,
        window_size: int = 10,
        stride: int = 1,
        truncate: bool = True,
        expand_last: bool = False,
        fill_value: Optional[Union[int, float, str, None]] = None,
    ):
        super().__init__(data, time_col, id_col)
        self.window_size = window_size
        self.stride = stride
        self.truncate = truncate
        self.expand_last = expand_last
        self.fill_value = fill_value

        # Sort data by time_col and id_col (if provided)
        self.data = self._sort_data()

    def get_partitions(self) -> List[Tuple[int, int]]:
        """Generates partitions using a sliding window approach.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        num_rows = self.data.shape[0]
        partitions = []

        for start in range(0, num_rows, self.stride):
            end = start + self.window_size

            if end > num_rows:
                if self.truncate:
                    break
                elif self.expand_last:
                    end = num_rows

            partitions.append((start, end))

        return partitions

    def apply_partition(
        self, partition: Tuple[int, int]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Applies a partition to the data and returns the corresponding subset.

        If the partition size is smaller than the window size and expand_last is True,
        the partition will be padded with the fill_value.

        :param partition: A tuple representing the start and end indices of the partition.
        :type partition: Tuple[int, int]
        :return: The partitioned DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        start, end = partition
        partition_data = self.data[start:end].clone()

        if self.expand_last and len(partition_data) < self.window_size:
            if isinstance(partition_data, pd.DataFrame):
                for _ in range(self.window_size - len(partition_data)):
                    empty_row = {col: self.fill_value for col in partition_data.columns}
                    partition_data = partition_data.append(empty_row, ignore_index=True)
            elif isinstance(partition_data, pl.DataFrame):
                fill_series = [
                    pl.Series(
                        name,
                        [self.fill_value] * (self.window_size - len(partition_data)),
                    )
                    for name in partition_data.columns
                ]
                partition_data = pl.concat([partition_data] + fill_series, rechunk=True)

        return partition_data

    def _sort_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        """Sorts the data by time_col and id_col (if provided).

        :return: The sorted DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        if isinstance(self.data, pd.DataFrame):
            if self.id_col:
                return self.data.sort_values(
                    by=[self.id_col, self.time_col]
                ).reset_index(drop=True)
            return self.data.sort_values(by=self.time_col).reset_index(drop=True)
        elif isinstance(self.data, pl.DataFrame):
            if self.id_col:
                return self.data.sort([self.id_col, self.time_col])
            return self.data.sort(self.time_col)

        raise TypeError(
            "Unsupported data type. Data must be a Pandas or Polars DataFrame."
        )
