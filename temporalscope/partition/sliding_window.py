"""
TemporalScope/temporalscope/partitioning/sliding_window.py

This module defines the SlidingWindowPartitioner class, a specific implementation of the
BaseTemporalPartitioner for creating contiguous, non-overlapping partitions using a sliding window mechanism.

Core Functionality:
-------------------
The SlidingWindowPartitioner divides a dataset into non-overlapping partitions using a fixed window size and
optional stride. The stride determines how far to move between the starting points of consecutive partitions, 
which can introduce gaps between them. Each partition can be further split into train, test, and validation sets.

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, Tuple, Optional
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.partition.base import BaseTemporalPartitioner


class SlidingWindowPartitioner(BaseTemporalPartitioner):
    """Sliding Window Partitioner for dividing time series data into contiguous, non-overlapping partitions.

    This class splits a dataset into partitions of a fixed window size. Users can define a stride to introduce gaps
    between consecutive partitions. Each partition can be further divided into train, test, and validation sets
    based on provided percentages.

    :param tf: The TimeFrame object containing the data to be partitioned.
    :type tf: TimeFrame
    :param window_size: The size of each partition (number of rows). If not provided, `num_partitions` is required.
    :type window_size: Optional[int]
    :param num_partitions: The number of partitions to divide the data into. If `window_size` is not provided, this parameter
                           is used to split the data evenly.
    :type num_partitions: Optional[int]
    :param stride: The number of rows to skip between the start points of consecutive partitions.
                   A stride larger than the window size creates gaps, while a stride equal to the window size results in no gaps.
    :type stride: int
    :param truncate: Whether to truncate the last partition if its size is smaller than the window size.
    :type truncate: bool
    :param train_pct: Percentage of data allocated for training within each partition.
    :type train_pct: float
    :param test_pct: Percentage of data allocated for testing within each partition.
    :type test_pct: Optional[float]
    :param val_pct: Percentage of data allocated for validation within each partition.
    :type val_pct: Optional[float]
    :param enable_warnings: Enable warnings for uneven partition sizes.
    :type enable_warnings: bool

    :raises ValueError: If neither `window_size` nor `num_partitions` is provided, or if train, test, and validation percentages do not sum to 1.0.

    Example Usage:
    --------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.partition.sliding_window import SlidingWindowPartitioner

        # Create a sample dataset using Pandas
        data = pd.DataFrame({
            'time': pd.date_range(start='2021-01-01', periods=20, freq='D'),
            'value': range(20)
        })

        # Create a TimeFrame object
        tf = TimeFrame(data, time_col='time', target_col='value', backend='pd')

        # Create a SlidingWindowPartitioner with window_size=5 and stride=5
        partitioner = SlidingWindowPartitioner(tf=tf, window_size=5, stride=5, truncate=True, train_pct=0.6, test_pct=0.3, val_pct=0.1)

        # Retrieve the partition indices
        partitions = partitioner.get_partition_indices()
        print(partitions)

        # Retrieve the actual data slices for each partition
        partitioned_data = partitioner.get_partition_data()
        print(partitioned_data)
    """

    def __init__(
        self,
        tf: TimeFrame,
        window_size: Optional[int] = None,
        num_partitions: Optional[int] = None,
        stride: int = 1,
        truncate: bool = True,
        train_pct: float = 0.7,
        test_pct: Optional[float] = 0.2,
        val_pct: Optional[float] = 0.1,
        enable_warnings: bool = False,
    ):
        """Initialize the SlidingWindowPartitioner with a TimeFrame."""
        super().__init__(tf, enable_warnings=enable_warnings)
        self.window_size = window_size
        self.num_partitions = num_partitions
        self.stride = stride
        self.truncate = truncate
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.val_pct = val_pct

        # Check that train, test, and validation percentages sum to 1
        if not (
            0 <= train_pct <= 1 and (test_pct or 0) + (val_pct or 0) + train_pct == 1.0
        ):
            raise ValueError("Train, test, and validation percentages must sum to 1.")

        if not window_size and not num_partitions:
            raise ValueError(
                "Either `window_size` or `num_partitions` must be specified."
            )

        # Sort the data using the TimeFrame class
        self.tf.sort_data(ascending=True)

    def _calculate_window_size(self, num_rows: int) -> int:
        """
        Calculate the window size based on either the number of partitions or a provided window size.

        If `self.num_partitions` is set, we calculate the window size by dividing the total number of rows by
        the number of partitions, ensuring the window size is at least 1. If `self.window_size` is provided,
        it will be used directly. Otherwise, we provide a fallback value.

        :param num_rows: The total number of rows in the dataset.
        :type num_rows: int
        :return: The calculated window size.
        :rtype: int
        """

        # If the number of partitions is specified, calculate the window size based on it
        if self.num_partitions:
            # Ensure the window size is at least 1 (to avoid windows of size 0)
            return max(1, num_rows // self.num_partitions)

        # If no partitions are specified, return the predefined window size.
        # If `self.window_size` is None, use a fallback (e.g., num_rows // 10 as default)
        return (
            self.window_size if self.window_size is not None else max(1, num_rows // 10)
        )

    def get_partition_indices(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Generate partition indices based on the window size or number of partitions.

        :return: Dictionary of partitions with indices for 'full', 'train', 'test', and optionally 'validation'.
        :rtype: Dict[str, Dict[str, Tuple[int, int]]]
        """
        num_rows = self.df.shape[0]
        window_size = self._calculate_window_size(num_rows)
        partitions = {}
        partition_count = 1

        for start in range(0, num_rows, self.stride):
            end = start + window_size

            if end > num_rows:
                if self.truncate:
                    break
                end = num_rows  # Include remaining data if truncate is False

            train_end = start + int(self.train_pct * (end - start))
            test_end = (
                train_end + int(self.test_pct * (end - start))
                if self.test_pct
                else train_end
            )
            validation_end = end if self.test_pct else train_end

            partitions[f"partition_{partition_count}"] = {
                "full": (start, end),
                "train": (start, train_end),
                "test": (train_end, test_end),
                "validation": (test_end, validation_end),
            }
            partition_count += 1

        return partitions

    def data_checks(self) -> None:
        """Perform any necessary validation checks for the sliding window approach."""
        # Example: Ensure no overlap between partitions, etc.
        pass
