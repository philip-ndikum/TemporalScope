""" TemporalScope/temporalscope/partitioning/base_temporal_partitioner.py

This module defines the BaseTemporalPartitioner class, an abstract base class for all temporal partitioning methods.
Each partitioning method must inherit from this class and implement the required methods.

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

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
import polars as pl
import modin.pandas as mpd
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.partitioning.partitioning_guidelines import check_sample_size


class BaseTemporalPartitioner(ABC):
    """Abstract base class for temporal partitioning methods.

    The BaseTemporalPartitioner operates on a TimeFrame object and returns partition indices for the dataset.

    :param tf: The TimeFrame object containing the pre-sorted time series data to be partitioned.
    :type tf: TimeFrame

    :example:

        .. code-block:: python

            from temporalscope.core.temporal_data_loader import TimeFrame
            from temporalscope.partitioning.base_temporal_partitioner import BaseTemporalPartitioner

            df = pd.DataFrame({
                'time': pd.date_range(start='2021-01-01', periods=20, freq='D'),
                'value': range(20)
            })

            # TimeFrame object created first
            tf = TimeFrame(df, time_col='time', target_col='value', backend='pd')

            # Subclass of BaseTemporalPartitioner must implement 'get_partitions'
            class CustomPartitioner(BaseTemporalPartitioner):
                def get_partitions(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
                    return {
                        "P1": {
                            'train': (0, 10),
                            'test': (10, 15),
                            'validation': (15, 20)
                        }
                    }

            partitioner = CustomPartitioner(tf=tf)
            print(partitioner.get_partitions())
    """

    def __init__(self, tf: TimeFrame):
        """Initialize the partitioner with the TimeFrame."""
        self.tf = tf
        self.df = self.tf.get_data()
        self.backend = self.tf.backend

    @abstractmethod
    def get_partitions(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Abstract method to generate partitions with keys like 'train', 'test', 'validation'.

        :return: Dictionary of partitions with partition indices.
        :rtype: Dict[str, Dict[str, Tuple[int, int]]]
        """
        pass

    def validate_partitions(self) -> None:
        """Optionally check if the partitions meet basic heuristics."""
        check_sample_size(self.df, backend=self.backend)

    def get_partition_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Optionally return the data for each partition.

        :return: Dictionary of partition names and their respective data slices.
        :rtype: Dict[str, Dict[str, pd.DataFrame]]
        """
        partitions = self.get_partitions()
        partitioned_data = {}
        for key, partition_dict in partitions.items():
            partitioned_data[key] = {
                part_name: self.df.iloc[start:end] if self.backend == 'pd' else self.df[start:end]
                for part_name, (start, end) in partition_dict.items()
            }
        return partitioned_data
