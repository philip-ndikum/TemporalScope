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

"""Base Temporal Partitioner.

This module defines the BaseTemporalPartitioner class, an abstract base class for all
temporal partitioning methods. Each partitioning method must inherit from this class
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Optional

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.temporal_data_loader import TimeFrame


class BaseTemporalPartitioner(ABC):
    """Abstract base class for temporal partitioning methods.

    The BaseTemporalPartitioner operates on a TimeFrame object and provides core
    functionality for retrieving partition indices and data. Subclasses must implement
    partitioning logic and optionally perform data validation checks.

    :param tf: TimeFrame object with sorted time-series data.
    :type tf: TimeFrame

    :param enable_warnings: Enable warnings for partition validations, defaults to False
    :type enable_warnings: bool, optional

    :ivar tf: The TimeFrame object containing the data to be partitioned.
    :ivar df: The DataFrame extracted from the TimeFrame.
    :ivar enable_warnings: Flag to enable or disable warnings during partition
                           validation.

    .. note::
       The partitions returned by each partitioning method will always include a "full"
       partition with index ranges. The "train", "test", and "validation" partitions are
       supported, and at least "train" and "test" must be defined for logical
       consistency.
    """

    def __init__(self, tf: TimeFrame, enable_warnings: Optional[bool] = False):
        """Initialize the partitioner with the TimeFrame object and optional warnings.

        :param tf: TimeFrame object with sorted time-series data.
        :type tf: TimeFrame
        :param enable_warnings: Enable warnings for partition validations,
                                defaults to False
        :type enable_warnings: bool, optional
        """
        self.tf = tf
        self.df = self.tf.get_data()  # Retrieve DataFrame from TimeFrame
        self.enable_warnings = enable_warnings

    @abstractmethod
    def get_partition_indices(self) -> dict[str, dict[str, tuple[int, int]]]:
        """Abstract method to generate partition indices.

        Includes 'full', 'train', 'test', 'validation'.

        :return: Dictionary of partitions with partition indices.
        :rtype: Dict[str, Dict[str, Tuple[int, int]]]

        .. note::
           - Each partition dictionary should contain "full", "train", "test",
             and optionally "validation" keys, where at least "train" and "test"
             must be defined for logical partitioning.
           - "validation" may be None if not required.

        .. rubric:: Example

        Example of a partition dictionary:

        .. code-block:: python

            {
                "partition_1": {
                    "full": (0, 10),
                    "train": (0, 8),
                    "test": (8, 10),
                    "validation": None,
                },
                "partition_2": {
                    "full": (5, 15),
                    "train": (5, 13),
                    "test": (13, 15),
                    "validation": None,
                },
            }
        """
        pass

    @abstractmethod
    def data_checks(self) -> None:
        """Perform data validation checks.

        This abstract method should be implemented by subclasses to perform
        specific data validation logic.

        Implementations should consider checks such as:

        - Ensuring sufficient sample size
        - Checking for window overlaps
        - Validating feature count
        - Any other relevant checks for the specific partitioning method

        :raises ValueError: If any validation check fails
        :raises NotImplementedError: If the method is not implemented by a subclass

        .. note::
           Subclasses must override this method with their own implementation.

        .. warning::
           Failure to implement proper data checks may lead to invalid partitions
           or unexpected behavior in downstream analysis.
        """
        pass

    def get_partition_data(
        self,
    ) -> dict[str, dict[str, pd.DataFrame | mpd.DataFrame | pl.DataFrame]]:
        """Return the data for each partition based on the partition indices.

        :return: Dictionary of partition names and their respective data slices.
        :rtype: dict[str, dict[str, pd.DataFrame | mpd.DataFrame | pl.DataFrame]]

        .. note::
           This method returns the actual data slices for each partition based on the
           indices generated by `get_partition_indices`. The returned structure mirrors
           the same dictionary format but contains actual data instead of index ranges.

        .. rubric:: Example

        Example of the returned data structure:

        .. code-block:: python

            {
                "partition_1": {
                    "full": DataFrame(...),
                    "train": DataFrame(...),
                    "test": DataFrame(...),
                    "validation": None,
                },
                "partition_2": {
                    "full": DataFrame(...),
                    "train": DataFrame(...),
                    "test": DataFrame(...),
                    "validation": None,
                },
            }
        """
        partitions = self.get_partition_indices()
        partitioned_data = {}
        for key, partition_dict in partitions.items():
            partitioned_data[key] = {
                part_name: self.df[start:end]  # Direct slicing here
                for part_name, (start, end) in partition_dict.items()
                if start is not None and end is not None
            }
        return partitioned_data
