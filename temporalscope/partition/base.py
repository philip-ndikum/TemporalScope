"""
TemporalScope/temporalscope/partition/base.py

This module defines the BaseTemporalPartitioner class, an abstract base class for all
temporal partitioning methods. Each partitioning method must inherit from this class
and implement the required methods.

Core Functionality:
-------------------
1. get_partition_indices: Must return the partition indices (row ranges) for the
   partitions ('train', 'test', 'validation', etc.).
2. get_partition_data: Must use the indices from get_partition_indices to return
   the actual partitioned data.

Each subclass must implement its own logic for partitioning the data and any
necessary validation.

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

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.temporal_data_loader import TimeFrame


class BaseTemporalPartitioner(ABC):
    """
    Abstract base class for temporal partitioning methods.

    The BaseTemporalPartitioner operates on a TimeFrame object and provides core
    functionality for retrieving partition indices and data. Subclasses must implement
    partitioning logic and optionally perform data validation checks.

    Parameters
    ----------
    tf : TimeFrame
        The TimeFrame object containing the pre-sorted time series data to be
        partitioned.
    enable_warnings : bool
        Whether to enable warnings during partition validation.

    Notes
    -----
    The partitions returned by each partitioning method will always include a "full"
    partition with index ranges. The "train", "test", and "validation" partitions are
    supported, and at least "train" and "test" must be defined for logical consistency.
    """

    def __init__(self, tf: TimeFrame, enable_warnings: bool = False):
        """
        Initialize the partitioner with the TimeFrame object and optional warnings.

        Parameters
        ----------
        tf : TimeFrame
            TimeFrame object with sorted time-series data.
        enable_warnings : bool, optional
            Enable warnings for partition validations, by default False.
        """
        self.tf = tf
        self.df = self.tf.get_data()  # Retrieve DataFrame from TimeFrame
        self.enable_warnings = enable_warnings

    @abstractmethod
    def get_partition_indices(self) -> dict[str, dict[str, tuple[int, int]]]:
        """
        Abstract method to generate partition indices.

        Includes 'full', 'train', 'test', 'validation'.

        Returns
        -------
        Dict[str, Dict[str, Tuple[int, int]]]
            Dictionary of partitions with partition indices.

        Notes
        -----
        - Each partition dictionary should contain "full", "train", "test",
        and optionally "validation" keys, where at least "train" and "test"
        must be defined for logical partitioning.
        - "validation" may be None if not required.

        Examples
        --------
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
        """
        Abstract method to perform data validation checks.

        Subclasses must implement their own data validation logic, such as ensuring
        sample size is sufficient, checking for window overlaps, or validating the
        feature count.
        """

    def get_partition_data(
        self,
    ) -> dict[str, dict[str, pd.DataFrame | mpd.DataFrame | pl.DataFrame]]:
        """
        Return the data for each partition based on the partition indices.

        Returns
        -------
        Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]]
            Dictionary of partition names and their respective data slices.

        Notes
        -----
        This method returns the actual data slices for each partition based on the
        indices generated by `get_partition_indices`. The returned structure mirrors the
        same dictionary format but contains actual data instead of index ranges.

        Examples
        --------
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