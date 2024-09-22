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

"""TemporalScope/src/temporalscope/partition/base_protocol.py.

This module defines the TemporalPartitionerProtocol, a protocol for all
temporal partitioning methods. Each partitioning method must implement
the required methods to comply with this protocol.

Core Functionality:
-------------------
1. fit: Must generate the partition indices (row ranges) for the
   partitions ('train', 'test', 'validation', etc.) in a memory-efficient manner.
   Implementations should leverage lazy-loading techniques to ensure that
   large datasets are handled efficiently, minimizing memory usage.
2. transform: Must use the indices from fit to return the actual partitioned data.
   This method should apply the calculated indices to retrieve specific data slices,
   maintaining the efficiency gained from lazy-loading in the fit stage.
3. check_data: Optional method to perform data validation checks.

Each implementing class must provide its own logic for partitioning the data and
any necessary validation, while adhering to the design principles of lazy-loading
and memory efficiency.
"""

from typing import Dict, Iterator, Protocol, Tuple, Union

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.temporal_data_loader import TimeFrame


class TemporalPartitionerProtocol(Protocol):
    """Protocol for temporal partitioning methods.

    The `TemporalPartitionerProtocol` operates on a `TimeFrame` object and provides core
    functionality for retrieving partition indices and data. Implementing classes must
    provide partitioning logic and optionally perform data validation checks, with a
    strong emphasis on memory efficiency through lazy-loading techniques.

    :ivar tf: The `TimeFrame` object containing the pre-sorted time series data to be partitioned.
    :vartype tf: TimeFrame
    :ivar df: The data extracted from the `TimeFrame`.
    :vartype df: Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]
    :ivar enable_warnings: Whether to enable warnings during partition validation.
    :vartype enable_warnings: bool

    .. note::
        The partitions returned by each partitioning method will always include a
        "full" partition with index ranges. The "train", "test", and "validation"
        partitions are supported, and at least "train" and "test" must be defined
        for logical consistency. To manage large datasets efficiently, implementations
        should focus on generating indices lazily to reduce memory footprint.
    """

    tf: TimeFrame
    df: Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]
    enable_warnings: bool

    def fit(
        self,
    ) -> Union[
        Dict[str, Dict[str, Tuple[int, int]]],
        Iterator[Dict[str, Dict[str, Tuple[int, int]]]],
    ]:
        """Generate partition indices.

        This method generates partition indices with keys like 'full', 'train',
        'test', and 'validation', utilizing lazy-loading techniques to ensure memory efficiency.

        :return: A dictionary of partitions with their respective indices, or an iterator over them.
        :rtype: Union[Dict[str, Dict[str, Tuple[int, int]]], Iterator[Dict[str, Dict[str, Tuple[int, int]]]]]

        .. note::
            Each partition dictionary should contain "full", "train", "test", and
            optionally "validation" keys, where at least "train" and "test" must
            be defined for logical partitioning.

            "Validation" may be ``None`` if not required.

            Implementations should focus on generating these indices lazily to
            optimize memory usage, particularly with large datasets.

        :example:

            .. code-block:: python

                {
                    "partition_1": {"full": (0, 10), "train": (0, 8), "test": (8, 10), "validation": None},
                    "partition_2": {"full": (5, 15), "train": (5, 13), "test": (13, 15), "validation": None},
                }
        """
        pass

    def transform(
        self,
    ) -> Union[
        Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]],
        Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]]],
    ]:
        """Return the data for each partition.

        This method returns the data slices for each partition based on the
        partition indices generated by the `fit` method.

        :return: A dictionary containing the data slices for each partition,
                 or an iterator over them.
        :rtype: Union[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]],
                      Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]]]]

        .. note::
            This method returns the actual data slices for each partition
            based on the indices generated by `fit`. The returned structure
            mirrors the same dictionary format but contains actual data
            instead of index ranges. The `transform` method should continue
            to optimize for memory efficiency by using the pre-calculated
            lazy indices to access only the necessary data.

        :example:

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
        pass

    def check_data(self) -> None:
        """Perform data validation checks.

        Implementing classes must provide their own data validation logic, such as ensuring sample size is sufficient,
        checking for window overlaps, or validating the feature count.
        """
        pass
