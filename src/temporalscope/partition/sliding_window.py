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

"""TemporalScope/temporalscope/partitioning/sliding_window.py.

This module defines the SlidingWindowPartitioner class, a specific implementation of the
TemporalPartitionerProtocol for creating contiguous, non-overlapping partitions using a sliding window mechanism.

Core Functionality:
-------------------
The SlidingWindowPartitioner divides a dataset into non-overlapping partitions using a fixed window size and
optional stride. The stride determines how far to move between the starting points of consecutive partitions,
which can introduce gaps between them. Each partition can be further split into train, test, and validation sets.

This class utilizes the generator pattern for memory efficiency, yielding partition indices and data slices one at a time.

The `SlidingWindowPartitioner` is intended for universal models, which assume flat partitioning across all entities.
Users are responsible for preprocessing steps such as deduplication or transforming `time_col` to numerical features.

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

import itertools
from typing import Dict, Iterator, Optional, Tuple, Union

import modin.pandas as mpd
import pandas as pd
import polars as pl

from temporalscope.core.core_utils import (
    BACKEND_MODIN,
    BACKEND_PANDAS,
    BACKEND_POLARS,
    SupportedBackendDataFrame,
    validate_backend,
)
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.partition.base_protocol import TemporalPartitionerProtocol
from temporalscope.partition.partition_validators import (
    check_class_balance,
    check_feature_to_sample_ratio,
    check_sample_size,
)


class SlidingWindowPartitioner(TemporalPartitionerProtocol):
    """Sliding Window Partitioner for dividing time series data into contiguous, non-overlapping partitions.

    This class splits a dataset into partitions using either a specified `window_size` or a calculated `window_size`
    based on the desired `num_partitions`. Users can define a stride to introduce gaps between consecutive partitions.
    Each partition can be further divided into train, test, and validation sets based on provided percentages.

    This class supports workflows for both machine learning (ML) and deep learning (DL) models. For ML, truncation or
    varying window sizes may be acceptable. However, in DL pipelines (e.g., TensorFlow, PyTorch, JAX), padding is often
    required to ensure uniform input shapes across batches, making the `truncate` parameter and padding behavior critical.

    The partitioning occurs globally across the entire dataset, maintaining the temporal order without grouping by entity.
    This design ensures compatibility with universal models, where the entire dataset is treated as a single unit for
    partitioning, aligning with the flexibility of the `TimeFrame` class. Users are responsible for any necessary preprocessing
    (e.g., deduplication or transformation of `time_col`).

    Assumptions:
    ------------
    - `train_pct` must be specified.
    - `test_pct` is optional, and if not provided, the remaining percentage after `train_pct` will implicitly be assigned to `test_pct`.
    - `val_pct` is also optional, and if provided, the sum of `train_pct`, `test_pct`, and `val_pct` must equal 1.0.
    - The total of `train_pct`, `test_pct`, and `val_pct` must sum to 1.0 exactly.
    - Partitioning occurs globally across the dataset, and users are responsible for preprocessing, such as deduplication
      or transformation of `time_col`.

    The class uses a generator pattern for `fit` and `transform` methods to yield partition indices and data slices
    one at a time, promoting memory efficiency and lazy loading.

    Example Usage:
    --------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.partition.sliding_window import SlidingWindowPartitioner

        # Create a sample dataset using Pandas
        data_df = pd.DataFrame({
            'time': pd.date_range(start='2021-01-01', periods=6, freq='D'),
            'value': range(6)
        })

        # Create a TimeFrame object
        data_tf = TimeFrame(data_df, time_col='time', target_col='value', backend='pd')

        # Create a SlidingWindowPartitioner with window_size=2 and stride=1
        partitioner = SlidingWindowPartitioner(
            tf=data_tf, window_size=2, stride=1, truncate=True, train_pct=0.7, test_pct=0.3
        )

        # Iterate over partition indices
        for partition in partitioner.fit():
            print(partition)

        # Iterate over data slices for each partition
        for partition_data in partitioner.transform():
            print(partition_data)

    Visualization:
    --------------
    .. note::

       Here's a conceptual 2D visualization of how the sliding window and stride work with a `time_col`:

       .. code-block:: text

           time        value
           -------     ------
           2021-01-01   0
           2021-01-02   1
           2021-01-03   2
           2021-01-04   3
           2021-01-05   4
           2021-01-06   5

       Partitioning with `window_size=2` and `stride=1`:

       - First partition:
           time        value
           -------     ------
           2021-01-01   0
           2021-01-02   1

       - Second partition:
           time        value
           -------     ------
           2021-01-02   1
           2021-01-03   2

       - Third partition:
           time        value
           -------     ------
           2021-01-03   2
           2021-01-04   3

       The sliding window moves across the entire dataset, maintaining the temporal order within each partition.

    :param tf: The TimeFrame object containing the data to be partitioned.
    :type tf: TimeFrame
    :param num_partitions: The desired number of partitions to create. If `window_size` is specified, this is ignored.
    :type num_partitions: Optional[int]
    :param window_size: The size of each partition (number of rows). If specified, it takes precedence over `num_partitions`.
    :type window_size: Optional[int]
    :param stride: The number of rows to skip between the start points of consecutive partitions.
                   A stride larger than the window size creates gaps, while a stride equal to the window size results in no gaps.
    :type stride: int
    :param reverse: Whether the sliding window should move in reverse (from the end to the start of the dataset).
                    If set to True, the window slides in reverse; if False (default), it slides forward.
    :type reverse: bool
    :param truncate: Whether to truncate the last partition if its size is smaller than the window size.
                     Note: For deep learning models, truncation can lead to varying input sizes and should be avoided.
    :type truncate: bool
    :param train_pct: Percentage of data allocated for training within each partition. Must be provided.
    :type train_pct: float
    :param test_pct: Percentage of data allocated for testing within each partition. Optional.
    :type test_pct: Optional[float]
    :param val_pct: Optional percentage of data allocated for validation within each partition. If provided, the sum of `train_pct`,
                    `test_pct`, and `val_pct` must equal 1.0.
    :type val_pct: Optional[float]
    :param enable_warnings: Enable warnings for uneven partition sizes.
    :type enable_warnings: bool
    :param verbose: If set to True, print partitioning details.
    :type verbose: bool

    :raises ValueError:
        - If neither `window_size` nor `num_partitions` is provided or valid.
        - If `stride` is not a positive integer.
        - If `train_pct`, `test_pct`, or `val_pct` are not within the range [0, 1].
        - If `train_pct`, `test_pct`, and `val_pct` do not sum to 1.0.
        - If the dataset cannot be sorted or retrieved properly from the TimeFrame.
        - If any required data is missing or invalid during the partitioning process.
    """

    def __init__(
        self,
        tf: TimeFrame,
        num_partitions: Optional[int] = 2,
        window_size: Optional[int] = None,
        stride: int = 1,
        reverse: bool = False,
        truncate: bool = True,
        train_pct: float = 0.7,
        test_pct: Optional[float] = 0.2,
        val_pct: Optional[float] = None,
        enable_warnings: bool = False,
        verbose: bool = False,
    ):
        """Initialize the SlidingWindowPartitioner with the given parameters.

        :param tf: TimeFrame object to partition.
        :param num_partitions: Number of partitions to create (ignored if `window_size` is provided).
        :param window_size: Size of each partition.
        :param stride: Number of rows to skip between partitions.
        :param reverse: Whether the sliding window should move in reverse.
        :param truncate: Whether to truncate the last partition if smaller than `window_size`.
        :param train_pct: Percentage of data allocated for training.
        :param test_pct: Percentage of data allocated for testing.
        :param val_pct: Percentage of data allocated for validation.
        :param enable_warnings: Enable warnings for uneven partition sizes.
        :param verbose: Enable verbose output.
        :raises ValueError: If input parameters are invalid.
        """
        validate_backend(tf.backend)
        num_rows = tf.get_data().shape[0]
        if window_size is None:
            if num_partitions is None or num_partitions <= 0:
                raise ValueError("`num_partitions` must be a positive integer.")
            window_size = num_rows // num_partitions

        if window_size <= 0:
            raise ValueError("`window_size` must be a positive integer.")
        if stride <= 0:
            raise ValueError("`stride` must be a positive integer.")

        # Validate percentage values
        if not (0 <= train_pct <= 1):
            raise ValueError("`train_pct` must be between 0 and 1.")
        if test_pct is not None and not (0 <= test_pct <= 1):
            raise ValueError("`test_pct` must be between 0 and 1.")
        if val_pct is not None and not (0 <= val_pct <= 1):
            raise ValueError("`val_pct` must be between 0 and 1.")
        if train_pct + (test_pct or 0) + (val_pct or 0) != 1.0:
            raise ValueError("Train, test, and validation percentages must sum to 1.0.")

        self.tf = tf
        self.window_size = window_size
        self.stride = stride
        self.reverse = reverse
        self.truncate = truncate
        self.verbose = verbose
        self.train_pct, self.test_pct, self.val_pct = self._precompute_percentages(train_pct, test_pct, val_pct)

        # Sort data by time column using TimeFrame method
        self.tf.sort_data(ascending=True)

        self._fit_executed = False
        self._transform_executed = False

    def _precompute_percentages(
        self,
        train_pct: float,
        test_pct: Optional[float],
        val_pct: Optional[float],
        precision: float = 1e-6,  # Default precision for floating-point comparisons
    ) -> Tuple[float, float, float]:
        """Precompute and validate train, test, and validation percentages.

        This function ensures that the sum of train, test, and validation percentages equals 1.0.
        If `test_pct` is not provided, it will be set to the remaining percentage after the train percentage.

        :param train_pct: Percentage of data allocated for training.
        :type train_pct: float
        :param test_pct: Optional. Percentage of data allocated for testing.
        :type test_pct: Optional[float]
        :param val_pct: Optional. Percentage of data allocated for validation.
        :type val_pct: Optional[float]
        :param precision: The tolerance level for floating-point imprecision, defaults to 1e-6.
        :type precision: float
        :return: A tuple containing the validated percentages for training, testing, and validation.
        :rtype: Tuple[float, float, float]
        :raises ValueError: If the percentages do not sum to 1.0 or are not within the valid range (0 to 1).
        """
        # Validate the train percentage
        if not (0 <= train_pct <= 1):
            raise ValueError("`train_pct` must be between 0 and 1.")

        # Ensure test_pct and val_pct are set correctly
        if test_pct is None and val_pct is None:
            test_pct = 1.0 - train_pct
            val_pct = 0.0
        elif test_pct is not None and val_pct is None:
            if not (0 <= test_pct <= 1):
                raise ValueError("`test_pct` must be between 0 and 1.")
            val_pct = 1.0 - train_pct - test_pct
        elif test_pct is None and val_pct is not None:
            if not (0 <= val_pct <= 1):
                raise ValueError("`val_pct` must be between 0 and 1.")
            test_pct = 1.0 - train_pct - val_pct
        else:
            # Both test_pct and val_pct are provided, ensure they are valid before comparison
            if test_pct is None or val_pct is None:
                raise ValueError("`test_pct` and `val_pct` cannot be None.")
            if not (0 <= test_pct <= 1):
                raise ValueError("`test_pct` must be between 0 and 1.")
            if not (0 <= val_pct <= 1):
                raise ValueError("`val_pct` must be between 0 and 1.")

        # Ensure they sum to 1.0
        total_pct = train_pct + (test_pct or 0) + (val_pct or 0)
        if not (abs(total_pct - 1.0) < precision):  # Use the precision parameter here
            raise ValueError("Train, test, and validation percentages must sum to 1.0.")

        # Ensure test_pct and val_pct are float types, not None
        return train_pct, float(test_pct), float(val_pct)

    def _pad_partition(
        self,
        df: Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame],
        window_size: int,
        end: int,
        reverse: bool,
    ) -> SupportedBackendDataFrame:
        """Pad the partition to the required window size by repeating the last row.

        This function ensures that the partition is padded to the full window size by repeating the last row of the
        partition until the desired window size is achieved.

        :param df: The DataFrame (Pandas, Modin, or Polars) to pad.
        :type df: Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]
        :param window_size: The target window size to pad the partition to.
        :type window_size: int
        :param end: The index indicating the end of the current partition.
        :type end: int
        :param reverse: If True, the padding is added to the start; otherwise, it's added at the end.
        :type reverse: bool
        :return: A DataFrame padded to the specified window size.
        :rtype: Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]
        """
        # Calculate how many rows to pad
        num_to_pad = window_size - df.shape[0]

        if num_to_pad <= 0:
            return df  # No need to pad

        # Handle Pandas or Modin DataFrames
        if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
            # Select the row to use for padding
            pad_row = df.iloc[[end - 1]] if not reverse else df.iloc[[0]]

            # Repeat the selected row for the required number of times
            pad_rows = pd.concat([pad_row] * num_to_pad, ignore_index=True)

            # Concatenate the original DataFrame with the padding
            if reverse:
                return pd.concat([pad_rows, df], ignore_index=True)
            else:
                return pd.concat([df, pad_rows], ignore_index=True)

        # Handle Polars DataFrames
        elif isinstance(df, pl.DataFrame):
            # Select the row to use for padding
            pad_row = df.slice(end - 1, 1) if not reverse else df.slice(0, 1)

            # Repeat the selected row for the required number of times
            pad_rows = pl.DataFrame([pad_row.to_dict(as_series=False)[0] for _ in range(num_to_pad)])

            # Concatenate the original DataFrame with the padding
            if reverse:
                return pad_rows.vstack(df)
            else:
                return df.vstack(pad_rows)

        raise TypeError("Unsupported DataFrame type.")

    def _fit_pandas_modin(
        self, df: Union[pd.DataFrame, mpd.DataFrame]
    ) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
        """Fit method specific to Pandas or Modin backends.

        :param df: Input DataFrame.
        :return: Iterator yielding partition indices for Pandas/Modin.
        """
        partition_count = 1

        num_rows = df.shape[0]
        start_range = list(range(0, num_rows, self.stride))

        if self.reverse:
            start_range.reverse()

        for start in start_range:
            end = start + self.window_size

            if end > num_rows:
                if self.truncate:
                    break
                end = num_rows

            train_end = start + int(self.train_pct * (end - start))
            test_end = train_end + int(self.test_pct * (end - start)) if self.test_pct else train_end
            validation_end = end if self.val_pct else test_end

            # Yield the partition indices
            yield {
                f"partition_{partition_count}": {
                    "full": (start, end),
                    "train": (start, train_end),
                    "test": (train_end, test_end),
                    "validation": ((test_end, validation_end) if self.val_pct else (0, 0)),
                }
            }
            partition_count += 1

    def _fit_polars(self, df: pl.DataFrame) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
        """Fit method specific to Polars backend.

        :param df: Input DataFrame.
        :return: Iterator yielding partition indices for Polars.
        """
        partition_count = 1

        num_rows = df.height
        start_range = list(range(0, num_rows, self.stride))

        if self.reverse:
            start_range.reverse()

        for start in start_range:
            end = start + self.window_size

            if end > num_rows:
                if self.truncate:
                    break
                end = num_rows

            train_end = start + int(self.train_pct * (end - start))
            test_end = train_end + int(self.test_pct * (end - start)) if self.test_pct else train_end
            validation_end = end if self.val_pct else test_end

            # Yield the partition indices
            yield {
                f"partition_{partition_count}": {
                    "full": (start, end),
                    "train": (start, train_end),
                    "test": (train_end, test_end),
                    "validation": ((test_end, validation_end) if self.val_pct else (0, 0)),
                }
            }
            partition_count += 1

    def _transform_pandas_modin(
        self, df: Union[pd.DataFrame, mpd.DataFrame]
    ) -> Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame]]]]:
        """Transform method for Pandas/Modin backend.

        This method transforms the partitioned dataset into slices, yielding the data slices corresponding to
        the partition indices generated by the `fit` method.

        It processes each partition and splits it into train, test, and optionally validation sets.
        If a partition's size is smaller than the specified `window_size`, padding is applied to ensure
        uniform size across partitions, unless `truncate` is set to True.

        :param df: Input DataFrame. This can be either Pandas or Modin DataFrame, depending on the backend.
        :type df: Union[pd.DataFrame, mpd.DataFrame]
        :return: Iterator yielding partitioned DataFrame slices for Pandas/Modin backends.
        :rtype: Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame]]]]

        Example Usage:
        --------------
        .. code-block:: python

            partitioner = SlidingWindowPartitioner(
                tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3
            )

            for partition_data in partitioner._transform_pandas_modin(df):
                print(partition_data)

        Output Format:
        --------------
        Each yielded partition has the following structure:

        .. code-block:: python

            {
                'partition_1': {
                    'full': <pd.DataFrame>,
                    'train': <pd.DataFrame>,
                    'test': <pd.DataFrame>,
                    'validation': <pd.DataFrame>  # (Optional, if val_pct is provided)
                }
            }

        Notes
        -----
        - Padding is applied when the size of a partition is smaller than the `window_size`, unless truncation is enabled.
        - Ensure that the input DataFrame is not empty to avoid runtime errors.

        Performance Considerations:
        ---------------------------
        - For very large datasets, the padding process may increase memory usage. Consider using Modin when handling
          large datasets to take advantage of distributed processing.

        """
        partition_count = 1

        for partition in self.fit():
            partitioned_data = {}

            # Ensure partition is a dictionary
            if isinstance(partition, dict):
                for key, partition_dict in partition.items():
                    partitioned_data[key] = {
                        part_name: df.iloc[start:end]
                        for part_name, (start, end) in partition_dict.items()
                        if start is not None and end is not None
                    }

                    # If the partition size is smaller than the window size, pad it
                    if partition_dict["full"][1] - partition_dict["full"][0] < self.window_size and not self.truncate:
                        partitioned_data[key]["full"] = self._pad_partition(
                            partitioned_data[key]["full"],
                            self.window_size,
                            partition_dict["full"][1],
                            self.reverse,
                        )
                yield partitioned_data

            partition_count += 1

    def _transform_polars(self, df: pl.DataFrame) -> Iterator[Dict[str, Dict[str, pl.DataFrame]]]:
        """Transform method for Polars backend.

        This method generates partitioned data slices for the Polars backend, yielding the data slices corresponding
        to the partition indices generated by the `fit` method. If the size of a partition is smaller than the
        specified `window_size`, padding is applied unless `truncate` is set to True.

        :param df: Input Polars DataFrame.
        :type df: pl.DataFrame
        :return: Iterator yielding partitioned DataFrame slices for Polars backend.
        :rtype: Iterator[Dict[str, Dict[str, pl.DataFrame]]]

        Example Usage:
        --------------
        .. code-block:: python

            partitioner = SlidingWindowPartitioner(
                tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3
            )

            for partition_data in partitioner._transform_polars(df):
                print(partition_data)

        Output Format:
        --------------
        Each yielded partition has the following structure:

        .. code-block:: python

            {
                'partition_1': {
                    'full': <pl.DataFrame>,
                    'train': <pl.DataFrame>,
                    'test': <pl.DataFrame>,
                    'validation': <pl.DataFrame>  # (Optional, if val_pct is provided)
                }
            }

        Notes
        -----
        - Padding is applied when the size of a partition is smaller than the `window_size`, unless truncation is enabled.
        - Polars DataFrames offer better performance with large datasets, especially for complex operations.

        Performance Considerations:
        ---------------------------
        - For very large datasets, Polars DataFrames are recommended due to their lower memory footprint and faster
          performance when compared to Pandas. Use Polars for more efficient partitioning and transformations.

        """
        partition_count = 1

        num_rows = df.height
        start_range = list(range(0, num_rows, self.stride))

        if self.reverse:
            start_range.reverse()

        for start in start_range:
            end = start + self.window_size

            if end > num_rows:
                if self.truncate:
                    break
                end = num_rows

            train_end = start + int(self.train_pct * (end - start))
            test_end = train_end + int(self.test_pct * (end - start)) if self.test_pct else train_end
            validation_end = end if self.val_pct else test_end

            # Yield the partitioned data slices
            partitioned_data = {
                part_name: df.slice(start, end - start)
                for part_name, (start, end) in {
                    "full": (start, end),
                    "train": (start, train_end),
                    "test": (train_end, test_end),
                    "validation": (test_end, validation_end),
                }.items()
            }

            # If partition size is smaller than window size, pad it
            if partitioned_data["full"].height < self.window_size and not self.truncate:
                partitioned_data["full"] = self._pad_partition(
                    partitioned_data["full"],
                    self.window_size,
                    partitioned_data["full"].height,
                    self.reverse,
                )

            # Wrap the partitioned_data in a dictionary to match the expected return type
            yield {f"partition_{partition_count}": partitioned_data}
            partition_count += 1

    def fit(self) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
        """Generate partition indices for the dataset.

        This method creates indices for sliding window partitions based on the specified `window_size`, `stride`,
        and other parameters. It yields the start and end indices for each partition, as well as train, test,
        and validation splits within each partition.

        :return: Iterator that yields partition indices for training, testing, and validation.
        :rtype: Iterator[Dict[str, Dict[str, Tuple[int, int]]]]
        :raises ValueError: If an unsupported backend is encountered.

        Example Usage:
        --------------
        .. code-block:: python

            partitioner = SlidingWindowPartitioner(
                tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3
            )

            for partition in partitioner.fit():
                print(partition)

        Output Format:
        --------------
        Each yielded partition has the following structure:

        .. code-block:: python

            {
                'partition_1': {
                    'full': (start_index, end_index),
                    'train': (train_start, train_end),
                    'test': (test_start, test_end),
                    'validation': (validation_start, validation_end)  # (Optional, if val_pct is provided)
                }
            }

        .. note::
           - The indices refer to row indices in the dataset, and the format remains the same regardless of the backend.
           - The partitioning occurs in a sliding window fashion with optional gaps, as specified by the stride.

        .. seealso::
           - :meth:`transform`: For generating the actual data slices corresponding to these indices.
        """
        df = self.tf.get_data()  # Get the dataset from the TimeFrame

        # Call backend-specific partitioning method
        if self.tf.backend in [BACKEND_PANDAS, BACKEND_MODIN]:
            return self._fit_pandas_modin(df)
        elif self.tf.backend == BACKEND_POLARS:
            return self._fit_polars(df)
        else:
            raise ValueError(f"Unsupported backend: {self.tf.backend}")

    def transform(self) -> Iterator[Dict[str, Dict[str, SupportedBackendDataFrame]]]:
        """Generate partitioned data slices for the dataset.

        This method yields the actual data slices corresponding to the partition indices generated by the `fit` method.
        The slices are returned as generic DataFrames, regardless of the backend (e.g., Pandas, Modin, or Polars).

        :return: Iterator yielding partitioned DataFrame slices.
        :rtype: Iterator[Dict[str, Dict[str, DataFrame]]]
        :raises ValueError: If an unsupported backend is encountered.

        Example Usage:
        --------------
        .. code-block:: python

            partitioner = SlidingWindowPartitioner(
                tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3
            )

            for partition_data in partitioner.transform():
                print(partition_data)

        Output Format:
        --------------
        Each yielded partition has the following structure:

        .. code-block:: python

            {
                'partition_1': {
                    'full': <DataFrame>,
                    'train': <DataFrame>,
                    'test': <DataFrame>,
                    'validation': <DataFrame>  # (Optional, if val_pct is provided)
                }
            }

        .. note::
           - This method transforms the dataset into partitioned slices based on indices created by `fit`.
           - Ensure the dataset is preprocessed properly to avoid errors during slicing.
           - The DataFrame format is agnostic to the backend.

        .. seealso::
           - :meth:`fit`: For generating the partition indices that are sliced in this method.
        """
        df = self.tf.get_data()  # Get the dataset from the TimeFrame

        # Call backend-specific transformation method
        if self.tf.backend in [BACKEND_PANDAS, BACKEND_MODIN]:
            return self._transform_pandas_modin(df)
        elif self.tf.backend == BACKEND_POLARS:
            return self._transform_polars(df)
        else:
            raise ValueError(f"Unsupported backend: {self.tf.backend}")

    def fit_transform(self) -> Iterator[Dict[str, Dict[str, SupportedBackendDataFrame]]]:
        """Fit and transform the dataset in a single step.

        This method combines the functionality of the `fit` and `transform` methods. It first generates partition indices
        using `fit`, and then returns the partitioned data slices using `transform`. The DataFrame format is backend-agnostic.

        :return: Iterator yielding partitioned DataFrame slices.
        :rtype: Iterator[Dict[str, Dict[str, DataFrame]]]
        :raises ValueError: If an unsupported backend is encountered.

        Example Usage:
        --------------
        .. code-block:: python

            partitioner = SlidingWindowPartitioner(
                tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3
            )

            for partition_data in partitioner.fit_transform():
                print(partition_data)

        Output Format:
        --------------
        Each yielded partition has the following structure:

        .. code-block:: python

            {
                'partition_1': {
                    'full': <DataFrame>,
                    'train': <DataFrame>,
                    'test': <DataFrame>,
                    'validation': <DataFrame>  # (Optional, if val_pct is provided)
                }
            }

        .. note::
           - This method is a convenient way to generate partition indices and their corresponding data slices in one step.
           - Ensure that the dataset is preprocessed properly to avoid issues during partitioning.

        .. seealso::
           - :meth:`fit`: For generating partition indices.
           - :meth:`transform`: For generating the actual partitioned slices.
        """
        for partition_indices in self.fit():
            yield from self.transform()

    def check_data(self, partition_index: Optional[int] = None) -> None:
        """Perform data checks on the entire dataset or a specific partition.

        This method performs validation checks on the dataset or a specific partition, ensuring that
        the sample size, feature-to-sample ratio, and class balance (if applicable) meet the expected criteria.

        If a partition index is provided, it checks only that partition; otherwise, it checks the entire dataset.

        :param partition_index: Index of the partition to check, or `None` to check the full dataset.
        :type partition_index: Optional[int]
        :raises ValueError: If the dataset or a partition fails validation checks.

        Example Usage:
        --------------
        .. code-block:: python

            partitioner = SlidingWindowPartitioner(
                tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3
            )

            # Perform checks on the full dataset
            partitioner.check_data()

            # Perform checks on the first partition
            partitioner.check_data(partition_index=0)

        .. note::
           - This method ensures that the data's structure and integrity (sample size, feature ratio, class balance)
             meet expectations for further processing.
           - Ensure the dataset or partition is not empty to avoid runtime errors.
        """
        df = self.tf.get_data()  # Get the DataFrame (could be Pandas, Modin, or Polars)

        if partition_index is not None:
            partition = next(itertools.islice(self.fit(), partition_index, None))
            start, end = partition[f"partition_{partition_index + 1}"]["full"]

            # Slice the DataFrame based on its type (Pandas/Modin vs Polars)
            if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
                df_to_check = df.iloc[start:end]
            elif isinstance(df, pl.DataFrame):
                df_to_check = df.slice(start, end - start)
            else:
                raise ValueError(f"Unsupported DataFrame type: {type(df)}")

            context = f"Partition {partition_index + 1}"
            min_samples = 100
        else:
            df_to_check = df
            context = "Full dataset"
            min_samples = 3000

        # Perform sample size, feature ratio, and class balance checks
        check_sample_size(
            df_to_check,
            backend=self.tf.backend,
            min_samples=min_samples,
            max_samples=100000,
            enable_warnings=True,
        )
        check_feature_to_sample_ratio(df_to_check, backend=self.tf.backend, max_ratio=0.2, enable_warnings=True)
        if self.tf.target_col:
            check_class_balance(
                df_to_check,
                target_col=self.tf.target_col,
                backend=self.tf.backend,
                enable_warnings=True,
            )

        if self.verbose:
            print(f"{context} checks completed with warnings where applicable.")
