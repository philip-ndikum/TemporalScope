""" TemporalScope/temporalscope/partitioning/sliding_window.py

This module defines the SlidingWindowPartitioner class, a specific implementation of the
TemporalPartitionerProtocol for creating contiguous, non-overlapping partitions using a sliding window mechanism.

Core Functionality:
-------------------
The SlidingWindowPartitioner divides a dataset into non-overlapping partitions using a fixed window size and
optional stride. The stride determines how far to move between the starting points of consecutive partitions, 
which can introduce gaps between them. Each partition can be further split into train, test, and validation sets.

This class utilizes the generator pattern for memory efficiency, yielding partition indices and data slices one at a time.

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

from typing import Dict, Tuple, Optional, Union, Iterator
import itertools
import warnings
import pandas as pd
import polars as pl
import modin.pandas as mpd
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.partition.base_protocol import TemporalPartitionerProtocol
from temporalscope.partition.partition_validators import (
    check_sample_size,
    check_feature_to_sample_ratio,
    check_class_balance,
)


class SlidingWindowPartitioner(TemporalPartitionerProtocol):
    """Sliding Window Partitioner for dividing time series data into contiguous, non-overlapping partitions.

    This class splits a dataset into partitions of a fixed window size. Users can define a stride to introduce gaps
    between consecutive partitions. Each partition can be further divided into train, test, and validation sets
    based on provided percentages.

    Assumptions:
    ------------
    - `train_pct` must be specified.
    - `test_pct` is optional, and if not provided, the remaining percentage after `train_pct` will implicitly be assigned to `test_pct`.
    - `val_pct` is also optional, and if provided, the sum of `train_pct`, `test_pct`, and `val_pct` must equal 1.0.
    - The total of `train_pct`, `test_pct`, and `val_pct` must sum to 1.0 exactly.

    The class uses a generator pattern for `fit` and `transform` methods to yield partition indices and data slices
    one at a time, promoting memory efficiency and lazy loading.

    :param tf: The TimeFrame object containing the data to be partitioned.
    :type tf: TimeFrame
    :param window_size: The size of each partition (number of rows).
    :type window_size: Optional[int]
    :param stride: The number of rows to skip between the start points of consecutive partitions.
                   A stride larger than the window size creates gaps, while a stride equal to the window size results in no gaps.
    :type stride: int
    :param reverse: Whether the sliding window should move in reverse (from the end to the start of the dataset).
                    If set to True, the window slides in reverse; if False (default), it slides forward.
    :type reverse: bool
    :param truncate: Whether to truncate the last partition if its size is smaller than the window size.
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
        - If `window_size` is not provided or is not a positive integer.
        - If `stride` is not a positive integer.
        - If `train_pct`, `test_pct`, or `val_pct` are not within the range [0, 1].
        - If `train_pct`, `test_pct`, and `val_pct` do not sum to 1.0.
        - If `train_pct` is provided without `test_pct` or `val_pct` summing to 1.0.
        - If the dataset cannot be sorted or retrieved properly from the TimeFrame.
        - If any required data is missing or invalid during the partitioning process.

    Example Usage:
    --------------
    .. code-block:: python

        import pandas as pd
        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.partition.sliding_window import SlidingWindowPartitioner

        # Create a sample dataset using Pandas
        data_df = pd.DataFrame({
            'time': pd.date_range(start='2021-01-01', periods=20, freq='D'),
            'value': range(20)
        })

        # Create a TimeFrame object
        data_tf = TimeFrame(data_df, time_col='time', target_col='value', backend='pd')

        # Create a SlidingWindowPartitioner with window_size=5 and stride=5
        partitioner = SlidingWindowPartitioner(
            tf=data_tf, window_size=5, stride=5, truncate=True, train_pct=0.8, test_pct=0.2, reverse=False
        )

        # Iterate over partition indices
        for partition in partitioner.fit():
            print(partition)

        # Iterate over data slices for each partition
        for partition_data in partitioner.transform():
            print(partition_data)

    Notes
    -----
    The sliding window can operate in two modes, depending on the `reverse` parameter:

    .. note::

       **Forward Sliding Window (reverse=False):**

       The sliding window starts from the beginning of the dataset and moves forward.

       Example:

       .. code-block:: text

           Data: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
           Window Size: 4, Stride: 3

           Window 1: [ 1, 2, 3, 4 ]
           Window 2: [ 4, 5, 6, 7 ]
           Window 3: [ 7, 8, 9, 10 ]

    .. seealso::

       **Reverse Sliding Window (reverse=True):**

       The sliding window starts from the end of the dataset and moves backward.

       Example:

       .. code-block:: text

           Data: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
           Window Size: 4, Stride: 3

           Window 1: [ 7, 8, 9, 10 ]
           Window 2: [ 4, 5, 6, 7 ]
           Window 3: [ 1, 2, 3, 4 ]
    """

    def __init__(
        self,
        tf: TimeFrame,
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
        if window_size is None or window_size <= 0:
            raise ValueError("`window_size` must be a positive integer.")
        if stride <= 0:
            raise ValueError("`stride` must be a positive integer.")
        if not (0 <= train_pct <= 1):
            raise ValueError("`train_pct` must be between 0 and 1.")
        if test_pct is not None and not (0 <= test_pct <= 1):
            raise ValueError("`test_pct` must be between 0 and 1.")
        if val_pct is not None and not (0 <= val_pct <= 1):
            raise ValueError("`val_pct` must be between 0 and 1.")
        if train_pct + (test_pct or 0) + (val_pct or 0) != 1.0:
            raise ValueError("Train, test, and validation percentages must sum to 1.0.")

        self.tf = tf  # Use TimeFrame directly
        self.window_size = window_size
        self.stride = stride
        self.reverse = reverse
        self.truncate = truncate
        self.verbose = verbose
        self.train_pct, self.test_pct, self.val_pct = self._precompute_percentages(
            train_pct, test_pct, val_pct
        )

        # Sort data by time column using TimeFrame method
        self.tf.sort_data(ascending=True)

        self._fit_executed = False
        self._transform_executed = False

    def _precompute_percentages(
        self, train_pct: float, test_pct: Optional[float], val_pct: Optional[float]
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """Calculate and validate the percentages for train, test, and validation splits.

        This method checks that the provided percentages for training, testing, and validation
        add up to 100%. It ensures that if a validation percentage is specified, both training
        and testing percentages are also provided. The method also prints out the calculated
        percentages if `verbose` mode is enabled.

        :param train_pct: The percentage of data allocated for training within each partition.
        :type train_pct: float
        :param test_pct: The percentage of data allocated for testing within each partition. If not provided,
                         it defaults to 1.0 minus `train_pct` and `val_pct`.
        :type test_pct: Optional[float]
        :param val_pct: The percentage of data allocated for validation within each partition, if any.
        :type val_pct: Optional[float]

        :return: A tuple containing the validated percentages for training, testing, and validation.
        :rtype: Tuple[float, Optional[float], Optional[float]]

        :raises ValueError: If the sum of `train_pct`, `test_pct`, and `val_pct` does not equal 100%, or
                            if `val_pct` is specified without both `train_pct` and `test_pct`.
        """
        total_pct = (train_pct or 0) + (test_pct or 0) + (val_pct or 0)
        if total_pct != 1.0:
            raise ValueError("Train, test, and validation percentages must sum to 1.0.")
        if val_pct is not None and (train_pct is None or test_pct is None):
            raise ValueError(
                "Validation percentage requires both train and test percentages to be provided."
            )
        if self.verbose:
            print(f"Train percentage: {train_pct}")
            print(f"Test percentage: {test_pct}")
            print(f"Validation percentage: {val_pct}")
        return train_pct, test_pct, val_pct

    def _validate_partitioning(self, num_rows: int, window_size: int) -> None:
        """Validate the feasibility of partitioning the dataset with the given window size and stride.

        This method checks if the dataset can be properly partitioned based on the provided `window_size` and `stride`.
        It ensures that:
        - The stride is not larger than the window size, which would cause partitions to be skipped.
        - The stride is a positive integer.
        - The dataset has enough rows to create at least one partition.

        :param num_rows: The total number of rows in the dataset.
        :type num_rows: int
        :param window_size: The window size to be used for each partition.
        :type window_size: int
        :raises ValueError: If partitioning is not possible due to any of the following conditions:
            - The stride is larger than the window size.
            - The stride is not a positive integer.
            - The dataset is too small to create even a single partition with the given window size and stride.
        """
        # Ensure the stride is not larger than the window size
        if self.stride > window_size:
            raise ValueError(
                f"Stride ({self.stride}) is larger than the window size ({window_size}). "
                "This would cause partitions to be skipped."
            )

        # Ensure the stride is a positive integer
        if self.stride <= 0:
            raise ValueError("Stride must be a positive integer.")

        # Calculate the number of possible partitions
        num_possible_partitions = (num_rows - window_size) // self.stride + 1

        # Ensure there are enough rows in the dataset for at least one partition
        if num_possible_partitions < 1:
            raise ValueError(
                f"Not enough rows ({num_rows}) to create partitions with window size {window_size} "
                f"and stride {self.stride}. Try reducing the number of partitions or adjusting the window size and stride."
            )

        # Print validation success message if verbose mode is enabled
        if self.verbose:
            print(
                f"Partitioning validated: {num_possible_partitions} possible partitions."
            )

    def _get_data_shape(self) -> Tuple[int, int]:
        """Get the number of rows and features from the dataset, ensuring compatibility with different backends.

        :return: A tuple containing the number of rows and features in the dataset.
        :rtype: Tuple[int, int]
        :raises ValueError: If the backend is unsupported.
        """
        backend = self.tf.backend  # Access the backend from the TimeFrame object
        if backend in ["pd", "mpd"]:
            num_rows, num_features = self.df.shape
        elif backend == "pl":
            num_rows = self.df.height
            num_features = self.df.width
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return num_rows, num_features

    def fit(self) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
        """Generate partition indices for the dataset, lazily yielding them one at a time.

        This method divides the dataset into partitions based on the specified window size and stride.
        It generates indices for the entire partition as well as for the training, testing, and validation splits
        within each partition.

        The method operates in a memory-efficient manner, generating and yielding each partition's indices
        only when needed.

        :yield: A dictionary where each key corresponds to a partition (e.g., 'partition_1'), and the value is another
                dictionary with keys 'full', 'train', 'test', and optionally 'validation', each mapping to a tuple of indices.
        :rtype: Iterator[Dict[str, Dict[str, Tuple[int, int]]]]
        :raises ValueError: If `window_size` is larger than the dataset or if the total number of partitions is insufficient.
        """
        num_rows, _ = (
            self._get_data_shape()
        )  # Retrieve the shape using backend-specific method
        window_size = self.window_size

        # Validate that the partitioning is possible with the given window size and stride
        self._validate_partitioning(num_rows, window_size)

        partition_count = 1

        # Ensure start_range is always a list to avoid type conflicts
        start_range = list(range(0, num_rows, self.stride))
        if self.reverse:
            start_range.reverse()

        # Iterate over the dataset to generate partition indices
        for start in start_range:
            end = start + window_size

            # Adjust the end if it exceeds the number of rows and truncate is False
            if end > num_rows:
                if self.truncate:
                    break  # Stop iteration if the last partition is smaller than the window size and truncate is True
                end = num_rows  # Adjust to include the remaining data

            # Compute the split points for train, test, and validation
            train_end = start + int(self.train_pct * (end - start))
            test_end = (
                train_end + int(self.test_pct * (end - start))
                if self.test_pct
                else train_end
            )
            validation_end = end if self.val_pct else test_end

            # Yield the partition indices
            yield {
                f"partition_{partition_count}": {
                    "full": (start, end),
                    "train": (start, train_end),
                    "test": (train_end, test_end),
                    "validation": (
                        (test_end, validation_end) if self.val_pct else (0, 0)
                    ),
                }
            }

            # If verbose is enabled, print details of the current partition
            if self.verbose:
                print(f"Partition {partition_count}: {start} to {end}")
                print(
                    f"Training: {start} to {train_end}, Testing: {train_end} to {test_end}"
                )

            partition_count += 1

        # Track that fit has been run
        self._fit_executed = True

    def transform(
        self,
    ) -> Iterator[
        Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]]
    ]:
        """Generate and yield the data slices for each partition.

        This method utilizes the partition indices generated by the `fit` method to extract and return
        the corresponding data slices from the original dataset. The data is returned for each partition,
        including the full partition as well as the training, testing, and validation subsets.

        The method is designed to be memory-efficient, generating and yielding each partition's data
        only when required.

        :yield: A dictionary where each key corresponds to a partition (e.g., 'partition_1'), and the value is another
                dictionary with keys 'full', 'train', 'test', and optionally 'validation', each mapping to a DataFrame slice.
        :rtype: Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame, pl.DataFrame]]]]
        :raises ValueError: If data slicing fails for any partition, which could occur if the indices are out of bounds.
        """
        # Generate partition indices using the fit method
        for partition in self.fit():
            partitioned_data = {}

            # Iterate over each partition and its corresponding indices
            for key, partition_dict in partition.items():
                partitioned_data[key] = {
                    # Slice the data using the appropriate backend method (pandas, Modin, or Polars)
                    part_name: (
                        self.df.iloc[start:end]
                        if isinstance(self.df, (pd.DataFrame, mpd.DataFrame))
                        else self.df.slice(start, end - start)
                    )  # Polars-specific slicing
                    for part_name, (start, end) in partition_dict.items()
                    if start is not None
                    and end is not None  # Ensure valid start and end indices
                }

            # Yield the partitioned data
            yield partitioned_data

        # Track that transform has been run
        self._transform_executed = True

    def check_data(self, partition_index: Optional[int] = None) -> None:
        """Perform data checks on the entire TimeFrame or a specific partition.

        This method validates whether the dataset or a specific partition meets
        recommended criteria based on sample size, feature-to-sample ratio, and class balance.

        - If `partition_index` is provided, checks are performed on the specified partition.
        - If `partition_index` is None, checks are performed on the entire TimeFrame.

        Assumptions:
        ------------
        - If the method is called without running `fit`, it checks the full dataset.
        - If `fit` has been run and `partition_index` is provided, it checks the specific partition.

        Warnings are raised instead of errors to allow users to proceed with caution.

        :param partition_index: Index of the partition to check, or None to check the entire dataset.
        :type partition_index: Optional[int]
        """
        if partition_index is not None:
            # Generate the required partition directly without assuming prior fit() call
            partition = next(itertools.islice(self.fit(), partition_index, None))
            start, end = partition[f"partition_{partition_index + 1}"]["full"]
            df_to_check = self.df[start:end]
            context = f"Partition {partition_index + 1}"
            min_samples = 100  # Lower threshold for partitions
        else:
            df_to_check = self.df
            context = "Full dataset"
            min_samples = 3000  # Higher threshold for the full dataset

        num_rows, num_features = df_to_check.shape
        target_col = self.tf.target_col

        # Perform checks with warnings enabled
        check_sample_size(
            df_to_check,
            backend=self.tf.backend,
            min_samples=min_samples,
            max_samples=100000,  # Standard large threshold
            enable_warnings=True,
        )

        check_feature_to_sample_ratio(
            df_to_check,
            backend=self.tf.backend,
            max_ratio=0.2,  # Standard ratio for features to samples
            enable_warnings=True,
        )

        if target_col:
            check_class_balance(
                df_to_check,
                target_col=target_col,
                backend=self.tf.backend,
                enable_warnings=True,
            )

        if self.verbose:
            print(f"{context} checks completed with warnings where applicable.")
