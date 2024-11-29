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

"""TemporalScope/src/temporalscope/partition/single_target/sliding_window.py

This module defines the SlidingWindowPartitioner class, a specific implementation
of the TemporalPartitionerProtocol for creating contiguous, non-overlapping
partitions using a sliding window mechanism.

The SlidingWindowPartitioner divides a dataset into non-overlapping partitions
using a fixed window size and optional stride. The stride determines how far to
move between the starting points of consecutive partitions, which can introduce
gaps between them. Each partition can be further split into train, test, and
validation sets.

This class utilizes the generator pattern for memory efficiency, yielding
partition indices and data slices one at a time. The `SlidingWindowPartitioner`
is intended for universal models, which assume flat partitioning across all
entities. Users are responsible for preprocessing steps such as deduplication or
transforming `time_col` to numerical features.

Engineering Design:
-------------------

+-------------------------+-------------------------------------------------------+
| Aspect                  | Description                                           |
+-------------------------+-------------------------------------------------------+
| Partial Temporal        | Follows a universal model design; allows overlapping  |
| Ordering                | labels within partitions while leaving strict         |
|                         | temporal ordering to the user.                        |
+-------------------------+-------------------------------------------------------+
| Narwhals API            | Leverages Narwhals backend for efficient operations.  |
|                         | Users can switch between supported backends (e.g.,    |
|                         | Pandas, Polars, Modin) using core_utils.              |
+-------------------------+-------------------------------------------------------+
| Dataset Accessibility   | Inspired by Dask and TensorFlow design. Provides      |
|                         | hierarchical access via `partitions[index]["train"]`, |
|                         | `test`, `validation`, or `full` labels.               |
+-------------------------+-------------------------------------------------------+
| Lazy/Eager Execution    | Narwhals backend supports lazy/eager evaluation;      |
|                         | generator pattern ensures memory-efficient `fit` and  |
|                         | `transform` operations.                               |
+-------------------------+-------------------------------------------------------+
| Human-Centric Design    | Combines human-readable labels (`train`, `test`) with |
|                         | indexing for scalable workflows, reducing cognitive   |
|                         | overhead when handling large numbers of partitions.   |
+-------------------------+-------------------------------------------------------+
| Padding Control         | Leaves padding decisions (e.g., zero-padding) to      |
|                         | users while allowing configurable truncation.         |
+-------------------------+-------------------------------------------------------+

.. note::

Visualization:
--------------
The table below illustrates how the sliding window mechanism works with overlapping
partitions. Each "X" represents a row included in the respective partition, based on
the configured window size and stride.

| Time         | Partition 1 | Partition 2 | Partition 3 | Partition 4 | Partition 5 |
|--------------|-------------|-------------|-------------|-------------|-------------|
| 2021-01-01   | X           |             |             |             |             |
| 2021-01-02   | X           | X           |             |             |             |
| 2021-01-03   |             | X           | X           |             |             |
| 2021-01-04   |             |             | X           | X           |             |
| 2021-01-05   |             |             |             | X           | X           |
| 2021-01-06   |             |             |             |             | X           |

.. seealso::

    1. Gu et al., 2021. The sliding window and SHAP theory applied to long
       short-term memory networks for state of charge prediction.
    2. Pham et al., 2023. Speech emotion recognition using overlapping sliding
       window and explainable neural networks.
    3. Van Zyl et al., 2024. Explainable AI for feature selection in time series
       energy forecasting with Grad-CAM and SHAP.
    4. Bi et al., 2020. Prediction model for identifying methylation sites with
       XGBoost and SHAP explainability.
    5. Zimmermann et al., 2022. Improving drift detection by monitoring SHAP
       loss values in pattern recognition workflows.
    6. Li et al., 2022. Visualizing distributional shifts using SHAP in machine
       learning models.
    7. Seiffer et al., 2021. Concept drift detection in manufacturing data with
       SHAP for error prediction improvement.
    8. Haug et al., 2022. Change detection for local explainability in evolving
       data streams.
    9. Zhao et al., 2020. Feature drift detection in evolving data streams with
       database applications.
"""

from temporalscope.partition.base_protocol import TemporalPartitionerProtocol

# Precision constant for floating-point comparisons
PRECISION = 1e-6


class SlidingWindowPartitioner(TemporalPartitionerProtocol):
    """Partition time series data into contiguous, non-overlapping windows.

    The `SlidingWindowPartitioner` divides a dataset into partitions of a fixed `window_size`
    with an optional `stride` to introduce gaps or overlaps. Each partition is further split
    into train, test, and validation subsets based on user-defined percentages.

    This class operates on a global dataset level, preserving temporal order without grouping
    by entities. It assumes preprocessed data and integrates seamlessly with the `TimeFrame`
    class for compatibility with various backends and workflows.

    Key Features:
    -------------
    - Preserves **weak temporal ordering**, allowing for duplicate entities or mixed-frequency
      time columns. This design ensures flexibility for advanced workflows while leaving
      strict sorting decisions to the user.
    - Configurable `window_size` and `stride` for flexible partitioning.
    - Supports train, test, and optional validation splits within each partition.
    - Lazy-loading with `fit` and `transform` methods for memory efficiency.
    - Designed for single-target, DataFrame-centric workflows.

    :param tf: The `TimeFrame` object containing validated and sorted time series data.
    :type tf: TimeFrame
    :param window_size: Number of rows in each partition. Required unless `num_partitions` is provided.
    :type window_size: Optional[int]
    :param stride: Number of rows to skip between consecutive partitions. Default is equal to `window_size`.
    :type stride: int, optional
    :param train_pct: Fraction of each partition allocated for training. Required.
    :type train_pct: float
    :param test_pct: Fraction allocated for testing. Defaults to remaining data after training.
    :type test_pct: Optional[float]
    :param val_pct: Fraction allocated for validation. Defaults to zero.
    :type val_pct: Optional[float]
    :param truncate: Whether to truncate the last partition if it is smaller than `window_size`.
    :type truncate: bool, optional
    :param verbose: Print partitioning details if True. Default is False.
    :type verbose: bool, optional

    :raises ValueError:
        - If `window_size` or `num_partitions` is invalid.
        - If train, test, and validation percentages do not sum to 1.0.
        - If `stride` is not positive or if required data is missing.

    Example Usage:
    --------------
    .. code-block:: python

        from temporalscope.core.temporal_data_loader import TimeFrame
        from temporalscope.partition.sliding_window import SlidingWindowPartitioner

        # Create a sample dataset
        data_df = pd.DataFrame(
            {
                "time": pd.date_range(start="2021-01-01", periods=6, freq="D"),
                "value": range(6),
            }
        )
        data_tf = TimeFrame(data_df, time_col="time", target_col="value")

        # Initialize partitioner
        partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=2, stride=1, train_pct=0.7, test_pct=0.3)

        # Iterate over partition indices
        for partition in partitioner.fit():
            print(partition)

        # Iterate over data slices for each partition
        for partition_data in partitioner.transform():
            print(partition_data)
    """

    # DEFAULT_PAD_SCHEME = "forward_fill"  # Define the default padding scheme

    # def __init__(
    #     self,
    #     tf: TimeFrame,
    #     num_partitions: Optional[int] = 2,
    #     window_size: Optional[int] = None,
    #     stride: int = 1,
    #     reverse: bool = False,
    #     truncate: bool = True,
    #     train_pct: float = 0.7,
    #     test_pct: Optional[float] = 0.3,
    #     val_pct: Optional[float] = None,
    #     enable_warnings: bool = False,
    #     verbose: bool = False,
    #     pad_scheme: str = DEFAULT_PAD_SCHEME,
    # ):
    #     """Initialize the SlidingWindowPartitioner with the given parameters.

    #     :param tf: TimeFrame object to partition. All columns except `time_col` must be numeric.
    #     :param num_partitions: Number of partitions to create (ignored if `window_size` is provided).
    #     :param window_size: Size of each partition.
    #     :param stride: Number of rows to skip between partitions.
    #     :param reverse: Whether the sliding window should move in reverse.
    #     :param truncate: Whether to truncate the last partition if smaller than `window_size`.
    #     :param train_pct: Percentage of data allocated for training.
    #     :param test_pct: Percentage of data allocated for testing.
    #     :param val_pct: Percentage of data allocated for validation.
    #     :param enable_warnings: Enable warnings for uneven partition sizes.
    #     :param verbose: Enable verbose output.
    #     :param pad_scheme: The padding scheme to use for filling partitions. Defaults to 'forward_fill'.
    #     :raises ValueError: If input parameters are invalid or columns (except `time_col`) are not numeric.
    #     """
    #     # Validate the backend and pad scheme
    #     is_valid_temporal_backend(tf.dataframe_backend)
    #     if pad_scheme not in PAD_SCHEMES:
    #         raise ValueError(f"Invalid pad_scheme: {pad_scheme}. Supported schemes: {PAD_SCHEMES}")

    #     # Check if all columns except `time_col` are numeric
    #     non_time_cols = [col for col in tf.get_data().columns if col != tf.time_col]
    #     non_numeric_cols = [col for col in non_time_cols if not pd.api.types.is_numeric_dtype(tf.get_data()[col])]

    #     if non_numeric_cols:
    #         raise ValueError(
    #             f"All columns except `time_col` must be numeric. Non-numeric columns found: {non_numeric_cols}"
    #         )

    #     # Get the number of rows from the TimeFrame object
    #     num_rows = tf.get_data().shape[0]

    #     # Determine window size if not provided
    #     if window_size is None:
    #         if num_partitions is None or num_partitions <= 0:
    #             raise ValueError("`num_partitions` must be a positive integer.")
    #         window_size = num_rows // num_partitions

    #     # Validate the window size and stride
    #     if window_size <= 0:
    #         raise ValueError("`window_size` must be a positive integer.")
    #     if stride <= 0:
    #         raise ValueError("`stride` must be a positive integer.")

    #     # Validate percentage values
    #     if not (0 <= train_pct <= 1):
    #         raise ValueError("`train_pct` must be between 0 and 1.")
    #     if test_pct is not None and not (0 <= test_pct <= 1):
    #         raise ValueError("`test_pct` must be between 0 and 1.")
    #     if val_pct is not None and not (0 <= val_pct <= 1):
    #         raise ValueError("`val_pct` must be between 0 and 1.")
    #     if train_pct + (test_pct or 0) + (val_pct or 0) != 1.0:
    #         raise ValueError("Train, test, and validation percentages must sum to 1.0.")

    #     # Assign attributes
    #     self.tf = tf
    #     self.window_size = window_size
    #     self.stride = stride
    #     self.reverse = reverse
    #     self.truncate = truncate
    #     self.verbose = verbose
    #     self.pad_scheme = pad_scheme  # Assign the chosen padding scheme

    #     # Precompute percentages
    #     self.train_pct, self.test_pct, self.val_pct = self.precompute_percentages(train_pct, test_pct, val_pct)

    #     # Sort the data using TimeFrame's sort_data method
    #     self.tf.sort_data(ascending=True)

    #     # Initialize internal state
    #     self._fit_executed = False
    #     self._transform_executed = False

    # def precompute_percentages(
    #     self,
    #     train_pct: float,
    #     test_pct: Optional[float],
    #     val_pct: Optional[float],
    #     precision: float = PRECISION,  # Now using the precision constant
    # ) -> Tuple[float, float, float]:
    #     """Precompute and validate train, test, and validation percentages.

    #     This function ensures that the sum of train, test, and validation percentages equals 1.0.
    #     If test_pct is not provided, it will be set to the remaining percentage after the train percentage.

    #     :param train_pct: Percentage of data allocated for training.
    #     :type train_pct: float
    #     :param test_pct: Optional. Percentage of data allocated for testing.
    #     :type test_pct: Optional[float]
    #     :param val_pct: Optional. Percentage of data allocated for validation.
    #     :type val_pct: Optional[float]
    #     :param precision: The tolerance level for floating-point imprecision, defaults to 1e-6.
    #     :type precision: float
    #     :return: A tuple containing the validated percentages for training, testing, and validation.
    #     :rtype: Tuple[float, float, float]
    #     :raises ValueError: If the percentages do not sum to 1.0 or are not within the valid range (0 to 1).
    #     """
    #     # Validate the train percentage
    #     if not (0 <= train_pct <= 1):
    #         raise ValueError("train_pct must be between 0 and 1.")
    #     # Handle test_pct and val_pct cases explicitly
    #     if test_pct is None and val_pct is None:
    #         test_pct = 1.0 - train_pct
    #         val_pct = 0.0
    #     elif test_pct is not None and val_pct is None:
    #         if not (0 <= test_pct <= 1):
    #             raise ValueError("test_pct must be between 0 and 1.")
    #         val_pct = 1.0 - train_pct - test_pct
    #     elif test_pct is None and val_pct is not None:
    #         if not (0 <= val_pct <= 1):
    #             raise ValueError("val_pct must be between 0 and 1.")
    #         test_pct = 1.0 - train_pct - val_pct
    #     else:
    #         # Both test_pct and val_pct are provided, ensure they are valid
    #         if test_pct is not None and not (0 <= test_pct <= 1):
    #             raise ValueError("test_pct must be between 0 and 1.")
    #         if val_pct is not None and not (0 <= val_pct <= 1):
    #             raise ValueError("val_pct must be between 0 and 1.")
    #     # Ensure they sum to 1.0, handling floating-point imprecision with precision constant
    #     total_pct = train_pct + (test_pct or 0) + (val_pct or 0)
    #     if not (abs(total_pct - 1.0) < precision):  # Compare with the precision constant
    #         raise ValueError("Train, test, and validation percentages must sum to 1.0.")
    #     # Ensure test_pct and val_pct are float types, not None
    #     return train_pct, float(test_pct or 0), float(val_pct or 0)

    # def _fit_pandas_modin(self) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
    #     """Fit method for partitioning using TimeFrame data.

    #     This method partitions the dataset retrieved from TimeFrame, irrespective of backend.

    #     :return: Iterator yielding partition indices.
    #     """
    #     df = self.tf.get_data()  # Get the DataFrame from TimeFrame
    #     partition_count = 1

    #     num_rows = df.shape[0]
    #     start_range = list(range(0, num_rows, self.stride))

    #     if self.reverse:
    #         start_range.reverse()

    #     for start in start_range:
    #         end = start + self.window_size

    #         if end > num_rows:
    #             if self.truncate:
    #                 break
    #             end = num_rows

    #         train_end = start + int(self.train_pct * (end - start))
    #         test_end = train_end + int(self.test_pct * (end - start)) if self.test_pct else train_end
    #         validation_end = end if self.val_pct else test_end

    #         # Yield the partition indices
    #         yield {
    #             f"partition_{partition_count}": {
    #                 "full": (start, end),
    #                 "train": (start, train_end),
    #                 "test": (train_end, test_end),
    #                 "validation": (test_end, validation_end) if self.val_pct else (0, 0),
    #             }
    #         }
    #         partition_count += 1

    # def _fit_polars(self) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
    #     """Fit method for partitioning using TimeFrame data.

    #     This method partitions the dataset retrieved from TimeFrame, irrespective of backend.

    #     :return: Iterator yielding partition indices.
    #     """
    #     df = self.tf.get_data()  # Get the DataFrame from TimeFrame
    #     partition_count = 1

    #     num_rows = df.shape[0]  # Use shape[0] to be consistent with other backends like Pandas/Modin
    #     start_range = list(range(0, num_rows, self.stride))

    #     if self.reverse:
    #         start_range.reverse()

    #     for start in start_range:
    #         end = start + self.window_size

    #         if end > num_rows:
    #             if self.truncate:
    #                 break
    #             end = num_rows

    #         train_end = start + int(self.train_pct * (end - start))
    #         test_end = train_end + int(self.test_pct * (end - start)) if self.test_pct else train_end
    #         validation_end = end if self.val_pct else test_end

    #         # Yield the partition indices
    #         yield {
    #             f"partition_{partition_count}": {
    #                 "full": (start, end),
    #                 "train": (start, train_end),
    #                 "test": (train_end, test_end),
    #                 "validation": (test_end, validation_end) if self.val_pct else (0, 0),
    #             }
    #         }
    #         partition_count += 1

    # def _transform_pandas_modin(self) -> Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame]]]]:
    #     """Transform method for Pandas/Modin backend.

    #     This method transforms the partitioned dataset into slices, yielding the data slices corresponding to
    #     the partition indices generated by the `fit` method.

    #     It processes each partition and splits it into train, test, and optionally validation sets.
    #     If a partition's size is smaller than the specified `window_size`, padding is applied using the selected
    #     padding scheme (`zero_pad`, `forward_fill_pad`, `backward_fill_pad`, or `mean_fill_pad`) to ensure
    #     uniform size across partitions, unless `truncate` is set to True.

    #     :return: Iterator yielding partitioned DataFrame slices for Pandas/Modin backends.
    #     :rtype: Iterator[Dict[str, Dict[str, Union[pd.DataFrame, mpd.DataFrame]]]]

    #     Example Usage:
    #     --------------
    #     .. code-block:: python

    #         partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3)

    #         for partition_data in partitioner._transform_pandas_modin():
    #             print(partition_data)

    #     Output Format:
    #     --------------
    #     Each yielded partition has the following structure:

    #     .. code-block:: python

    #         {
    #             'partition_1': {
    #                 'full': <pd.DataFrame>,
    #                 'train': <pd.DataFrame>,
    #                 'test': <pd.DataFrame>,
    #                 'validation': <pd.DataFrame>  # (Optional, if val_pct is provided)
    #             }
    #         }

    #     Notes
    #     -----
    #     .. note::
    #         - Padding is applied when the size of a partition is smaller than the `window_size`, unless truncation is enabled.
    #         - The padding scheme is determined by the `pad_scheme` parameter in the constructor (e.g., 'zero', 'forward_fill').
    #         - Ensure that the input DataFrame is not empty to avoid runtime errors.
    #         - For very large datasets, the padding process may increase memory usage. Consider using Modin when handling
    #           large datasets to take advantage of distributed processing.

    #     """
    #     partition_count = 1
    #     df = self.tf.get_data()  # Fetch the data from TimeFrame

    #     # Add a type check to ensure df is a DataFrame
    #     if not isinstance(df, (pd.DataFrame, mpd.DataFrame)):
    #         raise TypeError("Expected df to be a pandas or modin DataFrame")

    #     for partition in self.fit():  # Partition indices generated by fit()
    #         partitioned_data = {}

    #         # Iterate through the partition and generate train, test, validation sets
    #         if isinstance(partition, dict):
    #             for key, partition_dict in partition.items():
    #                 partitioned_data[key] = {
    #                     part_name: df.iloc[start:end]  # Slice based on indices
    #                     for part_name, (start, end) in partition_dict.items()
    #                     if start is not None and end is not None
    #                 }

    #                 # Check if padding is needed (partition size is smaller than window_size and truncate is False)
    #                 if partition_dict["full"][1] - partition_dict["full"][0] < self.window_size and not self.truncate:
    #                     # Apply the chosen padding scheme
    #                     if self.pad_scheme == "zero":
    #                         partitioned_data[key]["full"] = zero_pad(
    #                             partitioned_data[key]["full"], target_len=self.window_size
    #                         )
    #                     elif self.pad_scheme == "forward_fill":
    #                         partitioned_data[key]["full"] = forward_fill_pad(
    #                             partitioned_data[key]["full"],
    #                             target_len=self.window_size,
    #                             end=len(partitioned_data[key]["full"]),
    #                             reverse=False,
    #                         )
    #                     elif self.pad_scheme == "backward_fill":
    #                         partitioned_data[key]["full"] = backward_fill_pad(
    #                             partitioned_data[key]["full"],
    #                             target_len=self.window_size,
    #                             end=len(partitioned_data[key]["full"]),
    #                             reverse=False,
    #                         )
    #                     elif self.pad_scheme == "mean_fill":
    #                         partitioned_data[key]["full"] = mean_fill_pad(
    #                             partitioned_data[key]["full"],
    #                             target_len=self.window_size,
    #                             end=len(partitioned_data[key]["full"]),
    #                             reverse=False,
    #                         )

    #         yield partitioned_data
    #         partition_count += 1

    # def _transform_polars(self) -> Iterator[Dict[str, Dict[str, pl.DataFrame]]]:
    #     """Transform method for Polars backend.

    #     This method generates partitioned data slices for the Polars backend, yielding the data slices corresponding
    #     to the partition indices generated by the `fit` method. If the size of a partition is smaller than the
    #     specified `window_size`, padding is applied using the selected padding scheme (`zero_pad`, `forward_fill_pad`,
    #     `backward_fill_pad`, or `mean_fill_pad`), unless `truncate` is set to True.

    #     :return: Iterator yielding partitioned DataFrame slices for Polars backend.
    #     :rtype: Iterator[Dict[str, Dict[str, pl.DataFrame]]]

    #     Example Usage:
    #     --------------
    #     .. code-block:: python

    #         partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3)

    #         for partition_data in partitioner._transform_polars():
    #             print(partition_data)

    #     Output Format:
    #     --------------
    #     Each yielded partition has the following structure:

    #     .. code-block:: python

    #         {
    #             'partition_1': {
    #                 'full': <pl.DataFrame>,
    #                 'train': <pl.DataFrame>,
    #                 'test': <pl.DataFrame>,
    #                 'validation': <pl.DataFrame>  # (Optional, if val_pct is provided)
    #             }
    #         }

    #     Notes
    #     -----
    #     .. note::
    #         - Padding is applied when the size of a partition is smaller than the `window_size`, unless truncation is enabled.
    #         - The padding scheme is determined by the `pad_scheme` parameter in the constructor (e.g., 'zero', 'forward_fill').
    #         - Polars DataFrames offer better performance with large datasets, especially for complex operations.
    #         - For very large datasets, Polars DataFrames are recommended due to their lower memory footprint and faster
    #           performance when compared to Pandas. Use Polars for more efficient partitioning and transformations.

    #     """
    #     partition_count = 1
    #     df = self.tf.get_data()  # Fetch the data from TimeFrame

    #     for partition in self.fit():  # Partition indices generated by fit()
    #         partitioned_data = {}

    #         # Iterate through the partition and generate train, test, validation sets
    #         if isinstance(partition, dict):
    #             for key, partition_dict in partition.items():
    #                 partitioned_data[key] = {
    #                     part_name: df.slice(start, end - start)  # Slice based on indices
    #                     for part_name, (start, end) in partition_dict.items()
    #                     if start is not None and end is not None
    #                 }

    #                 # Apply padding if partition size is smaller than window_size and truncate is False
    #                 if partition_dict["full"][1] - partition_dict["full"][0] < self.window_size and not self.truncate:
    #                     # Apply the chosen padding scheme for Polars DataFrame
    #                     if self.pad_scheme == "zero":
    #                         partitioned_data[key]["full"] = zero_pad(
    #                             partitioned_data[key]["full"], target_len=self.window_size
    #                         )
    #                     elif self.pad_scheme == "forward_fill":
    #                         partitioned_data[key]["full"] = forward_fill_pad(
    #                             partitioned_data[key]["full"],
    #                             target_len=self.window_size,
    #                             end=len(partitioned_data[key]["full"]),
    #                             reverse=False,
    #                         )
    #                     elif self.pad_scheme == "backward_fill":
    #                         partitioned_data[key]["full"] = backward_fill_pad(
    #                             partitioned_data[key]["full"],
    #                             target_len=self.window_size,
    #                             end=len(partitioned_data[key]["full"]),
    #                             reverse=False,
    #                         )
    #                     elif self.pad_scheme == "mean_fill":
    #                         partitioned_data[key]["full"] = mean_fill_pad(
    #                             partitioned_data[key]["full"],
    #                             target_len=self.window_size,
    #                             end=len(partitioned_data[key]["full"]),
    #                             reverse=False,
    #                         )

    #         yield partitioned_data
    #         partition_count += 1

    # def fit(self) -> Iterator[Dict[str, Dict[str, Tuple[int, int]]]]:
    #     """Generate partition indices for the dataset.

    #     This method creates indices for sliding window partitions based on the specified `window_size`, `stride`,
    #     and other parameters. It yields the start and end indices for each partition, as well as train, test,
    #     and validation splits within each partition.

    #     :return: Iterator that yields partition indices for training, testing, and validation.
    #     :rtype: Iterator[Dict[str, Dict[str, Tuple[int, int]]]]
    #     :raises ValueError: If an unsupported backend is encountered.

    #     Example Usage:
    #     --------------
    #     .. code-block:: python

    #         partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3)

    #         for partition in partitioner.fit():
    #             print(partition)

    #     Output Format:
    #     --------------
    #     Each yielded partition has the following structure:

    #     .. code-block:: python

    #         {
    #             "partition_1": {
    #                 "full": (start_index, end_index),
    #                 "train": (train_start, train_end),
    #                 "test": (test_start, test_end),
    #                 "validation": (validation_start, validation_end),  # (Optional, if val_pct is provided)
    #             }
    #         }

    #     .. note::
    #        - The indices refer to row indices in the dataset, and the format remains the same regardless of the backend.
    #        - The partitioning occurs in a sliding window fashion with optional gaps, as specified by the stride.

    #     .. seealso::
    #        - :meth:`transform`: For generating the actual data slices corresponding to these indices.
    #     """
    #     # Call backend-specific partitioning method
    #     if self.tf.dataframe_backend in [BACKEND_PANDAS, BACKEND_MODIN]:
    #         return self._fit_pandas_modin()  # type: ignore[call-arg]
    #     elif self.tf.dataframe_backend == BACKEND_POLARS:
    #         return self._fit_polars()  # type: ignore[call-arg]
    #     else:
    #         raise ValueError(f"Unsupported backend: {self.tf.dataframe_backend}")

    # def transform(self) -> Iterator[Dict[str, Dict[str, SupportedBackendDataFrame]]]:
    #     """Generate partitioned data slices for the dataset.

    #     This method yields the actual data slices corresponding to the partition indices generated by the `fit` method.
    #     The slices are returned as generic DataFrames, regardless of the backend (e.g., Pandas, Modin, or Polars).

    #     :return: Iterator yielding partitioned DataFrame slices.
    #     :rtype: Iterator[Dict[str, Dict[str, DataFrame]]]
    #     :raises ValueError: If an unsupported backend is encountered.

    #     Example Usage:
    #     --------------
    #     .. code-block:: python

    #         partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3)

    #         for partition_data in partitioner.transform():
    #             print(partition_data)

    #     Output Format:
    #     --------------
    #     Each yielded partition has the following structure:

    #     .. code-block:: python

    #         {
    #             'partition_1': {
    #                 'full': <DataFrame>,
    #                 'train': <DataFrame>,
    #                 'test': <DataFrame>,
    #                 'validation': <DataFrame>  # (Optional, if val_pct is provided)
    #             }
    #         }

    #     .. note::
    #        - This method transforms the dataset into partitioned slices based on indices created by `fit`.
    #        - Ensure the dataset is preprocessed properly to avoid errors during slicing.
    #        - The DataFrame format is agnostic to the backend.

    #     .. seealso::
    #        - :meth:`fit`: For generating the partition indices that are sliced in this method.
    #     """
    #     df = self.tf.get_data()  # Get the dataset from the TimeFrame

    #     # Call backend-specific transformation method
    #     if self.tf.dataframe_backend in [BACKEND_PANDAS, BACKEND_MODIN]:
    #         return self._transform_pandas_modin(df)  # type: ignore[call-arg]
    #     elif self.tf.dataframe_backend == BACKEND_POLARS:
    #         return self._transform_polars(df)  # type: ignore[call-arg]
    #     else:
    #         raise ValueError(f"Unsupported backend: {self.tf.dataframe_backend}")

    # def fit_transform(self) -> Iterator[Dict[str, Dict[str, SupportedBackendDataFrame]]]:
    #     """Fit and transform the dataset in a single step.

    #     This method combines the functionality of the `fit` and `transform` methods. It first generates partition indices
    #     using `fit`, and then returns the partitioned data slices using `transform`. The DataFrame format is backend-agnostic.

    #     :return: Iterator yielding partitioned DataFrame slices.
    #     :rtype: Iterator[Dict[str, Dict[str, DataFrame]]]
    #     :raises ValueError: If an unsupported backend is encountered.

    #     Example Usage:
    #     --------------
    #     .. code-block:: python

    #         partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3)

    #         for partition_data in partitioner.fit_transform():
    #             print(partition_data)

    #     Output Format:
    #     --------------
    #     Each yielded partition has the following structure:

    #     .. code-block:: python
    #     `
    #         {
    #             'partition_1': {
    #                 'full': <DataFrame>,
    #                 'train': <DataFrame>,
    #                 'test': <DataFrame>,
    #                 'validation': <DataFrame>  # (Optional, if val_pct is provided)
    #             }
    #         }

    #     .. note::
    #        - This method is a convenient way to generate partition indices and their corresponding data slices in one step.
    #        - Ensure that the dataset is preprocessed properly to avoid issues during partitioning.

    #     .. seealso::
    #        - :meth:`fit`: For generating partition indices.
    #        - :meth:`transform`: For generating the actual partitioned slices.
    #     """
    #     for partition_indices in self.fit():
    #         yield from self.transform()

    # def check_data(self, partition_index: Optional[int] = None) -> None:
    #     """Perform data checks on the entire dataset or a specific partition.

    #     This method performs validation checks on the dataset or a specific partition, ensuring that
    #     the sample size, feature-to-sample ratio, and class balance (if applicable) meet the expected criteria.

    #     If a partition index is provided, it checks only that partition; otherwise, it checks the entire dataset.

    #     :param partition_index: Index of the partition to check, or `None` to check the full dataset.
    #     :type partition_index: Optional[int]
    #     :raises ValueError: If the dataset or a partition fails validation checks.

    #     Example Usage:
    #     --------------
    #     .. code-block:: python

    #         partitioner = SlidingWindowPartitioner(tf=data_tf, window_size=5, stride=2, train_pct=0.7, test_pct=0.3)

    #         # Perform checks on the full dataset
    #         partitioner.check_data()

    #         # Perform checks on the first partition
    #         partitioner.check_data(partition_index=0)

    #     .. note::
    #        - This method ensures that the data's structure and integrity (sample size, feature ratio, class balance)
    #          meet expectations for further processing.
    #        - Ensure the dataset or partition is not empty to avoid runtime errors.
    #     """
    #     df = self.tf.get_data()  # Get the DataFrame (could be Pandas, Modin, or Polars)

    #     if partition_index is not None:
    #         partition = next(itertools.islice(self.fit(), partition_index, None))
    #         start, end = partition[f"partition_{partition_index + 1}"]["full"]

    #         # Slice the DataFrame based on its type (Pandas/Modin vs Polars)
    #         if isinstance(df, (pd.DataFrame, mpd.DataFrame)):
    #             df_to_check = df.iloc[start:end]
    #         elif isinstance(df, pl.DataFrame):
    #             df_to_check = df.slice(start, end - start)
    #         else:
    #             raise ValueError(f"Unsupported DataFrame type: {type(df)}")

    #         context = f"Partition {partition_index + 1}"
    #         min_samples = 100
    #     else:
    #         df_to_check = df
    #         context = "Full dataset"
    #         min_samples = 3000

    #     # Perform sample size, feature ratio, and class balance checks
    #     check_sample_size(
    #         df_to_check,
    #         backend=self.tf.dataframe_backend,
    #         min_samples=min_samples,
    #         max_samples=100000,
    #         enable_warnings=True,
    #     )
    #     check_feature_to_sample_ratio(
    #         df_to_check, backend=self.tf.dataframe_backend, max_ratio=0.2, enable_warnings=True
    #     )
    #     if self.tf.target_col:
    #         check_class_balance(
    #             df_to_check,
    #             target_col=self.tf.target_col,
    #             backend=self.tf.dataframe_backend,
    #             enable_warnings=True,
    #         )

    #     if self.verbose:
    #         print(f"{context} checks completed with warnings where applicable.")
