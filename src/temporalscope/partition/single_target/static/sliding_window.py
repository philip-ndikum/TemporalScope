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
Here's the markdown table:

| Aspect | Description |
|--------|-------------|
| Partial Temporal Ordering | Follows a universal model design; allows overlapping labels within partitions while leaving strict temporal ordering to the user. |
| Narwhals API | Leverages Narwhals backend for efficient operations. Users can switch between supported backends (e.g., Pandas, Polars, Modin) using core_utils. |
| Dataset Accessibility | Inspired by Dask and TensorFlow design. Provides hierarchical access via `partitions[index]["train"]`, `test`, `validation`, or `full` labels. |
| Lazy/Eager Execution | Narwhals backend supports lazy/eager evaluation; generator pattern ensures memory-efficient `fit` and `transform` operations. |
| Human-Centric Design | Combines human-readable labels (`train`, `test`) with indexing for scalable workflows, reducing cognitive overhead when handling large numbers of partitions. |
| Padding Control | Leaves padding decisions (e.g., zero-padding) to users while allowing configurable truncation. |

Visualization:
--------------
The table below illustrates how the sliding window mechanism works with overlapping
partitions. Each "X" represents a row included in the respective partition, based on
the configured window size and stride.


|     Time     | Partition 1 | Partition 2 | Partition 3 | Partition 4 | Partition 5 |
|--------------|-------------|-------------|-------------|-------------|-------------|
| 2021-01-01   | X           |             |             |             |             |
| 2021-01-02   | X           | X           |             |             |             |
| 2021-01-03   |             | X           | X           |             |             |
| 2021-01-04   |             |             | X           | X           |             |
| 2021-01-05   |             |             |             | X           | X           |
| 2021-01-06   |             |             |             |             | X           |


See Also
--------
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

# from typing import Iterator, Optional

# # Narwhals Imports
# import narwhals as nw

# from temporalscope.core.core_utils import MODE_SINGLE_TARGET, SupportedTemporalDataFrame
# from temporalscope.core.temporal_data_loader import TimeFrame
# from temporalscope.partition.base_protocol import TemporalPartitionerProtocol

# # Precision constant for floating-point comparisons
# DEFAULT_PRECISION = 1e-6


# class SlidingWindowPartitioner(TemporalPartitionerProtocol):
#     """Partition time series data into contiguous, non-overlapping windows.

#     The `SlidingWindowPartitioner` divides a dataset into a configurable number of partitions (`num_partitions`).
#     Optionally, a fixed `window_size` can be specified to define the size of each partition. If both `num_partitions`
#     and `window_size` are provided, `num_partitions` takes precedence, and `window_size` will be inferred.

#     The partitioner is explicitly designed for **single-target workflows**. It does **not support multi-target modes**,
#     and passing any unsupported mode will result in an error. The class supports train, test, and validation splits
#     within each partition and operates in a memory-efficient manner using lazy evaluation.

#     Setup Process:
#     --------------
#     - Default sorting by `time_col` ensures temporal ordering.
#     - Explicit `mode` validation restricts functionality to `MODE_SINGLE_TARGET`.

#     Key Features:
#     -------------
#     - Configurable `num_partitions` as the primary partitioning parameter.
#     - Optional `window_size` for defining partition sizes directly.
#     - Flexible `stride` for overlapping or gapped partitions.
#     - Train, test, and validation splits within each partition.
#     - Fully Narwhals-compatible for backend-agnostic operation.
#     - Supports lazy-loading with `fit` and `transform` methods for memory efficiency.

#     Example Usage:
#     --------------
#     **Lazy Evaluation:**

#     .. code-block:: python

#         from temporalscope.core.temporal_data_loader import TimeFrame
#         from temporalscope.partition.sliding_window import SlidingWindowPartitioner
#         from temporalscope.core.core_utils import MODE_SINGLE_TARGET

#         # Create a sample dataset
#         data_df = pd.DataFrame(
#             {
#                 "time": pd.date_range(start="2021-01-01", periods=6, freq="D"),
#                 "value": range(6),
#             }
#         )
#         data_tf = TimeFrame(data_df, time_col="time", target_col="value")

#         # Initialize partitioner
#         partitioner = SlidingWindowPartitioner(
#             tf=data_tf,
#             num_partitions=3,
#             train_pct=0.7,
#             test_pct=0.3,
#             mode=MODE_SINGLE_TARGET,
#         )

#         # Lazily generate partition indices
#         for partition_indices in partitioner.fit():
#             print(partition_indices)

#         # Lazily transform data slices
#         for partition_data in partitioner.transform():
#             print(partition_data)

#     **Eager Evaluation:**

#     .. code-block:: python

#         from temporalscope.core.temporal_data_loader import TimeFrame
#         from temporalscope.partition.sliding_window import SlidingWindowPartitioner
#         from temporalscope.core.core_utils import MODE_SINGLE_TARGET

#         # Create a sample dataset
#         data_df = pd.DataFrame(
#             {
#                 "time": pd.date_range(start="2021-01-01", periods=6, freq="D"),
#                 "value": range(6),
#             }
#         )
#         data_tf = TimeFrame(data_df, time_col="time", target_col="value")

#         # Initialize partitioner
#         partitioner = SlidingWindowPartitioner(
#             tf=data_tf,
#             num_partitions=3,
#             train_pct=0.7,
#             test_pct=0.3,
#             mode=MODE_SINGLE_TARGET,
#         )

#         # Perform fit and transform in one step
#         for partition_data in partitioner.fit_transform():
#             print(partition_data)
#     """

#     def __init__(
#         self,
#         tf: TimeFrame,
#         num_partitions: Optional[int] = None,
#         window_size: Optional[int] = None,
#         stride: Optional[int] = None,
#         train_pct: float = 0.7,
#         test_pct: Optional[float] = None,
#         val_pct: Optional[float] = 0.0,
#         truncate: bool = True,
#         mode: str = MODE_SINGLE_TARGET,
#         precision: float = DEFAULT_PRECISION,  # Added precision parameter
#         verbose: bool = False,
#     ):
#         """Initialize the SlidingWindowPartitioner with validation of input parameters.

#         :param tf: The `TimeFrame` object containing validated and sorted time series data.
#         :type tf: TimeFrame
#         :param num_partitions: Number of partitions to create. Required unless `window_size` is provided.
#         :type num_partitions: Optional[int]
#         :param window_size: Size of each partition. Optional if `num_partitions` is provided.
#         :type window_size: Optional[int]
#         :param stride: Number of rows to skip between consecutive partitions. Defaults to `window_size`.
#         :type stride: int, optional
#         :param train_pct: Fraction of each partition allocated for training. Required.
#         :type train_pct: float
#         :param test_pct: Fraction allocated for testing. Defaults to remaining data after training.
#         :type test_pct: Optional[float]
#         :param val_pct: Fraction allocated for validation. Defaults to zero.
#         :type val_pct: Optional[float]
#         :param truncate: Whether to truncate the last partition if it is smaller than the calculated size.
#         :type truncate: bool, optional
#         :param mode: Operational mode, defaults to `MODE_SINGLE_TARGET`. Only `MODE_SINGLE_TARGET` is supported.
#         :type mode: str
#         :param precision: Tolerance for floating-point imprecision when validating percentages. Default is 1e-6.
#         :type precision: float, optional
#         :param verbose: Print partitioning details if True. Default is False.
#         :type verbose: bool, optional

#         :raises ValueError:
#             - If both `num_partitions` and `window_size` are invalid or missing.
#             - If train, test, and validation percentages do not sum to 1.0.
#             - If `stride` is not positive or if required data is missing.
#             - If `mode` is invalid or unsupported.
#         """
#         # Validate mode
#         if mode != MODE_SINGLE_TARGET:
#             raise ValueError(f"Unsupported mode: {mode}. This partitioner supports only `MODE_SINGLE_TARGET`.")
#         self.mode = mode

#         # Assign precision
#         self.precision = precision

#         # Validate percentages
#         if not (0 <= train_pct <= 1):
#             raise ValueError("`train_pct` must be between 0 and 1.")
#         if test_pct is not None and not (0 <= test_pct <= 1):
#             raise ValueError("`test_pct` must be between 0 and 1.")
#         if val_pct is not None and not (0 <= val_pct <= 1):
#             raise ValueError("`val_pct` must be between 0 and 1.")
#         total_pct = train_pct + (test_pct or 0) + (val_pct or 0)
#         if not abs(total_pct - 1.0) < self.precision:  # Use precision for comparison
#             raise ValueError("Train, test, and validation percentages must sum to 1.0.")

#         # Validate `num_partitions` or `window_size`
#         if num_partitions is None and window_size is None:
#             raise ValueError("Either `num_partitions` or `window_size` must be specified.")
#         if num_partitions is not None and num_partitions <= 0:
#             raise ValueError("`num_partitions` must be a positive integer.")
#         if window_size is not None and window_size <= 0:
#             raise ValueError("`window_size` must be a positive integer.")
#         self.num_partitions = num_partitions
#         self.window_size = window_size

#         # Default stride to window_size if not provided
#         self.stride = stride or self.window_size

#         if self.stride <= 0:
#             raise ValueError("`stride` must be a positive integer.")

#         # Assign other parameters
#         self.tf = tf
#         self.train_pct = train_pct
#         self.test_pct = test_pct
#         self.val_pct = val_pct
#         self.truncate = truncate
#         self.verbose = verbose

#         # Finally store metadata after fit
#         self.metadata = None  # Store metadata after fit

#     def setup(self) -> None:
#         """Prepare and validate input parameters for partitioning.

#         This method validates all input parameters, determines the partitioning scheme
#         (`num_partitions` or `window_size`), and ensures that the configuration is consistent
#         with the dataset's cardinality. Verbose mode provides a summary of the configuration
#         in tabular form.

#         :raises ValueError:
#             - If both `num_partitions` and `window_size` are `None`.
#             - If any percentage is outside the range [0, 1].
#             - If train, test, and validation percentages do not sum to 1.0.
#             - If `num_partitions` or `window_size` are invalid for the dataset's cardinality.
#         """
#         # Step 1: Validate percentages
#         if not (0 <= self.train_pct <= 1):
#             raise ValueError("`train_pct` must be between 0 and 1.")
#         if self.test_pct is not None and not (0 <= self.test_pct <= 1):
#             raise ValueError("`test_pct` must be between 0 and 1.")
#         if self.val_pct is not None and not (0 <= self.val_pct <= 1):
#             raise ValueError("`val_pct` must be between 0 and 1.")

#         # Compute missing percentages
#         if self.test_pct is None and self.val_pct is None:
#             self.test_pct = 1.0 - self.train_pct
#             self.val_pct = 0.0
#         elif self.test_pct is not None and self.val_pct is None:
#             self.val_pct = 1.0 - self.train_pct - self.test_pct
#         elif self.test_pct is None and self.val_pct is not None:
#             self.test_pct = 1.0 - self.train_pct - self.val_pct

#         # Ensure percentages sum to 1.0
#         total_pct = self.train_pct + self.test_pct + self.val_pct
#         if not abs(total_pct - 1.0) < self.precision:
#             raise ValueError("Train, test, and validation percentages must sum to 1.0.")

#         # Step 2: Validate and set partition scheme
#         if self.num_partitions is None and self.window_size is None:
#             raise ValueError("Either `num_partitions` or `window_size` must be specified.")

#         total_rows = self.tf.df.select([nw.col(self.tf._time_col)]).collect().shape[0]

#         if self.num_partitions is not None:
#             if self.num_partitions <= 0:
#                 raise ValueError("`num_partitions` must be a positive integer.")
#             self.window_size = total_rows // self.num_partitions
#             self.PARTITION_SCHEME = "num_partitions"
#         elif self.window_size is not None:
#             if self.window_size <= 0:
#                 raise ValueError("`window_size` must be a positive integer.")
#             self.num_partitions = (total_rows - self.window_size) // (self.stride or self.window_size) + 1
#             self.PARTITION_SCHEME = "window_size"

#         # Ensure valid cardinality
#         if self.num_partitions > total_rows:
#             raise ValueError(f"Insufficient rows ({total_rows}) for `num_partitions={self.num_partitions}`.")
#         if self.window_size > total_rows:
#             raise ValueError(f"Insufficient rows ({total_rows}) for `window_size={self.window_size}`.")

#         # Step 3: Set default stride if not provided
#         self.stride = self.stride or self.window_size
#         if self.stride <= 0:
#             raise ValueError("`stride` must be a positive integer.")

#         # Step 4: Verbose output
#         if self.verbose:
#             self._print_partition_table(total_rows)

#         # Reset metadata
#         self.metadata = None

#     @nw.narwhalify
#     def fit(self, df: SupportedTemporalDataFrame) -> None:
#         """Generate partition metadata using time column boundaries.

#         This method computes partition start and end boundaries based on the time column,
#         window size, and stride. These boundaries are stored as metadata for lazy partitioning.

#         :param df: Input DataFrame to compute partition metadata.
#         :type df: SupportedTemporalDataFrame

#         :raises ValueError:
#             - If the total rows in the DataFrame are insufficient for the requested partitions.
#             - If metadata already exists, indicating `setup` has not been completed.

#         .. note::
#             - Uses `select()` to retrieve time column values.
#             - Calculates boundaries using time-based filtering.
#             - Compatible with Narwhals backends (Pandas, Polars, PyArrow).
#         """
#         time_col_values = df.select([nw.col(self.tf._time_col)]).collect().to_numpy().flatten()
#         total_rows = len(time_col_values)

#         if total_rows < (self.window_size or total_rows // self.num_partitions):
#             raise ValueError(f"Total rows ({total_rows}) are insufficient for the requested partitions.")

#         window_size = self.window_size or total_rows // self.num_partitions
#         stride = self.stride or window_size

#         partitions = []
#         for i in range(0, total_rows - window_size + 1, stride):
#             start_time = time_col_values[i]
#             end_time = time_col_values[i + window_size - 1]
#             partitions.append(
#                 {
#                     "train": (start_time, start_time + (end_time - start_time) * self.train_pct),
#                     "test": (start_time + (end_time - start_time) * self.train_pct, end_time),
#                     "validation": None
#                     if not self.val_pct
#                     else (end_time, end_time + (end_time - start_time) * self.val_pct),
#                 }
#             )

#         self.metadata = {
#             "partitions": partitions,
#             "window_size": window_size,
#             "stride": stride,
#             "total_rows": total_rows,
#         }

#         if self.verbose:
#             print(f"Fit complete with {len(partitions)} partitions, window size {window_size}, stride {stride}.")

#     @nw.narwhalify
#     def transform(self, df: SupportedTemporalDataFrame) -> Iterator[dict[str, SupportedTemporalDataFrame]]:
#         """Apply time-based partitioning using metadata.

#         This method filters the DataFrame based on the time column boundaries stored
#         in the metadata. Partitions are lazily yielded for memory efficiency.

#         :param df: Input DataFrame to partition.
#         :type df: SupportedTemporalDataFrame
#         :return: An iterator yielding dictionaries of DataFrame partitions.
#         :rtype: Iterator[dict[str, SupportedTemporalDataFrame]]

#         :raises RuntimeError: If `fit` has not been called prior to `transform`.

#         .. note::
#             - Leverages `filter()` for time-based slicing.
#             - Maintains compatibility with all Narwhals-supported backends.
#             - Lazy evaluation minimizes memory usage.
#         """
#         if self.metadata is None:
#             raise RuntimeError("Call `fit` before `transform`.")

#         for partition in self.metadata["partitions"]:
#             yield {
#                 "train": df.filter(
#                     (nw.col(self.tf._time_col) >= partition["train"][0])
#                     & (nw.col(self.tf._time_col) < partition["train"][1])
#                 ),
#                 "test": df.filter(
#                     (nw.col(self.tf._time_col) >= partition["test"][0])
#                     & (nw.col(self.tf._time_col) < partition["test"][1])
#                 ),
#                 "validation": None
#                 if not partition["validation"]
#                 else df.filter(
#                     (nw.col(self.tf._time_col) >= partition["validation"][0])
#                     & (nw.col(self.tf._time_col) < partition["validation"][1])
#                 ),
#             }

#     @property
#     def partitions(self) -> list[dict[str, SupportedTemporalDataFrame]]:
#         """Provide indexed access to partitioned data.

#         This property allows users to access partitioned data slices by index, slice, or iteration.
#         Partitions are generated lazily and cached for efficient reuse.

#         :return: A list of dictionaries containing train, test, and validation DataFrame slices.
#         :rtype: list[dict[str, SupportedTemporalDataFrame]]

#         :raises RuntimeError: If `fit` has not been called prior to accessing partitions.

#         Example Usage:
#         --------------
#         .. code-block:: python

#             partition = partitioner.partitions[0]
#             print(partition["train"])
#         """
#         if self.metadata is None:
#             raise RuntimeError("Call `fit` before accessing `partitions`.")

#         # Generate all partitions lazily
#         return list(self.transform(self.tf.df))
