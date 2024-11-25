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
the required methods to comply with this protocol. Currently, only single-step
workflows (scalar targets) are supported. Multi-step workflows (sequence targets)
are planned for future releases, and the protocol is designed to be flexible
enough to accommodate both modes when multi-step support is added.

Partitioning in Modern Time-Series Analysis:
--------------------------------------------
Partitioning is foundational to modern time-series workflows. It ensures computational
efficiency, robust validation, and interpretable insights. Key use cases include:

    +----------------------------+----------------------------------------------------------------------------------+
    | Aspect                     | Details                                                                          |
    +----------------------------+----------------------------------------------------------------------------------+
    | Temporal Explainability    | Facilitates feature importance analyses by segmenting data for localized         |
    |                            | SHAP/WindowSHAP metrics.                                                         |
    +----------------------------+----------------------------------------------------------------------------------+
    | Robust Evaluation          | Respects temporal ordering in train-test splits, critical for time-series        |
    |                            | generalization.                                                                  |
    +----------------------------+----------------------------------------------------------------------------------+
    | Scalability and Efficiency | Supports sliding windows, expanding windows, and fixed partitions with           |
    |                            | lazy-loading and backend compatibility for large-scale datasets.                 |
    +----------------------------+----------------------------------------------------------------------------------+
    | Workflow Flexibility       | Supports both single-step and multi-step modes, enabling DataFrame operations    |
    |                            | and deep learning pipelines through flexible data structures.                    |
    +----------------------------+----------------------------------------------------------------------------------+

Core Functionality:
-------------------
The protocol consists of three main methods. fit and transform are mandatory, while check_data is optional for additional validation.

    +-------------+----------------------------------------------------------------------------------+--------------+
    | Method      | Description                                                                      | Required     |
    +-------------+----------------------------------------------------------------------------------+--------------+
    | fit         | Generates partition indices (row ranges) for datasets, supporting sliding        | Yes          |
    |             | windows, fixed-length, or expanding partitions.                                  |              |
    +-------------+----------------------------------------------------------------------------------+--------------+
    | transform   | Applies the partition indices to retrieve specific data slices, ensuring         | Yes          |
    |             | memory-efficient operation.                                                      |              |
    +-------------+----------------------------------------------------------------------------------+--------------+
    | check_data  | Validates input data to ensure required columns exist and are non-null.          | No           |
    |             | Optional since TimeFrame already guarantees clean data.                          |              |
    +-------------+----------------------------------------------------------------------------------+--------------+

.. seealso::

    1. Nayebi, A., Tipirneni, S., Reddy, C. K., et al. (2024). WindowSHAP: An efficient framework for
       explaining time-series classifiers based on Shapley values. Journal of Biomedical Informatics.
       DOI:10.1016/j.jbi.2023.104438.
    2. Gu, X., See, K. W., Wang, Y., et al. (2021). The sliding window and SHAP theory—an improved system
       with a long short-term memory network model for state of charge prediction in electric vehicles.
       Energies, 14(12), 3692. DOI:10.3390/en14123692.
    3. Van Ness, M., Shen, H., Wang, H., et al. (2023). Cross-Frequency Time Series Meta-Forecasting.
       arXiv preprint arXiv:2302.02077.

.. note::
    - Clean Data Guarantee: TimeFrame ensures all data is validated, sorted, and properly typed before partitioning.
      This includes numeric features, proper time column types, and null value checks.
    - Workflow Modes: Supports both single-step (scalar targets) and multi-step (sequence targets) modes:
      - Single-step: Traditional DataFrame operations with scalar targets
      - Multi-step: Deep learning workflows with sequence targets, potentially using PyTorch/TensorFlow formats
    - Data Flexibility: While TimeFrame loads data as DataFrames initially, the protocol supports transformation
      to other formats (tensors, datasets) based on the workflow mode.
    - Future plans: Support for multi-modal models and advanced workflows is a key design priority, ensuring this protocol
      remains adaptable to diverse datasets and state-of-the-art methods.
"""

from typing import Any, Dict, Iterator, Protocol, Tuple, Union

from temporalscope.core.core_utils import SupportedTemporalDataFrame
from temporalscope.core.temporal_data_loader import TimeFrame


class TemporalPartitionerProtocol(Protocol):
    """Protocol for temporal partitioning methods.

    The `TemporalPartitionerProtocol` operates on a `TimeFrame` object and provides core
    functionality for retrieving partition indices and data. Implementing classes must
    provide partitioning logic and optionally perform data validation checks, with a
    strong emphasis on memory efficiency through lazy-loading techniques.

    :ivar tf: The `TimeFrame` object containing pre-validated, sorted time series data.
             TimeFrame guarantees:
             - All features are numeric (except time column)
             - Time column is properly typed and sorted
             - No null values
             - Metadata preservation regardless of data format
    :vartype tf: TimeFrame
    :ivar df: The data to be partitioned. Format depends on the workflow mode:
             - Single-step mode: Typically DataFrame with scalar targets
             - Multi-step mode: Could be DataFrame, tensor, or dataset with sequence targets
    :vartype df: Any
    :ivar enable_warnings: Whether to enable warnings during partition validation.
    :vartype enable_warnings: bool

    .. note::
        The partitions returned by each partitioning method will always include a
        "full" partition with index ranges. The "train", "test", and "validation"
        partitions are supported, and at least "train" and "test" must be defined
        for logical consistency. To manage large datasets efficiently, implementations
        should focus on generating indices lazily to reduce memory footprint.

        The protocol is designed to be flexible, supporting both DataFrame operations
        and deep learning workflows through TimeFrame's metadata management.
    """

    tf: TimeFrame
    df: Union[SupportedTemporalDataFrame, Any]
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

            The data being partitioned is guaranteed to be clean and properly typed
            by TimeFrame, so implementations can focus on partitioning logic without
            additional data validation.

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
        Dict[str, Dict[str, Any]],
        Iterator[Dict[str, Dict[str, Any]]],
    ]:
        """Return the data for each partition.

        This method returns the data slices for each partition based on the
        partition indices generated by the `fit` method. The format of the data
        depends on the workflow mode (single-step vs multi-step).

        :return: A dictionary containing the data slices for each partition,
                 or an iterator over them.
        :rtype: Union[Dict[str, Dict[str, Any]], Iterator[Dict[str, Dict[str, Any]]]]

        .. note::
            This method returns the actual data slices for each partition
            based on the indices generated by `fit`. The returned structure
            mirrors the same dictionary format but contains actual data
            instead of index ranges.

            The data format depends on the workflow mode:
            - Single-step: Typically DataFrame slices with scalar targets
            - Multi-step: Could be DataFrame slices, tensors, or datasets with sequence targets

            TimeFrame maintains metadata (column names, target info) regardless of format.

            The transform method should continue to optimize for memory
            efficiency by using the pre-calculated lazy indices to access
            only the necessary data.

        :example:

            .. code-block:: python

                # Single-step mode (DataFrame example)
                {
                    "partition_1": {
                        "full": DataFrame(...),
                        "train": DataFrame(...),
                        "test": DataFrame(...),
                        "validation": None,
                    }
                }

                # Multi-step mode (PyTorch example)
                {
                    "partition_1": {
                        "full": TensorDataset(...),
                        "train": TensorDataset(...),
                        "test": TensorDataset(...),
                        "validation": None,
                    }
                }
        """
        pass

    def check_data(self) -> None:
        """Perform data validation checks.

        This method is optional since TimeFrame already guarantees clean, validated data.
        Implementing classes may provide additional validation specific to their partitioning
        strategy, such as:
        - Ensuring sufficient sample size for the chosen window/partition sizes
        - Checking for appropriate window overlaps
        - Validating feature counts or ratios
        - Custom domain-specific validation
        """
        pass

    def get_partition_data(self) -> Any:
        """Return the partitioned data."""
        pass
