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
the required methods to comply with this protocol. Currently, only single-target
workflows (scalar targets) are supported. Multi-target workflows (sequence or tensor
targets) are planned for future releases. This protocol is designed to be flexible
enough to accommodate both modes when multi-target support is added.

Partitioning for modern XAI Time-Series Pipelines:
--------------------------------------------------
Partitioning is foundational to modern time-series workflows. It ensures computational efficiency,
robust validation, and interpretable insights. Key use cases include:

+----------------------------+-----------------------------------------------------------------------------------+
| Aspect                     | Details                                                                           |
+----------------------------+-----------------------------------------------------------------------------------+
| Temporal Explainability    | Facilitates feature importance analyses by segmenting data for localized          |
|                            | SHAP/WindowSHAP metrics.                                                          |
+----------------------------+-----------------------------------------------------------------------------------+
| Robust Evaluation          | Respects temporal ordering in train-test splits, critical for time-series         |
|                            | generalization.                                                                   |
+----------------------------+-----------------------------------------------------------------------------------+
| Scalability and Efficiency | Supports sliding windows, expanding windows, and fixed partitions with            |
|                            | lazy-loading and backend compatibility for large-scale datasets.                  |
+----------------------------+-----------------------------------------------------------------------------------+
| Workflow Flexibility       | Supports both single-target and multi-target modes, enabling DataFrame            |
|                            | operations and deep learning pipelines through flexible partitioning methods.     |
+----------------------------+-----------------------------------------------------------------------------------+

Core Functionality:
-------------------
The protocol defines four mandatory methods, ensuring a strict and consistent lifecycle across all partitioning implementations.
Each method has a clear purpose and aligns with the goals of efficient partitioning:

+-----------------+-----------------------------------------------------------------------------------+
| Method          | Description                                                                       |
+-----------------+-----------------------------------------------------------------------------------+
| setup           | Prepares and validates input data, ensuring compatibility with the chosen         |
|                 | workflow (e.g., backend conversions, deduplication, parameter checks).            |
+-----------------+-----------------------------------------------------------------------------------+
| fit             | Generates partition indices (row ranges) for datasets, supporting sliding         |
|                 | windows, fixed-length, or expanding partitions.                                   |
+-----------------+-----------------------------------------------------------------------------------+
| transform       | Applies the partition indices to retrieve specific data slices, ensuring          |
|                 | memory-efficient operation using lazy evaluation techniques.                      |
+-----------------+-----------------------------------------------------------------------------------+
| fit_transform   | Combines `fit` and `transform` for eager workflows, directly producing            |
|                 | partitioned data slices.                                                          |
+-----------------+-----------------------------------------------------------------------------------+

Workflow Modes:
---------------
The protocol supports two primary modes:

1. Single-Target (DataFrame-Centric):
   - Operations focus on Narwhals-backed DataFrames (Pandas, Polars, or Modin).
   - Slices are returned in DataFrame formats, preserving metadata.

2. Multi-Target (Tensor/Dataset-Centric):
   - Designed for deep learning workflows (e.g., PyTorch, TensorFlow).
   - Handles transformations from DataFrame to tensor or dataset formats.
   - Ensures compatibility with sequence or tensor-target models.

Future Plans:
-------------
The protocol is designed for extensibility, ensuring advanced workflows like multi-modal models, cross-frequency partitioning,
or custom padding strategies can be integrated seamlessly.

.. seealso::

    1. Nayebi, A., Tipirneni, S., Reddy, C. K., et al. (2024). WindowSHAP: An efficient framework for
       explaining time-series classifiers based on Shapley values. Journal of Biomedical Informatics.
       DOI:10.1016/j.jbi.2023.104438.
    2. Gu, X., See, K. W., Wang, Y., et al. (2021). The sliding window and SHAP theory—an improved system
       with a long short-term memory network model for state of charge prediction in electric vehicles.
       Energies, 14(12), 3692. DOI:10.3390/en14123692.
    3. Van Ness, M., Shen, H., Wang, H., et al. (2023). Cross-Frequency Time Series Meta-Forecasting.
       arXiv preprint arXiv:2302.02077.
"""

# Ignore given that this is a protocol and does not require implementation.
# coverage: ignore

from typing import Any, Dict, Iterator, Protocol


class TemporalPartitionerProtocol(Protocol):
    """Protocol for temporal partitioning methods.

    This protocol defines the lifecycle for partitioning workflows, supporting both
    single-target (dataframe-centric) and multi-target (tensor/dataset-centric) use cases.
    """

    def setup(self) -> None:  # pragma: no cover
        """Prepare and validate input data for partitioning.

        This method performs preprocessing and ensures the data is compatible
        with the specific workflow. Example tasks include:
        - Sorting and deduplication for DataFrame workflows.
        - Conversion to tensors or datasets for multi-target workflows.
        - Validation of partitioning parameters (e.g., `num_partitions`, `stride`).

        This step ensures consistency across partitioning methods and minimizes
        runtime errors in subsequent stages.

        .. note::
            This method should be idempotent and isolated. While optional for
            end-users, implementations must ensure it is executed internally
            before partitioning begins.

        :raises ValueError: If any required input or parameter is invalid.
        """
        pass

    def fit(self) -> Iterator[Dict[str, Any]]:  # pragma: no cover
        """Compute partition indices for slicing.

        This method generates partition indices based on partitioning parameters
        such as `num_partitions`, `window_size`, and `stride`. It utilizes a lazy
        generator pattern to ensure memory efficiency, especially for large datasets.

        :return: Generator yielding partition indices structured as dictionaries.
        :rtype: Iterator[Dict[str, Any]]

        .. note::
            This method does not perform slicing; it only computes and returns indices.
        """
        pass

    def transform(self) -> Iterator[Dict[str, Any]]:
        """Retrieve data slices using computed indices.

        This method slices the data based on indices generated by `fit`. It ensures
        memory efficiency through lazy evaluation and supports various output formats
        depending on the workflow mode (e.g., DataFrame slices, tensors, or datasets).

        :return: Generator yielding dictionaries containing partitioned data slices.
        :rtype: Iterator[Dict[str, Any]]

        :raises ValueError: If `fit` has not been called prior to `transform`.
        """
        pass

    def fit_transform(self) -> Iterator[Dict[str, Any]]:  # pragma: no cover
        """Combine `fit` and `transform` for eager execution.

        This method computes partition indices and retrieves data slices in a
        single step. It is ideal for workflows requiring immediate access to
        partitioned data without intermediate steps.

        :return: Generator yielding dictionaries containing partitioned data slices.
        :rtype: Iterator[Dict[str, Any]]
        """
        pass
