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

"""TemporalScope/src/temporalscope/partition/single_target/static/__init__.py.

This module defines the namespace for static partitioning algorithms for single-target workflows in TemporalScope.
Static partitioning refers to user-defined, fixed, and deterministic strategies, contrasting with dynamic
(algorithmic or data-driven) methods. These methods are designed to work seamlessly with TimeFrame objects and
support backend-agnostic operations via the Narwhals ecosystem.

Static Partitioning:
--------------------
Static partitioning encompasses human-defined techniques, where partitions are specified based on domain knowledge
or explicit rules. These methods are suited for:
- Sliding Window Partitioning: Dividing data into fixed-size, non-overlapping or overlapping partitions.
- Expanding Window Partitioning: Creating sequentially increasing partitions for cumulative analysis.

Single-Target Workflows:
------------------------
This namespace supports DataFrame-centric workflows for scalar target models. Multi-target workflows, as described
in the TemporalPartitionerProtocol, will require further extensions to ensure compatibility with TensorFlow/PyTorch
datasets and sequence-target workflows.

Extensibility:
--------------
TemporalScope provides the foundation for users to implement or customize their own static partitioning strategies.
This flexibility ensures the framework remains adaptable to diverse requirements and emerging techniques in
time-series analysis.

Notes
-----
Users are encouraged to leverage the TemporalPartitionerProtocol for building custom static partitioning workflows
and refer to the foundational literature on partitioning techniques for guidance.

See Also
--------
1. Shah, A., DePavia, A., Hudson, N., Foster, I., & Stevens, R. (2024).
   Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning.
   *arXiv preprint arXiv:2406.06348.*
2. Nodoushan, A. N. (2023). Interpretability of Deep Learning Models for Time-Series Clinical Data.
   (Doctoral dissertation, The University of Arizona).
3. Saarela, M., & Podgorelec, V. (2024). Recent Applications of Explainable AI (XAI): A Systematic Literature Review.
   *Applied Sciences, 14(19), 8884.*
4. Nayebi, A., Tipirneni, S., Reddy, C. K., Foreman, B., & Subbian, V. (2023).
   WindowSHAP: An efficient framework for explaining time-series classifiers based on Shapley values.
   *Journal of Biomedical Informatics, 144, 104438.*
"""
