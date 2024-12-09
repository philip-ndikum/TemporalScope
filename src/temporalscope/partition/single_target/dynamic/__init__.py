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

"""TemporalScope/src/temporalscope/partition/single_target/dynamic/__init__.py.

This module defines the namespace for dynamic partitioning algorithms for single-target workflows in TemporalScope.
Dynamic partitioning refers to algorithmic or data-driven strategies, contrasting with static (human-defined) methods.
These approaches dynamically adjust partitioning based on data properties, causal relationships, or optimization techniques.

Dynamic Partitioning:
---------------------
Dynamic partitioning encompasses adaptive techniques where partitions are derived through algorithms rather than fixed
rules. These methods are suited for:
- Causal Discovery: Partitioning based on causal relationships in high-dimensional datasets.
- SHAP/Explainability-Driven Partitioning: Optimizing partitions to maximize insights from model explainability methods.
- Adaptive Strategies: Adjusting partition sizes or boundaries based on data drift, feature importance, or temporal changes.

Single-Target Workflows:
------------------------
This namespace supports DataFrame-centric workflows for scalar target models. Multi-target workflows, as described in
the TemporalPartitionerProtocol, will require further extensions to ensure compatibility with TensorFlow/PyTorch
datasets and sequence-target workflows.

Extensibility:
--------------
TemporalScope provides the foundation for users to implement their own dynamic partitioning algorithms. While no
dynamic algorithms are implemented in this module at this stage, the flexible architecture of TemporalScope allows
users to integrate bespoke methods tailored to their specific domain requirements.

.. note::

    Users are encouraged to leverage the TemporalPartitionerProtocol for building custom dynamic partitioning workflows
    and refer to the foundational literature on dynamic partitioning techniques for guidance.

.. seealso::

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
