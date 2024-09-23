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
"""Mean Absolute SHAP (MASV) analysis for temporal feature importance.

This module implements the Mean Absolute SHAP (MASV) analysis, which evaluates temporal feature importance across
different operational phases of a system.

The MASV metric provides insights into how feature importance varies over time or across different operational phases,
helping to identify key factors influencing system behavior at different stages.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from shap import Explainer

from temporalscope.partition.base_protocol import TemporalPartitionerProtocol


def calculate_masv(
    model: Callable, data: pd.DataFrame, partitioner: TemporalPartitionerProtocol
) -> dict[str, list[float]]:
    r"""Calculate Mean Absolute SHAP Values (MASV).

    Calculate MASV for temporal feature importance across partitions.

    :param model: Trained machine learning model.
    :type model: Callable

    :param data: The dataset used for analysis. Rows represent samples, and columns
        represent features.
    :type data: pd.DataFrame

    :param partitioner: The partitioner object to divide the data into phases.
    :type partitioner: TemporalPartitionerProtocol

    :return: A dictionary where keys are feature names and values are lists of
        MASV scores across partitions.
    :rtype: Dict[str, List[float]]

    .. note::
        The MASV is calculated as:

        .. math::

            MASV = \\frac{1}{n} \\sum |SHAP_i|

        Where `SHAP_i` is the SHAP value of feature `i` in a given phase, and `n` is the
        number of samples in that phase.

    .. seealso::
        Alomari, Y., & Ando, M. (2024). SHAP-based insights for aerospace
        PHM: Temporal feature importance, dependencies, robustness, and
        interaction analysis.

        For more information on SHAP values and their applications, refer to
        the `SHAP documentation <https://shap.readthedocs.io>`_.
    """
    # Initialize the SHAP explainer with the provided model
    explainer = Explainer(model)

    # Get the partitioned data from the partitioner
    partitions = partitioner.get_partition_data()

    # Initialize an empty dictionary to store MASV scores
    masv_dict: dict[str, list[float]] = {feature: [] for feature in data.columns}

    # Iterate over each partition
    for partition_data in partitions.values():
        # Extract the training data for the current partition
        phase_data = partition_data["train"]  # Assuming we're calculating MASV on the 'train' partition

        # Calculate SHAP values for the partition data
        shap_values = explainer(phase_data)

        # Compute the mean absolute SHAP values for the partition
        masv_phase = np.mean(np.abs(shap_values.values), axis=0)

        # Update the dictionary with MASV scores for each feature
        for i, feature in enumerate(data.columns):
            masv_dict[feature].append(masv_phase[i])

    # Return the dictionary containing MASV scores for each feature across partitions
    return masv_dict
