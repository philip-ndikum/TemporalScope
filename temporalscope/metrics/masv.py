"""temporalscope/methods/mean_absolute_shap.py

This module implements the Mean Absolute SHAP (MASV) analysis, which evaluates
temporal feature importance across different operational phases of a system.
"""

from shap import Explainer
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from temporalscope.partition.base_temporal_partioner import BaseTemporalPartitioner


def calculate_masv(
    model: Callable, data: pd.DataFrame, partitioner: BaseTemporalPartitioner
) -> Dict[str, List[float]]:
    """
    Calculate Mean Absolute SHAP Values (MASV) for temporal feature importance across partitions.

    :param model: Trained machine learning model.
    :param data: The dataset used for analysis. Rows represent samples, and columns represent features.
    :param partitioner: The partitioner object to divide the data into phases.
    :return: A dictionary where keys are feature names and values are lists of MASV scores across partitions.

    .. note::
        The MASV is calculated as:

        .. math::

            MASV = \\frac{1}{n} \\sum |SHAP_i|

        Where `SHAP_i` is the SHAP value of feature `i` in a given phase, and `n` is the number of samples in that phase.

    References:
        - Alomari, Y., & Ando, M. (2024). SHAP-based insights for aerospace
          PHM: Temporal feature importance, dependencies, robustness, and
          interaction analysis. ELTE Eötvös Loránd University, Faculty of
          Informatics, Institute of Computer Science, Budapest, Hungary.
    """

    # Initialize the SHAP explainer with the provided model
    explainer = Explainer(model)

    # Get the partitions from the partitioner
    partitions = partitioner.get_partitions()

    # Initialize an empty dictionary to store MASV scores
    masv_dict = {feature: [] for feature in data.columns}

    # Iterate over each partition
    for partition in partitions:
        # Extract data for the current partition
        phase_data = data.iloc[partition[0] : partition[1]]

        # Calculate SHAP values for the partition data
        shap_values = explainer(phase_data)

        # Compute the mean absolute SHAP values for the partition
        masv_phase = np.mean(np.abs(shap_values.values), axis=0)

        # Update the dictionary with MASV scores for each feature
        for i, feature in enumerate(data.columns):
            masv_dict[feature].append(masv_phase[i])

    # Return the dictionary containing MASV scores for each feature across partitions
    return masv_dict
