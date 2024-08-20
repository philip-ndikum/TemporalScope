"""
This module implements the Mean Absolute SHAP (MASV) analysis, which evaluates
temporal feature importance across different operational phases of a system.
"""

from shap import Explainer
import numpy as np
import pandas as pd
from typing import Callable


def calculate_masv(
    model: Callable, data: pd.DataFrame, phases: list[tuple[int, int]]
) -> dict[str, list[float]]:
    """
    Calculate Mean Absolute SHAP Values (MASV) for temporal feature importance.

    :param model: Trained machine learning model.
    :param data: The dataset used for analysis. Rows represent samples, and
        columns represent features.
    :param phases: A list of tuples, where each tuple represents the start and
        end of a phase (time window).
    :return: A dictionary where keys are feature names and values are lists of
        MASV scores across phases.

    .. note::
        The MASV is calculated as:

        .. math::

            MASV = \frac{1}{n} \sum |SHAP_i|

        Where `SHAP_i` is the SHAP value of feature `i` in a given phase, and
        `n` is the number of samples in that phase.

    .. references::
        - Alomari, Y., & Ando, M. (2024). SHAP-based insights for aerospace
          PHM: Temporal feature importance, dependencies, robustness, and
          interaction analysis. ELTE Eötvös Loránd University, Faculty of
          Informatics, Institute of Computer Science, Budapest, Hungary.
    """

    # Initialize the SHAP explainer with the provided model
    explainer = Explainer(model)

    # Calculate SHAP values for the entire dataset
    shap_values = explainer(data)

    # Initialize an empty dictionary to store MASV scores
    masv_dict = {}

    # Iterate over each phase
    for phase in phases:
        start, end = phase

        # Extract data for the current phase using slicing
        phase_data = data.iloc[start:end]

        # Calculate SHAP values for the phase data
        shap_vals = explainer.shap_values(phase_data)

        # Compute the mean absolute SHAP values for the phase
        masv_phase = np.mean(np.abs(shap_vals), axis=0)

        # Update the dictionary with MASV scores for each feature
        for i, feature in enumerate(data.columns):
            if feature not in masv_dict:
                masv_dict[feature] = []
            masv_dict[feature].append(masv_phase[i])

    # Return the dictionary containing MASV scores for each feature across phases
    return masv_dict
