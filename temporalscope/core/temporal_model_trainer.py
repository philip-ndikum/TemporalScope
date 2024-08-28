""" temporalscope/temporal_model_trainer.py

This module implements the TemporalModelTrainer class, which provides functionality to train machine learning models
on data partitioned by temporal methods. Users can pass their custom models or use a default lightweight model.
"""

from typing import Callable, Optional, List, Dict, Union
import pandas as pd
import numpy as np
import lightgbm as lgb
from temporalscope.methods.base_temporal_partitioner import BaseTemporalPartitioner


class TemporalModelTrainer:
    """ Trains models on temporally partitioned data. Users can specify a custom model 
    or use the default LightGBM model.

    :param partitioner: An instance of a class that inherits from BaseTemporalPartitioner.
    :type partitioner: BaseTemporalPartitioner
    :param model: Optional. A custom model with `fit` and `predict` methods. Defaults to LightGBM.
    :type model: Optional[Callable]
    :param model_params: Optional. Parameters for the default model (LightGBM). Ignored if a custom model is provided.
    :type model_params: Optional[Dict[str, Union[str, int, float]]]
    """

    def __init__(
        self,
        partitioner: BaseTemporalPartitioner,
        model: Optional[Callable] = None,
        model_params: Optional[Dict[str, Union[str, int, float]]] = None
    ):
        self.partitioner = partitioner
        self.model = model or self._initialize_default_model(model_params)

    def _initialize_default_model(self, model_params: Optional[Dict[str, Union[str, int, float]]]):
        """Initialize a default LightGBM model with specified or default parameters."""
        params = model_params or {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'verbosity': -1,
        }
        return lgb.LGBMRegressor(**params)

    def train_and_evaluate(self) -> Dict[str, List[float]]:
        """
        Train the model on each temporal partition and return predictions.

        :return: Dictionary containing predictions for each partition.
        :rtype: Dict[str, List[float]]
        """
        partitioned_data = self.partitioner.get_partitioned_data()
        phase_predictions = {}

        for i, phase_data in enumerate(partitioned_data):
            trained_model = self.train_model_on_phase(phase_data)
            X_phase = phase_data.drop(columns=[self.partitioner.target])
            phase_predictions[f"Phase {i}"] = trained_model.predict(X_phase).tolist()

        return phase_predictions

    def train_model_on_phase(self, phase_data: pd.DataFrame):
        """Train the model on the provided phase data."""
        X = phase_data.drop(columns=[self.partitioner.target])
        y = phase_data[self.partitioner.target]
        self.model.fit(X, y)
        return self.model

