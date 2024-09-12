"""
temporalscope/temporal_model_trainer.py

This module implements the TemporalModelTrainer class, which provides functionality to
train machine learning models on data partitioned by temporal methods.
Users can pass their custom models or use a default lightweight model.

TemporalScope is Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from typing import Any, Dict, List, Optional, Protocol, Union

import lightgbm as lgb
import pandas as pd

from temporalscope.methods.base_temporal_partitioner import BaseTemporalPartitioner


class Fittable(Protocol):
    def fit(self, x: Any, y: Any) -> None:
        """Model Object should have a fit method."""
        ...


class TemporalModelTrainer:
    """
    Train models on temporally partitioned data.

    Users can specify a custom model or use the default LightGBM model.

    Parameters
    ----------
    partitioner : BaseTemporalPartitioner
        An instance of a class that inherits from BaseTemporalPartitioner.
    model : Fittable, optional
        A custom model with `fit` and `predict` methods. Defaults to LightGBM.
    model_params : Optional[Dict[str, Union[str, int, float]]], optional
        Parameters for the default model (LightGBM).
        Ignored if a custom model is provided.

    """

    def __init__(
        self,
        partitioner: BaseTemporalPartitioner,
        model: Optional[Fittable] = None,
        model_params: Optional[Dict[str, Union[str, int, float]]] = None,
    ):
        self.partitioner = partitioner
        self.model = model or self._initialize_default_model(model_params)

    def _initialize_default_model(
        self, model_params: Optional[Dict[str, Union[str, int, float]]]
    ) -> lgb.LGBMRegressor:
        """Initialize a default LightGBM model with specified or default parameters."""
        params = model_params or {
            "objective": "regression",
            "boosting_type": "gbdt",
            "metric": "rmse",
            "verbosity": -1,
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

        # TODO: Fix type hints for this method

    def train_model_on_phase(self, phase_data: pd.DataFrame) -> Any:
        """Train the model on the provided phase data."""
        X = phase_data.drop(columns=[self.partitioner.target])
        y = phase_data[self.partitioner.target]
        self.model.fit(X, y)
        return self.model