# """Implements the `TemporalModelTrainer` for training on temporally partitioned data.

# This module provides functionality to train machine learning models on data
# partitioned by temporal methods. Users can pass their custom models or use a
# default lightweight model (LightGBM).
# """

# from typing import Any, Dict, List, Optional, Protocol, Union

# import lightgbm as lgb
# import pandas as pd

# from temporalscope.partition.base import BaseTemporalPartitioner

# TODO: Update class

# class Fittable(Protocol):
#     """Protocol for objects with `fit` method."""

#     def fit(self, x: Any, y: Any) -> None:
#         """Model Object should have a fit method."""
#         ...


# class TemporalModelTrainer:
#     """Train models on temporally partitioned data.

#     Users can specify a custom model or use the default LightGBM model.

#     :param partitioner: An instance of a class that inherits from
#                         `BaseTemporalPartitioner`.
#     :type partitioner: BaseTemporalPartitioner
#     :param model: A custom model with `fit` and `predict` methods. Defaults to LightGBM.
#     :type model: Fittable, optional
#     :param model_params: Parameters for the default model (LightGBM).
#                          Ignored if a custom model is provided.
#     :type model_params: dict, optional
#     """

#     def __init__(
#         self,
#         partitioner: BaseTemporalPartitioner,
#         model: Optional[Fittable] = None,
#         model_params: Optional[Dict[str, Union[str, int, float]]] = None,
#     ):
#         """Initialize the TemporalModelTrainer with a partitioner and model.

#         param partitioner: An instance of a class that inherits from
#                             BaseTemporalPartitioner.
#         type partitioner: BaseTemporalPartitioner
#         param model: A custom model with fit and predict methods. Defaults to LightGBM.
#         type model: Fittable, optional
#         param model_params: Parameters for the default model (LightGBM).
#                              Ignored if a custom model is provided.
#         type model_params: dict, optional
#         """
#         self.partitioner = partitioner
#         self.model = model or self._initialize_default_model(model_params)

#     def _initialize_default_model(
#         self, model_params: Optional[Dict[str, Union[str, int, float]]]
#     ) -> lgb.LGBMRegressor:
#         """Initialize a default LightGBM model with specified or default parameters."""
#         params = model_params or {
#             "objective": "regression",
#             "boosting_type": "gbdt",
#             "metric": "rmse",
#             "verbosity": -1,
#         }
#         return lgb.LGBMRegressor(**params)

#     def train_and_evaluate(self) -> Dict[str, List[float]]:
#         """Train the model on each temporal partition and return predictions.

#         :return: Dictionary containing predictions for each partition.
#         :rtype: Dict[str, List[float]]
#         """
#         # partitioned_data = self.partitioner.get_partition_data()
#         phase_predictions: Dict[str, List[float]] = {}

#         # for i, phase_data in enumerate(partitioned_data):
#         #     trained_model = self.train_model_on_phase(phase_data)
#         #     X_phase = phase_data.drop(columns=[self.partitioner.target])
#         #     phase_predictions[f"Phase {i}"] = trained_model.predict(X_phase).tolist()

#         return phase_predictions

#         # TODO: Fix type hints for this method

#     def train_model_on_phase(self, phase_data: pd.DataFrame) -> Any:
#         """Train the model on the provided phase data."""
#         # Todo fix
#         # X = phase_data.drop(columns=[self.partitioner.target])
#         # y = phase_data[self.partitioner.target]
#         # self.model.fit(X, y)
#         # return self.model
#         return None
