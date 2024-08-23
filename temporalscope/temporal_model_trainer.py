"""temporalscope/temporal_model_trainer.py

This module implements the TemporalModelTrainer class, designed to provide an easy-to-use interface for training machine learning models on temporal data. It builds on top of the TemporalDataLoader class, allowing users to seamlessly train models on data that has been preprocessed and validated for temporal consistency.

Design Considerations:
1. Model Flexibility: Users can specify various machine learning models (e.g., LightGBM, MLP with JAX/Flax) and hyperparameters. Default models are provided for quick use cases.
2. Temporal Partitioning: The class supports temporal partitioning schemes (e.g., sliding windows, expanding windows) to ensure models are trained and validated on temporally consistent data.
3. Integration with SHAP: The class can easily be extended or integrated with SHAP to provide insights into temporal feature importance, helping users understand how model predictions change over time.
"""

from typing import Union, Optional, Callable, List, Dict, Any
import lightgbm as lgb
import numpy as np
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from temporalscope.temporal_data_loader import TemporalDataLoader


class TemporalModelTrainer:
    """Trains machine learning models on temporal data using various partitioning schemes.

    This class provides functionalities to train models on time series data, applying temporal partitioning
    techniques like sliding windows or expanding windows. It supports integration with multiple models, including
    traditional models like LightGBM and neural networks using JAX/Flax.

    :param data_loader: An instance of TemporalDataLoader.
    :type data_loader: TemporalDataLoader
    :param model_type: Type of model to train ('lightgbm', 'mlp'). Default is 'lightgbm'.
    :type model_type: str
    :param model_params: Hyperparameters for the model. Default is None.
    :type model_params: Optional[Dict[str, Any]]
    """

    def __init__(self, data_loader: TemporalDataLoader, model_type: str = 'lightgbm', model_params: Optional[Dict[str, Any]] = None):
        self.data_loader = data_loader
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on the specified type and parameters."""
        if self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        elif self.model_type == 'mlp':
            return self._initialize_mlp()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initialize_mlp(self):
        """Initialize a simple MLP model using JAX/Flax."""
        class MLP(nn.Module):
            features: List[int]

            def setup(self):
                self.layers = [nn.Dense(feat) for feat in self.features]

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = nn.relu(layer(x))
                return self.layers[-1](x)

        # Example MLP configuration, can be modified based on `self.model_params`
        mlp = MLP(features=self.model_params.get('features', [64, 32, 1]))
        return mlp

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model on the provided training data.

        :param X_train: Training features.
        :type X_train: Union[np.ndarray, jnp.ndarray]
        :param y_train: Training target values.
        :type y_train: Union[np.ndarray, jnp.ndarray]
        :param X_val: Optional. Validation features.
        :type X_val: Optional[Union[np.ndarray, jnp.ndarray]]
        :param y_val: Optional. Validation target values.
        :type y_val: Optional[Union[np.ndarray, jnp.ndarray]]
        """
        if self.model_type == 'lightgbm':
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None)
        elif self.model_type == 'mlp':
            self._train_mlp(X_train, y_train, X_val, y_val)

    def _train_mlp(self, X_train, y_train, X_val=None, y_val=None):
        """Train the MLP model using JAX/Flax."""
        # Set up training loop, optimizer, etc.
        seed = random.PRNGKey(0)
        params = self.model.init(seed, X_train)
        optimizer = optax.adam(learning_rate=self.model_params.get('learning_rate', 1e-3))
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, X, y):
            def loss_fn(params):
                logits = self.model.apply(params, X)
                loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, num_classes=logits.shape[-1]))
                return loss.mean()

            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        for epoch in range(self.model_params.get('epochs', 10)):
            params, opt_state = step(params, opt_state, X_train, y_train)
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate_mlp(params, X_val, y_val)
                print(f'Epoch {epoch}, Validation Loss: {val_loss}')

        self.params = params

    def _evaluate_mlp(self, params, X_val, y_val):
        """Evaluate the MLP model on validation data."""
        logits = self.model.apply(params, X_val)
        val_loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y_val, num_classes=logits.shape[-1])).mean()
        return val_loss

    def predict(self, X_test):
        """Generate predictions using the trained model.

        :param X_test: Test features.
        :type X_test: Union[np.ndarray, jnp.ndarray]
        :return: Predicted values.
        :rtype: Union[np.ndarray, jnp.ndarray]
        """
        if self.model_type == 'lightgbm':
            return self.model.predict(X_test)
        elif self.model_type == 'mlp':
            return self.model.apply(self.params, X_test)

