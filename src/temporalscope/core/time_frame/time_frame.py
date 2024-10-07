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

"""TemporalScope/src/temporalscope/core/time_frame.py.

This module provides a flexible data loader for time series forecasting, allowing users to define their own
preprocessing, loss functions, and explainability workflows. The core assumption is that features are organized
in a context window prior to the target column, making the system compatible with SHAP and other explainability methods.
Given the variance in pre-processing techniques, meta-learning & loss-functions, TemporalScope explicitly does not
impose constraints on the end-user in the engineering design.

Engineering Design
--------------------

    TemporalScope is designed with several key assumptions to ensure performance, scalability, and flexibility
    across a wide range of time series forecasting and XAI workflows.

    1. Preprocessed Data Assumption:
       TemporalScope assumes that the user provides clean, preprocessed data. This includes handling categorical
       encoding, missing data imputation, and feature scaling prior to using TemporalScope's partitioning and explainability
       methods. Similar assumptions are seen in popular packages such as TensorFlow and GluonTS, which expect the
       user to manage data preprocessing outside the core workflow.

    2. Time Column Constraints:
       The `time_col` must be either a numeric index or a timestamp. TemporalScope relies on this temporal ordering for
       key operations like sliding window partitioning and temporal explainability workflows (e.g., SHAP).

    3. Numeric Features Requirement:
       Aside from the `time_col`, all other features in the dataset must be numeric. This ensures compatibility with machine
       learning and deep learning models that require numeric inputs. As seen in frameworks like TensorFlow, users are expected
       to preprocess categorical features (e.g., one-hot encoding or embeddings) before applying modeling or partitioning algorithms.

    4. Universal Model Assumption:
       TemporalScope is designed with the assumption that models trained will operate on the entire dataset without
       automatically applying hidden groupings or segmentations (e.g., for mixed-frequency data). This ensures that users
       can leverage frameworks like SHAP, Boruta-SHAP, and LIME for model-agnostic explainability without limitations.

    5. Supported Data Modes:

       TemporalScope also integrates seamlessly with model-agnostic explainability techniques like SHAP, LIME, and
       Boruta-SHAP, allowing insights to be extracted from most machine learning and deep learning models.

       The following table illustrates the two primary modes supported by TemporalScope and their typical use cases:

       +--------------------+----------------------------------------------------+--------------------------------------------+
       | **Mode**           | **Description**                                    | **Compatible Frameworks**                  |
       +--------------------+----------------------------------------------------+--------------------------------------------+
       | Single-step mode   | Suitable for scalar target machine learning tasks. | Scikit-learn, XGBoost, LightGBM, SHAP      |
       |                    | Each row represents a single time step.            | TensorFlow (for standard regression tasks) |
       +--------------------+----------------------------------------------------+--------------------------------------------+
       | Multi-step mode    | Suitable for deep learning tasks like sequence     | TensorFlow, PyTorch, Keras, SHAP, LIME     |
       |                    | forecasting. Input sequences (`X`) and output      | (for seq2seq models, sequence forecasting) |
       |                    | sequences (`Y`) are handled as separate datasets.  |                                            |
       +--------------------+----------------------------------------------------+--------------------------------------------+


       These modes follow standard assumptions in time series forecasting libraries, allowing for seamless integration
       with different models while requiring the user to manage their own data preprocessing.

    By enforcing these constraints, TemporalScope focuses on its core purpose—time series partitioning, explainability,
    and scalability—while leaving more general preprocessing tasks to the user. This follows industry standards seen in
    popular time series libraries.

.. seealso::

    1. Van Ness, M., Shen, H., Wang, H., Jin, X., Maddix, D.C., & Gopalswamy, K.
       (2023). Cross-Frequency Time Series Meta-Forecasting. arXiv preprint
       arXiv:2302.02077.

    2. Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024).
       Unified training of universal time series forecasting transformers. arXiv
       preprint arXiv:2402.02592.

    3. Trirat, P., Shin, Y., Kang, J., Nam, Y., Na, J., Bae, M., Kim, J., Kim, B., &
       Lee, J.-G. (2024). Universal time-series representation learning: A survey.
       arXiv preprint arXiv:2401.03717.

    4. Xu, Q., Zhuo, X., Jiang, C., & Liu, Y. (2019). An artificial neural network
       for mixed frequency data. Expert Systems with Applications, 118, pp.127-139.

    5. Filho, L.L., de Oliveira Werneck, R., Castro, M., Ribeiro Mendes Júnior, P.,
       Lustosa, A., Zampieri, M., Linares, O., Moura, R., Morais, E., Amaral, M., &
       Salavati, S. (2024). A multi-modal approach for mixed-frequency time series
       forecasting. Neural Computing and Applications, pp.1-25.

"""

from __future__ import annotations

import polars as pl
from polars.interchange.protocol import SupportsInterchange

from temporalscope.core import exceptions
from temporalscope.core.time_frame.validation import validate_time_frame


class TimeFrame:
    """Core class for the TemporalScope package.

    The `TimeFrame` class is designed to handle time series data across various DataFrame libraries
    that adhere to the Python DataFrame Interchange Protocol. It facilitates workflows for machine
    learning, deep learning, and explainability methods, while abstracting away backend-specific
    implementation details.

    This class validates the data, sorts it by time (if specified), and ensures compatibility with
    temporal XAI techniques (such as SHAP, Boruta-SHAP, and LIME), supporting larger data workflows
    in production environments.

    Key features
    ------------
    1. Backend Agnostic: Works with any DataFrame library implementing the Python DataFrame
       Interchange Protocol (e.g., Polars, Pandas, Modin).
    2. Data Validation: Performs checks on time and target columns, ensuring data integrity.
    3. Automatic Sorting: Optionally sorts data by the specified time column.
    4. XAI Compatibility: Designed to work seamlessly with various explainable AI techniques.


    Engineering Design Assumptions
    ------------------
    - Universal Models: This class is designed assuming the user has pre-processed their data for compatibility with
      deep learning models. Across the TemporalScope utilities (e.g., target shifter, padding, partitioning algorithms),
      it is assumed that preprocessing tasks, such as categorical feature encoding, will be managed by the user or
      upstream modules. Thus, the model will learn global weights and will not groupby categorical variables.
    - Mixed Time Frequency supported: Given the flexibility of deep learning models to handle various time frequencies,
      this class allows `time_col` to contain mixed frequency data, assuming the user will manage any necessary preprocessing
      or alignment outside of this class.
    - The `time_col` should be either numeric or timestamp-like for proper temporal ordering. Any mixed or invalid data
      types will raise validation errors.
    - All non-time columns are expected to be numeric. Users are responsible for handling non-numeric features
      (e.g., encoding categorical features).

    Example Usage
    -------------
    .. code-block:: python

       import polars as pl

       data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=100, interval="1d"), "value": range(100)})
       tf = TimeFrame(data, time_col="time", target_col="value")
       print(tf.get_data().head())

    .. seealso::
       - `polars` documentation: https://pola-rs.github.io/polars/
       - `pandas` documentation: https://pandas.pydata.org/
       - `modin` documentation: https://modin.readthedocs.io/
    """

    def __init__(
        self,
        df: SupportsInterchange,
        time_col: str,
        target_col: str,
        bypass_sort: bool = False,
    ) -> None:
        """Initialize a TimeFrame object with required validations and backend handling.

        This constructor validates the provided DataFrame and performs checks on the required columns (`time_col`,
        `target_col`).

        :param df: The input DataFrame. Must adhere to the Python (dataframe interchange protocol)[https://data-apis.org/dataframe-protocol/latest/index.html].
        :type df: SupportsInterchange
        :param time_col: The name of the column representing time. Should be numeric or timestamp-like for sorting.
        :type time_col: str
        :param target_col: The column representing the target variable. Must be a valid column in the DataFrame.
        :type target_col: str
        :param bypass_sort: If True, the data will be treated as sorted and sorting will be bypassed in the validation checks. Default is False.
        :type bypass_sort: bool

        :raises InterchangeProtocolNotSupported:
            - If the DataFrame type does not adhere to the Python dataframe interchange protocol.

        .. note::
            - The `time_col` must be numeric or timestamp-like to ensure proper temporal ordering.
            - Sorting is automatically performed by `time_col` unless disabled via `bypass_sort=True`.

        Example Usage:
        --------------
        .. code-block:: python

            import polars as pl
            from temporalscope.core.temporal_data_loader import TimeFrame

            data = pl.DataFrame({"time": pl.date_range(start="2021-01-01", periods=5, interval="1d"), "value": range(5)})

            tf = TimeFrame(data, time_col="time", target_col="value")
            print(tf.get_data().head())
        """
        self._time_col = time_col
        self._target_col = target_col
        self._df = self._convert_dataframe(df)

        validate_time_frame(self, bypass_sort)

    @property
    def time_col(self) -> str:
        """Return the column name representing time."""
        return self._time_col

    @property
    def target_col(self) -> str:
        """Return the column name representing the target variable."""
        return self._target_col

    @property
    def df(self) -> pl.DataFrame:
        """Return the DataFrame in its current state."""
        return self._df

    @staticmethod
    def _convert_dataframe(df: SupportsInterchange) -> pl.DataFrame:
        """Convert the DataFrame before downstream validation.

        This method checks if the input DataFrame adheres to the Python DataFrame Interchange Protocol.
        If the DataFrame does not adhere to the protocol, an exception is raised.

        :param df: The input DataFrame. Must adhere to the Python (dataframe interchange protocol)[https://data-apis.org/dataframe-protocol/latest/index.html].
        :type df: SupportsInterchange

        raises InterchangeProtocolNotSupported:
            - If the DataFrame type does not adhere to the Python dataframe interchange protocol.

        returns: pl.DataFrame
            - The preprocessed DataFrame.
        """
        if hasattr(df, "__dataframe__"):
            return pl.from_dataframe(df.__dataframe__())
        else:
            error_message = "Object `df` must adhere to the Python dataframe interchange protocol. See https://data-apis.org/dataframe-api/draft/index.html"
            raise exceptions.InterchangeProtocolNotSupported(error_message)
