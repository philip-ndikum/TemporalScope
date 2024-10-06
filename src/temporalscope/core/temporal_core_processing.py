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

"""TemporalScope/src/temporalscope/core/temporal_core_processing.py

Core Dataset Conversion and Interoperability Layer

This module provides core functionalities for dataset preparation and conversion, primarily
focused on handling multi-step workflows and ensuring interoperability between backends like
Pandas, TensorFlow, Modin, and Polars. It facilitates conversions required for downstream
tasks such as those used by the `temporal_target_shifter.py` module, ensuring multi-step
processing is smooth and integrated with deep learning and machine learning frameworks.

The module is fully functional, avoiding object-oriented over-complication, following a
functional approach for ease of use and extensibility.

Key Features:
-------------
- **Dataset Conversion**: Functions for converting between formats (e.g., Pandas, TensorFlow).
- **Interoperability**: Manages conversions between different backends for multi-step workflows.
- **Support for Future Extensions**: Stubbed for future implementations of key features required
  by downstream tasks like multi-step target handling and TensorFlow dataset conversion.

Example Usage:
--------------
.. code-block:: python

    from temporal_core_processing import convert_to_tensorflow, convert_to_pandas

    # Example DataFrame
    df = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'feature_1': range(100),
        'target': range(100)
    })

    # Convert DataFrame to TensorFlow Dataset
    tf_dataset = convert_to_tensorflow(df)

    # Convert TensorFlow Dataset back to Pandas
    df_back = convert_to_pandas(tf_dataset)
"""

from typing import Union
import pandas as pd
import polars as pl
import modin.pandas as mpd
import tensorflow as tf

from temporalscope.core.core_utils import SupportedBackendDataFrame


def convert_to_tensorflow(df: SupportedBackendDataFrame) -> tf.data.Dataset:
    """
    Stub: Convert a DataFrame to a TensorFlow Dataset.

    This function will convert Pandas, Modin, or Polars DataFrames into a TensorFlow Dataset
    to enable compatibility with deep learning frameworks like TensorFlow.

    :param df: The input DataFrame to convert.
    :return: A TensorFlow `tf.data.Dataset` object.
    """
    pass


def convert_to_pandas(df: SupportedBackendDataFrame) -> pd.DataFrame:
    """
    Stub: Convert a DataFrame or TensorFlow Dataset to a Pandas DataFrame.

    This function will handle converting Modin, Polars, or TensorFlow Datasets back to Pandas
    DataFrames to ensure interoperability across backends and downstream tasks.

    :param df: The input DataFrame or TensorFlow Dataset.
    :return: A Pandas DataFrame.
    """
    pass


def handle_multi_step_conversion(df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    """
    Stub: Prepare DataFrame for multi-step forecasting.

    This function will handle the preparation of multi-step targets by expanding the target
    column into sequences of the specified length, suitable for sequential models.

    :param df: The input DataFrame containing single-step targets.
    :param sequence_length: The length of the target sequence for multi-step forecasting.
    :return: A DataFrame with expanded target sequences.
    """
    pass
