# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from typing import Any

import pytest

from temporalscope.core.core_utils import TEMPORALSCOPE_CORE_BACKEND_TYPES
from temporalscope.core.exceptions import TimeColumnError
from temporalscope.core.temporal_data_loader import TimeFrame, MODE_SINGLE_STEP
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants - start with stable backends first
VALID_BACKENDS = [backend for backend in TEMPORALSCOPE_CORE_BACKEND_TYPES.keys() if backend != "dask"]


def is_dask_df(obj: Any) -> bool:
    """Check if object is a Dask DataFrame."""
    return type(obj).__module__.startswith("dask.")


def get_shape(df: Any) -> tuple[int, int]:
    """Get shape of DataFrame, handling different backends."""
    if is_dask_df(df):
        nrows = int(df.shape[0].compute())  # type: ignore
        ncols = df.shape[1]
        return (nrows, ncols)
    return df.shape


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_timeframe_basic_initialization(backend: str) -> None:
    """Test basic TimeFrame initialization with minimal data across backends."""
    # Generate minimal test data
    df = generate_synthetic_time_series(
        backend=backend,
        num_samples=5,
        num_features=1,
        time_col_numeric=True,  # Start with numeric time for simplicity
    )

    # Basic TimeFrame initialization
    tf = TimeFrame(df=df, time_col="time", target_col="target", mode=MODE_SINGLE_STEP)

    # Verify basic properties
    shape = get_shape(tf.df)
    assert shape[0] == 5, "Expected 5 rows"
    assert "time" in tf.df.columns, "Expected time column"
    assert "target" in tf.df.columns, "Expected target column"
    assert tf.mode == MODE_SINGLE_STEP, "Expected single step mode"
    assert tf.backend == backend, f"Expected {backend} backend"


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_timeframe_missing_columns(backend: str) -> None:
    """Test error handling for missing columns."""
    df = generate_synthetic_time_series(backend=backend, num_samples=5)

    # Test missing time column
    with pytest.raises(TimeColumnError, match=r".*must exist.*"):
        TimeFrame(df, time_col="invalid_time", target_col="target")

    # Test missing target column
    with pytest.raises(TimeColumnError, match=r".*must exist.*"):
        TimeFrame(df, time_col="time", target_col="invalid_target")
