# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    convert_to_backend,
    is_valid_temporal_backend,
)
from temporalscope.core.exceptions import UnsupportedBackendError
from temporalscope.datasets.datasets import DatasetLoader

# Constants
DEFAULT_DATASET_NAME = "macrodata"
VALID_BACKENDS = list(TEMPORALSCOPE_CORE_BACKEND_TYPES.keys())
INVALID_BACKEND = "unsupported_backend"


@pytest.fixture
def dataset_loader():
    """Fixture to create DatasetLoader instance for the default dataset."""
    return DatasetLoader(dataset_name=DEFAULT_DATASET_NAME)


@pytest.fixture(params=VALID_BACKENDS)
def backend(request):
    """Parametrized fixture for all supported backends in TEMPORALSCOPE_CORE_BACKEND_TYPES."""
    return request.param


# ========================= Dataset Tests =========================
def test_invalid_dataset_name():
    """Test that initializing DatasetLoader with an invalid dataset name raises a ValueError."""
    with pytest.raises(ValueError, match="Dataset 'invalid' is not supported"):
        DatasetLoader(dataset_name="invalid")


def test_load_dataset_structure(dataset_loader):
    """Test loading dataset and verifying structure."""
    df, target_col = dataset_loader._load_dataset_and_target()
    assert isinstance(df, pd.DataFrame)
    assert target_col == "realgdp"
    assert "ds" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])


# ========================= Backend Validation Tests =========================
@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_valid_backends_with_load_data(dataset_loader, backend):
    """Test loading data and converting to each backend specified in TEMPORALSCOPE_CORE_BACKEND_TYPES."""
    is_valid_temporal_backend(backend)  # Ensure the backend is supported
    data = dataset_loader.load_data(backend=backend)
    expected_type = TEMPORALSCOPE_CORE_BACKEND_TYPES[backend]
    assert isinstance(data, expected_type), f"Data is not of type {expected_type} for backend '{backend}'"


def test_invalid_backend_raises_error(dataset_loader):
    """Test that using an invalid backend raises an UnsupportedBackendError."""
    with pytest.raises(UnsupportedBackendError, match="is not supported"):
        dataset_loader.load_data(backend=INVALID_BACKEND)


def test_load_data_with_backend_conversion(mocker, dataset_loader):
    """Test load_data conversion by mocking convert_to_backend function."""
    mocker.patch("temporalscope.core.core_utils.convert_to_backend", side_effect=convert_to_backend)
    backend = "pandas"
    data = dataset_loader.load_data(backend=backend)
    assert isinstance(data, pd.DataFrame)
