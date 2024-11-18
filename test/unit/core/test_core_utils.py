# Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
# See the NOTICE file for additional information regarding copyright ownership.
# The ASF licenses this file to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    TEMPORALSCOPE_OPTIONAL_BACKENDS,
    UnsupportedBackendError,
    convert_to_backend,
    get_api_keys,
    get_dataframe_backend,
    get_default_backend_cfg,
    get_narwhals_backends,
    get_temporalscope_backends,
    is_valid_temporal_backend,
    is_valid_temporal_dataframe,
    print_divider,
)
from temporalscope.datasets.synthetic_data_generator import generate_synthetic_time_series

# Constants
VALID_BACKENDS = ["pandas", "modin", "pyarrow", "polars", "dask"]
INVALID_BACKEND = "unsupported_backend"

# ========================= Fixtures =========================


@pytest.fixture(params=VALID_BACKENDS)
def synthetic_df(request):
    """Fixture providing synthetic DataFrames for each backend.

    :param request: pytest request object containing the backend parameter
    :return: DataFrame in the specified backend format
    """
    return generate_synthetic_time_series(backend=request.param, num_samples=10, num_features=2)


@pytest.fixture
def narwhalified_df():
    """Fixture providing a narwhalified DataFrame."""
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=2)
    return nw.from_native(df)


# ========================= Tests for get_narwhals_backends =========================


def test_get_narwhals_backends():
    """Test the retrieval of Narwhals-supported backends."""
    backends = get_narwhals_backends()
    assert "pandas" in backends, "Expected 'pandas' backend in Narwhals backends list."
    assert "modin" in backends, "Expected 'modin' backend in Narwhals backends list."


# ========================= Tests for get_default_backend_cfg =========================


def test_get_default_backend_cfg():
    """Test retrieval of the default backend configuration for Narwhals."""
    cfg = get_default_backend_cfg()
    assert isinstance(cfg, dict), "Expected default backend configuration to be a dictionary."
    assert "BACKENDS" in cfg, "Expected 'BACKENDS' key in default configuration."
    assert all(
        backend in cfg["BACKENDS"] for backend in get_narwhals_backends()
    ), "Mismatch in default backend configuration."


# ========================= Tests for get_temporalscope_backends =========================


def test_get_temporalscope_backends():
    """Test that only TemporalScope-compatible backends are returned."""
    backends = get_temporalscope_backends()
    assert all(
        backend in VALID_BACKENDS for backend in backends
    ), "Non-compatible backend found in TemporalScope backends."


# ========================= Tests for is_valid_temporal_backend =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_is_valid_temporal_backend_supported(backend):
    """Test that is_valid_temporal_backend passes for supported backends."""
    try:
        is_valid_temporal_backend(backend)
    except UnsupportedBackendError:
        pytest.fail(f"is_valid_temporal_backend raised UnsupportedBackendError for valid backend '{backend}'.")


def test_is_valid_temporal_backend_unsupported():
    """Test that is_valid_temporal_backend raises error for unsupported backend."""
    with pytest.raises(UnsupportedBackendError):
        is_valid_temporal_backend(INVALID_BACKEND)


def test_is_valid_temporal_backend_optional_warning():
    """Test that is_valid_temporal_backend issues a warning for optional backends if available."""
    # Check if "cudf" is optional in TemporalScope
    if "cudf" in TEMPORALSCOPE_OPTIONAL_BACKENDS:
        # Expect a warning if "cudf" is not installed
        with pytest.warns(UserWarning, match="optional and requires additional setup"):
            is_valid_temporal_backend("cudf")
    else:
        pytest.skip("Skipping test as 'cudf' is not an optional backend in this configuration.")


# ========================= Tests for is_valid_temporal_dataframe =========================


def test_is_valid_temporal_dataframe_supported(synthetic_df):
    """Test that is_valid_temporal_dataframe returns True for supported DataFrame types."""
    # Check if the DataFrame is valid
    is_valid, df_type = is_valid_temporal_dataframe(synthetic_df)
    assert is_valid, "Expected DataFrame to be valid."
    assert df_type == "native", "Expected DataFrame type to be 'native'."


def test_is_valid_temporal_dataframe_narwhalified(narwhalified_df):
    """Test that is_valid_temporal_dataframe handles narwhalified DataFrames."""
    # Check if the narwhalified DataFrame is valid
    is_valid, df_type = is_valid_temporal_dataframe(narwhalified_df)
    assert is_valid, "Expected narwhalified DataFrame to be valid."
    assert df_type == "narwhals", "Expected DataFrame type to be 'narwhals' for narwhalified DataFrame."


def test_is_valid_temporal_dataframe_unsupported():
    """Test that is_valid_temporal_dataframe returns False for unsupported DataFrame types."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()

    is_valid, df_type = is_valid_temporal_dataframe(df)
    assert not is_valid, "Expected DataFrame to be invalid for unsupported type."
    assert df_type is None, "Expected DataFrame type to be None for unsupported type."


# ========================= Tests for get_api_keys =========================


def test_get_api_keys_present(monkeypatch):
    """Test retrieval of API keys when they are set in the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_openai")
    monkeypatch.setenv("CLAUDE_API_KEY", "test_key_claude")
    api_keys = get_api_keys()
    assert api_keys["OPENAI_API_KEY"] == "test_key_openai", "Expected OPENAI_API_KEY to match environment variable."
    assert api_keys["CLAUDE_API_KEY"] == "test_key_claude", "Expected CLAUDE_API_KEY to match environment variable."


def test_get_api_keys_absent(monkeypatch, capsys):
    """Test warnings when API keys are missing from environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    get_api_keys()
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY is not set" in captured.out, "Expected warning for missing OPENAI_API_KEY."
    assert "CLAUDE_API_KEY is not set" in captured.out, "Expected warning for missing CLAUDE_API_KEY."


# ========================= Tests for print_divider =========================


def test_print_divider_default(capsys):
    """Test default divider output."""
    print_divider()
    captured = capsys.readouterr()
    assert captured.out == "=" * 70 + "\n", "Expected default divider output."


def test_print_divider_custom(capsys):
    """Test custom character and length in divider output."""
    print_divider(char="*", length=30)
    captured = capsys.readouterr()
    assert captured.out == "*" * 30 + "\n", "Expected custom divider output."


# ========================= Tests for convert_to_backend =========================


def test_convert_to_backend_valid(synthetic_df, request):
    """Test DataFrame conversion to each valid backend."""
    # Get current backend from fixture param
    current_backend = request.node.callspec.params["synthetic_df"]

    # Convert to same backend should work
    converted_df = convert_to_backend(synthetic_df, current_backend)

    # Validate the type of the converted DataFrame
    expected_type = TEMPORALSCOPE_CORE_BACKEND_TYPES[current_backend]
    assert isinstance(converted_df, expected_type), f"Expected {expected_type} for backend '{current_backend}'."


def test_convert_lazy_frame_to_pandas(synthetic_df):
    """Test conversion of LazyFrame to pandas."""
    if hasattr(synthetic_df, "compute"):
        # Convert to pandas
        pandas_df = convert_to_backend(synthetic_df, "pandas")
        assert isinstance(pandas_df, TEMPORALSCOPE_CORE_BACKEND_TYPES["pandas"])


def test_convert_to_backend_invalid(synthetic_df):
    """Test that convert_to_backend raises UnsupportedBackendError with unsupported backend."""
    # Ensure UnsupportedBackendError is raised for invalid backend
    with pytest.raises(UnsupportedBackendError, match=f"Backend '{INVALID_BACKEND}' is not supported"):
        convert_to_backend(synthetic_df, INVALID_BACKEND)


def test_convert_to_backend_unsupported_dataframe_type():
    """Test that convert_to_backend raises UnsupportedBackendError for unsupported DataFrame types."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()

    with pytest.raises(UnsupportedBackendError, match="Input DataFrame type 'UnsupportedDataFrame' is not supported"):
        convert_to_backend(df, "pandas")


def test_is_valid_temporal_dataframe_exception():
    """Test that is_valid_temporal_dataframe handles exceptions gracefully."""

    class BrokenDataFrame:
        @property
        def __class__(self):
            raise Exception("Simulated error")

    df = BrokenDataFrame()
    is_valid, df_type = is_valid_temporal_dataframe(df)
    assert not is_valid
    assert df_type is None


def test_convert_to_backend_fallbacks():
    """Test convert_to_backend fallback conversion paths."""
    # Get a dask DataFrame which has compute() method
    df = generate_synthetic_time_series(backend="dask", num_samples=10, num_features=2)

    # Convert to pandas - this will trigger the compute() fallback
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, TEMPORALSCOPE_CORE_BACKEND_TYPES["pandas"])


def test_convert_to_backend_conversion_methods():
    """Test convert_to_backend fallback conversion methods."""
    # Use existing synthetic data generator
    df = generate_synthetic_time_series(backend="dask", num_samples=10, num_features=2)

    # Convert to pandas - this will use the to_pandas method
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, TEMPORALSCOPE_CORE_BACKEND_TYPES["pandas"])


def test_convert_to_backend_narwhalified():
    """Test converting narwhalified DataFrame."""
    # Get narwhalified DataFrame
    df = generate_synthetic_time_series(backend="pandas", num_samples=10, num_features=2)
    df_narwhals = nw.from_native(df)

    # Convert to polars
    result = convert_to_backend(df_narwhals, "polars")
    assert isinstance(result, TEMPORALSCOPE_CORE_BACKEND_TYPES["polars"])


def test_convert_to_backend_array_method():
    """Test converting DataFrame with __array__ method."""
    # Use synthetic data generator to create a DataFrame
    df = generate_synthetic_time_series(backend="pandas", num_samples=2, num_features=1)
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, pd.DataFrame)


def test_convert_to_backend_numpy_method():
    """Test converting DataFrame with to_numpy method."""
    # Use synthetic data generator to create a DataFrame
    df = generate_synthetic_time_series(backend="pandas", num_samples=2, num_features=1)
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, pd.DataFrame)


def test_convert_to_backend_compute_validation():
    """Test validation after compute() method."""
    # Use synthetic data generator to create a dask DataFrame
    df = generate_synthetic_time_series(backend="dask", num_samples=2, num_features=1)
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, pd.DataFrame)


def test_convert_to_backend_compute_error():
    """Test error handling when DataFrame computation fails."""

    class BrokenComputeDataFrame:
        def __init__(self):
            self._df = pd.DataFrame({"col": [1, 2, 3]})

        def to_pandas(self):
            return self._df

        def compute(self):
            raise Exception("Computation failed")

        @property
        def __class__(self):
            class_mock = type("MockClass", (), {"__module__": "dask.dataframe.core"})
            return class_mock

    df = BrokenComputeDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Failed to convert DataFrame: Computation failed"):
        convert_to_backend(df, "pandas")


def test_convert_to_backend_compute_validation_failure():
    """Test validation failure after compute."""

    class InvalidComputeDataFrame:
        def __init__(self):
            self._df = pd.DataFrame({"col": [1, 2, 3]})

        def to_pandas(self):
            return self._df

        def compute(self):
            # Return something that's not a DataFrame
            return [1, 2, 3]

        @property
        def __class__(self):
            class_mock = type("MockClass", (), {"__module__": "dask.dataframe.core"})
            return class_mock

    df = InvalidComputeDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Failed to compute"):
        convert_to_backend(df, "pandas")


def test_convert_to_backend_invalid_with_pandas():
    """Test converting invalid DataFrame with to_pandas method."""
    base_df = generate_synthetic_time_series(backend="pandas", num_samples=2, num_features=1)

    class CustomDataFrame:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

        @property
        def __class__(self):
            # Force an exception in is_valid_temporal_dataframe to make it return False
            raise Exception("Simulated error")

    df = CustomDataFrame(base_df)

    # Test returning pandas directly when target is pandas
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, pd.DataFrame)

    # Test using pandas as intermediate for another backend
    result = convert_to_backend(df, "polars")
    assert isinstance(result, TEMPORALSCOPE_CORE_BACKEND_TYPES["polars"])


def test_convert_to_backend_direct_pandas():
    """Test converting invalid DataFrame directly to pandas."""
    base_df = generate_synthetic_time_series(backend="pandas", num_samples=2, num_features=1)

    class CustomDataFrame:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

        @property
        def __class__(self):
            # Make it look like a pandas DataFrame
            class_mock = type("DataFrame", (), {"__module__": "pandas.core.frame"})
            return class_mock

        def __getattr__(self, name):
            # Forward all other attributes to the underlying DataFrame
            return getattr(self._df, name)

    df = CustomDataFrame(base_df)
    result = convert_to_backend(df, "pandas")
    assert isinstance(result, pd.DataFrame)


def test_convert_to_backend_pandas_intermediate():
    """Test using pandas as intermediate when converting invalid DataFrame."""
    base_df = generate_synthetic_time_series(backend="pandas", num_samples=2, num_features=1)

    class CustomDataFrame:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

        @property
        def __class__(self):
            # Make it look like a pandas DataFrame
            class_mock = type("DataFrame", (), {"__module__": "pandas.core.frame"})
            return class_mock

        def __getattr__(self, name):
            # Forward all other attributes to the underlying DataFrame
            return getattr(self._df, name)

    df = CustomDataFrame(base_df)
    result = convert_to_backend(df, "polars")
    assert isinstance(result, TEMPORALSCOPE_CORE_BACKEND_TYPES["polars"])


# ========================= Tests for get_dataframe_backend =========================


def test_get_dataframe_backend_supported(synthetic_df, request):
    """Test get_dataframe_backend returns correct backend for supported types."""
    expected_backend = request.node.callspec.params["synthetic_df"]
    backend = get_dataframe_backend(synthetic_df)
    assert backend == expected_backend, f"Expected backend {expected_backend}, got {backend}"


def test_get_dataframe_backend_narwhalified(narwhalified_df):
    """Test get_dataframe_backend handles narwhalified DataFrames."""
    # Should return pandas since our narwhalified fixture uses pandas underneath
    backend = get_dataframe_backend(narwhalified_df)
    assert backend == "pandas", "Expected 'pandas' backend for narwhalified DataFrame"


def test_get_dataframe_backend_unsupported():
    """Test get_dataframe_backend raises error for unsupported types."""

    class UnsupportedDataFrame:
        pass

    df = UnsupportedDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Unknown DataFrame type"):
        get_dataframe_backend(df)


def test_get_dataframe_backend_broken():
    """Test get_dataframe_backend handles broken DataFrames."""

    class BrokenDataFrame:
        @property
        def __class__(self):
            raise Exception("Simulated error")

    df = BrokenDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Unknown DataFrame type"):
        get_dataframe_backend(df)
