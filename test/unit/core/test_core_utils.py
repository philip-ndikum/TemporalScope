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

from typing import Any, Callable, Dict

import narwhals as nw
import pandas as pd
import pytest

from temporalscope.core.core_utils import (
    TEMPORALSCOPE_CORE_BACKEND_TYPES,
    TEMPORALSCOPE_OPTIONAL_BACKENDS,
    TimeColumnError,
    UnsupportedBackendError,
    check_dataframe_empty,
    check_dataframe_nulls_nans,
    convert_to_backend,
    convert_to_datetime,
    convert_to_numeric,
    get_api_keys,
    get_dataframe_backend,
    get_default_backend_cfg,
    get_narwhals_backends,
    get_temporalscope_backends,
    is_lazy_evaluation,
    is_valid_temporal_backend,
    is_valid_temporal_dataframe,
    print_divider,
    validate_and_convert_time_column,
    validate_column_type,
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


def test_convert_to_backend_invalid_type_no_to_pandas():
    """Test convert_to_backend with invalid type that doesn't have to_pandas."""

    class InvalidDataFrame:
        def __init__(self):
            pass

        @property
        def __class__(self):
            # Force df_type to be None in is_valid_temporal_dataframe
            raise Exception("Simulated error")

    df = InvalidDataFrame()
    with pytest.raises(UnsupportedBackendError, match="Input DataFrame type 'InvalidDataFrame' is not supported"):
        convert_to_backend(df, "pandas")


def test_get_dataframe_backend_valid_but_unknown():
    """Test get_dataframe_backend with DataFrame that's valid but doesn't match known backends."""

    class CustomDataFrame:
        def __init__(self):
            self._df = pd.DataFrame({"col": [1, 2, 3]})

        def to_pandas(self):
            return self._df

        @property
        def __class__(self):
            # Make it look like a valid DataFrame by using a known module path
            class_mock = type("DataFrame", (), {"__module__": "pandas.core.frame"})
            return class_mock

        def __getattr__(self, name):
            # Prevent isinstance checks from returning True
            if name == "__class__":
                class_mock = type("NotDataFrame", (), {})
                return class_mock
            return getattr(self._df, name)

    df = CustomDataFrame()
    # This DataFrame will pass is_valid_temporal_dataframe but fail all isinstance checks
    with pytest.raises(UnsupportedBackendError, match="Unknown DataFrame type"):
        get_dataframe_backend(df)


# ========================= Tests for is_lazy_evaluation =========================


def test_is_lazy_evaluation_dask(synthetic_df, request):
    """Test lazy evaluation detection for dask backend."""
    if request.node.callspec.params["synthetic_df"] == "dask":
        assert is_lazy_evaluation(synthetic_df), "Expected dask DataFrame to use lazy evaluation"


def test_is_lazy_evaluation_polars_lazy(synthetic_df, request):
    """Test lazy evaluation detection for polars lazy backend."""
    if request.node.callspec.params["synthetic_df"] == "polars-lazy":
        assert is_lazy_evaluation(synthetic_df), "Expected polars lazy DataFrame to use lazy evaluation"


def test_is_lazy_evaluation_eager(synthetic_df, request):
    """Test lazy evaluation detection for eager backends."""
    backend = request.node.callspec.params["synthetic_df"]
    if backend not in ["dask", "polars-lazy"]:
        assert not is_lazy_evaluation(synthetic_df), f"Expected {backend} DataFrame to use eager evaluation"


def test_is_lazy_evaluation_narwhalified(narwhalified_df):
    """Test lazy evaluation detection for narwhalified DataFrame."""
    assert not is_lazy_evaluation(narwhalified_df), "Expected narwhalified DataFrame to use eager evaluation"


@pytest.fixture(params=get_temporalscope_backends())
def backend(request) -> str:
    """Fixture providing all supported backends for testing."""
    return request.param


@pytest.fixture
def data_config(backend: str) -> Callable[..., Dict[str, Any]]:
    """Base fixture for data generation configuration."""

    def _config(**kwargs) -> Dict[str, Any]:
        default_config = {
            "num_samples": 3,
            "num_features": 2,
            "with_nulls": False,
            "with_nans": False,
            "backend": backend,
            "drop_time": True,
            "random_seed": 42,
        }
        default_config.update(kwargs)
        return default_config

    return _config


# ========================= check_dataframe_empty Tests =========================


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_empty_df(backend: str) -> None:
    """Test check_dataframe_empty returns True for empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    df = nw.from_native(df)
    assert check_dataframe_empty(df) is True


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_data(backend: str) -> None:
    """Test check_dataframe_empty returns False for non-empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    df = nw.from_native(df)
    assert check_dataframe_empty(df) is False


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_empty_with_lazy_evaluation(backend: str) -> None:
    """Test check_dataframe_empty works with lazy evaluation."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    df = nw.from_native(df)
    if hasattr(df.to_native(), "lazy"):
        df = df.to_native().lazy()
        df = nw.from_native(df)
    assert check_dataframe_empty(df) is True


@pytest.mark.parametrize("backend", ["dask"])
def test_check_dataframe_empty_lazy_compute(backend: str) -> None:
    """Test check_dataframe_empty handles lazy evaluation with compute."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    df = nw.from_native(df)

    if hasattr(df, "compute"):
        df.compute = lambda: nw.from_native({"col": []})
        assert check_dataframe_empty(df) is True


# Tests for error handling
def test_check_dataframe_empty_error_handling() -> None:
    """Test check_dataframe_empty error handling for None input."""
    with pytest.raises(ValueError, match="DataFrame cannot be None"):
        check_dataframe_empty(None)


def test_check_dataframe_empty_unsupported_type() -> None:
    """Test check_dataframe_empty raises ValueError for unsupported DataFrame type."""

    class UnsupportedDataFrame:
        """Custom DataFrame type not supported by Narwhals API."""

        pass

    df = UnsupportedDataFrame()
    with pytest.raises(ValueError, match="Unsupported DataFrame type"):
        check_dataframe_empty(df)


# ========================= check_dataframe_nulls_nans Tests =========================


@pytest.fixture
def sample_df():
    """Generate sample DataFrame for testing check_dataframe_nulls_nans function.

    Uses synthetic_data_generator to create consistent test data across backends.
    """
    return generate_synthetic_time_series(
        backend="pandas", num_samples=3, num_features=2, with_nulls=True, null_percentage=0.3, drop_time=True
    )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_no_nulls(backend: str) -> None:
    """Test check_dataframe_nulls_nans with DataFrame containing no null values.

    Uses synthetic data generator with no nulls to verify check_dataframe_nulls_nans correctly
    identifies when there are no null values in specified columns.
    """
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, with_nulls=False, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert result == {"feature_1": 0}


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_with_nulls(backend: str) -> None:
    """Test check_dataframe_nulls_nans with DataFrame containing null values.

    Uses synthetic data generator with nulls to verify check_dataframe_nulls_nans correctly
    counts null values in specified columns.
    """
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=1, with_nulls=True, null_percentage=0.5, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert 4 <= result["feature_1"] <= 6  # ~50% nulls


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_empty_dataframe(backend: str) -> None:
    """Test check_dataframe_nulls_nans with an empty DataFrame."""
    df = generate_synthetic_time_series(backend=backend, num_samples=0, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Empty"):  # Generic pattern
        check_dataframe_nulls_nans(df, ["feature_1"])


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_nonexistent_column(backend: str) -> None:
    """Test check_dataframe_nulls_nans with nonexistent column."""
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    with pytest.raises(ValueError, match="Column 'nonexistent' not found."):  # Match the specific message
        check_dataframe_nulls_nans(df, ["nonexistent"])


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_empty_column_list(backend: str) -> None:
    """Test check_dataframe_nulls_nans with empty list of columns.

    Verifies check_dataframe_nulls_nans returns empty dict for empty column list.
    """
    df = generate_synthetic_time_series(backend=backend, num_samples=3, num_features=1, drop_time=True)
    result = check_dataframe_nulls_nans(df, [])
    assert result == {}


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_check_dataframe_nulls_nans_all_nulls(backend: str) -> None:
    """Test check_dataframe_nulls_nans with columns containing all null values.

    Uses synthetic data generator with 100% null percentage to verify
    check_dataframe_nulls_nans correctly identifies when all values are null.
    """
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, with_nulls=True, null_percentage=1.0, drop_time=True
    )
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    assert result == {"feature_1": 3}


@pytest.mark.parametrize("backend", ["dask"])
def test_check_dataframe_nulls_nans_lazy_compute(backend: str) -> None:
    """Test check_dataframe_nulls_nans handles lazy evaluation with compute."""
    # Generate a real synthetic Dask DataFrame with nulls
    df = generate_synthetic_time_series(
        backend=backend, num_samples=10, num_features=1, with_nulls=True, null_percentage=0.5, drop_time=True
    )

    # Perform the test directly without mocking
    result = check_dataframe_nulls_nans(df, ["feature_1"])
    print(f"Result from check_dataframe_nulls_nans:\n{result}")  # Debug: Verify result

    # Verify null counts (depends on generator behavior)
    assert "feature_1" in result
    assert 0 <= result["feature_1"] <= 5  # Adjust based on null_percentage


# ========================= validate_and_convert_time_column Tests =========================


@pytest.mark.parametrize("backend", ["pandas", "modin", "polars", "pyarrow", "dask"])
def test_validate_and_convert_time_column_numeric_single(backend):
    """Test validation and numeric conversion of time column for a single backend."""
    # Generate synthetic data with the specified backend
    df = generate_synthetic_time_series(
        backend=backend, num_samples=3, num_features=1, drop_time=False, time_col_numeric=False
    )

    # Convert the time column to numeric
    result = validate_and_convert_time_column(df, "time", conversion_type="numeric")

    # Ensure the column was converted properly to numeric
    if backend in ["pyarrow"]:
        # PyArrow: Check if `time` exists in the schema (field names)
        assert "time" in [
            field.name for field in result.schema
        ], f"'time' column not found in result for backend: {backend}"
    else:
        assert "time" in result.columns, f"'time' column not found in result for backend: {backend}"

    # Backend-specific dtype validation
    if backend in ["pandas", "modin"]:
        # Pandas/Modin: Use DataFrame dtypes directly
        resolved_dtype = result["time"].dtype
        assert resolved_dtype in ["float64", "float32"], f"Expected numeric dtype for 'time', got {resolved_dtype}"

    elif backend == "polars":
        # Polars: Use `schema` for dtype resolution
        resolved_dtype = result.schema.get("time")
        assert str(resolved_dtype).lower() in [
            "float64",
            "float32",
        ], f"Expected numeric dtype for Polars, got {resolved_dtype}"

    elif backend == "pyarrow":
        # PyArrow: Validate `ChunkedArray` dtype
        resolved_dtype = result.schema.field("time").type
        assert str(resolved_dtype).lower() in [
            "double",
            "float64",
        ], f"Expected numeric dtype for PyArrow, got {resolved_dtype}"

    elif backend == "dask":
        # Dask: Resolve dtype from `_meta`
        resolved_dtype = result["time"]._meta.dtype
        assert resolved_dtype in ["float64", "float32"], f"Expected numeric dtype for Dask, got {resolved_dtype}"


# ========================= Edge case tests =========================


def test_convert_to_numeric_with_timezones():
    """Test convert_to_numeric with timezone-aware and naive datetime columns.

    This test ensures that:
    - Timezone-aware datetime columns are correctly converted to numeric timestamps.
    - Naive datetime columns are handled correctly.
    - Columns with invalid types raise appropriate errors.
    """
    # Create a DataFrame with different datetime columns
    df = pd.DataFrame(
        {
            "naive_datetime": pd.date_range("2023-01-01", periods=3),
            "aware_datetime": pd.date_range("2023-01-01", periods=3, tz="UTC"),
            "invalid_column": ["not_a_datetime", "still_not", "nope"],
        }
    )

    # Update col_dtype to match actual Pandas dtypes
    naive_dtype = df["naive_datetime"].dtype
    aware_dtype = df["aware_datetime"].dtype

    # Test naive datetime column
    result = convert_to_numeric(df, "naive_datetime", nw.col("naive_datetime"), naive_dtype)
    assert "naive_datetime" in result.columns
    assert pd.api.types.is_float_dtype(result["naive_datetime"]), "Expected numeric dtype for naive datetime."

    # Test timezone-aware datetime column
    result = convert_to_numeric(df, "aware_datetime", nw.col("aware_datetime"), aware_dtype)
    assert "aware_datetime" in result.columns
    assert pd.api.types.is_float_dtype(result["aware_datetime"]), "Expected numeric dtype for timezone-aware datetime."

    # Test invalid column type
    with pytest.raises(ValueError, match="not a datetime column"):
        convert_to_numeric(df, "invalid_column", nw.col("invalid_column"), df["invalid_column"].dtype)


def test_convert_to_numeric_error_handling():
    """Test convert_to_numeric error handling for invalid column types."""
    # Create a DataFrame with an invalid column type
    df = pd.DataFrame({"invalid_col": ["not_a_datetime", "nope", "still_not"]})

    # Test invalid column type
    with pytest.raises(ValueError, match="is not a datetime column"):
        convert_to_numeric(df, "invalid_col", nw.col("invalid_col"), df["invalid_col"].dtype)


def test_validate_column_type():
    """Test validate_column_type for various scenarios."""
    # Test valid numeric column
    validate_column_type("numeric_col", "float64")  # Should not raise an error

    # Test valid datetime column
    validate_column_type("datetime_col", "datetime64[ns]")  # Should not raise an error

    # Test invalid column type
    with pytest.raises(ValueError, match="neither numeric nor datetime"):
        validate_column_type("invalid_col", "string")

    # Test mixed-type column (invalid)
    with pytest.raises(ValueError, match="neither numeric nor datetime"):
        validate_column_type("mixed_col", "object")

    # Test custom/user-defined types (invalid)
    with pytest.raises(ValueError, match="neither numeric nor datetime"):
        validate_column_type("custom_col", "custom_type")


def test_convert_to_datetime_error_handling():
    """Test convert_to_datetime error handling for invalid column types."""
    # Create a DataFrame with an invalid column type
    df = pd.DataFrame({"invalid_col": ["not_a_datetime", "nope", "still_not"]})

    # Test invalid column type
    with pytest.raises(ValueError, match="neither string nor numeric"):
        convert_to_datetime(df, "invalid_col", nw.col("invalid_col"), df["invalid_col"].dtype)


def test_validate_column_type_edge_cases():
    """Test validate_column_type for edge cases."""
    # Test very long column name
    long_col_name = "a" * 300
    validate_column_type(long_col_name, "datetime64[ns]")  # Should not raise an error

    # Test numeric column with unusual dtype
    validate_column_type("unusual_numeric", "float128")  # Should not raise an error

    # Test datetime column with timezone
    validate_column_type("tz_datetime", "datetime64[ns, UTC]")  # Should not raise an error


def test_convert_to_datetime_with_string_column():
    """Test convert_to_datetime with string datetime column."""
    # Create a DataFrame with a string datetime column
    df = pd.DataFrame(
        {
            "string_datetime": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }
    )

    # Call convert_to_datetime for the string column
    result = convert_to_datetime(df, "string_datetime", nw.col("string_datetime"), "string")

    # Check that the column was converted correctly
    assert "string_datetime" in result.columns, "Column 'string_datetime' not found in result."
    assert pd.api.types.is_datetime64_any_dtype(
        result["string_datetime"]
    ), "Expected column 'string_datetime' to be converted to datetime."


def test_convert_to_datetime_with_numeric_column():
    """Test convert_to_datetime with numeric timestamp column."""
    # Create a DataFrame with a numeric timestamp column
    df = pd.DataFrame(
        {
            "numeric_timestamp": [1672531200, 1672617600, 1672704000],  # Unix timestamps for Jan 1-3, 2023
        }
    )

    # Call convert_to_datetime for the numeric column
    result = convert_to_datetime(df, "numeric_timestamp", nw.col("numeric_timestamp"), "float")

    # Check that the column was converted correctly
    assert "numeric_timestamp" in result.columns, "Column 'numeric_timestamp' not found in result."
    assert pd.api.types.is_datetime64_any_dtype(
        result["numeric_timestamp"]
    ), "Expected column 'numeric_timestamp' to be converted to datetime."


def test_validate_and_convert_time_column_missing_time_column():
    """Test for missing time column in validate_and_convert_time_column."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(TimeColumnError, match="Column 'time' does not exist in the DataFrame."):
        validate_and_convert_time_column(df, "time", conversion_type="numeric")


def test_validate_and_convert_time_column_invalid_conversion_type():
    """Test for invalid conversion type in validate_and_convert_time_column."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    with pytest.raises(
        ValueError, match="Invalid conversion_type 'invalid_type'. Must be one of 'numeric', 'datetime', or None."
    ):
        validate_and_convert_time_column(df, "time", conversion_type="invalid_type")


def test_validate_and_convert_time_column_to_numeric():
    """Test validate_and_convert_time_column with numeric conversion."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    result = validate_and_convert_time_column(df, "time", conversion_type="numeric")
    assert "time" in result.columns
    assert pd.api.types.is_float_dtype(result["time"]), "Expected numeric dtype for 'time' column."


def test_validate_and_convert_time_column_to_datetime():
    """Test validate_and_convert_time_column with datetime conversion."""
    df = pd.DataFrame({"time": [1672531200, 1672617600, 1672704000]})
    result = validate_and_convert_time_column(df, "time", conversion_type="datetime")
    assert "time" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["time"]), "Expected datetime dtype for 'time' column."


def test_validate_and_convert_time_column_validation_only():
    """Test validate_and_convert_time_column with validation-only path."""
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=3)})
    result = validate_and_convert_time_column(df, "time", conversion_type=None)
    assert "time" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["time"]), "Expected datetime dtype for 'time' column."
