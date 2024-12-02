"""Unit tests for single target partition utilities."""

import pytest

from temporalscope.partition.single_target.utils import (
    determine_partition_scheme,
    print_config,
    validate_cardinality,
    validate_percentages,
)

# =========== validate_percentages ==============


def test_validate_percentages_basic():
    """Test basic validation with valid percentages."""
    train, test, val = validate_percentages(0.7, 0.2, 0.1)
    assert train == pytest.approx(0.7)
    assert test == pytest.approx(0.2)
    assert val == pytest.approx(0.1)


def test_validate_percentages_compute_test():
    """Test computation of missing test percentage."""
    train, test, val = validate_percentages(0.7, None, 0.1)
    assert train == pytest.approx(0.7)
    assert test == pytest.approx(0.2)  # 1.0 - 0.7 - 0.1
    assert val == pytest.approx(0.1)


def test_validate_percentages_compute_val():
    """Test computation of missing validation percentage."""
    train, test, val = validate_percentages(0.7, 0.2, None)
    assert train == pytest.approx(0.7)
    assert test == pytest.approx(0.2)
    assert val == pytest.approx(0.1)  # 1.0 - 0.7 - 0.2


def test_validate_percentages_compute_both():
    """Test computation of missing test and validation percentages."""
    train, test, val = validate_percentages(0.7, None, None)
    assert train == pytest.approx(0.7)
    assert test == pytest.approx(0.3)  # 1.0 - 0.7
    assert val == pytest.approx(0.0)


def test_validate_percentages_invalid_train():
    """Test validation of invalid train percentage."""
    with pytest.raises(ValueError, match=r"`train_pct` must be between 0 and 1."):
        validate_percentages(1.5, 0.2, 0.1)
    with pytest.raises(ValueError, match=r"`train_pct` must be between 0 and 1."):
        validate_percentages(-0.1, 0.2, 0.1)


def test_validate_percentages_invalid_test():
    """Test validation of invalid test percentage."""
    with pytest.raises(ValueError, match=r"`test_pct` must be between 0 and 1."):
        validate_percentages(0.7, 1.5, 0.1)
    with pytest.raises(ValueError, match=r"`test_pct` must be between 0 and 1."):
        validate_percentages(0.7, -0.1, 0.1)


def test_validate_percentages_invalid_val():
    """Test validation of invalid validation percentage."""
    with pytest.raises(ValueError, match=r"`val_pct` must be between 0 and 1."):
        validate_percentages(0.7, 0.2, 1.5)
    with pytest.raises(ValueError, match=r"`val_pct` must be between 0 and 1."):
        validate_percentages(0.7, 0.2, -0.1)


def test_validate_percentages_sum_not_one():
    """Test validation of percentages not summing to 1.0."""
    with pytest.raises(ValueError, match=r"Train, test, and validation percentages must sum to 1.0."):
        validate_percentages(0.7, 0.7, 0.1)


def test_validate_percentages_precision():
    """Test validation with custom precision."""
    # Should pass with default precision
    train, test, val = validate_percentages(0.7, 0.2, 0.1 + 1e-7)
    assert train == pytest.approx(0.7)
    assert test == pytest.approx(0.2)
    assert val == pytest.approx(0.1 + 1e-7)

    # Should fail with stricter precision
    with pytest.raises(ValueError, match=r"Train, test, and validation percentages must sum to 1.0."):
        validate_percentages(0.7, 0.2, 0.1 + 1e-7, precision=1e-8)


# =========== determine_partition_scheme ==============


def test_determine_partition_scheme_invalid_configuration():
    """Test validation of invalid configuration that reaches the final error."""
    with pytest.raises(ValueError, match=r"Either `num_partitions` or `window_size` must be specified"):
        determine_partition_scheme(None, None, total_rows=100, stride=None)


def test_determine_partition_scheme_zero_num_partitions():
    """Test validation of zero num_partitions."""
    with pytest.raises(ValueError, match=r"`num_partitions` must be a positive integer."):
        determine_partition_scheme(num_partitions=0, window_size=None, total_rows=100, stride=None)


def test_determine_partition_scheme_negative_num_partitions():
    """Test validation of negative num_partitions."""
    with pytest.raises(ValueError, match=r"`num_partitions` must be a positive integer"):
        determine_partition_scheme(num_partitions=-1, window_size=None, total_rows=100, stride=None)


def test_determine_partition_scheme_num_partitions():
    """Test partition scheme determination using num_partitions."""
    scheme, num_parts, window = determine_partition_scheme(
        num_partitions=5, window_size=None, total_rows=100, stride=None
    )
    assert scheme == "num_partitions"
    assert num_parts == 5
    assert window == 20  # 100 // 5


def test_determine_partition_scheme_window_size():
    """Test partition scheme determination using window_size."""
    scheme, num_parts, window = determine_partition_scheme(
        num_partitions=None, window_size=20, total_rows=100, stride=10
    )
    assert scheme == "window_size"
    assert num_parts == 9  # (100 - 20) // 10 + 1
    assert window == 20


def test_determine_partition_scheme_window_size_default_stride():
    """Test partition scheme determination using window_size with default stride."""
    scheme, num_parts, window = determine_partition_scheme(
        num_partitions=None, window_size=20, total_rows=100, stride=None
    )
    assert scheme == "window_size"
    assert num_parts == 5  # (100 - 20) // 20 + 1
    assert window == 20


def test_determine_partition_scheme_invalid_num_partitions():
    """Test validation of invalid num_partitions."""
    with pytest.raises(ValueError, match=r"`num_partitions` must be a positive integer."):
        determine_partition_scheme(num_partitions=0, window_size=None, total_rows=100, stride=None)
    with pytest.raises(ValueError, match=r"`num_partitions` must be a positive integer."):
        determine_partition_scheme(num_partitions=-1, window_size=None, total_rows=100, stride=None)


def test_determine_partition_scheme_invalid_window_size():
    """Test validation of invalid window_size."""
    with pytest.raises(ValueError, match=r"`window_size` must be a positive integer."):
        determine_partition_scheme(num_partitions=None, window_size=0, total_rows=100, stride=None)
    with pytest.raises(ValueError, match=r"`window_size` must be a positive integer."):
        determine_partition_scheme(num_partitions=None, window_size=-1, total_rows=100, stride=None)


def test_determine_partition_scheme_missing_params():
    """Test validation when both parameters are missing."""
    with pytest.raises(ValueError, match=r"Either `num_partitions` or `window_size` must be specified."):
        determine_partition_scheme(num_partitions=None, window_size=None, total_rows=100, stride=None)


# =========== validate_cardinality ==============


def test_validate_cardinality_valid():
    """Test validation of valid cardinality."""
    # Should not raise any exceptions
    validate_cardinality(num_partitions=5, window_size=20, total_rows=100)


def test_validate_cardinality_invalid_num_partitions():
    """Test validation of invalid num_partitions cardinality."""
    with pytest.raises(ValueError, match=r"Insufficient rows \(10\) for `num_partitions=20`"):
        validate_cardinality(num_partitions=20, window_size=5, total_rows=10)


def test_validate_cardinality_invalid_window_size():
    """Test validation of invalid window_size cardinality."""
    with pytest.raises(ValueError, match=r"Insufficient rows \(10\) for `window_size=20`"):
        validate_cardinality(num_partitions=2, window_size=20, total_rows=10)


# =========== print_config ==============


def test_print_config_valid(capsys):
    """Test printing of valid configuration."""
    config = {
        "num_partitions": 5,
        "window_size": 20,
        "train_pct": 0.7,
        "test_pct": 0.2,
        "val_pct": 0.1,
        "truncate": True,
        "mode": "single_target",
    }
    print_config(config)
    captured = capsys.readouterr()
    assert "Configuration Details:" in captured.out
    assert "Parameter" in captured.out
    assert "Value" in captured.out
    assert "num_partitions" in captured.out
    assert "5" in captured.out


def test_print_config_invalid_types():
    """Test validation of invalid configuration types."""
    config = {
        "num_partitions": 5,
        "invalid_list": [1, 2, 3],  # Invalid type
        "invalid_dict": {"key": "value"},  # Invalid type
    }
    with pytest.raises(TypeError, match=r"Invalid data types in config:"):
        print_config(config)


def test_print_config_empty(capsys):
    """Test printing of empty configuration."""
    print_config({})
    captured = capsys.readouterr()
    assert "Configuration Details:" in captured.out
    assert "Parameter" in captured.out
    assert "Value" in captured.out
