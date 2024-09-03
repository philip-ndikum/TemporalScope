""" tests/test_temporalscope.py
"""

import pytest


def test_import_temporalscope():
    """
    Test to ensure that the 'temporalscope' package imports correctly.
    """
    try:
        import temporalscope
    except ImportError:
        pytest.fail("Failed to import 'temporalscope'")
