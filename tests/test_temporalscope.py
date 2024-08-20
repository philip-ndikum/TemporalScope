"""
tests/test_temporalscope.py
"""


def test_import_temporalscope():
    """
    Test to ensure that the 'temporalscope' package imports correctly.
    """
    try:
        import temporalscope
    except ImportError:
        assert False, "Failed to import 'temporalscope'"
    assert True
