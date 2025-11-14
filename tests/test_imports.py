"""Test that modules can be imported correctly."""

from __future__ import annotations


def test_import_process_input() -> None:
    """Test that process_input module can be imported."""
    import src.process_input  # noqa: F401


def test_import_random_forest_regression() -> None:
    """Test that random_forest_regression module can be imported."""
    import src.random_forest_regression  # noqa: F401


def test_import_streamlit_app() -> None:
    """Test that streamlit_app module can be imported."""
    import src.streamlit_app  # noqa: F401


def test_import_src_package() -> None:
    """Test that src package can be imported."""
    import src

    assert src.__version__ == "0.1.0"


def test_import_functions_from_process_input() -> None:
    """Test that functions can be imported from process_input."""
    from src.process_input import process_and_aggregate  # noqa: F401


def test_import_functions_from_random_forest() -> None:
    """Test that functions can be imported from random_forest_regression."""
    from src.random_forest_regression import (  # noqa: F401
        determine_target_type,
        random_forest_processing,
        train_models,
    )
