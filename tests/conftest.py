from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked with long."""
    if not config.getoption("markexpr", "False"):
        config.option.markexpr = "not long"
