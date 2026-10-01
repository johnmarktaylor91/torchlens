"""Shared machinery for the real-model tree (testing MEMO; lanes P04+B1).

Registers the realism SELECTION markers (memo D8: orthogonal to the
smoke/heavy/slow/rare cost tiers). The pyproject marker declarations are
P03's single-owner territory (declared once in its first commit); this
in-tree registration keeps the tree self-contained and collision-free --
``config.addinivalue_line`` is additive and idempotent.
"""

from __future__ import annotations

import pytest

from tests.real_model.registry import Registry, load_registry


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_model: real-model realism tests (any band; selected by the R0/R1 gates)",
    )
    config.addinivalue_line(
        "markers",
        "real_class: R0 band -- real upstream class, constructed weights, zero network",
    )
    config.addinivalue_line(
        "markers",
        "real_checkpoint: R1/R2 bands -- pinned released weights, offline verified cache",
    )


@pytest.fixture(scope="session")
def artifact_registry() -> Registry:
    """The loaded, validated artifact registry (session-scoped)."""

    return load_registry()
