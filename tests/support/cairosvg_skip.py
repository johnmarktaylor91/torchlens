"""Skip helper for tests that rasterize SVG through ``cairosvg``."""

from __future__ import annotations

from types import ModuleType

import pytest

from torchlens._visualization._cairosvg import cairosvg_unavailable_reason, get_cairosvg_support


def require_cairosvg() -> ModuleType:
    """Return the ``cairosvg`` module, or skip naming the missing package or library.

    ``pytest.importorskip`` alone only catches ``ImportError``; a present package
    whose ``libcairo`` is missing raises ``OSError`` and would fail the test, so the
    capability flag ``HAS_CAIROSVG`` decides first and its reason becomes the skip.
    """

    if not get_cairosvg_support():
        pytest.skip(
            f"cairosvg cannot rasterize here: {cairosvg_unavailable_reason()}",
            allow_module_level=True,
        )
    return pytest.importorskip("cairosvg")
