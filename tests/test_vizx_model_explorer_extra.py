"""The ``model-explorer`` extra exists and matches its teaching refusal (U10.4).

The serve refusal names ``torchlens[model-explorer]``; this pins that the
extra exists, installs the vendor app package, and is floored at the
vendor-contract pin, the same shape as the ``netron`` extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    tomllib = pytest.importorskip("tomli", reason="need tomllib or tomli to parse pyproject.toml")

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_model_explorer_extra_matches_the_netron_extra() -> None:
    """Both viewer extras exist, each one bounded requirement on its vendor package."""

    with PYPROJECT.open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    assert extras["netron"] == ["netron>=9.2,<10"]
    assert extras["model-explorer"] == ["ai-edge-model-explorer>=0.1.33,<0.2"]
