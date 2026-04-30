"""Executable documentation tests for intervention examples."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import runpy
import warnings

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "intervention"
EXAMPLE_FILES = tuple(sorted(EXAMPLE_DIR.glob("[0-9][0-9]_*.py")))


def _example_ids() -> Iterator[str]:
    """Yield stable pytest IDs for intervention example scripts.

    Yields
    ------
    str
        Example file stem.
    """

    for path in EXAMPLE_FILES:
        yield path.stem


@pytest.mark.slow
@pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=tuple(_example_ids()))
def test_intervention_example_runs(example_path: Path) -> None:
    """Import and run one intervention worked example.

    Parameters
    ----------
    example_path:
        Path to the example script.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        namespace = runpy.run_path(str(example_path))
        namespace["main"]()
