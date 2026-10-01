"""Executable gate over ``docs/reference/checks_kit.md`` (memo item 11).

Every Python fence on the checks-kit page executes top to bottom, statement
by statement, in ONE shared namespace with ZERO ambient injections: the page
must define everything it references, so a recipe that drifts from the real
surface fails this gate (the executed-recipes law -- docs recipes run in CI,
never rot silently).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "reference" / "checks_kit.md"

_FENCE_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _fences() -> list[tuple[int, str]]:
    """Return (1-based ordinal, code) for every Python fence on the page."""

    text = DOC_PATH.read_text(encoding="utf-8")
    return [(i + 1, m.group("code")) for i, m in enumerate(_FENCE_RE.finditer(text))]


def test_page_has_the_committed_recipe_set() -> None:
    """The page keeps every memo-item-11 recipe (anti-vacuity guard)."""

    text = DOC_PATH.read_text(encoding="utf-8")
    fences = _fences()
    assert len(fences) >= 7, f"checks-kit page collapsed to {len(fences)} fences"
    for required in (
        "error_if_nonfinite",  # the torch-tripwire lead-in
        "optimizer.state",  # optimizer-state mapping scan
        "grad.",  # one-shot grad triage
        "torch.isfinite(loss)",  # loss-finiteness guard
        "tl.dead",  # dead-unit recipe (verdict stays with tl.dead)
        "clip_grad_norm_",  # the silent-zeroing warning
        "capture.hooks",  # the phase-seam spelling
    ):
        assert required in text, f"checks-kit page lost its {required!r} recipe"


def test_every_fence_executes_in_one_namespace() -> None:
    """Execute the page's fences top to bottom; a drifted recipe fails here.

    The compiled filename carries the fence ordinal, so a failure's traceback
    names the offending fence without any exception wrapping.
    """

    namespace: dict[str, object] = {}
    for ordinal, code in _fences():
        exec(compile(code, f"{DOC_PATH.name}:fence{ordinal}", "exec"), namespace)
