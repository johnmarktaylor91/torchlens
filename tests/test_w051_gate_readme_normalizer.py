"""AUD-CODE 3.10 regression: the README gate's volatile normalizer.

The README first screen prints ``log.summary()``, whose Memory line carries
the host-RSS forward peak (a runtime measurement documented as never
portable). Two cold runs printed ``78.9 MB`` and ``80 MB`` and the gate went
red non-deterministically. The normalizer now declares exactly that TOKEN
volatile; the activation byte counts on the same line stay pinned.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_GATE = Path(__file__).resolve().parent / "test_quickstart_readme_gate.py"


def _normalize():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("readme_gate_under_test", _GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._normalize


_LINE_A = (
    "Memory   forward peak 78.9 MB (process RSS growth (NOT a tensor peak)) | "
    "activations at capture 33.6 MB, retained now 33.6 MB"
)
_LINE_B = (
    "Memory   forward peak 80 MB (process RSS growth (NOT a tensor peak)) | "
    "activations at capture 33.6 MB, retained now 33.6 MB"
)
_LINE_C = (
    "Memory   forward peak 80 MB (process RSS growth (NOT a tensor peak)) | "
    "activations at capture 34.0 MB, retained now 33.6 MB"
)


def test_the_two_observed_peak_lines_normalize_equal() -> None:
    normalize = _normalize()
    assert normalize(_LINE_A) == normalize(_LINE_B)
    assert "<volatile>" in normalize(_LINE_A)


def test_activation_byte_counts_stay_pinned() -> None:
    """The normalizer is token-scoped: a drifted activation count still fails."""

    normalize = _normalize()
    assert normalize(_LINE_B) != normalize(_LINE_C)
    assert "activations at capture 33.6 MB" in normalize(_LINE_B)


def test_unavailable_peak_wording_is_not_normalized() -> None:
    """Only a measured peak is volatile; the unavailable disclosure is pinned text."""

    normalize = _normalize()
    line = "Memory   forward peak unavailable (not measured) | activations at capture 1 MB"
    assert normalize(line) == line
