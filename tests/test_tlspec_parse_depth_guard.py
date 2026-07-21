"""r55 CLASS 4 immunizer -- bounded JSON/literal parse at every manifest boundary.

r54 ``free_2`` (LOW-MED): a deeply-nested ``manifest.json`` blew the C recursion
stack in stdlib ``json.load`` at load / format-detection -- an uncaught
``RecursionError`` that escaped ``tl.load(path)`` before the descriptor-parse
graceful-degradation net ran. The class is closed by ONE bounded reader
(``_io/_json``: byte ceiling + string-aware depth prescan BEFORE ``json.loads``)
routed at every manifest boundary, plus an independent depth counter through
``runnable_load._parse_literal``. Over-limit degrades typed; never crashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import _json
from torchlens._io.runnable_load import _MAX_LITERAL_NESTING_DEPTH, _parse_literal

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# (a) bounded JSON reader                                                      #
# --------------------------------------------------------------------------- #


def test_bounded_json_refuses_deep_nesting_without_recursion() -> None:
    """A depth far past the ceiling is a typed ``JSONDecodeError``, never a crash."""

    deep = "[" * 5000 + "]" * 5000
    with pytest.raises(json.JSONDecodeError):
        _json.loads_bounded(deep)


def test_bounded_json_refuses_oversize_payload() -> None:
    """An over-size payload is a typed ``JSONDecodeError`` before parsing."""

    with pytest.raises(json.JSONDecodeError):
        _json.loads_bounded("[]", max_bytes=1)


def test_bounded_json_parses_normal_manifest() -> None:
    """A normal shallow object parses identically to stdlib ``json``."""

    obj = {"a": 1, "b": [1, 2, {"c": [3, 4]}], "d": "text with [brackets] {inside}"}
    text = json.dumps(obj)
    assert _json.loads_bounded(text) == obj


def test_bounded_json_string_aware_depth() -> None:
    """Brackets inside string literals do not inflate the measured depth."""

    text = json.dumps({"k": "[[[[[[[[[[ not real nesting ]]]]]]]]]]"})
    assert _json.loads_bounded(text, max_depth=3) == {"k": "[[[[[[[[[[ not real nesting ]]]]]]]]]]"}


# --------------------------------------------------------------------------- #
# (b) independent literal-depth counter                                       #
# --------------------------------------------------------------------------- #


def test_parse_literal_refuses_over_depth_without_recursion() -> None:
    """A nested literal past the ceiling raises ``ValueError``, not ``RecursionError``."""

    node: dict = {"kind": "int", "value": 0}
    for _ in range(_MAX_LITERAL_NESTING_DEPTH + 50):
        node = {"kind": "list", "items": [node]}
    with pytest.raises(ValueError):
        _parse_literal(node)


def test_parse_literal_accepts_shallow_nesting() -> None:
    """A shallow nested literal parses cleanly (no over-refusal)."""

    node: dict = {"kind": "int", "value": 0}
    for _ in range(20):
        node = {"kind": "list", "items": [node]}
    parsed = _parse_literal(node)
    assert parsed is not None


# --------------------------------------------------------------------------- #
# (c) end-to-end: deep manifest degrades typed, never RecursionError           #
# --------------------------------------------------------------------------- #


class _M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def test_deeply_nested_manifest_does_not_crash_load(tmp_path: Path) -> None:
    """A depth-900 nested ``manifest.json`` never escapes ``tl.load`` as a crash."""

    trace = tl.trace(_M().eval(), torch.randn(2, 4), intervention_ready=True)
    bundle = tmp_path / "deep.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)

    manifest_path = bundle / "manifest.json"
    text = manifest_path.read_text()
    # Splice a depth-900 nested array as a new top-level key WITHOUT recursing in
    # the builder (raw string construction, mirroring the free_2 repro).
    depth = 900
    raw = "[" * depth + "0" + "]" * depth
    injected = '{"__deep__": ' + raw + ", " + text[1:]
    manifest_path.write_text(injected)

    try:
        tl.load(str(bundle))
    except RecursionError:  # pragma: no cover - the exact failure we forbid
        pytest.fail("uncaught RecursionError escaped tl.load() on a deep manifest")
    except Exception:
        pass  # any typed disposition (TorchLensIOError / analysis-only) is acceptable


def test_normal_bundle_still_loads(tmp_path: Path) -> None:
    """The bounded reader does not perturb a legitimate load."""

    trace = tl.trace(_M().eval(), torch.randn(2, 4), intervention_ready=True)
    bundle = tmp_path / "clean.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    loaded = tl.load(str(bundle))
    assert loaded is not None


# --------------------------------------------------------------------------- #
# (d) source-scan: no bare json.load(s) at an _io/io artifact boundary          #
# --------------------------------------------------------------------------- #


def test_no_bare_json_read_in_io_packages() -> None:
    """Every ``_io``/``io`` JSON READ routes through the bounded helper."""

    roots = [Path(tl.__file__).parent / "_io", Path(tl.__file__).parent / "io"]
    offenders: list[str] = []
    for root in roots:
        for source_path in sorted(root.rglob("*.py")):
            if source_path.name == "_json.py":
                continue
            for lineno, line in enumerate(source_path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "json.load(" in line or "json.loads(" in line:
                    offenders.append(f"{source_path.name}:{lineno}: {stripped}")
    assert not offenders, "bare json.load(s) at an artifact boundary (use _json.*_bounded): " + str(
        offenders
    )
