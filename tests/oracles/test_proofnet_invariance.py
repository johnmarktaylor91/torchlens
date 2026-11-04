"""Invariance witnesses for declared cache-key dont-cares (item 6b; wave 1-2).

D14 inverted the burden: every field declared irrelevant to the capture-cache
key carries a WITNESS SLOT -- a registered test proving the field cannot
change capture semantics. This module funds the two mechanically witnessable
empty slots (cache, cache_dir) and enforces the closure rule: a dont-care
row's slot is either a witness node that EXISTS on disk or a dated
``invariance_witness`` KNOWN-GAP row. An empty, un-gapped slot is red.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from tests.oracles._known_gaps import load_known_gaps
from tests.oracles._lints import load_dont_care

REPO_ROOT = Path(__file__).resolve().parents[2]


def _capture_digest(**capture_overrides: Any) -> str:
    """Structural + payload digest of a fixture capture under options."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    torch.manual_seed(1)
    trace = tl.trace(
        model, torch.randn(2, 4), capture=tl.options.CaptureOptions(**capture_overrides)
    )
    try:
        digest = hashlib.sha256()
        for label in trace.layer_labels:
            digest.update(label.encode())
            payload = trace[label].out
            if isinstance(payload, torch.Tensor):
                digest.update(payload.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    finally:
        trace.cleanup()


def test_cache_toggle_is_capture_invariant(tmp_path: Any) -> None:
    """cache=True (cold store) and cache=False capture identical products."""

    baseline = _capture_digest()
    cached_cold = _capture_digest(cache=True, cache_dir=str(tmp_path / "store-a"))
    assert cached_cold == baseline, (
        "flipping cache= changed the captured product -- the dont-care"
        " declaration for 'cache' is FALSE; key it or fix the capture path"
    )


def test_cache_dir_is_capture_invariant(tmp_path: Any) -> None:
    """Two different cold cache stores capture identical products."""

    first = _capture_digest(cache=True, cache_dir=str(tmp_path / "store-b"))
    second = _capture_digest(cache=True, cache_dir=str(tmp_path / "store-c"))
    assert first == second, (
        "the cache STORE LOCATION changed the captured product -- the"
        " dont-care declaration for 'cache_dir' is FALSE"
    )


def _witness_exists(node: str) -> bool:
    if "::" in node:
        path_part, test_name = node.split("::", 1)
        path = REPO_ROOT / path_part
        if not path.is_file():
            return False
        return bool(re.search(rf"def {re.escape(test_name.split('::')[-1])}\b", path.read_text()))
    return (REPO_ROOT / node).is_file()


def test_every_dont_care_slot_is_witnessed_or_gapped() -> None:
    """Closure (6b): witness node on disk XOR dated KNOWN-GAP row."""

    gapped_keys = {gap.key for gap in load_known_gaps() if gap.predicate == "invariance_witness"}
    problems: list[str] = []
    for field, witness in load_dont_care().items():
        if witness:
            if not _witness_exists(witness):
                problems.append(f"{field}: witness node missing from tree ({witness})")
            if field in gapped_keys:
                problems.append(
                    f"{field}: BOTH a witness and a KNOWN-GAP row -- delete the"
                    " stale gap row (monotone burn-down)"
                )
        elif field not in gapped_keys:
            problems.append(
                f"{field}: empty witness slot with NO invariance_witness gap"
                " row -- fund the witness or ledger the debt (D14)"
            )
    assert not problems, "\n".join(problems)


def test_show_edge_shapes_is_capture_invariant() -> None:
    """show_edge_shapes is a draw-only flag: no capture path can read it.

    The flag (walk ruling names.1) lives on ``VisualizationOptions`` and
    ``Trace.draw``; ``tl.trace`` takes no visualization group, and no module
    under ``torchlens/_capture`` names it, so it cannot change a captured
    product or the capture-cache key's inputs.
    """

    import inspect

    import torchlens as tl

    trace_params = inspect.signature(tl.trace).parameters
    assert "show_edge_shapes" not in trace_params
    assert not any(
        "VisualizationOptions" in str(param.annotation) for param in trace_params.values()
    ), "tl.trace now accepts VisualizationOptions; key show_edge_shapes instead"
    readers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "torchlens" / "_capture").rglob("*.py"))
        if "show_edge_shapes" in path.read_text("utf-8")
    ]
    assert readers == [], (
        f"capture code reads show_edge_shapes ({readers}) -- the dont-care"
        " declaration for 'VisualizationOptions.show_edge_shapes' is FALSE; key it"
    )
