"""F09 CP4: costreport items 18 + 22 -- the device-attribution docs wave
and the ladder generator's rung-name assertion.

D28: every python code block on docs/reference/device_attribution.md
executes in CI (this panel shipped a wrong emit_nvtx cite in its own
round 1 -- an untested docs page is a second source of truth about the
API), plus the two-sided NVTX effect witness: ranges appear when enabled
and not when disabled.

D26.4 (item 22): the ladder generator asserts every rung name it renders
exists in the artifact it was handed (`rerun_no_save` silently becoming
`rerun_metadata_only` is the measured regression class).
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest
import torch
import torch.nn as nn

import torchlens as tl

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC_PAGE = REPO_ROOT / "docs" / "reference" / "device_attribution.md"

_PY_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_every_docs_code_block_executes() -> None:
    """D28: the page's python blocks run top-to-bottom in one namespace."""

    blocks = _PY_FENCE.findall(DOC_PAGE.read_text(encoding="utf-8"))
    assert len(blocks) >= 4
    namespace: dict[str, object] = {}
    for block in blocks:
        exec(compile(block, str(DOC_PAGE), "exec"), namespace)  # noqa: S102 -- docs CI
    log = namespace.get("log")
    assert log is not None
    log.cleanup()  # type: ignore[attr-defined]


def test_docs_page_banned_words_hold() -> None:
    """Memo 3.8: no bare 'achieved'/'throughput' claims outside the ban text."""

    text = DOC_PAGE.read_text(encoding="utf-8").lower()
    # The words appear ONLY in sentences stating they are banned/gated.
    for word in ("achieved", "throughput"):
        for line in text.splitlines():
            if word in line:
                assert any(
                    marker in line
                    for marker in ("ban", "without joined", "not called", "prints", "no ")
                ), line


def test_nvtx_two_sided_effect_witness(monkeypatch) -> None:
    """D28: ranges appear when enabled and NOT when disabled."""

    calls: list[str] = []

    def _record_push(name: str) -> None:
        calls.append(str(name))

    def _record_pop() -> None:
        return None

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", _record_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", _record_pop)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    x = torch.randn(2, 4)

    off = tl.trace(model, x)
    off.cleanup()
    assert not [name for name in calls if name.startswith("torchlens::")]

    on = tl.trace(model, x, capture=tl.options.CaptureOptions(emit_nvtx=True))
    on.cleanup()
    ranges = [name for name in calls if name.startswith("torchlens::")]
    assert ranges, "emit_nvtx=True emitted no torchlens:: NVTX ranges"
    assert any("linear" in name for name in ranges)


def test_generator_asserts_rung_names_exist() -> None:
    """D26.4 (item 22): prose never renders for a vanished rung."""

    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    try:
        from generate_perf_numbers import REFERENCED_RUNGS, render_numbers_markdown
    finally:
        sys.path.pop(0)

    good_rows = [
        {
            "model": "tinynet",
            "device": "cpu",
            "operation": name,
            "status": "ok",
            "passes": {"timing": {"median_ms": 1.0}},
        }
        for name in sorted(REFERENCED_RUNGS)
    ]
    canonical = {"rows": good_rows, "date": "2026-01-01", "baseline_status": "canonical"}
    assert "raw_forward" in render_numbers_markdown(canonical)

    renamed_rows = [dict(row, operation=row["operation"] + "_renamed") for row in good_rows]
    with pytest.raises(ValueError, match="absent from the canonical artifact"):
        render_numbers_markdown(
            {"rows": renamed_rows, "date": "2026-01-01", "baseline_status": "canonical"}
        )
    # A provisional/partial payload legitimately omits tiers: no refusal.
    assert render_numbers_markdown(
        {"rows": renamed_rows, "date": "2026-01-01", "baseline_status": "provisional"}
    )


def test_shipped_baseline_passes_the_rung_assertion() -> None:
    """The canonical 196-row artifact satisfies the generator's references."""

    import json

    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    try:
        from generate_perf_numbers import render_numbers_markdown
    finally:
        sys.path.pop(0)
    payload = json.loads(
        (REPO_ROOT / "benchmarks" / "perf_baselines" / "linux-cpu.json").read_text()
    )
    assert "raw_forward" in render_numbers_markdown(payload)
