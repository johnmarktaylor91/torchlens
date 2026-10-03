"""Observe-kit item 4: the pass-level-peak publication gate + its source lint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.observe import format_pass_peak, pass_peak_facts

TORCHLENS_DIR = Path(tl.__file__).resolve().parent

#: Display packages whose readers of a pass-level peak must route through the
#: gate. Capture/schema internals legitimately read the raw fields.
_DISPLAY_PACKAGES = ("report", "export", "visualization", "notebook", "bridge", "observe")

#: Files sanctioned to read the raw fields inside display packages: the gate
#: itself.
_GATE_MODULE = "observe/_peaks.py"


class _FakeTrace:
    """Minimal trace stub carrying the two publication-gated fields."""

    def __init__(self, backend: str | None, peak: int | None) -> None:
        self.forward_memory_backend = backend
        self.forward_peak_memory = peak


def test_cuda_zero_renders_as_high_water_fact() -> None:
    """A CUDA 0 is 'no high-water advance', never 'used 0 bytes'."""

    facts = pass_peak_facts(_FakeTrace("cuda", 0))
    assert facts.value_bytes == 0
    assert facts.ratio_eligible is False
    line = format_pass_peak(_FakeTrace("cuda", 0))
    assert "high-water" in line
    assert "0 B advance" in line


def test_cpu_renders_labeled_and_never_ratio_eligible() -> None:
    """CPU values carry the RSS meaning and are ratio-ineligible."""

    facts = pass_peak_facts(_FakeTrace("cpu", 332_000))
    assert facts.ratio_eligible is False
    assert "RSS" in facts.meaning
    line = format_pass_peak(_FakeTrace("cpu", 332_000))
    assert "cpu basis" in line
    assert "NOT a tensor peak" in line


def test_mps_and_unknown_render_labeled() -> None:
    """MPS values carry the allocator-delta meaning; unknown stays unavailable."""

    assert "allocator delta" in pass_peak_facts(_FakeTrace("mps", 1024)).meaning
    unavailable = pass_peak_facts(_FakeTrace(None, None))
    assert unavailable.value_bytes is None
    assert "unavailable" in format_pass_peak(_FakeTrace("unknown", 5))


def test_positive_cuda_peak_is_the_only_ratio_eligible_value() -> None:
    """Only a positive CUDA device peak may feed a derived ratio."""

    assert pass_peak_facts(_FakeTrace("cuda", 4096)).ratio_eligible is True
    for backend, value in (("cuda", 0), ("cpu", 4096), ("mps", 4096), (None, None)):
        assert pass_peak_facts(_FakeTrace(backend, value)).ratio_eligible is False


def test_summary_footer_routes_through_the_gate() -> None:
    """The shipped summary footer renders through format_pass_peak."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    captured = tl.trace(model, torch.randn(2, 4))
    try:
        summary_text = captured.summary(level="memory")
        assert "Live forward-memory peak" in summary_text
        # On CPU the line must carry the labeled RSS meaning or honest absence,
        # never a bare unlabeled "measured" claim.
        assert ("NOT a tensor peak" in summary_text) or ("unavailable" in summary_text)
    finally:
        captured.cleanup()


def test_lint_display_surfaces_read_peaks_only_through_the_gate() -> None:
    """LINT: display packages reading raw pass-peak fields must use the gate.

    A new print site that interpolates ``forward_peak_memory`` /
    ``backward_peak_memory`` without routing through
    ``torchlens.observe._peaks`` fails here with the teaching message.
    """

    offenders: list[str] = []
    for package in _DISPLAY_PACKAGES:
        root = TORCHLENS_DIR / package
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            relative = str(path.relative_to(TORCHLENS_DIR))
            if relative.endswith(_GATE_MODULE.replace("/", str(Path("/")))) or relative.replace(
                "\\", "/"
            ).endswith(_GATE_MODULE):
                continue
            text = path.read_text(encoding="utf-8")
            reads_peaks = "forward_peak_memory" in text or "backward_peak_memory" in text
            uses_gate = "format_pass_peak" in text or "pass_peak_facts" in text
            if reads_peaks and not uses_gate:
                offenders.append(relative)
    assert not offenders, (
        "display surfaces reading pass-level peaks without the publication "
        "gate (route through torchlens.observe._peaks.format_pass_peak / "
        f"pass_peak_facts -- the basis must always be named): {offenders}"
    )
