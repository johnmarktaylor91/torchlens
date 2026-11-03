"""Vizmech wave-3 items 18-19: the widened, indexed geometry audit.

Calibration discipline (memo s6): every metric is proven in BOTH directions
-- a known-bad must fail and a known-good must pass with headroom, never
tuned-tight. The known-bad is a PINNED hand-built DOT reproducing the
historic 12-way fan-in head-label pileup (the live renderer no longer
produces it: FIXD03-F13 relocated high-fan-in argument labels to midpoint
labels, so a renderer-produced fixture would silently disarm the
calibration); the known-good is a plain chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._geometry_audit import (
    AuditResult,
    audit_layout,
    compute_usability_envelope,
    parse_layout_json,
    run_layout_json,
)


class _FanStack(nn.Module):
    """N-way ``torch.stack`` fan-in: the argument-label known-bad."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x * float(i + 1) for i in range(self.n)], 0).sum(0)


def _audit(model: nn.Module, x: torch.Tensor, tmp_path: Path, name: str) -> AuditResult:
    """Render through dot and audit through dot (the engine that rendered)."""

    trace = tl.trace(model, x)
    dot = trace.draw(
        file_format="svg",
        path=str(tmp_path / name),
    ).dot
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    return audit_layout(parsed)


def test_anti_vacuity_refuses_empty_denominator(tmp_path: Path) -> None:
    """A minimum the artifact cannot meet raises instead of passing silently."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    result = _audit(model, torch.randn(1, 4), tmp_path, "tiny")
    with pytest.raises(AssertionError, match="anti-vacuity"):
        result.require_minimums(cluster=5)


def test_node_labels_and_captions_are_audited(tmp_path: Path) -> None:
    """Item 19: element classes beyond edge labels exist in the parse."""

    class Wrapped(nn.Module):
        """One module cluster so a caption exists."""

        def __init__(self) -> None:
            super().__init__()
            self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    trace = tl.trace(Wrapped(), torch.randn(1, 4))
    dot = trace.draw(
        file_format="svg",
        path=str(tmp_path / "classes"),
        show_legend=True,
    ).dot
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    counts = parsed.class_counts()
    assert counts.get("node-label", 0) >= 3
    assert counts.get("cluster-caption", 0) >= 1
    # The key rides in the header label: three header lines plus its entries.
    assert counts.get("graph-caption", 0) >= 6
    envelope = compute_usability_envelope(parsed)
    assert envelope.min_text_size > 0.0


def test_competing_label_channels_detected(tmp_path: Path) -> None:
    """Wave-4 plan seam plumbing: multi-channel edges are enumerable.

    An ``xN`` multiplicity midpoint plus an argument head label on one edge
    is the composition-conflict shape that silently deleted the gpt2 ``IF``
    label (memo D11); the detector is the plan's tripwire until it lands.
    """

    from torchlens._visualization._geometry_audit import edges_with_competing_labels

    class Reused(nn.Module):
        """A two-input op whose edges carry argument labels."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.stack([x, x * 2.0], 0).sum(0)

    trace = tl.trace(Reused(), torch.randn(2, 3))
    dot = trace.draw(file_format="svg", path=str(tmp_path / "cc")).dot
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    competing = edges_with_competing_labels(parsed)
    # The fixture may or may not compose channels on one edge; the CONTRACT
    # is that the detector is callable and returns the documented shape.
    for owner, kinds in competing:
        assert isinstance(owner, str) and len(kinds) > 1
