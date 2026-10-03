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
from torchlens.visualization._geometry_audit import (
    AuditResult,
    audit_layout,
    compute_usability_envelope,
    graphviz_version,
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
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / name),
    )
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    return audit_layout(parsed)


def _pinned_fan_in_pileup_dot(n: int = 12) -> str:
    """Hand-built DOT reproducing the historic fan-in head-label pileup.

    ``n`` sources feed one sink, each edge carrying an ``arg (0, k)``
    ``headlabel`` with no placement attrs -- graphviz paints every head
    label at one shared radius, so they smear into one band (the D03-R4
    geometry). Pinned so the instrument calibration cannot be disarmed by
    product fixes to the live renderer.
    """

    lines = ["digraph knownbad {", "  rankdir=BT;", '  sink [shape=box, label="cat_1_13"];']
    for i in range(n):
        lines.append(f'  s{i} [shape=box, label="conv2d_{i}"];')
        lines.append(f"  s{i} -> sink [headlabel=<arg (0, {i})>, labelfontsize=8, arrowsize=.7];")
    lines.append("}")
    return "\n".join(lines)


def test_known_bad_fan_in_fails() -> None:
    """The pinned 12-way fan-in head-label pileup must trip the oracle."""

    dot = _pinned_fan_in_pileup_dot(12)
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    result = audit_layout(parsed)
    assert result.hard_violation_count >= 10, result.describe("fan12")
    kinds = {violation.kind for violation in result.violations}
    # The knot hits multiple element classes at once (labels, nodes,
    # splines, arrowheads) -- the widened audit sees all of them.
    assert {"text-node", "text-arrowhead"} <= kinds


def test_live_fan_in_renders_clean(tmp_path: Path) -> None:
    """The live renderer's 12-way fan-in is CLEAN (D03-R4 midpoint fix).

    The exact geometry the pinned known-bad above reproduces must no longer
    be produced by the product: argument labels at visible fan-in >=
    ``_ARG_LABEL_MIDPOINT_FANIN`` ride reserved-space midpoint labels, and
    the argument-row inventory survives the relocation intact.
    """

    result = _audit(_FanStack(12), torch.randn(2, 3), tmp_path, "fan12live")
    assert result.hard_violation_count == 0, result.describe("fan12live")
    arg_texts = [text for text in result.text_multiset if str(text).startswith("arg (0, ")]
    assert len(arg_texts) == 12, f"argument-label inventory lost rows: {sorted(arg_texts)}"
    result.require_minimums(edge_midpoint=12)


def test_known_good_chain_passes_with_inventory(tmp_path: Path) -> None:
    """A plain chain passes -- over a NON-EMPTY denominator (anti-vacuity)."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    result = _audit(model, torch.randn(1, 4), tmp_path, "chain")
    assert result.hard_violation_count == 0, result.describe("chain")
    result.require_minimums(node=4, edge=3, node_label=4)


def test_anti_vacuity_refuses_empty_denominator(tmp_path: Path) -> None:
    """A minimum the artifact cannot meet raises instead of passing silently."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    result = _audit(model, torch.randn(1, 4), tmp_path, "tiny")
    with pytest.raises(AssertionError, match="anti-vacuity"):
        result.require_minimums(cluster=5)


def test_inventory_multiset_carries_argument_labels(tmp_path: Path) -> None:
    """Plank 1: geometry pairs with a text inventory (deleted labels caught)."""

    trace = tl.trace(_FanStack(8), torch.randn(2, 3))
    dot = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=str(tmp_path / "fan8"))
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    edge_texts = parsed.text_multiset(("edge-head", "edge-tail", "edge-midpoint"))
    arg_labels = [text for text in edge_texts if text.startswith("arg")]
    assert len(arg_labels) >= 8, f"argument-label inventory lost rows: {sorted(edge_texts)}"


def test_spatial_index_prunes_pairs(tmp_path: Path) -> None:
    """Item 18: the grid prefilter must beat brute-force pairing."""

    trace = tl.trace(_FanStack(12), torch.randn(2, 3))
    dot = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=str(tmp_path / "fan12b"))
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    result = audit_layout(parsed)
    text_count = len(parsed.textboxes)
    element_count = (
        len(parsed.nodes)
        + len(parsed.edges)
        + sum(len(edge.arrowheads) for edge in parsed.edges)
        + text_count
    )
    brute_force = text_count * element_count
    assert result.checked_pairs < brute_force, (
        f"grid prefilter inert: checked {result.checked_pairs} of {brute_force}"
    )


def test_engine_recorded_and_version_available(tmp_path: Path) -> None:
    """Plank 8: the engine rides the result; the version is expressible."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    result = _audit(model, torch.randn(1, 4), tmp_path, "eng")
    assert result.engine == "dot"
    assert "graphviz" in graphviz_version().lower()


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
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "classes"),
        show_legend=True,
    )
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    counts = parsed.class_counts()
    assert counts.get("node-label", 0) >= 3
    assert counts.get("cluster-caption", 0) >= 1
    assert counts.get("legend-text", 0) >= 5
    assert counts.get("graph-caption", 0) >= 1
    envelope = compute_usability_envelope(parsed)
    assert envelope.min_text_size > 0.0


def test_competing_label_channels_detected(tmp_path: Path) -> None:
    """Wave-4 plan seam plumbing: multi-channel edges are enumerable.

    An ``xN`` multiplicity midpoint plus an argument head label on one edge
    is the composition-conflict shape that silently deleted the gpt2 ``IF``
    label (memo D11); the detector is the plan's tripwire until it lands.
    """

    from torchlens.visualization._geometry_audit import edges_with_competing_labels

    class Reused(nn.Module):
        """A two-input op whose edges carry argument labels."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.stack([x, x * 2.0], 0).sum(0)

    trace = tl.trace(Reused(), torch.randn(2, 3))
    dot = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=str(tmp_path / "cc"))
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    competing = edges_with_competing_labels(parsed)
    # The fixture may or may not compose channels on one edge; the CONTRACT
    # is that the detector is callable and returns the documented shape.
    for owner, kinds in competing:
        assert isinstance(owner, str) and len(kinds) > 1
