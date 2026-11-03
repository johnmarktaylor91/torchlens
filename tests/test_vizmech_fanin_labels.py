"""FIXD03-F13 regression gates: D03-R4 fan-in labels + D03-R5 oracle scope.

D03-R4 (head-label pileup): argument labels on children with visible fan-in
>= ``_ARG_LABEL_MIDPOINT_FANIN`` relocate from ``headlabel`` (painted
post-layout at one shared radius -- the densenet smear) to a midpoint
``label`` (reserved layout space); same-pair parallel arg edges fold into
ONE edge listing every argument row. Below the threshold the historical
head-label channel is preserved byte-for-byte.

D03-R5 (instrument scope): ``_geometry_audit._is_own_pair`` no longer
exempts an edge's head/tail label from its OWN spline/arrowheads (the
demonstrated toy_branchy false negative), and the Stage-0 lens oracle now
reads ``headlabel``/``taillabel`` glyph boxes exactly from the
``_hldraw_``/``_tldraw_`` draw ops (it was structurally blind to the
channel the ``arg N`` family rides).

Calibration discipline: every sharpened check is proven in BOTH directions
against PINNED hand-built DOT fixtures (renderer-independent, so product
fixes can never disarm the calibration) and against live renders.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._edge_multiplicity import _ARG_LABEL_MIDPOINT_FANIN
from torchlens._visualization._geometry_audit import (
    AuditResult,
    ParsedLayout,
    audit_layout,
    parse_layout_json,
    run_layout_json,
)


class _FanCat(nn.Module):
    """N distinct conv branches concatenated: the densenet ``cat_*`` geometry."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(nn.Conv2d(3, 2, kernel_size=1) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class _Branches(nn.Module):
    """One module producing N tensors (collapses to ONE rendered box)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(3, 2, kernel_size=1) for _ in range(n))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [conv(x) for conv in self.convs]


class _CollapsedFan(nn.Module):
    """A collapsed block feeding one ``cat`` at N slots: parallel same-pair edges."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.block = _Branches(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.block(x), dim=1)


class _FanStack(nn.Module):
    """N-way ``torch.stack`` fan-in from distinct scaled copies."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x * float(i + 1) for i in range(self.n)], 0).sum(0)


def _render_and_parse(
    model: nn.Module, x: torch.Tensor, tmp_path: Path, name: str, **kwargs: object
) -> tuple[str, ParsedLayout, AuditResult]:
    """Render through dot and audit through dot (the engine that rendered)."""

    torch.manual_seed(0)
    trace = tl.trace(model, x)
    dot = trace.draw(
        file_format="svg",
        path=str(tmp_path / name),
        **kwargs,
    ).dot
    parsed = parse_layout_json(run_layout_json(dot, "dot"), engine="dot", dot_source=dot)
    return dot, parsed, audit_layout(parsed)


def _arg_texts(parsed: ParsedLayout, kinds: tuple[str, ...]) -> list[str]:
    """Argument-row texts in the given label channels."""

    return sorted(text for text in parsed.text_multiset(kinds) if str(text).startswith("arg (0, "))


# ---------------------------------------------------------------------------
# D03-R4: the fix
# ---------------------------------------------------------------------------


def test_high_fanin_arg_labels_relocate_to_midpoint(tmp_path: Path) -> None:
    """Fan-in 12 distinct sources: zero violations, full row inventory."""

    dot, parsed, result = _render_and_parse(_FanCat(12), torch.zeros(1, 3, 8, 8), tmp_path, "fan12")
    assert result.hard_violation_count == 0, result.describe("fan12")
    # The relocation must conserve the inventory (a fix that DELETES labels
    # must never score a free zero) and land on the midpoint channel.
    assert _arg_texts(parsed, ("edge-midpoint",)) == sorted(f"arg (0, {k})" for k in range(12))
    assert not _arg_texts(parsed, ("edge-head", "edge-tail"))
    result.require_minimums(edge_midpoint=12, node=13)


def test_low_fanin_arg_labels_keep_headlabel_channel(tmp_path: Path) -> None:
    """Below the threshold the historical head-label channel is untouched."""

    below = _ARG_LABEL_MIDPOINT_FANIN - 1
    dot, parsed, result = _render_and_parse(_FanStack(below), torch.randn(2, 3), tmp_path, "fanlow")
    assert len(_arg_texts(parsed, ("edge-head",))) == below
    assert not _arg_texts(parsed, ("edge-midpoint",))


def test_threshold_boundary_relocates_exactly_at_gate(tmp_path: Path) -> None:
    """Fan-in exactly ``_ARG_LABEL_MIDPOINT_FANIN`` relocates; one less keeps."""

    at = _ARG_LABEL_MIDPOINT_FANIN
    dot, parsed, result = _render_and_parse(_FanStack(at), torch.randn(2, 3), tmp_path, "fanat")
    assert len(_arg_texts(parsed, ("edge-midpoint",))) == at
    assert not _arg_texts(parsed, ("edge-head",))
    assert result.hard_violation_count == 0, result.describe("fanat")


def test_parallel_same_pair_arg_edges_merge_into_one_edge(tmp_path: Path) -> None:
    """A collapsed block feeding cat at N slots renders ONE merged edge.

    N parallel edges between the same two boxes carry no distinguishing
    per-edge geometry, so the merged reserved-space midpoint label listing
    every argument row is the same information without the N-way band (the
    densenet ``collapse="auto"`` shape: 302 arg-label violations -> 0).
    """

    n = 6
    dot, parsed, result = _render_and_parse(
        _CollapsedFan(n), torch.zeros(1, 3, 8, 8), tmp_path, "samepair", depth=1
    )
    assert result.hard_violation_count == 0, result.describe("samepair")
    # Full row inventory on the midpoint channel...
    assert _arg_texts(parsed, ("edge-midpoint",)) == sorted(f"arg (0, {k})" for k in range(n))
    # ...carried by exactly ONE rendered edge into the cat node (the merge),
    # not N parallel occurrence edges.
    cat_heads = [edge for edge in parsed.edges if edge.head.startswith("cat")]
    arg_owners = {
        box.owner
        for box in parsed.textboxes
        if box.kind == "edge-midpoint" and str(box.text).startswith("arg (0, ")
    }
    assert len(cat_heads) == 1, [edge.name for edge in cat_heads]
    assert len(arg_owners) == 1


def test_rolled_mode_high_fanin_relocates(tmp_path: Path) -> None:
    """The rolled (Layer) counting branch honors the same fan-in gate."""

    dot, parsed, result = _render_and_parse(
        _FanCat(8), torch.zeros(1, 3, 8, 8), tmp_path, "fanrolled", rolled=True
    )
    assert result.hard_violation_count == 0, result.describe("fanrolled")
    assert len(_arg_texts(parsed, ("edge-midpoint",))) == 8
    assert not _arg_texts(parsed, ("edge-head",))


# ---------------------------------------------------------------------------
# D03-R5a: own-edge endpoint labels are no longer exempt in the audit
# ---------------------------------------------------------------------------


def test_own_node_label_exemption_retained(tmp_path: Path) -> None:
    """The narrowing is scoped: a node's own label stays exempt (clean chain)."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    dot, parsed, result = _render_and_parse(model, torch.randn(1, 4), tmp_path, "chain")
    assert result.hard_violation_count == 0, result.describe("chain")
    result.require_minimums(node_label=4)


# ---------------------------------------------------------------------------
# D03-R5b: the Stage-0 lens oracle sees head/tail-label text
# ---------------------------------------------------------------------------
