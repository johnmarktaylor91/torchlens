"""Op-segment span disclosure names hidden atomic module calls."""

from __future__ import annotations

from pathlib import Path

import torch

from tests.test_auto_collapse_metrics import SegmentToyBlock, _draw_source, _trace


class _AtomicGapStack(torch.nn.Module):
    """Two-op blocks with a bare atomic activation wedged between them."""

    def __init__(self) -> None:
        """Initialize blocks around one atomic top-level child."""

        super().__init__()
        self.layers = torch.nn.Sequential(
            SegmentToyBlock(),
            torch.nn.ReLU(),
            SegmentToyBlock(),
            SegmentToyBlock(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the atomic-gap stack forward pass."""

        return self.layers(x)


def test_op_segment_span_disclosure_counts_atomic_module_members(tmp_path: Path) -> None:
    """R19-5 span disclosure must name hidden ATOMIC module calls too.

    T9 (grind-p3, MED) red pin: the spanned-homes walk used the renderer's
    effective module stack, which drops an atomic module's own innermost
    level (a presentation choice — the renderer keeps the op and drops the
    box). Containment disclosure must not inherit that drop: a segment
    hiding a bare ``nn.ReLU`` child's op silently omitted its module call
    from the ``-- spans @...`` list, undercounting the hidden module calls.
    """

    trace = _trace(_AtomicGapStack(), torch.randn(1, 4, 8, 8))
    try:
        source = _draw_source(trace, tmp_path, "atomic_gap_spans", "max")

        assert "-- spans" in source
        spans_lines = [line for line in source.splitlines() if "-- spans" in line]
        assert any("layers.1:1" in line for line in spans_lines), (
            "the atomic nn.ReLU child's module call is missing from the "
            f"span disclosure: {spans_lines}"
        )
    finally:
        trace.cleanup()
