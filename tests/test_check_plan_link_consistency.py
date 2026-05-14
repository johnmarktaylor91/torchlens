"""Tests for documentation-plan structural link checks."""

from __future__ import annotations

from scripts.check_plan_link_consistency import find_invalid_s2_6_citations


def _v33_like_plan(citation_range: str) -> list[str]:
    """Build a plan fixture whose §2.6 table matches v33 line placement.

    Parameters
    ----------
    citation_range:
        Absolute range used by the synthetic §2.6 comparison-table citation.

    Returns
    -------
    list[str]
        Markdown lines with the comparison-table header at line 1166 and
        data rows at lines 1168-1184.
    """

    lines = [""] * 1191
    lines[1157] = "### §2.6 Comparison table"
    lines[1165] = (
        "| Feature | TorchLens | TransformerLens | nnsight | captum | torchexplorer | baukit |"
    )
    lines[1166] = "|---|---|---|---|---|---|---|"
    data_rows = [
        "| Captures every eager op (not just module outputs) | yes | no | partial | no | partial | no |",
        "| Works on arbitrary `nn.Module` (no rewrite) | yes | no | partial | yes | yes | yes |",
        "| Stable op-level graph metadata | yes | partial | partial | no | limited | limited |",
        "| Forward AND backward in one Trace object | yes | no | partial | no | no | no |",
        "| Backward intervention DSL (grad helpers) | yes | partial | partial | no | no | no |",
        "| Reusable selector DSL | yes | partial | partial | partial | no | partial |",
        "| Built-in helper transforms (forward) | yes | partial | partial | partial | no | partial |",
        "| Multi-run comparison built in | yes | no | partial | no | no | no |",
        "| Sparse predicate capture | yes | no | no | no | no | no |",
        "| Early abort | yes | no | no | no | no | no |",
        "| Portable save/load | yes | no | no | no | no | no |",
        "| Visualisation built in | yes | no | no | partial | yes | no |",
        "| FLOPs / MACs aggregation | yes | no | no | no | no | no |",
        "| Validation against autograd | yes | no | no | no | no | no |",
        "| Apple Silicon MLX backend | yes | no | no | no | no | no |",
        "| Compatibility report (`tl.compat.report`) | yes | no | no | no | no | no |",
        "| Mature transformer-internals shortcuts | no | yes | partial | no | no | partial |",
    ]
    for offset, row in enumerate(data_rows, start=1167):
        lines[offset] = row
    lines.append(f"The §2.6 comparison table at plan lines {citation_range} has 17 data rows.")
    return lines


def test_stale_s2_6_line_range_is_rejected() -> None:
    """The v33 stale range misses the first §2.6 data rows."""

    errors = find_invalid_s2_6_citations(_v33_like_plan("1172-1191"))

    assert [error.message for error in errors] == ["range 1172-1191 misses first §2.6 data row"]


def test_full_s2_6_line_range_is_accepted() -> None:
    """The v33 header-inclusive range contains both §2.6 boundary rows."""

    errors = find_invalid_s2_6_citations(_v33_like_plan("1166-1184"))

    assert errors == []


def test_historical_s2_6_line_range_is_ignored() -> None:
    """Historical §12 audit prose can quote stale ranges without failing."""

    lines = _v33_like_plan("1172-1191")
    lines[-1] = "### §12.35 Round 32 critique integration log"
    lines.append("The §2.6 comparison table at plan lines 1172-1191 had a stale range.")

    errors = find_invalid_s2_6_citations(lines)

    assert errors == []
