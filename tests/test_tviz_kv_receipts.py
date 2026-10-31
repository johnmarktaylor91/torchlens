"""KV-group receipts: grouped-query ablations say which query heads share a cell.

Under grouped-query attention a key or value site stores one tensor per
KV group, read by several query heads, so a grid over that site measures
groups, not query heads. The receipt must carry the grouping, every
rendering must name each group's query heads, and nothing may present a
group's effect as one query head's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens
import torchlens.tviz as tviz
from torchlens.mechinterp import PatchGrid

GQA = tviz.GqaInfo(n_query_heads=8, n_kv_heads=2)


def _grid(n_heads: int) -> PatchGrid:
    """A two-layer patch grid over ``n_heads`` head coordinates."""

    return PatchGrid(
        axes=("layer", "head"),
        coordinates={"layer": ("blk.0.attn", "blk.1.attn"), "head": tuple(range(n_heads))},
        values=torch.arange(2 * n_heads, dtype=torch.float32).reshape(2, n_heads) / 4 - 0.5,
        receipts={
            "engine": "rerun",
            "fires": 2 * n_heads,
            "metric_provenance": "logit_diff",
            "clean_metric": 1.0,
            "corrupted_metric": 0.0,
        },
    )


def _kv_receipt(layer: str = "blk.0.attn") -> tviz.CausalReceipt:
    return tviz.receipt_from_patch_grid(
        _grid(2),
        layer=layer,
        negative_control=0.0,
        positive_control=3.0,
        joint_effect=0.4,
        intervention="kv group value patched",
        kv_groups=GQA,
    )


def test_receipt_names_the_query_heads_of_each_group() -> None:
    """The receipt's cells are groups and it says who shares each one."""

    receipt = _kv_receipt()
    assert receipt.unit == "kv group"
    assert receipt.heads == (0, 1)
    assert receipt.sharing_lines() == (
        "kv group 1 of 2, shared by query heads 0-3",
        "kv group 2 of 2, shared by query heads 4-7",
    )
    assert GQA.query_heads(1) == (4, 5, 6, 7)
    # The per-query-head header the attention panels use is unchanged.
    assert GQA.header(5) == "kv group 2 of 2, shared by query heads 4-7"


def test_query_head_grid_refuses_the_kv_grouping() -> None:
    """A grid over all eight query heads is not a kv-site grid."""

    with pytest.raises(tviz.TvizError) as caught:
        tviz.receipt_from_patch_grid(
            _grid(8),
            layer="blk.0.attn",
            negative_control=0.0,
            positive_control=3.0,
            kv_groups=GQA,
        )
    assert caught.value.fields["code"] == "tv_record_invalid"
    assert "2 kv groups" in str(caught.value)


def test_grid_figure_discloses_the_sharing(tmp_path: Path) -> None:
    """The rendered grid states the unit, the sharing and the group labels."""

    rows = [_kv_receipt("blk.0.attn"), _kv_receipt("blk.1.attn")]
    artifact = torchlens.viz.ablation_grid(
        rows, tmp_path / "kv.svg", joint_effect=-0.2, svg_fonttype="none"
    )
    lines = "\n".join(artifact.disclosure_lines)
    assert (
        "kv-group receipt: each cell intervenes on one key/value group shared by 4 query heads "
        "(8 query heads over 2 groups); no cell is a per-query-head effect"
    ) in lines
    assert (
        "kv group 1 of 2, shared by query heads 0-3; kv group 2 of 2, shared by query heads 4-7"
    ) in lines
    assert (
        "joint effect of the same intervention on all 4 measured kv groups together: -0.2" in lines
    )
    assert "measured heads" not in lines
    svg = artifact.paths[0].read_text()
    for label in ("kv 1", "q0-3", "kv 2", "q4-7", "measured kv-group effects on logit_diff"):
        assert label in svg, label


def test_query_head_receipts_render_as_before(tmp_path: Path) -> None:
    """Without a grouping nothing kv-specific appears."""

    receipt = tviz.receipt_from_patch_grid(
        _grid(4),
        layer="blk.0.attn",
        negative_control=0.0,
        positive_control=3.0,
        joint_effect=0.1,
    )
    assert receipt.unit == "head" and receipt.sharing_lines() == ()
    artifact = torchlens.viz.ablation_grid(receipt, tmp_path / "q.svg", svg_fonttype="none")
    lines = "\n".join(artifact.disclosure_lines)
    assert "kv" not in lines
    assert "measured 4 heads together" not in lines
    assert "all 4 measured heads together" in lines
    assert "h3" in artifact.paths[0].read_text()


def test_mixed_units_refuse(tmp_path: Path) -> None:
    """One grid shows one unit; query-head and kv-group rows never mix."""

    heads = tviz.receipt_from_patch_grid(
        _grid(2), layer="blk.1.attn", negative_control=0.0, positive_control=3.0
    )
    with pytest.raises(tviz.TvizError) as caught:
        torchlens.viz.ablation_grid([_kv_receipt(), heads], tmp_path / "m.png", joint_effect=0.0)
    assert caught.value.fields["code"] == "tv_receipt_invalid"
    assert not (tmp_path / "m.png").exists()


def test_kv_receipt_cannot_annotate_query_head_panels() -> None:
    """A group's effect never lands on a single query head's panel header."""

    with pytest.raises(tviz.TvizError) as caught:
        tviz.Annotation.from_receipt(_kv_receipt())
    assert caught.value.fields["code"] == "tv_annotation_invalid"
    assert "render_receipt_grid" in str(caught.value)
