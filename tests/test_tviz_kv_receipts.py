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
import torchlens.errors
import torchlens.transformer as tviz
from torchlens.mechinterp import PatchGrid

GQA = tviz.GqaInfo(num_query_heads=8, num_key_value_heads=2)


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
    with pytest.raises(torchlens.errors.RenderError) as caught:
        torchlens.viz.ablation_grid([_kv_receipt(), heads], tmp_path / "m.png", joint_effect=0.0)
    assert caught.value.fields["code"] == "tv_receipt_invalid"
    assert not (tmp_path / "m.png").exists()


def test_kv_receipt_cannot_annotate_query_head_panels() -> None:
    """A group's effect never lands on a single query head's panel header."""

    with pytest.raises(torchlens.errors.RenderError) as caught:
        tviz.Annotation.from_receipt(_kv_receipt())
    assert caught.value.fields["code"] == "tv_annotation_invalid"
    assert "render_receipt_grid" in str(caught.value)
