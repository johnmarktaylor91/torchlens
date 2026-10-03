"""tviz bridge rows (memo D18 + composition row 14).

The CircuitsVis payload carries identical coordinates/values to the static
records with bridge provenance attached; oversized payloads and the missing
dependency refuse typed; and the native static path is unaffected by any
bridge state. The BertViz tuple adapter's numeric parity oracle runs in the
real-model suite (R4).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

import torchlens.tviz as tviz


def _views(n_layers: int = 2) -> list[tviz.AttentionView]:
    """Deterministic self-attention views over one shared token axis."""

    tokens = tuple(f"t{i}" for i in range(5))
    generator = torch.Generator().manual_seed(0)
    return [
        tviz.AttentionView(
            pattern=torch.softmax(torch.randn(3, 5, 5, generator=generator), dim=-1),
            query_tokens=tviz.TokenAxis(role="query", tokens=tokens),
            key_tokens=tviz.TokenAxis(role="key", tokens=tokens),
            heads=(0, 1, 2),
            layer=f"blk.{layer}.attn",
        )
        for layer in range(n_layers)
    ]


def test_payload_matches_record_values_with_bridge_provenance() -> None:
    """Identical coordinates and values, different provenance (comp row 1)."""

    views = _views()
    payload = tviz.circuitsvis_payload(views)
    assert payload["tokens"] == [f"t{i}" for i in range(5)]
    assert payload["layers"] == ["blk.0.attn", "blk.1.attn"]
    reconstructed = torch.tensor(payload["attention"][0])
    assert torch.allclose(reconstructed, views[0].pattern, atol=1e-6)
    assert "dormant" in payload["disclosure"]


def test_rectangular_axes_refuse_the_bridge() -> None:
    """Cross-attention stays on the native renderers."""

    view = tviz.AttentionView(
        pattern=torch.softmax(torch.randn(2, 3, 5), dim=-1),
        query_tokens=tviz.TokenAxis(role="query", tokens=("a", "b", "c")),
        key_tokens=tviz.TokenAxis(role="key", tokens=("v", "w", "x", "y", "z")),
        heads=(0, 1),
        layer="dec.0.cross",
    )
    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.circuitsvis_payload([view])
    assert excinfo.value.fields["code"] == "tv_record_invalid"


def test_oversized_payload_refuses_typed() -> None:
    """Above the ceiling the handoff refuses with the measured size."""

    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.circuitsvis_attention(_views(), max_bytes=64)
    error = excinfo.value
    assert error.fields["code"] == "tv_bridge_payload_too_large"
    assert error.fields["payload_bytes"] > error.fields["limit_bytes"]


@pytest.mark.smoke
@pytest.mark.skipif(
    importlib.util.find_spec("circuitsvis") is not None,
    reason="the refusal row needs circuitsvis absent",
)
def test_missing_bridge_dependency_refuses_typed() -> None:
    """circuitsvis absent -> tv_bridge_unavailable naming the native path."""

    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.circuitsvis_attention(_views())
    assert excinfo.value.fields["code"] == "tv_bridge_unavailable"


def test_native_path_unaffected_by_bridge_state(tmp_path: Path) -> None:
    """Composition row 14: static rendering never depends on the bridge."""

    artifact = tviz.render_attention(_views()[0], tmp_path / "native.svg")
    assert artifact.paths[0].exists()
    assert "<image" not in artifact.paths[0].read_text()


@pytest.mark.smoke
def test_bertviz_tuple_layout() -> None:
    """One [1, heads, dst, src] tensor per layer (the HF tuple layout)."""

    views = _views()
    tensors = tviz.bertviz_tuple(views)
    assert len(tensors) == 2
    assert tensors[0].shape == (1, 3, 5, 5)
    assert torch.allclose(tensors[1][0], views[1].pattern)
    unbatched = tviz.bertviz_tuple(views, batch=False)
    assert unbatched[0].shape == (3, 5, 5)
