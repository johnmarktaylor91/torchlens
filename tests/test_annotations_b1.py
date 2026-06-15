"""Sprint B1 annotation persistence and API regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace


class _TinyAnnotatedModel(nn.Module):
    """Small model with stable relu and linear sites."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple relu graph.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output after relu.
        """

        return self.fc(torch.relu(x))


class _ManyReluModel(nn.Module):
    """Model with more than eight relu sites."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relu repeatedly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Repeated relu result.
        """

        for _ in range(10):
            x = torch.relu(x)
        return x


def _trace_tiny() -> tuple[Trace, _TinyAnnotatedModel, torch.Tensor]:
    """Capture a tiny annotation fixture.

    Returns
    -------
    tuple[Trace, _TinyAnnotatedModel, torch.Tensor]
        Captured trace, live model, and input.
    """

    torch.manual_seed(901)
    model = _TinyAnnotatedModel()
    x = torch.randn(2, 4)
    return tl.trace(model, x, layers_to_save="all"), model, x


def _first_compute_op(trace: Trace) -> Any:
    """Return the first non-input/output op.

    Parameters
    ----------
    trace:
        Captured trace.

    Returns
    -------
    Any
        First compute op.
    """

    return next(op for op in trace.layer_list if not op.is_input and not op.is_output)


def test_annotation_tensor_blob_round_trips_tlspec(tmp_path: Path) -> None:
    """Tensor annotations under ``_annotation_blobs`` should save and load."""

    trace, _model, _x = _trace_tiny()
    tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    trace.annotate(tl.func("relu"), data=tensor)
    key = f"layer:{trace.resolve_sites(tl.func('relu'), max_fanout=10).first().layer_label}"
    assert torch.equal(trace._annotation_blobs[key], tensor)

    bundle_path = tmp_path / "annotated.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)

    assert isinstance(loaded, Trace)
    assert loaded._annotation_blobs is not None
    assert torch.equal(loaded._annotation_blobs[key], tensor)


def test_json_breadcrumb_round_trips_and_ndarray_rejects(tmp_path: Path) -> None:
    """JSON user breadcrumbs persist and raw ndarrays are not stringified."""

    trace, _model, _x = _trace_tiny()
    breadcrumb = {"kind": "note", "scores": [1, 2, 3]}

    trace.annotate(tl.func("relu"), data=breadcrumb)
    relu = trace.resolve_sites(tl.func("relu"), max_fanout=10).first()
    assert relu.annotations["user"]["data"] == breadcrumb
    assert trace.layer_logs[relu.layer_label].annotations["user"]["data"] == breadcrumb

    bundle_path = tmp_path / "breadcrumb.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)
    loaded_relu = loaded.resolve_sites(tl.func("relu"), max_fanout=10).first()

    assert loaded_relu.annotations["user"]["data"] == breadcrumb
    assert loaded.layer_logs[loaded_relu.layer_label].annotations["user"]["data"] == breadcrumb
    with pytest.raises(ValueError, match="JSON-serializable or a torch.Tensor"):
        trace.annotate(tl.func("relu"), data=np.zeros((2, 2)))


def test_annotate_uses_large_default_fanout() -> None:
    """The public annotate default should not inherit resolver max_fanout=8."""

    trace = tl.trace(_ManyReluModel(), torch.randn(1, 4), layers_to_save="none")

    trace.annotate(tl.func("relu"), data={"fanout": True})

    annotated = [
        op
        for op in trace.layer_list
        if op.func_name == "relu" and op.annotations.get("user", {}).get("data") == {"fanout": True}
    ]
    assert len(annotated) == 10


def test_annotate_rerun_preserves_user_annotations_and_refreshes_internals() -> None:
    """Same-shape rerun keeps user annotations and blob data only."""

    trace, model, x = _trace_tiny()
    relu = trace.resolve_sites(tl.func("relu"), max_fanout=10).first()
    key = f"layer:{relu.layer_label}"
    trace.annotations["user"] = {"trace_note": "keep"}
    trace.input_annotations["user"] = {"input_note": "keep"}
    trace.annotate(tl.func("relu"), data=torch.ones(2, 2))
    trace.annotate(tl.func("relu"), data={"node_note": "keep"})
    relu.annotations["dedup_reference_label"] = "stale"
    trace.layer_logs[relu.layer_label].annotations["dedup_reference_label"] = "stale"

    result = trace.rerun(model, x + 0.125)
    rerun_relu = result.resolve_sites(tl.func("relu"), max_fanout=10).first()

    assert result is trace
    assert result.annotations["user"] == {"trace_note": "keep"}
    assert result.input_annotations["user"] == {"input_note": "keep"}
    assert result._annotation_blobs is not None
    assert torch.equal(result._annotation_blobs[key], torch.ones(2, 2))
    assert rerun_relu.annotations["user"]["data"] == {"node_note": "keep"}
    assert result.layer_logs[rerun_relu.layer_label].annotations["user"]["data"] == {
        "node_note": "keep"
    }
    assert "dedup_reference_label" not in rerun_relu.annotations
    assert "dedup_reference_label" not in result.layer_logs[rerun_relu.layer_label].annotations


def test_non_torch_backend_rejects_torch_tensor_annotation() -> None:
    """Tensor annotation blobs are gated to torch traces for B1."""

    trace, _model, _x = _trace_tiny()
    trace.backend = "jax"  # Simulate a non-torch active payload codec without optional runtimes.

    with pytest.raises(ValueError, match="supported only for torch traces"):
        trace.annotate(tl.func("relu"), data=torch.ones(1))


def test_with_annotations_returns_owned_copy() -> None:
    """with_annotations should leave the source trace untouched."""

    trace, _model, _x = _trace_tiny()

    copied = trace.with_annotations(tl.func("relu"), data={"owned": True})

    assert copied is not trace
    assert trace._annotation_blobs is None
    copied_relu = copied.resolve_sites(tl.func("relu"), max_fanout=10).first()
    source_relu = trace.resolve_sites(tl.func("relu"), max_fanout=10).first()
    assert copied_relu.annotations["user"]["data"] == {"owned": True}
    assert "user" not in source_relu.annotations


def test_annotation_image_sets_nodespec_image_hook(tmp_path: Path) -> None:
    """Image annotations should render through the existing NodeSpec image path."""

    image_module = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "annotation.png"
    image_module.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    trace, _model, _x = _trace_tiny()

    trace.annotate(tl.func("relu"), image=image_path)
    dot = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
    )

    assert str(image_path) in dot
