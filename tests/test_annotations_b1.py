"""Sprint B1 annotation persistence and API regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.intervention.errors import MultiMatchWarning


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
    return tl.trace(model, x, save="all"), model, x


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

    return next(op for op in trace._layer_list if not op.is_input and not op.is_output)


def test_annotation_tensor_blob_round_trips_tlspec(tmp_path: Path) -> None:
    """Tensor annotations under ``_annotation_blobs`` should save and load."""

    trace, _model, _x = _trace_tiny()
    tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    trace.annotate(tl.func("relu"), data=tensor)
    key = f"layer:{trace[trace[tl.func('relu')].labels[0]].layer_label}"
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
    relu = trace[trace[tl.func("relu")].labels[0]]
    assert relu.annotations["user"]["data"] == breadcrumb
    assert trace._layer_logs[relu.layer_label].annotations["user"]["data"] == breadcrumb

    bundle_path = tmp_path / "breadcrumb.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)
    loaded_relu = loaded[loaded[tl.func("relu")].labels[0]]

    assert loaded_relu.annotations["user"]["data"] == breadcrumb
    assert loaded._layer_logs[loaded_relu.layer_label].annotations["user"]["data"] == breadcrumb
    with pytest.raises(ValueError, match="JSON-serializable or a torch.Tensor"):
        trace.annotate(tl.func("relu"), data=np.zeros((2, 2)))


def test_annotate_uses_large_default_fanout() -> None:
    """The public annotate default should not cap matches at a handful of sites."""

    trace = tl.trace(
        _ManyReluModel(),
        torch.randn(1, 4),
        save="none",
    )

    with pytest.warns(MultiMatchWarning, match="matched 10 sites"):
        trace.annotate(tl.func("relu"), data={"fanout": True})

    annotated = [
        op
        for op in trace._layer_list
        if op.func_name == "relu" and op.annotations.get("user", {}).get("data") == {"fanout": True}
    ]
    assert len(annotated) == 10


def test_annotate_rerun_preserves_user_annotations_and_refreshes_internals() -> None:
    """Same-shape rerun keeps user annotations and blob data only."""

    trace, model, x = _trace_tiny()
    relu = trace[trace[tl.func("relu")].labels[0]]
    key = f"layer:{relu.layer_label}"
    trace.annotations["user"] = {"trace_note": "keep"}
    trace.provenance.input_annotations["user"] = {"input_note": "keep"}
    trace.annotate(tl.func("relu"), data=torch.ones(2, 2))
    trace.annotate(tl.func("relu"), data={"node_note": "keep"})
    relu.annotations["dedup_reference_label"] = "stale"
    trace._layer_logs[relu.layer_label].annotations["dedup_reference_label"] = "stale"

    result = trace.run(model, x + 0.125)
    rerun_relu = result[result[tl.func("relu")].labels[0]]

    assert result is trace
    assert result.annotations["user"] == {"trace_note": "keep"}
    assert result.provenance.input_annotations["user"] == {"input_note": "keep"}
    assert result._annotation_blobs is not None
    assert torch.equal(result._annotation_blobs[key], torch.ones(2, 2))
    assert rerun_relu.annotations["user"]["data"] == {"node_note": "keep"}
    assert result._layer_logs[rerun_relu.layer_label].annotations["user"]["data"] == {
        "node_note": "keep"
    }
    assert "dedup_reference_label" not in rerun_relu.annotations
    assert "dedup_reference_label" not in result._layer_logs[rerun_relu.layer_label].annotations


def test_non_torch_backend_rejects_torch_tensor_annotation() -> None:
    """Tensor annotation blobs are gated to torch traces for B1."""

    trace, _model, _x = _trace_tiny()
    trace._provenance_backend = (
        "jax"  # Simulate a non-torch active payload codec without optional runtimes.
    )

    with pytest.raises(ValueError, match="supported only for torch traces"):
        trace.annotate(tl.func("relu"), data=torch.ones(1))


def test_annotate_copy_returns_owned_copy() -> None:
    """annotate(..., copy=True) should leave the source trace untouched."""

    trace, _model, _x = _trace_tiny()

    copied = trace.annotate(tl.func("relu"), data={"owned": True}, copy=True)

    assert copied is not trace
    assert trace._annotation_blobs is None
    copied_relu = copied[copied[tl.func("relu")].labels[0]]
    source_relu = trace[trace[tl.func("relu")].labels[0]]
    assert copied_relu.annotations["user"]["data"] == {"owned": True}
    assert "user" not in source_relu.annotations


def test_annotation_image_sets_nodespec_image_hook(tmp_path: Path) -> None:
    """Image annotations should render through the existing NodeSpec image path."""

    from PIL import Image as image_module

    image_path = tmp_path / "annotation.png"
    image_module.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    trace, _model, _x = _trace_tiny()

    trace.annotate(tl.func("relu"), image=image_path)
    dot = trace.draw(
        file_format="svg",
        path=str(tmp_path / "graph"),
    ).dot

    assert str(image_path) in dot
    assert "annotation_images" not in dot  # a PNG is drawn as is, never copied


def _drawn_image_sources(trace: Trace, tmp_path: Path) -> list[str]:
    """Draw ``trace`` and return every node-image path in its DOT source.

    Parameters
    ----------
    trace:
        Annotated trace to draw.
    tmp_path:
        Folder receiving the rendered file.

    Returns
    -------
    list[str]
        The ``SRC`` of every ``<IMG>`` cell in the DOT source.
    """

    import re

    dot = trace.draw(file_format="svg", path=str(tmp_path / "graph")).dot
    return re.findall(r'<IMG SRC="([^"]+)"', dot)


def test_jpeg_annotation_draws_from_a_png_copy(tmp_path: Path) -> None:
    """A JPEG annotation keeps its path on the record and draws from a PNG copy."""

    from PIL import Image as image_module

    jpeg_path = tmp_path / "annotation.jpg"
    image_module.new("RGB", (12, 9), color=(200, 40, 10)).save(jpeg_path, format="JPEG")
    trace, _model, _x = _trace_tiny()

    trace.annotate(tl.func("relu"), image=jpeg_path)
    relu = trace[trace[tl.func("relu")].labels[0]]
    assert relu.annotations["user"]["image"] == str(jpeg_path)

    sources = _drawn_image_sources(trace, tmp_path)
    assert len(sources) == 1, sources
    png_path = Path(sources[0])
    assert png_path.suffix == ".png" and png_path.is_file(), sources
    assert png_path.parent == Path(trace._visualizer_dir) / "annotation_images"
    with image_module.open(png_path) as drawn:
        assert drawn.format == "PNG" and drawn.size == (12, 9)


def test_pil_image_annotation_draws_from_a_png(tmp_path: Path) -> None:
    """A PIL image annotation is written once as a PNG and drawn from it."""

    from PIL import Image as image_module

    picture = image_module.new("CMYK", (6, 5), color=(0, 80, 80, 0))
    trace, _model, _x = _trace_tiny()

    trace.annotate(tl.func("relu"), image=picture)
    relu = trace[trace[tl.func("relu")].labels[0]]
    stored = Path(relu.annotations["user"]["image"])
    assert stored.suffix == ".png" and stored.is_file()
    assert _drawn_image_sources(trace, tmp_path) == [str(stored)]
    with image_module.open(stored) as drawn:
        assert drawn.mode == "RGB" and drawn.size == (6, 5)


@pytest.mark.parametrize("kind", ["not-an-image", "missing"])
def test_unreadable_annotation_image_refuses_typed(tmp_path: Path, kind: str) -> None:
    """A file Pillow cannot open refuses ``image_path_unreadable`` and annotates nothing."""

    from torchlens._errors import InvalidArgumentError

    bad_path = tmp_path / "broken.jpg"
    if kind == "not-an-image":
        bad_path.write_text("not a picture", encoding="utf-8")
    trace, _model, _x = _trace_tiny()

    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.annotate(tl.func("relu"), image=bad_path)
    assert excinfo.value.fields["code"] == "image_path_unreadable"
    relu = trace[trace[tl.func("relu")].labels[0]]
    assert "user" not in relu.annotations


def test_saved_jpeg_annotation_keeps_the_path_not_the_pixels(tmp_path: Path) -> None:
    """A saved trace carries the annotation path only and draws again after loading."""

    from PIL import Image as image_module

    noise = torch.randint(0, 256, (96, 96, 3), dtype=torch.uint8).numpy()
    jpeg_path = tmp_path / "noise.jpg"
    image_module.fromarray(noise).save(jpeg_path, format="JPEG", quality=95)
    assert jpeg_path.stat().st_size > 8_000
    trace, _model, _x = _trace_tiny()

    def _bundle_bytes(name: str) -> int:
        """Save ``trace`` under ``name`` and return the bundle's total size."""

        bundle = tmp_path / name
        trace.save(bundle)
        files = [bundle] if bundle.is_file() else [p for p in bundle.rglob("*") if p.is_file()]
        return sum(path.stat().st_size for path in files)

    plain_bytes = _bundle_bytes("plain.tlspec")
    trace.annotate(tl.func("relu"), image=jpeg_path)
    annotated_bytes = _bundle_bytes("annotated.tlspec")
    assert annotated_bytes - plain_bytes < 4_000, (plain_bytes, annotated_bytes)

    loaded = tl.load(tmp_path / "annotated.tlspec")
    loaded_relu = loaded[loaded[tl.func("relu")].labels[0]]
    assert loaded_relu.annotations["user"]["image"] == str(jpeg_path)
    sources = _drawn_image_sources(loaded, tmp_path)
    assert len(sources) == 1 and sources[0].endswith(".png"), sources
    assert Path(sources[0]).is_file()
