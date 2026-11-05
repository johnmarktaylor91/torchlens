"""A trace captured from a photo draws that photo on its input node.

``tl.trace(model, "cat.jpg")`` and ``tl.trace(model, pil_image)`` keep the photo
as ``trace.input``; ``trace.draw()`` puts a PNG thumbnail of it above the input
node's own rows. Tensor inputs draw as before, ``batch_render="shape_only"``
turns the photo off, and a saved trace keeps a bounded copy that still draws.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._visualization._input_photo import (
    INPUT_PHOTO_FILE,
    INPUT_PHOTO_MAX_EDGE,
    _raw_input_visualizer_dir,
    single_photo,
)

pytest.importorskip("PIL")
Image = importlib.import_module("PIL.Image")

_IMG_ROW = re.compile(r'<IMG SRC="raw_inputs/' + re.escape(INPUT_PHOTO_FILE) + '" SCALE="TRUE"/>')


class _Pixels:
    """Processor output carrying ``pixel_values``."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        """Hold the pixel tensor."""

        self.pixel_values = pixel_values


class _TinyVision(nn.Module):
    """A one-conv model with an attached image processor (the photo's preprocessing)."""

    def __init__(self) -> None:
        """Build the conv."""

        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3)

    def image_processor(self, image: Any, *, return_tensors: str) -> _Pixels:
        """Turn any photo into an 8x8 pixel batch."""

        del image, return_tensors
        return _Pixels(torch.ones(1, 3, 8, 8))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the conv."""

        return self.conv(pixel_values).mean(dim=(2, 3))


def _input_node(dot: str) -> str:
    """Return the input node's DOT statement."""

    match = re.search(r'^\t"?input_1\w*"? \[.*?\]\n', dot, re.M | re.S)
    assert match is not None, "no input node in the DOT"
    return match.group(0)


def _node(dot: str, name: str) -> str:
    """Return one node's DOT statement (single-pass ids carry a ``pass1`` suffix)."""

    match = re.search(rf'^\t"?{re.escape(name)}(?:pass1)?"? \[.*?\]\n', dot, re.M | re.S)
    assert match is not None, f"no node {name} in the DOT"
    return match.group(0)


def _assert_rounded_box(node: str) -> None:
    """The node is a rounded box (a picture or a label list never sits in an oval)."""

    assert "shape=box" in node, node
    style = re.search(r'style="?([\w,]+)', node)
    assert style is not None and "rounded" in style.group(1).split(","), node


def _photo(width: int = 300, height: int = 200) -> Any:
    """Return an RGB test photo."""

    return Image.new("RGB", (width, height), (140, 110, 80))


def _assert_photo_drawn(trace: tl.Trace, tmp_path: Path) -> None:
    """The input node carries the photo row above its own rows, as a small PNG."""

    node = _input_node(trace.draw(path=str(tmp_path / "graph"), file_format="svg").dot)
    assert _IMG_ROW.search(node), node
    assert node.index("<IMG") < node.index("input_1</B>"), node
    _assert_rounded_box(node)
    assert "fillcolor=" in node and " image=" not in node, node
    thumb_path = _raw_input_visualizer_dir(trace) / INPUT_PHOTO_FILE
    with Image.open(thumb_path) as thumb:
        assert thumb.format == "PNG"
        assert max(thumb.size) <= INPUT_PHOTO_MAX_EDGE


def test_jpeg_path_input_draws_the_photo(tmp_path: Path) -> None:
    """A JPEG path input is drawn on the input node as a PNG thumbnail."""

    path = tmp_path / "cat.jpg"
    _photo(640, 480).save(path, format="JPEG")
    trace = tl.trace(_TinyVision(), str(path))
    assert isinstance(trace.input, Image.Image)
    _assert_photo_drawn(trace, tmp_path)


def test_pil_input_draws_the_photo(tmp_path: Path) -> None:
    """A PIL image input is drawn on the input node."""

    _assert_photo_drawn(tl.trace(_TinyVision(), _photo()), tmp_path)


def test_tensor_input_draws_no_photo(tmp_path: Path) -> None:
    """A raw tensor input keeps its plain node."""

    trace = tl.trace(_TinyVision(), torch.ones(1, 3, 8, 8))
    node = _input_node(trace.draw(path=str(tmp_path / "graph"), file_format="svg").dot)
    assert "<IMG" not in node, node


def test_shape_only_turns_the_photo_off(tmp_path: Path) -> None:
    """``batch_render="shape_only"``, at capture or set afterwards, draws no photo."""

    captured = tl.trace(
        _TinyVision(), _photo(), capture=tl.options.CaptureOptions(batch_render="shape_only")
    )
    node = _input_node(captured.draw(path=str(tmp_path / "a"), file_format="svg").dot)
    assert "<IMG" not in node, node
    later = tl.trace(_TinyVision(), _photo())
    later.batch_render = "shape_only"
    node = _input_node(later.draw(path=str(tmp_path / "b"), file_format="svg").dot)
    assert "<IMG" not in node, node


def test_saved_trace_keeps_a_bounded_photo_that_still_draws(tmp_path: Path) -> None:
    """Save and load keep a bounded copy of a large photo, and the loaded trace draws it."""

    big = Image.effect_noise((2400, 1800), 64).convert("RGB")
    with_photo = tl.trace(_TinyVision(), big, save="none")
    with_tensor = tl.trace(_TinyVision(), torch.ones(1, 3, 8, 8), save="none")
    photo_file, tensor_file = tmp_path / "photo.tlspec", tmp_path / "tensor.tlspec"
    with_photo.save(photo_file)
    with_tensor.save(tensor_file)
    assert photo_file.stat().st_size - tensor_file.stat().st_size < 300_000
    loaded = tl.load(photo_file)
    assert isinstance(loaded.input, Image.Image)
    assert max(loaded.input.size) <= 256
    _assert_photo_drawn(loaded, tmp_path)


@pytest.mark.parametrize(
    ("wrap", "found"),
    [
        (lambda photo: photo, True),
        (lambda photo: [photo], True),
        (lambda photo: (photo,), True),
        (lambda photo: [photo, photo], False),
        (lambda photo: "cat.jpg", False),
        (lambda photo: {"image": photo}, False),
    ],
    ids=["bare", "list-of-one", "tuple-of-one", "list-of-two", "path-string", "mapping"],
)
def test_single_photo_accepts_one_image_only(wrap: Any, found: bool) -> None:
    """One image, alone or in a one-item list or tuple, counts as the input photo."""

    photo = _photo(4, 4)
    assert (single_photo(wrap(photo)) is photo) is found


def test_annotation_image_sits_in_its_own_cell_above_the_rows(tmp_path: Path) -> None:
    """An annotated picture is the top cell of a rounded box that keeps the node's fill."""

    picture = tmp_path / "relu.png"
    _photo(400, 100).save(picture)
    trace = tl.trace(_TinyVision(), torch.ones(1, 3, 8, 8))
    conv = next(label for label in trace.op_labels if label.startswith("conv2d")).split(":")[0]
    trace.annotate(conv, image=str(picture))
    node = _node(trace.draw(path=str(tmp_path / "graph"), file_format="svg").dot, conv)
    assert f'<IMG SRC="{picture}" SCALE="TRUE"/>' in node, node
    assert node.index("<IMG") < node.index(f"{conv}</B>"), node
    _assert_rounded_box(node)
    assert "fillcolor=" in node and "labelloc" not in node and "imagescale" not in node, node
    width = re.search(r'FIXEDSIZE="TRUE" WIDTH="(\d+)"', node)
    assert width is not None and int(width.group(1)) <= 144 + 8, node


def test_decoded_output_list_draws_as_a_rounded_box(tmp_path: Path) -> None:
    """A top-k label list on the output node draws in a rounded box, not an oval."""

    options = tl.options.CaptureOptions(
        output_transform=lambda out: [("tabby", 0.6), ("lynx", 0.3)]
    )
    trace = tl.trace(_TinyVision(), torch.ones(1, 3, 8, 8), capture=options)
    node = _node(trace.draw(path=str(tmp_path / "graph"), file_format="svg").dot, "output_1")
    assert "tabby" in node, node
    _assert_rounded_box(node)
    plain = _node(
        tl.trace(_TinyVision(), torch.ones(1, 3, 8, 8))
        .draw(path=str(tmp_path / "plain"), file_format="svg")
        .dot,
        "output_1",
    )
    assert "shape=box" not in plain, plain
