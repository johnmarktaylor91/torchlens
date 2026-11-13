"""Repgeom node visuals on module outputs: reused-module passes and collapsed boxes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens import repgeom
from torchlens._visualization.node_spec import NodeSpec

_N_STIMULI = 8
_WIDTH = 6


class _Block(nn.Module):
    """Residual block whose one ReLU module runs twice from one loop line.

    The ReLU's two calls are two passes of one layer, so the block output is the later
    pass. (torchvision's BasicBlock calls its ReLU from two lines: two layers.)
    """

    def __init__(self) -> None:
        """Initialize the two linear layers and the shared activation."""

        super().__init__()
        self.fc1 = nn.Linear(_WIDTH, _WIDTH)
        self.fc2 = nn.Linear(_WIDTH, _WIDTH)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block output: the shared ReLU's second pass."""

        out = x
        for fc, skip in ((self.fc1, None), (self.fc2, x)):
            out = fc(out) if skip is None else fc(out) + skip
            out = self.relu(out)
        return out


class _TwoStages(nn.Module):
    """Two residual stages, each ending in a reused module's later pass."""

    def __init__(self) -> None:
        """Initialize the stem, both stages and the head."""

        super().__init__()
        self.stem = nn.Linear(_WIDTH, _WIDTH)
        self.stage1 = _Block()
        self.stage2 = _Block()
        self.head = nn.Linear(_WIDTH, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run stem, both stages and head."""

        return self.head(self.stage2(self.stage1(self.stem(x))))


class _SharedStage(nn.Module):
    """One residual block called twice: a multi-call module."""

    def __init__(self) -> None:
        """Initialize the stem, the shared block and the head."""

        super().__init__()
        self.stem = nn.Linear(_WIDTH, _WIDTH)
        self.block = _Block()
        self.head = nn.Linear(_WIDTH, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared block twice between stem and head."""

        return self.head(self.block(self.block(self.stem(x))))


def _stimuli() -> torch.Tensor:
    """Return a deterministic stimulus batch with distinct rows."""

    generator = torch.Generator().manual_seed(1105)
    return torch.randn(_N_STIMULI, _WIDTH, generator=generator)


def _annotated_trace(model: nn.Module, save: Any) -> tl.Trace:
    """Capture ``model`` and store MDS coordinates for the ``save`` sites."""

    torch.manual_seed(1105)
    trace = tl.trace(model.eval(), _stimuli(), save=save)
    repgeom.mds_evolution(trace, save=save, min_stimuli=_N_STIMULI)
    return trace


def _per_call_output_trace() -> tl.Trace:
    """Capture the shared block and store distinct coordinates on each call's output pass.

    Both call outputs are passes of one layer, and ``mds_evolution`` takes one
    pass per layer in a call, so the coordinates are stored directly under the
    ``mds:op:<pass label>`` keys the producer writes.
    """

    torch.manual_seed(1105)
    trace = tl.trace(_SharedStage().eval(), _stimuli(), save=tl.module("block"))
    calls = trace.modules["block"].calls
    outputs = [str(label) for index in range(len(calls)) for label in calls[index].output_ops]
    assert len(outputs) == 2
    generator = torch.Generator().manual_seed(1105)
    trace._annotation_blobs = dict(trace._annotation_blobs or {})
    for label in outputs:
        trace._annotation_blobs[f"mds:op:{label}"] = torch.randn(_N_STIMULI, 2, generator=generator)
    return trace


def _node_images(svg_text: str) -> dict[str, str]:
    """Return node title to embedded image href for every image-bearing node."""

    images = {}
    for block in re.split(r"<!-- ", svg_text):
        title = re.search(r"<title>([^<]+)</title>", block)
        href = re.search(r'<image\b[^>]+(?:xlink:href|href)="([^"]+)"', block)
        if title is not None and href is not None:
            images[title.group(1)] = href.group(1)
    return images


def _stage_save() -> Any:
    """Return the module-output selector for both stages."""

    return tl.module("stage1") | tl.module("stage2")


def test_module_output_keys_are_later_passes_of_a_reused_relu() -> None:
    """The regression's premise: each stage output is stored under a pass-qualified op key."""

    trace = _annotated_trace(_TwoStages(), _stage_save())

    assert trace._annotation_blobs is not None
    mds_keys = sorted(key for key in trace._annotation_blobs if key.startswith("mds:"))
    assert len(mds_keys) == 2
    assert all(re.fullmatch(r"mds:op:relu_\d+_\d+:2", key) for key in mds_keys)


def test_unrolled_draw_decorates_the_reused_module_pass(tmp_path: Path) -> None:
    """``node_spec_fn`` decorates the pass a node draws, not only single-pass layers."""

    trace = _annotated_trace(_TwoStages(), _stage_save())
    output_path = tmp_path / "unrolled.svg"

    trace.draw(
        path=str(output_path),
        file_format="svg",
        node_spec_fn=tl.viz.nodes.mds_scatter(thumbnail_size=24, canvas_size=120),
    )

    images = _node_images(output_path.read_text(encoding="utf-8"))
    assert len(images) == 2
    assert all(re.fullmatch(r"relu_\d+_\d+pass2", title) for title in images)
    assert len(set(images.values())) == 2


def test_aggregate_layer_without_rendered_pass_stays_undecorated() -> None:
    """Called outside a draw, a multi-pass Layer has no single pass and gets no picture."""

    trace = _annotated_trace(_TwoStages(), _stage_save())
    hook = tl.viz.nodes.mds_scatter()
    multipass = next(layer for layer in trace.layers if layer.num_passes == 2)

    assert hook(multipass, NodeSpec(lines=["x"])) is None


def test_collapsed_boxes_draw_their_call_output_scatter(tmp_path: Path) -> None:
    """``collapsed_node_spec_fn`` decorates each collapsed stage box with its output's scatter."""

    trace = _annotated_trace(_TwoStages(), _stage_save())
    output_path = tmp_path / "collapsed.svg"

    trace.draw(
        path=str(output_path),
        file_format="svg",
        depth=1,
        collapsed_node_spec_fn=tl.viz.nodes.mds_scatter(thumbnail_size=24, canvas_size=120),
    )

    svg_text = output_path.read_text(encoding="utf-8")
    images = _node_images(svg_text)
    assert sorted(images) == ["stage1pass1", "stage2pass1"]
    assert len(set(images.values())) == 2
    assert ">stage1</text>" in svg_text
    assert ">stage2</text>" in svg_text


def test_unrolled_multicall_boxes_each_draw_their_own_call(tmp_path: Path) -> None:
    """Each unrolled call box of a reused module shows the scatter of its own call."""

    trace = _per_call_output_trace()
    output_path = tmp_path / "multicall.svg"

    trace.draw(
        path=str(output_path),
        file_format="svg",
        depth=1,
        collapsed_node_spec_fn=tl.viz.nodes.mds_scatter(thumbnail_size=24, canvas_size=120),
    )

    svg_text = output_path.read_text(encoding="utf-8")
    assert sorted(_node_images(svg_text)) == ["blockpass1", "blockpass2"]
    calls = trace.modules["block"].calls
    blocks = {
        title.group(1): block
        for block in svg_text.split("<!-- ")
        if (title := re.search(r"<title>([^<]+)</title>", block)) is not None
    }
    # The two point-cloud fallbacks may look alike; the tooltip names the key each box drew.
    for index, title in enumerate(["blockpass1", "blockpass2"]):
        (output_label,) = calls[index].output_ops
        assert f"MDS thumbnail scatter for op:{output_label} " in blocks[title]


def test_rolled_multicall_box_has_no_single_output(tmp_path: Path) -> None:
    """A rolled box standing for two calls is ambiguous, like a rolled multi-pass layer."""

    trace = _per_call_output_trace()
    output_path = tmp_path / "rolled.svg"

    trace.draw(
        path=str(output_path),
        file_format="svg",
        depth=1,
        rolled=True,
        collapsed_node_spec_fn=tl.viz.nodes.mds_scatter(thumbnail_size=24, canvas_size=120),
    )

    assert _node_images(output_path.read_text(encoding="utf-8")) == {}
