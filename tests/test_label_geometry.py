"""Regression gate: rolled-view edge labels must never collide with anything.

Renders the same 16-model rolled inspection set used to tune the edge-label
placement constants, lays each graph out with ``dot -Tjson``, and runs the
exact per-label geometry audit (``tests/support/label_geometry.py``).  Any
hard violation -- a label penetrating another label, a node outline, an edge
spline, an arrowhead, or a cluster border by more than the calibrated
threshold -- fails the gate with a self-explaining message.
"""

from __future__ import annotations

import io
from pathlib import Path
import re

import pytest
import torch
from torch import nn
from PIL import Image

cairosvg = pytest.importorskip("cairosvg")

import torchlens as tl  # noqa: E402
import test_loop_module_rolling as demos  # noqa: E402
from support.label_geometry import audit_gv_source  # noqa: E402

_CONTAINER_FRAME_GUTTER_PX = 5

# The 16 rolled demo configs the placement sweep was calibrated against:
# the 12 committed fixtures plus 4 extra inspection models.
DEMO_CONFIGS: list[tuple[str, type[nn.Module], dict[str, object]]] = [
    ("inside_outside_relu", demos.InsideOutsideRelu, {"vis_call_depth": 1}),
    ("inside_outside_relu_separable", demos.InsideOutsideReluSeparable, {"vis_call_depth": 1}),
    ("inside_outside_block_collapsed", demos.InsideOutsideBlock, {"vis_call_depth": 1}),
    ("inside_outside_block_expanded", demos.InsideOutsideBlock, {"vis_call_depth": 1000}),
    ("deep_loop_body", demos.DeepLoopBody, {"vis_call_depth": 1000}),
    ("rnn_cell", demos.TanhRNNCellLoop, {"vis_call_depth": 1000}),
    ("repeated_block_stack_collapsed", demos.RepeatedBlockStack, {"vis_call_depth": 1}),
    ("repeated_block_stack_expanded", demos.RepeatedBlockStack, {"vis_call_depth": 1000}),
    ("two_distinct_loops", demos.TwoDistinctLoops, {"vis_call_depth": 1}),
    ("buffer_loop", demos.BufferRewriteLoops, {"show_buffer_layers": "always"}),
    ("nested_loop", demos.NestedLoopBlock, {"vis_call_depth": 1}),
    ("parallel_fanout", demos.ParallelFanout, {"vis_call_depth": 1}),
    ("collapsed_block_recurrence", demos.CollapsedBlockRecurrence, {"vis_call_depth": 1000}),
    ("inside_outside_loop", demos.InsideOutsideLoop, {"vis_call_depth": 1}),
    ("parallel_siblings_loop", demos.ParallelSiblingsLoop, {"vis_call_depth": 1000}),
    ("shared_two_site_recurrences", demos.SharedTwoSiteRecurrences, {"vis_call_depth": 1}),
]


@pytest.mark.parametrize(
    ("name", "model_cls", "kwargs"),
    DEMO_CONFIGS,
    ids=[config[0] for config in DEMO_CONFIGS],
)
def test_rolled_edge_labels_have_zero_hard_violations(
    name: str, model_cls: type[nn.Module], kwargs: dict[str, object], tmp_path: Path
) -> None:
    """Every rolled demo graph lays out with zero hard label-geometry violations."""

    trace = demos._trace(model_cls())
    gv_source = trace.draw(
        vis_mode="rolled",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / name),
        **kwargs,
    )
    result = audit_gv_source(gv_source)
    assert result.hard_violation_count == 0, result.describe_violations(name)


_NEGATIVE_CONTROL_GV = """
digraph negative_control {
\trankdir=BT
\ta [shape=box]
\tb [shape=box]
\ta -> b [headlabel="overlapping head label" labeldistance=0]
}
"""


def test_audit_negative_control_flags_label_node_overlap() -> None:
    """The audit itself catches a deliberately overlapping head label.

    ``labeldistance=0`` centers the head label on the edge's head point, so it
    must penetrate node ``b`` -- if the audit reports this graph clean, the
    gate has gone blind and the positive tests above prove nothing.
    """

    result = audit_gv_source(_NEGATIVE_CONTROL_GV)
    assert result.hard_violation_count >= 1
    violation_types = {violation["type"] for violation in result.violations}
    assert "label-node" in violation_types
    # Failure messages must be self-explaining: graph, edge, label, type, depth.
    description = result.describe_violations("negative_control")
    assert "label-node" in description
    assert "a->b" in description
    assert "overlapping head label" in description
    assert "penetration" in description


class _ContainerLabelModel(nn.Module):
    """Small model with a key-labeled dictionary output."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model."""

        return {"left": x + 1, "right": x + 2}


class _DemoModelOutput(dict[str, torch.Tensor | tuple[tuple[torch.Tensor, torch.Tensor], ...]]):
    """Minimal HuggingFace ``ModelOutput`` stand-in with a wide class name."""

    def __init__(
        self,
        **kwargs: torch.Tensor | tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """Create a mapping output with attribute access."""

        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _HFLikeOutputModel(nn.Module):
    """Return a nested HF-like output container with a wide node label."""

    def forward(self, x: torch.Tensor) -> _DemoModelOutput:
        """Run the model."""

        return _DemoModelOutput(
            logits=x + 1,
            past_key_values=((x + 2, x + 3),),
        )


class _ThreadedCacheModel(nn.Module):
    """Thread an unchanged cache container through repeated modules."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.layers = nn.ModuleList([_CacheLayer() for _ in range(3)])

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Pass the cache through multiple module boundaries."""

        for layer in self.layers:
            x, past_key_values = layer(x, past_key_values)
        return x, past_key_values


class _CacheLayer(nn.Module):
    """Layer that consumes and returns an unchanged cache container."""

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Return ``x`` and ``past_key_values`` unchanged."""

        return x + 1, past_key_values


def _native_svg_outer_frame_min_brightness(svg_path: Path) -> dict[str, int]:
    """Return minimum RGB brightness in the outermost SVG frame."""

    svg_text = svg_path.read_text()
    match = re.search(r'viewBox="([^"]+)"', svg_text)
    if match is None:
        raise AssertionError(f"{svg_path} did not include an SVG viewBox.")
    _x0, _y0, view_width, view_height = [float(value) for value in match.group(1).split()]
    width = round(view_width)
    height = round(view_height)
    png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode(),
        output_width=width,
        output_height=height,
    )
    rgba_image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    image = Image.new("RGB", rgba_image.size, (255, 255, 255))
    image.paste(rgba_image, mask=rgba_image.getchannel("A"))
    width, height = image.size
    gutter = min(_CONTAINER_FRAME_GUTTER_PX, width // 2, height // 2)
    edge_ranges = {
        "left": ((x, y) for x in range(gutter) for y in range(height)),
        "right": ((x, y) for x in range(width - gutter, width) for y in range(height)),
        "top": ((x, y) for x in range(width) for y in range(gutter)),
        "bottom": ((x, y) for x in range(width) for y in range(height - gutter, height)),
    }
    brightness: dict[str, int] = {}
    for edge_name, pixels in edge_ranges.items():
        brightness[edge_name] = min(min(image.getpixel((x, y))) for x, y in pixels)
    return brightness


def test_container_edge_labels_are_midpoint_and_geometry_clean(tmp_path: Path) -> None:
    """Container key labels use midpoint labels without endpoint placement attrs."""

    trace = tl.trace(_ContainerLabelModel(), torch.ones(2), intervention_ready=True)
    gv_source = trace.draw(
        show_containers="labels",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "container_labels"),
    )

    assert '<FONT POINT-SIZE="8">left</FONT>' in gv_source
    assert '<FONT POINT-SIZE="8">right</FONT>' in gv_source
    assert "headlabel=" not in gv_source
    assert "taillabel=" not in gv_source
    assert "labeldistance=" not in gv_source
    assert "labelangle=" not in gv_source
    result = audit_gv_source(gv_source)
    assert result.hard_violation_count == 0, result.describe_violations("container_labels")


@pytest.mark.parametrize(
    ("name", "model", "args"),
    [
        (
            "threaded_kv",
            _ThreadedCacheModel(),
            (
                torch.tensor([3.0]),
                tuple((torch.ones(1), torch.zeros(1)) for _ in range(2)),
            ),
        ),
        ("hf_output", _HFLikeOutputModel(), torch.tensor([1.0])),
    ],
)
def test_container_nodes_have_native_svg_frame_margin(
    name: str,
    model: nn.Module,
    args: object,
    tmp_path: Path,
) -> None:
    """Container-node labels do not put ink on the native SVG frame."""

    trace = tl.trace(
        model,
        args,
        capture_container_structure=True,
    )
    trace.draw(
        show_containers="nodes",
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / name),
    )

    edge_brightness = _native_svg_outer_frame_min_brightness(tmp_path / f"{name}.svg")
    assert edge_brightness == {"left": 255, "right": 255, "top": 255, "bottom": 255}
