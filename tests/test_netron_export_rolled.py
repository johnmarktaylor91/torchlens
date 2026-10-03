"""Rolled-granularity netron export (lane F14, netron memo D-13).

Repeated passes merge by rolled identity; boundary edges union and
deduplicate in stable order; recurrent feedback is DISCLOSED via the
``recurrence`` attribute (drawn literally it forms a dataflow cycle
``check_model(full_check=True)`` rejects -- probed against onnx 1.22, so
the D-07 validity gate wins over a literal self-loop edge); an edge is
TYPED only when all member passes agree, otherwise ``shape_variants``/
``dtype_variants`` properties and an untyped edge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx", reason="rolled contract asserts checker validity")
json_format = pytest.importorskip("google.protobuf.json_format")

import torchlens as tl  # noqa: E402


class _Loop(nn.Module):
    """Three-pass tanh(linear) recurrence."""

    def __init__(self) -> None:
        """Build the reused linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same two ops three times."""

        for _ in range(3):
            x = torch.tanh(self.fc(x))
        return x


def _export(log: Any, path: Path, **kwargs: Any) -> dict[str, Any]:
    """Export, checker-validate, and parse one rolled artifact."""

    artifact = tl.export.netron(log, path, granularity="rolled", **kwargs)
    model = json_format.Parse(artifact.read_text(encoding="utf-8"), onnx.ModelProto())
    onnx.checker.check_model(model, full_check=True)
    return json.loads(artifact.read_text(encoding="utf-8"))


def test_passes_merge_with_feedback_disclosed(tmp_path: Path) -> None:
    """Six op passes roll into two nodes; the feedback edge is disclosed."""

    log = tl.trace(_Loop(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "rolled.json")
    nodes = {node["name"]: node for node in payload["graph"]["node"]}
    assert set(nodes) == {"linear_1_1", "tanh_1_2"}
    attrs = {a["name"]: a for a in nodes["linear_1_1"].get("attribute", [])}
    assert int(attrs["passes"]["i"]) == 3
    import base64

    recurrence = base64.b64decode(attrs["recurrence"]["s"]).decode()
    assert "tanh_1_2" in recurrence, "feedback source named in the disclosure"
    assert nodes["linear_1_1"]["input"] == ["input_1"], "feedback never drawn as an edge"
    typed = {row["name"] for row in payload["graph"].get("valueInfo", [])}
    assert "linear_1_1" in typed, "shape-stable rolled values stay typed"


def test_variance_gates_the_type() -> None:
    """Cross-pass shape/dtype drift strips the type and discloses variants.

    Plain torch-function recurrence groups only shape-identical passes, so
    the natural drift case lives at episode scale (the ``shape_summary``
    "2->4" family); the belt is pinned here at the projection layer with
    stub pass records -- asserting a false invariant is worse than clutter.
    """

    from types import SimpleNamespace

    from torchlens.export._netron_fields import rolled_variance_attrs

    members = [
        SimpleNamespace(dtype=torch.float32, shape=(2, 8)),
        SimpleNamespace(dtype=torch.float16, shape=(2, 16)),
    ]
    attrs = {attr.name: attr for attr in rolled_variance_attrs(members, feedback_sources=[])}
    assert attrs["shape_variants"].value == "2x8 | 2x16"
    assert attrs["dtype_variants"].value == "float32 | float16"
    assert attrs["passes"].value == 2
    # And the type gate itself: agreeing passes keep the type, drifting drop it.
    stable = [SimpleNamespace(dtype=torch.float32, shape=(2, 8))] * 2
    assert "shape_variants" not in {
        attr.name for attr in rolled_variance_attrs(stable, feedback_sources=[])
    }


@pytest.mark.smoke
def test_projection_drops_value_info_for_drifting_group() -> None:
    """The rolled projection emits NO valueInfo for a shape-drifting group."""

    from types import SimpleNamespace

    from torchlens.export._netron_records import project_rolled

    def _entry(label: str, index: int, shape: tuple[int, ...], **kwargs: Any) -> Any:
        """One duck-typed layer-pass record."""

        return SimpleNamespace(
            layer_label=label,
            label=f"{label}:{index}",
            layer_type=label.split("_")[0],
            func_name=label.split("_")[0],
            parents=kwargs.get("parents", ()),
            children=kwargs.get("children", ()),
            is_input=kwargs.get("is_input", False),
            is_output=kwargs.get("is_output", False),
            is_buffer=False,
            shape=shape,
            dtype=torch.float32,
            num_passes=kwargs.get("num_passes", 1),
            pass_index=index,
            module_call_stack=(),
            code_context=[],
            func_qualname=label.split("_")[0],
            arg_names=(),
            parent_arg_positions={},
            num_params=0,
            func_duration=None,
            flops_forward=None,
            activation_memory=None,
        )

    entries = [
        _entry("input_1", 1, (2, 4), is_input=True, children=("grow_1_1",)),
        _entry("grow_1_1", 1, (2, 8), parents=("input_1",), num_passes=2),
        _entry("grow_1_1", 2, (2, 16), parents=("grow_1_1:1",), num_passes=2),
        _entry("output_1", 1, (2, 16), is_output=True, parents=("grow_1_1:2",)),
    ]
    log = SimpleNamespace(layer_list=entries, intervention_audit=[])
    projection = project_rolled(log, "meaningful")
    assert [node.name for node in projection.nodes] == ["grow_1_1"]
    assert projection.value_infos == [], "drifting group carries no type claim"
    attrs = {attr.name: attr.value for attr in projection.nodes[0].attrs}
    assert attrs["shape_variants"] == "2x8 | 2x16"
    assert "feedback" in attrs["recurrence"], "pass-2 self-input disclosed"


@pytest.mark.slow
def test_ten_thousand_op_stress_case(tmp_path: Path) -> None:
    """The 10,000-op stress case exports valid and compact (memo B5).

    Capture itself is the slow part (measured ~13 s at 5k ops on this tier's
    box); the export must stay in single-digit seconds and switch to compact
    separators above 1 MB.
    """

    class _Chain(nn.Module):
        """A 10,000-op linear chain (the cheapest possible layout)."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Chain ten thousand adds."""

            for _ in range(10_000):
                x = x + 1.0
            return x

    import time
    import warnings

    log = tl.trace(_Chain(), torch.randn(4))
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the extent budget fires, correctly
        path = tl.export.netron(log, tmp_path / "stress.json", granularity="op")
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"10k-op export took {elapsed:.1f}s"
    text = path.read_text(encoding="utf-8")
    assert "\n" not in text[1_000:2_000], "compact separators above ~1 MB"
    payload = json.loads(text)
    assert len(payload["graph"]["node"]) == 10_000
    model = json_format.Parse(text, onnx.ModelProto())
    onnx.checker.check_model(model, full_check=True)
