"""Every drawn op sits inside its own module frame, and edit marks never move frames.

Two structural pins (frame membership and frame boxes, not byte goldens):

- rolled drawing of one module instance shared by two parents: each parent's
  ops are inside that parent's frame. The shared layer's later pass runs under
  the second parent, and the rolled edge from it used to be placed by the
  first pass's modules only, so the second parent's frame was never drawn.
- rank layout of a residual stack: drawing intervention marks leaves every
  module frame box exactly as it is without edits. The rank engine measures a
  table label as one line holding every row, so the edit row used to widen the
  edited node and every frame around it; it now sizes nodes without edit rows.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.visualization._rank_layout_internal.layout as layout_mod
from torchlens.visualization._geometry_audit import run_layout_json


class _SharedChildParent(nn.Module):
    """Applies its (possibly shared) child, then a relu."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.child(x))


class _TwoParents(nn.Module):
    """Two parent modules that hold the SAME child instance."""

    def __init__(self) -> None:
        super().__init__()
        shared = nn.Linear(4, 4)
        self.p1 = _SharedChildParent(shared)
        self.p2 = _SharedChildParent(shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p2(self.p1(x))


class _ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.fc(x))


class _ResStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[_ResBlock() for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def _owner(modules: Any, is_atomic_module: bool) -> list[str]:
    """Module path an op is drawn in: an atomic module's output sits one level up."""

    path = list(modules)
    return path[:-1] if is_atomic_module else path


def _innermost_frames(dot_source: str) -> dict[str, str]:
    """Map each laid-out node name to the innermost cluster that holds it."""

    objects = run_layout_json(dot_source, "dot")["objects"]
    clusters = [obj for obj in objects if str(obj.get("name", "")).startswith("cluster")]
    innermost: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for cluster in clusters:
        members = cluster.get("nodes", [])
        for index in members:
            name = str(objects[index]["name"])
            if name not in sizes or len(members) < sizes[name]:
                innermost[name] = str(cluster["name"])
                sizes[name] = len(members)
    return innermost


def test_rolled_shared_child_puts_each_parents_ops_in_its_frame(tmp_path: Path) -> None:
    """Rolled: the p2 relu sits inside the p2 frame, the p1 relu inside the p1 frame."""

    torch.manual_seed(0)
    trace = tl.trace(_TwoParents(), torch.randn(2, 4))
    dot = str(
        trace.draw(
            vis_mode="rolled",
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "rolled"),
        )
    )
    frames = _innermost_frames(dot)
    relus = [layer for layer in trace.layer_logs.values() if layer.layer_type == "relu"]
    assert len(relus) == 2
    expected = {}
    for layer in trace.layer_logs.values():
        path = _owner([m.split(":")[0] for m in layer.modules], layer.is_atomic_module)
        if path:
            expected[layer.layer_label] = f"cluster_{path[-1]}"
    assert {expected[relu.layer_label] for relu in relus} == {"cluster_p1", "cluster_p2"}
    assert {name: frames.get(name) for name in expected} == expected


def test_unrolled_shared_child_puts_every_pass_in_its_frame(tmp_path: Path) -> None:
    """Unrolled control: every pass of every op sits inside its own module-call frame."""

    torch.manual_seed(0)
    trace = tl.trace(_TwoParents(), torch.randn(2, 4))
    dot = str(
        trace.draw(
            vis_mode="unrolled",
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "unrolled"),
        )
    )
    frames = _innermost_frames(dot)
    expected = {}
    for layer in trace.layer_logs.values():
        for pass_index, op in layer.ops.items():
            path = _owner(op.modules, op.is_atomic_module)
            if path:
                name = f"{layer.layer_label}pass{pass_index}"
                expected[name] = f"cluster_{path[-1].replace(':', '_pass')}"
    assert {name: frames.get(name) for name in expected} == expected


def _rank_frame_boxes(
    trace: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> dict[str, str]:
    """Render through the rank path with neato stubbed; return each cluster's ``bb``."""

    captured: dict[str, str] = {}

    def _fake_neato(**neato_kwargs: object) -> SimpleNamespace:
        captured["source"] = Path(str(neato_kwargs["source_path"])).read_text()
        Path(str(neato_kwargs["rendered_path"])).write_text("<svg></svg>")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(layout_mod, "_run_neato_with_fallbacks", _fake_neato)
    trace.draw(
        vis_node_placement="rank",
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "rank"),
        **kwargs,
    )
    boxes: dict[str, str] = {}
    current: list[str] = []
    for line in captured["source"].splitlines():
        stripped = line.strip()
        if stripped.startswith("subgraph cluster_"):
            current.append(stripped.split()[1])
        elif stripped.startswith("bb=") and current:
            boxes[current[-1]] = stripped
        elif stripped == "}" and current:
            current.pop()
    return boxes


@pytest.mark.parametrize("show_cone", [False, True], ids=["marks", "marks_and_cone"])
def test_rank_layout_edit_marks_keep_module_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, show_cone: bool
) -> None:
    """Rank layout: the edited stack draws exactly the frames the unedited stack draws."""

    torch.manual_seed(0)
    trace = tl.trace(
        _ResStack(),
        torch.randn(1, 8),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    plain = _rank_frame_boxes(trace, tmp_path, monkeypatch)
    fork = trace.fork("edited")
    fork.do(tl.label("linear_1_1"), tl.zero_ablate())
    fork.do(tl.label("relu_2_5"), tl.scale(0.5))
    edited = _rank_frame_boxes(fork, tmp_path, monkeypatch, vis_show_cone=show_cone)
    assert len(plain) == 4
    assert edited == plain


def test_rank_size_estimate_ignores_edit_rows() -> None:
    """An edited node is sized from its unedited rows, whatever its edit rows say."""

    from torchlens.visualization._edit_marks import EditRow
    from torchlens.visualization.node_spec import render_lines_to_html

    rows = ["linear_1_1", "(1, 8), 32 B", "params: weight (8, 8)"]
    plain = render_lines_to_html(rows)
    edited = render_lines_to_html(
        [*rows, EditRow("\u0394 zero_ablate", "#7B1FA2"), EditRow("+2 more", "#7B1FA2")]
    )
    assert layout_mod._without_edit_rows(edited) == plain
    assert layout_mod._without_edit_rows(plain) == plain
