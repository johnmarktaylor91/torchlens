"""A ``module=`` focused draw: its input and output stubs, and ``depth=`` counted from the focus.

Each stub stands for one op outside the focused module: an input stub reads
``from <op>``, the shape and size of the tensor it brings, and that op's
module; an output stub reads ``to <op>`` and its module. Stubs sit outside
the focused module's frame. ``depth=`` counts module nesting from the focused
module, so a focused draw at ``depth=N`` shows the subtree as a whole-model
draw at ``depth=N`` shows a model rooted there.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._label_format import format_memory, format_shape

pytestmark = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot not installed")

_NODE = re.compile(r"^\s*(\S+) \[label=<(.*?)> ", re.MULTILINE)
_TAG = re.compile(r"<[^>]*>")
_CLUSTER = re.compile(r'subgraph "?cluster_([^"\s{]+)')


class _Swish(nn.Module):
    """A three-op leaf module: ``x * sigmoid(x) + 0.1``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x) + 0.1


class _Inner(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = _Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class _Mixer(nn.Module):
    """Two inputs from two modules; its output feeds a module and the model output."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([x, y], dim=-1)) * 0.5)


class _Toy(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.pre_x = _Inner(width)
        self.pre_y = _Inner(width)
        self.mixer = _Mixer(width)
        self.shared = _Inner(width)
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = self.mixer(self.pre_x(x), self.pre_y(y))
        return self.head(self.shared(self.shared(mixed))), mixed


@pytest.fixture(scope="module")
def trace() -> Iterator[tl.Trace]:
    torch.manual_seed(0)
    captured = tl.trace(_Toy().eval(), (torch.randn(2, 4), torch.randn(2, 4)))
    try:
        yield captured
    finally:
        captured.cleanup()


def _dot(trace: tl.Trace, folder: Path, name: str, **kwargs: object) -> str:
    return trace.draw(str(folder / f"{name}.svg"), **kwargs).dot


def _node_rows(dot: str) -> dict[str, list[str]]:
    """Map each drawn node name to the text rows of its label."""

    rows: dict[str, list[str]] = {}
    for name, label in _NODE.findall(dot):
        text = _TAG.sub("\n", label)
        rows[name] = [row.strip() for row in text.split("\n") if row.strip()]
    return rows


def _stubs(dot: str, kind: str) -> dict[str, list[str]]:
    return {
        name: rows
        for name, rows in _node_rows(dot).items()
        if name.startswith(f"__module_focus_{kind}_")
    }


def _cluster_body(dot: str, cluster: str) -> str:
    """Return the text inside one cluster's braces (nested clusters included)."""

    start = dot.index(f"cluster_{cluster}")
    open_brace = dot.index("{", start)
    depth = 0
    for index in range(open_brace, len(dot)):
        if dot[index] == "{":
            depth += 1
        elif dot[index] == "}":
            depth -= 1
            if depth == 0:
                return dot[open_brace : index + 1]
    raise AssertionError(f"unbalanced cluster {cluster}")


def _tensor_row(trace: tl.Trace, label: str) -> str:
    """The ``(shape), size`` row of an op named by its label or its DOT node name."""

    op = trace[label.replace("pass", ":")]
    return f"{format_shape(op.shape)}, {format_memory(op.activation_memory)}"


# -- stubs name the outside op, its tensor and its module ----------------------


def test_input_stubs_name_the_outside_op_its_tensor_and_its_module(
    trace: tl.Trace, tmp_path: Path
) -> None:
    dot = _dot(trace, tmp_path, "mixer", module="mixer")
    stubs = _stubs(dot, "input")
    feeders = set(trace.modules["mixer"].input_ops)
    assert len(stubs) == 2
    for name, rows in stubs.items():
        label = name.removeprefix("__module_focus_input_")
        assert label in feeders
        module = trace[label].modules[-1].split(":")[0]
        assert rows == [f"from {label}", _tensor_row(trace, label), f"@{module}"], rows
    assert {rows[-1] for rows in stubs.values()} == {"@pre_x.act", "@pre_y.act"}


def test_output_stubs_name_where_the_output_goes(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "mixer", module="mixer")
    stubs = _stubs(dot, "output")
    assert len(stubs) == 2, stubs
    by_first_row = {rows[0]: rows for rows in stubs.values()}
    # One reader is the first call of ``shared`` (a module called twice, so its
    # op's address carries the pass); the other is the model's own output.
    (to_shared,) = [rows for rows in stubs.values() if rows[-1].startswith("@shared.fc")]
    assert to_shared[-1] == "@shared.fc:1"
    assert to_shared[0].startswith("to linear_")
    (to_output,) = [rows for rows in stubs.values() if rows[-1].startswith("model output")]
    assert to_output[0].startswith("to output_")
    assert to_output[-1] == "model output 1"
    # No size row: the tensor leaving is the inner op's own, already drawn.
    assert all(len(rows) == 2 for rows in stubs.values()), by_first_row


def test_stubs_sit_outside_the_focused_frame(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "mixer", module="mixer")
    body = _cluster_body(dot, "mixer_pass1")
    assert "__module_focus_" not in body
    assert "cat_1_" in body
    assert dot.count("__module_focus_input_") >= 2


def test_model_input_stub_reads_the_input_role(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "pre_x", module="pre_x")
    (rows,) = _stubs(dot, "input").values()
    assert rows[0] == "from input_1"
    assert rows[1] == _tensor_row(trace, "input_1")
    assert rows[2] == "model input x"  # the role in words, not an ``@`` module address


def test_leaf_focus_draws_its_ops_between_its_stubs(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "act", module="pre_x.act")
    body = _cluster_body(dot, "pre_x.act_pass1")
    assert "sigmoid_1_" in body and "mul_1_" in body and "add_1_" in body
    (into,) = _stubs(dot, "input").values()
    (out,) = _stubs(dot, "output").values()
    # Addresses leave off the scope line's ``pre_x``; ``mixer`` is outside it.
    assert into[0].startswith("from linear_") and into[-1] == "@fc"
    assert out[0].startswith("to cat_") and out[-1] == "@mixer"
    assert "<B>@act</B>" in dot and "@pre_x.act" not in dot
    notes = _dot(trace, tmp_path, "act_notes", module="pre_x.act", show_legend=True)
    assert "@name: a module address under pre_x" in notes


def test_focus_on_a_module_called_twice_draws_both_passes(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "shared", module="shared")
    clusters = set(_CLUSTER.findall(dot))
    assert {"shared_pass1", "shared_pass2"} <= clusters
    (into,) = _stubs(dot, "input").values()
    (out,) = _stubs(dot, "output").values()
    assert into[-1] == "@mixer.norm"
    assert out[-1] == "@head"


def test_reading_notes_explain_from_and_to_when_asked(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "mixer", module="mixer", show_legend=True)
    header = dot[dot.index("label=<") : dot.index("labeljust")]
    assert "from name, to name: the op outside this module" in header
    plain = _dot(trace, tmp_path, "mixer_plain", module="mixer")
    assert "from name, to name" not in plain  # the reading key is opt-in
    whole = _dot(trace, tmp_path, "whole", show_legend=True)
    assert "from name, to name" not in whole


@pytest.mark.parametrize("rolled", [False, True])
def test_rolled_and_unrolled_focus_share_the_stub_rows(
    trace: tl.Trace, tmp_path: Path, rolled: bool
) -> None:
    dot = _dot(trace, tmp_path, f"rolled{rolled}", module="mixer", rolled=rolled)
    for rows in _stubs(dot, "input").values():
        assert rows[0].startswith("from gelu_") or rows[0].startswith("from add_")
        assert rows[-1] in {"@pre_x.act", "@pre_y.act"}


@pytest.mark.parametrize("module", ["mixer", "pre_x.act"])
def test_every_default_op_node_carries_name_shape_size_and_params(
    trace: tl.Trace, tmp_path: Path, module: str
) -> None:
    """A focused draw's op nodes keep the whole-model rows: name, ``(shape), size``, params.

    The shape row is never left off because it repeats an input's: mul and
    norm keep the projection's shape and still show it.
    """

    dot = _dot(trace, tmp_path, module.replace(".", "_"), module=module)
    rows = _node_rows(dot)
    drawn = {name: node_rows for name, node_rows in rows.items() if not name.startswith("__")}
    assert drawn
    for name, node_rows in drawn.items():
        label = name.replace("pass", ":")
        assert node_rows[0].startswith(name.rsplit("pass", 1)[0]), (name, node_rows)
        assert _tensor_row(trace, label) in node_rows, (name, node_rows)
        if trace[label].has_params:
            assert any(row.startswith("params:") for row in node_rows), (name, node_rows)
    (into,) = _stubs(dot, "input").values() if module == "pre_x.act" else (None,)
    if into is not None:  # the input stub still brings its tensor's shape and size
        assert into[1] == _tensor_row(trace, "linear_1_1")


def test_whole_model_draw_keeps_every_shape_row(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "whole")
    for name, node_rows in _node_rows(dot).items():
        if name.startswith(("mul_", "layer", "sigmoid_", "add_")):
            assert any(row.startswith("(") for row in node_rows), (name, node_rows)


def test_leaf_rows_under_the_scope_leave_the_scope_prefix_off(
    trace: tl.Trace, tmp_path: Path
) -> None:
    """A focused leaf draws its op with a ``@name`` row relative to the scope line."""

    norm = _dot(trace, tmp_path, "norm", module="mixer.norm")
    assert ">@norm</TD>" in norm and "@mixer.norm" not in norm
    (into,) = _stubs(norm, "input").values()
    assert into[-1] == "@mixer"  # the scope module itself keeps its own name
    whole = _dot(trace, tmp_path, "whole")
    assert ">@mixer.norm</TD>" in whole


# -- depth counts from the focused module --------------------------------------


def test_default_focus_depth_opens_the_subtree_like_the_whole_model_draw(
    trace: tl.Trace, tmp_path: Path
) -> None:
    focus = _dot(trace, tmp_path, "pre_x", module="pre_x")
    whole = _dot(trace, tmp_path, "whole")
    under_focus = set(_CLUSTER.findall(focus))
    under_whole = {c for c in _CLUSTER.findall(whole) if c.startswith("pre_x")}
    assert under_focus == under_whole == {"pre_x_pass1", "pre_x.act_pass1"}


def test_focus_depth_one_draws_the_children_as_one_box_each(
    trace: tl.Trace, tmp_path: Path
) -> None:
    focus = _dot(trace, tmp_path, "pre_x_d1", module="pre_x", depth=1)
    clusters = set(_CLUSTER.findall(focus))
    assert "pre_x_pass1" in clusters
    assert "pre_x.act_pass1" not in clusters
    assert "sigmoid_1_" not in focus
    assert "box3d" in focus
    # The whole-model draw collapses ``act`` at the depth that reaches it (2).
    whole = _dot(trace, tmp_path, "whole_d2", depth=2)
    assert "pre_x.act_pass1" not in set(_CLUSTER.findall(whole))
    assert "sigmoid_1_" not in whole


def test_focus_depth_two_opens_the_children(trace: tl.Trace, tmp_path: Path) -> None:
    focus = _dot(trace, tmp_path, "pre_x_d2", module="pre_x", depth=2)
    assert "pre_x.act_pass1" in set(_CLUSTER.findall(focus))
    assert "sigmoid_1_" in focus
    assert "box3d" not in focus
