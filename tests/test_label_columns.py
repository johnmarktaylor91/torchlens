"""Step 8 by column: label parts as integers, the same labels as the per-op pass.

The column pass (``_postprocess/_label_columns.py``) must give every op the
labels, label parts and lookup keys the facade pass gives it, and the trace
the same label maps and remapped trace-level references. Each model below runs
twice, once through each pass, and every public label value is compared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from _oracle_env import expect_bundle_minor_version_mismatch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _label_columns, labeling
from torchlens._postprocess._final_info_columns import _NotColumnar
from torchlens._postprocess._pipeline import PIPELINE

_OLD_ARTIFACT = Path(__file__).resolve().parent / "agent_surface_goldens" / "clean.tlspec"


class _Chain(nn.Module):
    """Linear and relu layers in a row."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class _Cell(nn.Module):
    """One cell called three times, so its ops have later passes."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.cell(x)) + x
        return x


class _Buffers(nn.Module):
    """A registered buffer read twice and a batch norm in eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offset", torch.ones(4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm(x + self.offset)
        return y * self.offset, torch.sin(y)


class _NoParamLoop(nn.Module):
    """A loop of parameter-free ops of one repeated type."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(4):
            x = torch.sin(x) + 1
        return x


_MODELS = {
    "chain": (_Chain, (2, 4)),
    "cell": (_Cell, (2, 4)),
    "buffers": (_Buffers, (3, 4)),
    "no_param_loop": (_NoParamLoop, (2, 4)),
}

_OP_FIELDS = (
    "label",
    "layer_label",
    "short_label",
    "short_layer_label",
    "type_index",
    "step_index",
    "pass_number",
    "num_passes",
    "lookup_keys",
    "parents",
    "children",
)
_TRACE_FIELDS = (
    "input_layers",
    "output_layers",
    "buffer_layers",
    "layers_with_params",
    "layer_labels",
    "op_labels",
    "_raw_to_final_layer_labels",
    "_raw_to_final_parent_layer_labels",
    "_raw_to_final_op_labels",
    "_final_to_raw_layer_labels",
    "_graph_op_equivalence_classes",
)


def _build(name: str) -> nn.Module:
    """One seeded instance of a model of ``_MODELS``, in eval mode."""

    torch.manual_seed(0)
    return _MODELS[name][0]().eval()


def _trace(name: str, recurrence: bool = True, model: nn.Module | None = None) -> Any:
    """Trace one model of ``_MODELS`` (a fresh instance unless given) on a fixed input."""

    if model is None:
        model = _build(name)
    torch.manual_seed(0)
    inputs = torch.randn(*_MODELS[name][1])
    capture = tl.options.CaptureOptions(recurrence_detection=recurrence)
    return tl.trace(model, inputs, capture=capture)


def _repr_without_trace_name(trace: Any) -> str:
    """``repr(trace)`` with the trace name in its identity line replaced.

    The default name carries a per-trace counter (``_noparamloop_1``, then
    ``_noparamloop_2``), so it is the one value two traces of one model
    instance spell differently. Only the ``Trace <name>`` prefix of the
    repr's first line is replaced; every other character is compared.
    """

    text = repr(trace)
    prefix = f"Trace {trace.name}"
    assert text.startswith(prefix), text
    return "Trace <name>" + text[len(prefix) :]


def _by_position(value: Any, positions: dict[str, str]) -> Any:
    """``value`` with each raw label replaced by its op's position.

    Raw labels carry a per-trace counter, so two traces of one model spell
    them differently; positions make the two traces comparable.
    """

    if isinstance(value, str):
        return positions.get(value, value)
    if isinstance(value, dict):
        return {
            _by_position(key, positions): _by_position(item, positions)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_by_position(item, positions) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    return value


def _snapshot(trace: Any) -> dict[str, Any]:
    """Every op's label values and the trace's label-valued fields."""

    positions = {op._label_raw: f"#op{index}" for index, op in enumerate(trace)}
    ops = [{field: getattr(op, field) for field in _OP_FIELDS} for op in trace]
    fields = {field: getattr(trace, field) for field in _TRACE_FIELDS}
    return _by_position({"ops": ops, "trace": fields}, positions)


def _trace_both_ways(
    monkeypatch: pytest.MonkeyPatch, name: str, recurrence: bool = True
) -> tuple[Any, Any, list[bool]]:
    """Trace ``name`` through the column pass and through the facade pass.

    Both passes trace ONE model instance, so instance-derived values (the
    parameter hashes in the equivalence-class keys) are equal as they are.
    """

    model = _build(name)
    outcomes: list[bool] = []
    column_pass = _label_columns.map_labels_by_column

    def spy(trace: Any) -> bool:
        outcome = column_pass(trace)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(labeling, "map_labels_by_column", spy)
    by_column = _trace(name, recurrence, model)
    monkeypatch.setattr(labeling, "map_labels_by_column", lambda _trace: False)
    by_facade = _trace(name, recurrence, model)
    return by_column, by_facade, outcomes


@pytest.mark.parametrize("recurrence", [True, False], ids=["loops", "params_only"])
@pytest.mark.parametrize("name", sorted(_MODELS))
def test_column_pass_equals_facade_pass(
    monkeypatch: pytest.MonkeyPatch, name: str, recurrence: bool
) -> None:
    """Both passes give equal labels, label parts, lookup keys and trace references."""

    by_column, by_facade, outcomes = _trace_both_ways(monkeypatch, name, recurrence)
    assert outcomes == [True], "the column pass must run on a building store"
    assert _snapshot(by_column) == _snapshot(by_facade)
    assert _repr_without_trace_name(by_column) == _repr_without_trace_name(by_facade)


def test_later_passes_inherit_the_first_pass_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pass of one layer shares the layer label, type and type index."""

    by_column, _by_facade, outcomes = _trace_both_ways(monkeypatch, "cell")
    assert outcomes == [True]
    multi = [op for op in by_column if op.num_passes > 1]
    assert multi, "the cell model must have later passes"
    for op in multi:
        first = by_column[f"{op.layer_label}:1"]
        assert (op.type, op.type_index) == (first.type, first.type_index)
        assert first.layer_label == op.layer_label
        assert op.label == f"{op.layer_label}:{op.pass_number}"


def test_every_label_spelling_finds_its_op() -> None:
    """``trace[...]`` resolves each op's label, short label and single-pass layer labels."""

    trace = _trace("buffers")
    for op in trace:
        assert trace[op.label].raw_index == op.raw_index
        if op.num_passes == 1:
            assert trace[op.layer_label].raw_index == op.raw_index
    for label in (*trace.input_layers, *trace.output_layers):
        assert trace[label].layer_label == label


def test_label_parts_number_inputs_buffers_and_outputs() -> None:
    """Inputs and buffers take step 0; outputs and ops take the running step."""

    kinds = ["input", "linear", "buffer", "add", "output"]
    passes = [1, 1, 1, 1, 1]
    raws = ["input_raw", "linear_raw", "buffer_raw", "add_raw", "output_raw"]
    groups: list[Any] = [None] * 5
    out_kinds, type_indices, step_indices = _label_columns.label_parts(kinds, passes, raws, groups)
    assert out_kinds == kinds
    assert type_indices == [1, 1, 1, 1, 1]
    assert step_indices == [0, 1, 0, 2, 3]


def test_label_parts_later_pass_inherits_its_head() -> None:
    """A later pass takes its group head's type and indices, not its own type."""

    kinds = ["input", "linear", "relu", "linear", "relu"]
    passes = [1, 1, 1, 2, 2]
    raws = ["i", "l1", "r1", "l2", "r2"]
    groups: list[Any] = [None, ["l1", "l2"], ["r1", "r2"], ["l1", "l2"], ["r1", "r2"]]
    out_kinds, type_indices, step_indices = _label_columns.label_parts(kinds, passes, raws, groups)
    assert out_kinds == kinds
    assert type_indices == [1, 1, 1, 1, 1]
    assert step_indices == [0, 1, 2, 1, 2]
    groups[3] = ["r1", "l2"]
    out_kinds, type_indices, step_indices = _label_columns.label_parts(kinds, passes, raws, groups)
    assert out_kinds[3] == "relu"
    assert (type_indices[3], step_indices[3]) == (1, 2)


@pytest.mark.parametrize(
    "group",
    [None, [], ["missing", "l2"], ["l2", "l1"]],
    ids=["none", "empty", "missing_head", "head_after"],
)
def test_label_parts_defers_unreadable_groups(group: Any) -> None:
    """A later pass whose head is absent or not earlier goes to the per-op pass."""

    kinds = ["linear", "linear"]
    passes = [1, 2]
    raws = ["l1", "l2"]
    with pytest.raises(_NotColumnar):
        _label_columns.label_parts(kinds, passes, raws, [None, group])


def test_rename_step_is_gone_from_the_step_list() -> None:
    """The trace-level remap runs inside ``map_final_labels``; no step of its own."""

    names = [step.name for step in PIPELINE]
    assert "rename_layer_references" not in names
    assert names.index("map_final_labels") + 1 == names.index("log_final_layer_info")
    trace = _trace("chain")
    assert "rename_layer_references" not in trace._phase_timings
    assert "map_final_labels" in trace._phase_timings


def test_saved_file_with_the_old_timing_key_loads_and_reads_equal() -> None:
    """A file saved while the rename step existed loads; its labels read as saved."""

    with expect_bundle_minor_version_mismatch():
        loaded = tl.load(str(_OLD_ARTIFACT))
    assert "rename_layer_references" in loaded._phase_timings
    labels = [op.label for op in loaded]
    assert "relu_1_2:1" in labels
    for op in loaded:
        assert loaded[op.label].raw_index == op.raw_index
    assert loaded.output_layers == [loaded[label].layer_label for label in loaded.output_layers]
