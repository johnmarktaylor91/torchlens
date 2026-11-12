"""Soundness of the per-capture fixed-cost cuts (lane FXE).

Four cuts, each with a reference it must equal:

* The output-node copy reads the raw row cell for every generated row-cell
  descriptor kind (plain, relation, group, dataflow, fact) and calls the
  descriptor only for an unset cell, a transformed cell class or a transformed
  sentinel. Reference: reading every field through its descriptor, checked on
  every output node of live captures.
* ``OpColumns`` keeps a read plan per field and ``RowIngestPlan`` keeps the last
  layout's field-id tables. Reference: the same captures with both caches
  bypassed, field by field, on either side of ``MIN_COLUMN_OPS``, repeated, and
  after a capture that failed mid-postprocess.
* ``run_pipeline`` records step timings without the first-sample ``KeyError``.
  Reference: ``_record_phase_timing``.
* ``validate_amendment`` tests the identity prefixes with one ``startswith``.
  Reference: the refusal it must still raise.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_mod
from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER
from torchlens._data_classes import _op_columns, _op_output_copy as ooc
from torchlens._data_classes._op_row_release import _OP_FACADE
from torchlens._data_classes.op import _OP_SLOT_NAMES, _OP_STORE_LAYOUT, Op
from torchlens._ir import op_record
from torchlens._postprocess import _pipeline, _row_ingest
from torchlens._postprocess._step_views import MIN_COLUMN_OPS
from torchlens._trace_core.op_store import _CSR, _FACT, _MISSING, OpStoreLayout
from torchlens._utils._record_group_table import op_attr
from torchlens._utils.display import _record_phase_timing

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
#: Fields that differ between two separate traces by design: the owning trace, the
#: measured wall time of each call and the id of each autograd node.
_PER_TRACE_FIELDS = frozenset({"source_trace", "func_duration", "grad_fn_object_id"})


class _Chain(nn.Module):
    """``depth`` elementwise ops on one input (the perf gate's model shape)."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.depth):
            x = x + 1
        return x


class _Recurrent(nn.Module):
    """One linear layer applied repeatedly, a branch, a buffer and two outputs."""

    def __init__(self, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.fc = nn.Linear(4, 4)
        self.norm = nn.BatchNorm1d(4)
        self.register_buffer("offset", torch.ones(4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.steps):
            x = torch.relu(self.fc(x))
        hidden = self.norm(x) + self.offset
        if hidden.sum() > -1e9:
            hidden = hidden * 1.5
        return torch.tanh(hidden), {"twice": hidden * 2}["twice"]


def _models() -> list[tuple[str, nn.Module, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("chain_small", _Chain(3), torch.ones(8)),
        ("chain_column", _Chain(MIN_COLUMN_OPS + 8), torch.ones(8)),
        ("recurrent_small", _Recurrent(2).eval(), torch.randn(2, 4)),
        ("recurrent_column", _Recurrent(12).eval(), torch.randn(2, 4)),
    ]


def _frozen(value: Any) -> Any:
    """Return a comparison key for ``value`` (tensors by exact value)."""

    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), value.detach().flatten().tolist())
    if isinstance(value, dict):
        return ("dict", tuple((repr(key), _frozen(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(_frozen(item) for item in value))
    if isinstance(value, set | frozenset):
        return ("set", tuple(sorted(repr(item) for item in value)))
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return _ADDRESS.sub("0x", repr(value))


def _trace_fields(trace: Any) -> dict[tuple[str, str], Any]:
    """Freeze every stored field of every op, plus the trace's op labels."""

    fields: dict[tuple[str, str], Any] = {
        ("<trace>", "labels"): [str(op.label) for op in trace.ops]
    }
    for op in trace.ops:
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            if field not in _PER_TRACE_FIELDS:
                fields[(str(op.label), field)] = _frozen(getattr(op, op_attr(field), None))
    return fields


def _capture(model: nn.Module, x: torch.Tensor) -> dict[tuple[str, str], Any]:
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=5))
    try:
        return _trace_fields(trace)
    finally:
        trace.cleanup()


def _assert_same_fields(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    mismatched = [key for key in left if left[key] != right[key]]
    assert not mismatched, mismatched[:10]


def _bypass_plan_caches(patch: pytest.MonkeyPatch) -> None:
    """Make ``OpColumns`` and ``RowIngestPlan`` plan from scratch on every call."""

    def uncached_read_plan(name: str) -> tuple[Any, tuple[type, ...], Any]:
        attribute = _op_columns._stored_attribute(name)
        return (attribute, *_op_columns._slow_cell_types(name, attribute))

    def uncached_layout_tables(layout: Any) -> tuple[Any, ...]:
        _row_ingest._LAYOUT_TABLES.clear()
        return original_layout_tables(layout)

    original_layout_tables = _row_ingest._layout_tables
    patch.setattr(_op_columns, "_read_plan", uncached_read_plan)
    patch.setattr(
        _op_columns, "_planned_attribute", lambda name: (uncached_read_plan(name)[0], None)
    )
    patch.setattr(_row_ingest, "_layout_tables", uncached_layout_tables)


# --- cut 1: output-node copy reads raw cells for every generated descriptor kind


def _reference_fields(op: Op, plan: tuple[tuple[Any, ...], ...]) -> dict[str, Any]:
    """Read every planned field through its descriptor (or ``getattr``)."""

    cls = type(op)
    return {
        field: getattr(op, attr, None)
        if descriptor is None
        else ooc._descriptor_read(descriptor, op, cls)
        for field, descriptor, attr, _shared, _fid in plan
    }


@pytest.mark.parametrize("index", range(4))
def test_output_copy_raw_reads_equal_descriptor_reads_in_live_capture(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    """Every output node's copied fields equal a read of every field through its descriptor."""

    name, model, x = _models()[index]
    original = ooc._read_parent_fields
    seen: dict[str, int] = {"copies": 0, "raw_reads": 0}

    def checked(op: Op, plan: tuple[tuple[Any, ...], ...]) -> dict[str, Any]:
        fast = original(op, plan)
        reference = _reference_fields(op, plan)
        cells = ooc._plain_cells(op)
        for field, _descriptor, _attr, shared, fid in plan:
            assert _frozen(fast[field]) == _frozen(reference[field]), (name, field)
            if not shared and isinstance(reference[field], list | dict | set):
                assert fast[field] is not reference[field], (name, field)
            seen["raw_reads"] += cells is not None and fid is not None
        seen["copies"] += 1
        return fast

    monkeypatch.setattr(ooc, "_read_parent_fields", checked)
    trace = tl.trace(model, x)
    trace.cleanup()
    assert seen["copies"] >= 1
    assert seen["raw_reads"] > 0, "the raw-cell path was never taken"


def test_output_copy_plan_takes_raw_cells_for_every_generated_kind() -> None:
    """Relation, group, dataflow and fact fields get a raw-cell column id, not only plain ones."""

    shared = ooc._ALWAYS_SHARED_FIELDS | ooc._op_copy._OUTPUT_NODE_REPLACED_FIELDS
    plan = ooc._field_plan(Op, shared, ooc._op_copy._OUTPUT_COPY_UNREAD_DERIVED_FIELDS)
    kinds = {
        _op_columns._DESCRIPTOR_KINDS[type(descriptor).__name__]
        for _field, descriptor, _attr, _shared, fid in plan
        if fid is not None
    }
    assert {"plain", "relation", "group", "dataflow", "fact"} <= kinds


def test_output_copy_transformed_cells_cover_every_descriptor_transform() -> None:
    """Each cell class or sentinel a generated descriptor transforms sends the copy to the descriptor."""

    checked = 0
    for name in _OP_SLOT_NAMES:
        attribute = vars(Op).get(op_attr(name))
        if type(attribute).__name__ not in _op_columns._DESCRIPTOR_KINDS:
            continue
        slow_types, sentinel = _op_columns._slow_cell_types(name, attribute)
        assert set(slow_types) <= ooc._TRANSFORMED_CELLS, name
        assert sentinel is _MISSING or sentinel in ooc._TRANSFORMED_SENTINELS, name
        checked += 1
    assert checked > 0
    assert set(ooc._TRANSFORMED_SENTINELS) == {_CSR, _FACT}


# --- cut 2 and 3: OpColumns read plans and row-ingest layout tables


@pytest.mark.parametrize("index", range(4))
def test_cached_plans_equal_fresh_plans_field_by_field(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    """Two warm captures and a capture with both caches bypassed are identical, field by field."""

    _name, model, x = _models()[index]
    first = _capture(model, x)
    second = _capture(model, x)
    with monkeypatch.context() as patch:
        _bypass_plan_caches(patch)
        fresh = _capture(model, x)
    _assert_same_fields(first, second)
    _assert_same_fields(first, fresh)


def test_interleaved_small_and_column_captures_stay_equal() -> None:
    """Alternating facade-path and column-path captures repeat their own fields exactly."""

    models = _models()
    baseline = {name: _capture(model, x) for name, model, x in models}
    for _ in range(2):
        for name, model, x in reversed(models):
            _assert_same_fields(baseline[name], _capture(model, x))


def test_capture_after_a_failed_postprocess_equals_one_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step that raises mid-pipeline leaves no plan that changes the next capture."""

    _name, model, x = _models()[1]
    before = _capture(model, x)

    class _Boom(RuntimeError):
        pass

    def failing_step(*args: Any, **kwargs: Any) -> None:
        raise _Boom("mid-postprocess")

    with monkeypatch.context() as patch:
        patch.setattr(postprocess_mod, "_freeze_relation_views", failing_step)
        with pytest.raises(Exception) as excinfo:
            tl.trace(model, x, capture=tl.options.CaptureOptions(seed=5))
    chain: list[BaseException] = []
    error: BaseException | None = excinfo.value
    while error is not None and len(chain) < 10:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(item, _Boom) for item in chain)
    _assert_same_fields(before, _capture(model, x))


def test_nested_trace_refusal_leaves_the_next_capture_unchanged() -> None:
    """A nested trace still raises, and the outer model's next capture is unchanged."""

    _name, model, x = _models()[0]
    before = _capture(model, x)

    class _Nests(nn.Module):
        def forward(self, y: torch.Tensor) -> torch.Tensor:
            tl.trace(_Chain(2), y)
            return y + 1

    with pytest.raises(RuntimeError, match="not re-entrant"):
        tl.trace(_Nests(), x)
    _assert_same_fields(before, _capture(model, x))


def test_read_plan_is_rebuilt_when_the_class_attribute_or_facade_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan is reused only while the facade binding and the class attribute are the same objects."""

    name = next(
        field
        for field in _OP_SLOT_NAMES
        if type(vars(Op).get(op_attr(field))).__name__ == "_OpField"
    )
    attribute, slow_types, sentinel = _op_columns._read_plan(name)
    assert _op_columns._read_plan(name)[0] is attribute
    assert _op_columns._READ_PLANS[name][2] is attribute

    replacement = copy.copy(attribute)
    with monkeypatch.context() as patch:
        patch.setattr(Op, op_attr(name), replacement)
        assert _op_columns._read_plan(name) == (replacement, slow_types, sentinel)
        assert _op_columns._READ_PLANS[name][2] is replacement
    assert _op_columns._read_plan(name)[0] is attribute
    assert _op_columns._READ_PLANS[name][2] is attribute

    bound = _OP_FACADE[0]
    rebound = tuple(bound)
    _OP_FACADE[0] = rebound
    try:
        assert _op_columns._read_plan(name) == (attribute, slow_types, sentinel)
        assert _op_columns._READ_PLANS[name][0] is rebound
    finally:
        _OP_FACADE[0] = bound
    _op_columns._read_plan(name)
    assert _op_columns._READ_PLANS[name][0] is _OP_FACADE[0]


def test_layout_tables_are_reused_only_for_the_same_layout() -> None:
    """Equal tables for an equal layout, the same tables for the same one, a fresh dict per plan."""

    tables = _row_ingest._layout_tables(_OP_STORE_LAYOUT)
    assert _row_ingest._layout_tables(_OP_STORE_LAYOUT) is tables
    other = OpStoreLayout(_OP_STORE_LAYOUT.names)
    other_tables = _row_ingest._layout_tables(other)
    assert other_tables is not tables
    assert other_tables == tables
    assert _row_ingest._LAYOUT_TABLES[0][0] is other
    plan_a = _row_ingest.RowIngestPlan(_OP_STORE_LAYOUT)
    plan_b = _row_ingest.RowIngestPlan(_OP_STORE_LAYOUT)
    assert plan_a.omit_empty == plan_b.omit_empty
    assert plan_a.omit_empty is not plan_b.omit_empty


# --- cut 4: step timings


@pytest.mark.parametrize("start", [None, {}, {"step": {"total_s": 0.5, "count": 2}}])
def test_step_timing_equals_phase_timing(start: dict | None) -> None:
    """``_record_step_timing`` leaves the same ``_phase_timings`` as ``_record_phase_timing``."""

    class _Holder:
        pass

    left, right = _Holder(), _Holder()
    if start is not None:
        left._phase_timings = copy.deepcopy(start)
        right._phase_timings = copy.deepcopy(start)
    for bucket, elapsed in (("step", 0.25), ("other", 0.125), ("step", 1.0)):
        _pipeline._record_step_timing(left, bucket, elapsed)
        _record_phase_timing(right, bucket, elapsed)
    assert left.__dict__ == right.__dict__


# --- cut 5: amendment identity prefixes


@pytest.mark.parametrize("prefix", op_record._UNPATCHABLE_PREFIXES)
def test_amendment_identity_prefixes_are_still_refused(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """A forged amendment patching any identity path (or one below it) is refused."""

    for path in (prefix, prefix + ".inner"):
        monkeypatch.setitem(op_record.AMENDMENT_FAMILIES, "forged", ((path, (object,)),))
        forged = op_record.OpAmendment(
            seq=0,
            run_nonce=0,
            target_seq=0,
            target_label_raw="x",
            family="forged",
            patch=((path, 1),),
        )
        with pytest.raises(op_record.AmendmentValidationError, match="unpatchable"):
            op_record.validate_amendment(forged)


def test_amendment_non_identity_path_is_not_refused_as_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(op_record.AMENDMENT_FAMILIES, "forged", (("core.output_note", (object,)),))
    forged = op_record.OpAmendment(
        seq=0,
        run_nonce=0,
        target_seq=0,
        target_label_raw="x",
        family="forged",
        patch=(("core.output_note", 1),),
    )
    op_record.validate_amendment(forged)
