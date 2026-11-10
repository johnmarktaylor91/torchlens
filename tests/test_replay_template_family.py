"""Replay argument templates are a derived family, resolved on first read.

Step 9 stores each captured ``args_template`` and ``kwargs_template`` as a
carried cell (``torchlens/_postprocess/_replay_template_cells.py``); the first
read renames its labels and binds each ``ParentRef``'s display key, the work
steps 9 and 17 used to do for every op. These tests pin that nothing in
capture, ``to_pandas``, ``repr`` or a default save resolves a template, that a
read answers final labels with bound display keys, and that pickling, forks
and the per-op step 9 path keep the same answers.
"""

from __future__ import annotations

import gc
import pickle
import types
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._postprocess import labeling
from torchlens._postprocess._replay_template_cells import (
    REPLAY_TEMPLATES,
    carry_entry_templates,
)
from torchlens._trace_core.cell_states import (
    _MISSING,
    STATE_DERIVED,
    STATE_SET,
    LazyCell,
    cell_state,
)
from torchlens._trace_core.op_store import DetachedOpStore
from torchlens._utils._display_keys import display_key
from torchlens.intervention._types import CapturedArgTemplate, ParentRef

pytestmark = pytest.mark.filterwarnings("ignore::torchlens.errors.InterventionWarning")

FIELDS = ("args_template", "kwargs_template")


class _Looped(torch.nn.Module):
    """One layer applied twice (multi-pass keys) and a keyword-argument call."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.fc(x))
        y = self.fc(y)
        return torch.sum(y * 2.0, dim=1, keepdim=True)


def _trace(**kwargs: Any) -> Any:
    torch.manual_seed(0)
    return tl.trace(_Looped(), torch.randn(3, 4), **kwargs)


def _cell(op: Any, name: str) -> tuple[Any, int, int]:
    """Return ``(store, row, fid)`` of one stored field of an op."""

    store = op._core
    return store, op._row, store.layout.fid_by_name[name]


def _states(trace: Any) -> list[str]:
    """Return the state of every template cell that holds a template."""

    states = []
    for op in trace._layer_list:
        for name in FIELDS:
            store, row, fid = _cell(op, name)
            raw = store.cell_get(row, fid, True)
            if raw is None or raw is _MISSING:
                continue
            if raw.__class__ is LazyCell and raw.state != STATE_DERIVED:
                continue  # a default-omitted ``None``
            states.append(cell_state(store, row, fid))
    return states


def _parent_refs(value: Any, found: list[Any]) -> list[Any]:
    """Collect the ``ParentRef`` leaves of a template in walk order."""

    if isinstance(value, ParentRef):
        found.append(value)
    elif isinstance(value, CapturedArgTemplate):
        _parent_refs(value.args, found)
        _parent_refs(value.kwargs, found)
    elif isinstance(value, (tuple, list)):
        for member in value:
            _parent_refs(member, found)
    elif isinstance(value, dict):
        for member in value.values():
            _parent_refs(member, found)
    return found


def _plain(value: Any) -> Any:
    """Return a template as comparable plain data (tensors by shape and dtype)."""

    if isinstance(value, ParentRef):
        return ("ref", value.parent_label, getattr(value, "_display_key", None))
    if isinstance(value, CapturedArgTemplate):
        return ("template", _plain(value.args), _plain(value.kwargs))
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), value.dtype)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [_plain(member) for member in value])
    if isinstance(value, dict):
        return ("dict", [(key, _plain(member)) for key, member in value.items()])
    return value


def _snapshot(trace: Any) -> list[tuple[Any, ...]]:
    """Return every op's resolved templates, as plain data, with their display keys."""

    rows = []
    for op in trace._layer_list:
        for name in FIELDS:
            template = getattr(op.call, name)
            keys = [
                (ref.parent_label, getattr(ref, "_display_key", None))
                for ref in _parent_refs(template, [])
            ]
            rows.append((op.layer_label, name, _plain(template), keys))
    return rows


def test_capture_leaves_every_template_carried() -> None:
    """No postprocess step reads a template: each one is still unresolved."""

    states = _states(_trace())
    assert states
    assert set(states) == {STATE_DERIVED}


def test_to_pandas_repr_and_save_resolve_no_template(tmp_path: Any) -> None:
    """The C7 readers never force the family."""

    trace = _trace()
    trace.to_pandas()
    repr(trace)
    trace.save(tmp_path / "t.tlspec")
    assert set(_states(trace)) == {STATE_DERIVED}


def test_read_answers_final_labels_and_binds_display_keys() -> None:
    """A resolved template names final labels; each reference carries its display key."""

    trace = _trace()
    records = trace._layer_dict_all_keys
    keys = [key for *_rest, op_keys in _snapshot(trace) for key in op_keys]
    assert keys
    for label, key in keys:
        assert label in records
        assert key == display_key(records[label])
    assert any(":" in key for _label, key in keys)
    assert set(_states(trace)) == {STATE_SET}


def test_reads_are_stable_and_args_and_kwargs_stay_distinct() -> None:
    """``op.x is op.x``; an op's args and kwargs templates are equal, separate objects."""

    trace = _trace()
    with_kwargs = [op for op in trace._layer_list if op.call.kwargs_template is not None]
    assert with_kwargs
    for op in with_kwargs:
        args, kwargs = op.call.args_template, op.call.kwargs_template
        assert args == kwargs and args is not kwargs
        assert op.call.args_template is args


def test_column_and_per_op_step_9_paths_give_the_same_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-op path (off the column path) carries the same templates."""

    column = _snapshot(_trace())
    monkeypatch.setattr(labeling, "log_final_info_by_column", lambda *_args: False)
    trace = _trace()
    assert set(_states(trace)) == {STATE_DERIVED}
    assert _snapshot(trace) == column


@pytest.mark.parametrize("replayable", (False, True))
def test_pickle_and_deepcopy_with_unresolved_templates(replayable: bool) -> None:
    """A trace whose templates are unresolved pickles; the copy answers the same."""

    import copy

    options = tl.options.CaptureOptions(replayable=True) if replayable else None
    trace = _trace(capture=options) if options is not None else _trace()
    restored = pickle.loads(pickle.dumps(trace))
    copied = copy.deepcopy(trace)
    assert set(_states(trace)) == {STATE_DERIVED}
    expected = [(label, name, template) for label, name, template, _keys in _snapshot(trace)]
    for other in (restored, copied):
        got = [
            (op.layer_label, name, _plain(getattr(op.call, name)))
            for op in other._layer_list
            for name in FIELDS
        ]
        if trace.is_replayable:
            assert got == expected
        else:
            assert all(template is None for _label, _name, template in got)


def test_fork_resolves_into_its_own_view() -> None:
    """A fork's read answers the parent's template and leaves the parent's cell carried."""

    trace = _trace()
    fork = trace.fork()
    fork_snapshot = _snapshot(fork)
    assert set(_states(trace)) == {STATE_DERIVED}
    assert fork_snapshot == _snapshot(trace)


def _raw_material(op: Any) -> Any:
    """Return the raw template an op's carried ``args_template`` cell holds, else ``None``."""

    store, row, fid = _cell(op, "args_template")
    material = REPLAY_TEMPLATES.material_of(store.cell_get(row, fid, True))
    return None if material is _MISSING else material[0]


def test_carried_cell_pickled_away_from_its_trace_keeps_display_keys() -> None:
    """A pickled carried cell carries the label map and display keys, not the trace."""

    trace = _trace()
    op = next(op for op in reversed(trace._layer_list) if _parent_refs(_raw_material(op), []))
    store, row, fid = _cell(op, "args_template")
    layout = store.layout
    payload = pickle.dumps(store.cell_get(row, fid, True))
    expected = [
        (ref.parent_label, getattr(ref, "_display_key", None))
        for ref in _parent_refs(op.call.args_template, [])
    ]
    assert all(key is not None for _label, key in expected)
    del trace, op, store
    gc.collect()
    cell = pickle.loads(payload)
    assert cell.material[1].trace_ref is None
    seed: list[Any] = [_MISSING] * layout.n_fields
    seed[fid] = cell
    detached = DetachedOpStore(layout)
    detached.adopt_row(seed)
    resolved = detached.cell_get(0, fid)
    got = [
        (ref.parent_label, getattr(ref, "_display_key", None)) for ref in _parent_refs(resolved, [])
    ]
    assert got == expected


def test_record_without_a_row_store_is_renamed_at_once() -> None:
    """The per-op path renames a test double's templates instead of carrying them."""

    class _Trace:
        _raw_to_final_layer_labels = {"raw_a": "relu_1_2"}

    template = CapturedArgTemplate(args=(ParentRef("raw_a"), ParentRef("other")))
    double = types.SimpleNamespace(args_template=template, kwargs_template=None)
    carry_entry_templates(
        _Trace(), double, lambda name, value: setattr(double, name, value), {}, ParentRef
    )
    assert double.args_template == CapturedArgTemplate(
        args=(ParentRef("relu_1_2"), ParentRef("other"))
    )
    assert double.kwargs_template is None
