"""Deterministic counters for ``tl.record``'s per-capture bookkeeping.

A one-module ``tl.record`` read used to pay, on every capture, a full walker
call per empty hook slot of every module (twice: the pre-forward ownership
snapshot and the session-end clear), a ``dataclasses.replace`` per facet for
every journal record cloned into the recorder and the replay projection, a
dict and a set rebuilt over every context and record on every module exit,
and a wrapped ``untyped_storage`` read under ``pause_logging`` per label stamp.
These tests pin the cheaper paths by what they visit and by call counts; the
recorded facts they produce are checked equal to the plain forms they replace.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from collections import OrderedDict
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._ir.capture_events import _clone_op_event_for_replay
from torchlens._ir.op_record import OpRecord
from torchlens._ir.refs import _clone_frozen_record
from torchlens.backends.torch import _session_cleanup
from torchlens.backends.torch._module_boundary_adoption import collect_pre_forward_tensor_ids
from torchlens.backends.torch._session_cleanup import (
    _clear_module_attribute_tensor_metadata,
    _clear_session_tensor_metadata,
)
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch


def _block() -> nn.Sequential:
    return nn.Sequential(OrderedDict(fc=nn.Linear(8, 8), act=nn.ReLU()))


def _model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        OrderedDict(stem=nn.Linear(8, 8), block1=_block(), block2=_block(), head=nn.Linear(8, 4))
    ).eval()


def _record_one_module(model: nn.Module) -> Any:
    with torch.no_grad():
        return tl.record(model, torch.randn(2, 8), save=tl.module("block1"))


# ---------------------------------------------------------------------------
# Module ``__dict__`` sweep: inline leaf skips visit exactly what the walker did
# ---------------------------------------------------------------------------


class _Stashing(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.stash_list = [torch.zeros(2)]
        self.stash_dict = {"k": torch.ones(2)}
        self.empty_list: list[Any] = []
        self.empty_dict: dict[str, Any] = {}
        self.count = 3
        self.name = "stash"
        self.nothing = None
        self.flag = True
        self.shape = (3, 3)  # all-scalar tuple: skipped inline like ``kernel_size``
        self.scalar_list = [1, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _reference_sweep(model: nn.Module) -> set[int]:
    """The plain per-attribute walker loop the shared entry replaced."""

    seen: set[int] = set()
    found: set[int] = set()
    for submodule in model.modules():
        for attr_val in submodule.__dict__.values():
            _clear_session_tensor_metadata(attr_val, seen, visit=lambda t: found.add(id(t)))
    return found


def test_module_sweep_visits_the_same_tensors_as_the_plain_walker() -> None:
    model = _Stashing()
    found: set[int] = set()
    seen: set[int] = set()
    for submodule in model.modules():
        _clear_module_attribute_tensor_metadata(submodule, seen, visit=lambda t: found.add(id(t)))
    assert found == _reference_sweep(model)
    assert found == {id(model.stash_list[0]), id(model.stash_dict["k"])}


def test_module_sweep_skips_scalar_and_empty_leaves_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    real = _session_cleanup._clear_session_tensor_metadata

    def counting(value: Any, seen: set[int], depth: int = 0, visit: Any = None) -> None:
        if depth == 0:  # the sweep's own calls; the walker's recursion passes depth + 1
            calls.append(value)
        real(value, seen, depth, visit)

    monkeypatch.setattr(_session_cleanup, "_clear_session_tensor_metadata", counting)
    _clear_module_attribute_tensor_metadata(nn.Linear(4, 4), set())
    # ``_parameters`` is the only non-empty, non-scalar slot of a bare Linear.
    assert len(calls) == 1 and isinstance(calls[0], dict) and "weight" in calls[0]
    calls.clear()
    model = _Stashing()
    _clear_module_attribute_tensor_metadata(model, set())
    # The root owns no Parameter of its own: ``_modules`` and the two stashes with
    # content are its only slots that can carry a stamp.
    assert [type(v) for v in calls] == [dict, list, dict]


def test_pre_forward_snapshot_matches_the_plain_sweep() -> None:
    model = _Stashing()
    owned, _callable_only = collect_pre_forward_tensor_ids(model)
    assert set(owned) >= _reference_sweep(model)


# ---------------------------------------------------------------------------
# Journal record clone: slot copy equals ``dataclasses.replace``
# ---------------------------------------------------------------------------


def _op_records(recording: Any) -> list[OpRecord]:
    events = object.__getattribute__(recording, "_capture_events")
    return [e for e in events.amended_op_records() if isinstance(e, OpRecord)]


def test_slot_clone_equals_dataclasses_replace_on_every_record() -> None:
    records = _op_records(_record_one_module(_model()))
    assert records, "the capture journals shell records for every op"
    for record in records:
        clone = _clone_op_event_for_replay(record)
        assert clone == record and clone is not record
        assert clone.core == dataclasses.replace(record.core) and clone.core is not record.core
        assert clone.graph is not None and record.graph is not None
        assert clone.graph.parent_arg_positions == record.graph.parent_arg_positions
        assert clone.graph.parent_arg_positions is not record.graph.parent_arg_positions
        for domain in record.graph.parent_arg_positions:
            assert (
                clone.graph.parent_arg_positions[domain]
                is not record.graph.parent_arg_positions[domain]
            )
        changed = _clone_frozen_record(record.core, seq=record.core.seq + 1000)
        assert changed == dataclasses.replace(record.core, seq=record.core.seq + 1000)


# ---------------------------------------------------------------------------
# Module-exit retention indexes equal the fresh comprehensions they replaced
# ---------------------------------------------------------------------------


def test_boundary_indexes_equal_fresh_comprehensions(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens._capture.projections import RecordingState
    from torchlens.backends.torch import _predicate_forward

    # Unselected op contexts leave ``all_contexts`` once committed, so the fresh
    # comprehension is taken over every context (``keep_all_contexts``, as a dry
    # run keeps them) at each module exit, where the retention step reads the index.
    real_post_init = RecordingState.__post_init__

    def _keep_every_context(self: Any) -> None:
        real_post_init(self)
        self.keep_all_contexts = True

    real_boundary = _predicate_forward._record_predicate_module_boundary_outputs
    exits: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def _spy(trace: Any, state: Any, out: Any, **kwargs: Any) -> None:
        fresh = {
            ctx.raw_label: ctx
            for ctx in state.all_contexts
            if ctx.kind == "op" and ctx.raw_label is not None
        }
        exits.append((dict(state.op_context_by_raw_label), fresh))
        real_boundary(trace, state, out, **kwargs)

    monkeypatch.setattr(RecordingState, "__post_init__", _keep_every_context)
    monkeypatch.setattr(_predicate_forward, "_record_predicate_module_boundary_outputs", _spy)
    recording = _record_one_module(_model())
    state = object.__getattribute__(recording, "_recording_state")
    assert exits and any(index for index, _ in exits), "module exits read a live index"
    for index, fresh in exits:
        assert index == fresh
    assert not state.op_context_by_raw_label, "a finished pass keeps no op context index"
    expected = {
        (record.ctx.raw_label, record.ctx.pass_number)
        for record in recording.records
        if record.ctx.raw_label is not None
    }
    assert expected, "the selected module retained its output"
    assert state.retained_keys.sync(recording.records) == expected


# ---------------------------------------------------------------------------
# Call counters over one capture (gates; wall clock is never asserted)
# ---------------------------------------------------------------------------


def _count_capture(model: nn.Module) -> dict[str, int]:
    counts = {"walker": 0, "replace": 0, "wrapped_untyped_storage": 0, "wrapped_total": 0}
    walker = _clear_session_tensor_metadata.__code__
    replace_code = dataclasses.replace.__code__
    # Every installed torch wrapper is one closure of ``wrapped_func``: one code object.
    wrapped_code = torch.Tensor.untyped_storage.__code__

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event != "call":
            return
        code = frame.f_code
        if code is walker:
            counts["walker"] += 1
        elif code is replace_code:
            counts["replace"] += 1
        elif code is wrapped_code:
            counts["wrapped_total"] += 1
            if frame.f_locals.get("func_name") == "untyped_storage":
                counts["wrapped_untyped_storage"] += 1

    sys.setprofile(profile)
    try:
        _record_one_module(model)
    finally:
        sys.setprofile(None)
    return counts


def test_one_module_record_bookkeeping_counters() -> None:
    model = _model()
    _record_one_module(model)  # warm: wrappers installed, caches primed
    n_modules = sum(1 for _ in model.modules())
    counts = _count_capture(model)
    # Walker calls: per module, the non-empty ``_parameters``/``_modules`` slots
    # plus the forward-callable sweep (defaults, kwdefaults, closure, globals),
    # for the pre-forward snapshot and the session-end clear (170 measured for
    # these 9 modules; 2,559 before, when every ``forward.__annotations__``
    # descended into the ``torch.Tensor`` class namespace).
    assert counts["walker"] <= 24 * n_modules, counts
    # No journal record is cloned through ``dataclasses.replace`` any more; what
    # remains is the per-module-exit boundary context and predicate aliasing
    # (54 measured for these 9 modules; 279 before).
    assert counts["replace"] <= 8 * n_modules, counts
    # Label stamps outside a live forward read storage through the original
    # accessor; only the stamps minted while the run collector is active still
    # go through the wrapper (one per saved module output here; 46 before).
    assert counts["wrapped_untyped_storage"] <= 4, counts


def test_shadow_escape_detector_relu_trace_has_no_unattributed_args() -> None:
    """The original-accessor storage read stays invisible to the shadow detector.

    The docs/performance.md shadow example: a plain ``nn.ReLU`` trace must attribute
    its input. Reading storage through the raw accessor while logging was live made the
    detector report it and the capture then flagged ``relu`` arg0 as source-less.
    """

    saved_escape_detector = _state._escape_detector_mode
    saved_completeness_witness = _state._completeness_witness_mode
    unwrap_torch()
    try:
        wrap_torch(escape_detector="shadow")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trace = tl.trace(nn.ReLU(), torch.randn(4))
    finally:
        unwrap_torch()
        wrap_torch(
            escape_detector=saved_escape_detector,
            completeness_witness=saved_completeness_witness,
        )
    messages = [str(item.message) for item in caught]
    assert not [m for m in messages if "no graph/source provenance" in m], messages
    assert not [m for m in messages if "untyped_storage" in m], messages
    assert not trace.__dict__.get("escape_diagnostics"), trace.__dict__.get("escape_diagnostics")
