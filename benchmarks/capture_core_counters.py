"""Deterministic instruments behind ``capture_core_cost.py`` (counters C1 to C9).

Everything here counts; nothing times. The instruments are:

* :func:`count_phases`: a ``sys.setprofile`` counter that attributes every
  Python and C call made on the current thread to the innermost marked phase of
  ``tl.trace`` (C1), tallies named call sites (C8), and runs observers at a few
  named calls and returns;
* :func:`count_calls`: the same counter without phases (C7);
* :func:`retained_after`: allocation blocks, collector-tracked objects and
  live record objects per kind while only the trace is held (C2, C4);
* :class:`StoreWriteObserver`: op-store cells written by capture, by ingest
  and by the passes (C3);
* :class:`TensorBytesObserver`: tensor bytes strongly reachable from the trace
  (its capture journal and store included) at op boundaries of the forward,
  split into saved outputs, argument tensors and the rest (C9).

The observers read frame arguments of named functions; they never patch
product code. An observer runs inside the profile callback, where Python
suspends profiling, so its own calls are not counted.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref
from collections import Counter
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

#: ``(file suffix, function name, phase)``: a call of the function opens the phase.
PHASE_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("backends/torch/wrappers.py", "wrapped_func", "wrapper"),
    ("backends/torch/_ops_emission.py", "log_function_output_tensors", "wrapper.emit"),
    ("backends/torch/_thin_emit.py", "thin_emit", "wrapper.emit"),
    ("backends/torch/_ops_finalize.py", "_make_layer_log_entry", "wrapper.emit.make_entry"),
    (
        "backends/torch/_ops_shared_fields.py",
        "_build_shared_fields_dict",
        "wrapper.emit.shared_fields",
    ),
    ("_capture/projections.py", "commit_op", "wrapper.emit.commit_op"),
    ("_postprocess/__init__.py", "_postprocess_body", "post"),
    ("_data_classes/_compaction.py", "compact_op_metadata", "post.compaction"),
    ("_capture/finalize/arms.py", "_complete_trace_path", "finalize"),
    ("_capture/run.py", "_run_forward", "forward.other"),
    ("_user_funcs.py", "trace", "entry"),
)

#: ``(file suffix, function name, tag)``: a call of the function increments the tag.
TAG_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("backends/torch/wrappers.py", "wrapped_func", "wrapper_calls"),
    ("backends/torch/_ops_emission.py", "log_function_output_tensors", "emit_calls"),
    ("_capture/projections.py", "commit_op", "commit_ops"),
    ("_capture/run.py", "_run_forward", "run_forward"),
    ("_postprocess/__init__.py", "_postprocess_body", "postprocess_body"),
    ("_postprocess/_materialize.py", "materialize_from_events", "materialize"),
    ("_trace_core/op_store.py", "freeze", "store_freeze"),
    ("_trace_core/op_store.py", "cell_set", "cell_set"),
    ("_trace_core/op_store.py", "cell_del", "cell_del"),
)

#: The wrapper fast lane does not exist at the baseline. The lane that adds it
#: names its per-call entry function(s) in ``FAST_LANE_COUNTER_NAMES`` of this
#: module; until then these defaults match nothing and C8 reads 0.
FAST_LANE_FILE = "backends/torch/_call_plan.py"
FAST_LANE_DEFAULT_NAMES: tuple[str, ...] = ("fast_lane_call",)

_STEP_FILE = "_postprocess/_executor.py"

Hook = Callable[[Any], None]


def fast_lane_names() -> tuple[str, ...]:
    """Return the fast lane's entry function names, as the product declares them."""

    try:
        from torchlens.backends.torch import _call_plan  # type: ignore[attr-defined]
    except ImportError:
        return FAST_LANE_DEFAULT_NAMES
    return tuple(getattr(_call_plan, "FAST_LANE_COUNTER_NAMES", FAST_LANE_DEFAULT_NAMES))


def _classifier() -> Callable[[types.CodeType], tuple[str | None, str | None] | None]:
    """Build the code-object classifier: ``code -> (phase, tag)`` or ``None``."""

    fast_names = fast_lane_names()

    def classify(code: types.CodeType) -> tuple[str | None, str | None] | None:
        filename = code.co_filename.replace("\\", "/")
        if "torchlens" not in filename:
            return None
        name = code.co_name
        phase = None
        tag = None
        for suffix, func_name, marker_phase in PHASE_MARKERS:
            if name == func_name and filename.endswith(suffix):
                phase = marker_phase
                break
        if phase is None and name.startswith("_run_") and filename.endswith(_STEP_FILE):
            phase = "post." + name[5:]
        for suffix, func_name, marker_tag in TAG_MARKERS:
            if name == func_name and filename.endswith(suffix):
                tag = marker_tag
                break
        if tag is None and name in fast_names and filename.endswith(FAST_LANE_FILE):
            tag = "fast_lane_calls"
        if phase is None and tag is None:
            return None
        return phase, tag

    return classify


def count_phases(
    fn: Callable[[], Any],
    call_hooks: dict[str, Hook] | None = None,
    return_hooks: dict[str, Hook] | None = None,
) -> tuple[Any, dict[str, int], dict[str, int]]:
    """Run ``fn`` and attribute every Python and C call to its innermost phase.

    Parameters
    ----------
    fn
        The code to measure.
    call_hooks, return_hooks
        ``tag -> hook(frame)`` run when a tagged function is called or returns.

    Returns
    -------
    tuple
        ``fn``'s result, calls per phase (``"outside"`` is everything before
        the first marker) and calls per tag.
    """

    classify = _classifier()
    call_hooks = call_hooks or {}
    return_hooks = return_hooks or {}
    counts: dict[str, int] = {}
    tags: dict[str, int] = {}
    stack = ["outside"]
    marks: list[tuple[Any, bool, Hook | None]] = []
    cache: dict[int, tuple[types.CodeType, Any]] = {}

    def profile(frame: Any, event: str, _arg: Any) -> None:
        if event == "call":
            code = frame.f_code
            entry = cache.get(id(code))
            if entry is None or entry[0] is not code:
                entry = (code, classify(code))
                cache[id(code)] = entry
            info = entry[1]
            if info is not None:
                phase, tag = info
                on_return = None
                if tag is not None:
                    tags[tag] = tags.get(tag, 0) + 1
                    hook = call_hooks.get(tag)
                    if hook is not None:
                        hook(frame)
                    on_return = return_hooks.get(tag)
                if phase is not None:
                    stack.append(phase)
                if phase is not None or on_return is not None:
                    marks.append((frame, phase is not None, on_return))
            current = stack[-1]
            counts[current] = counts.get(current, 0) + 1
        elif event == "c_call":
            current = stack[-1]
            counts[current] = counts.get(current, 0) + 1
        elif event == "return" and marks and marks[-1][0] is frame:
            _frame, pops_phase, on_return = marks.pop()
            if pops_phase:
                stack.pop()
            if on_return is not None:
                on_return(frame)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = fn()
    finally:
        sys.setprofile(previous)
    return result, counts, tags


def count_calls(fn: Callable[[], Any]) -> tuple[Any, int]:
    """Run ``fn`` and return its result and the Python plus C calls it made."""

    total = [0]

    def profile(_frame: Any, event: str, _arg: Any) -> None:
        if event == "call" or event == "c_call":
            total[0] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = fn()
    finally:
        sys.setprofile(previous)
    return result, total[0]


def code_context_lookups(fn: Callable[[], Any]) -> tuple[Any, dict[str, int]]:
    """Run ``fn`` and count the emit's code-context lookups (the full frame walk).

    ``lookups`` counts calls of ``_get_code_context`` made by the call-site step
    of the emit (``backends.torch._ops_call_site_fields``); ``repeat_lookups``
    counts those whose result (the same ``FuncCallLocation`` records and the
    same call-site chain) an earlier lookup of the same trace already returned,
    that is lookups for a site the capture had already seen.
    """

    from torchlens.backends.torch import _ops_call_site_fields as site_module

    original = site_module._get_code_context
    seen: set[tuple[Any, ...]] = set()
    counts = {"lookups": 0, "repeat_lookups": 0}

    def counted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        out = kwargs.get("call_site_out")
        chain = out[-1] if out else None
        key = (tuple(id(item) for item in result), id(chain))
        counts["lookups"] += 1
        if key in seen:
            counts["repeat_lookups"] += 1
        seen.add(key)
        return result

    site_module._get_code_context = counted
    try:
        result = fn()
    finally:
        site_module._get_code_context = original
    return result, counts


def flatten_code_context(document: dict[str, Any]) -> dict[str, float]:
    """``code_context.<model>.<count>_per_op`` for each model record that has the block."""

    flat: dict[str, float] = {}
    for model, record in document.get("models", {}).items():
        block = record.get("code_context") if isinstance(record, dict) else None
        if block and record.get("rows"):
            for name, count in block.items():
                flat[f"code_context.{model}.{name}_per_op"] = count / record["rows"]
    return flat


def record_kinds() -> tuple[type, ...]:
    """Return the public record classes whose live instances C4 counts."""

    from torchlens._data_classes.buffer import Buffer
    from torchlens._data_classes.grad_fn import GradFn
    from torchlens._data_classes.layer import Layer
    from torchlens._data_classes.module import Module, ModuleCall
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.param import Param

    return (Op, Layer, Module, ModuleCall, Param, Buffer, GradFn)


def _kind_census(objects: list[Any], kinds: tuple[type, ...]) -> Counter[str]:
    return Counter(type(obj).__name__ for obj in objects if isinstance(obj, kinds))


_PROXY_TYPES = (weakref.ProxyType, weakref.CallableProxyType)


def _is_dead_weakref(obj: Any) -> bool:
    """Return whether ``obj`` is a weak reference or proxy whose referent is gone."""

    if isinstance(obj, weakref.ReferenceType):
        return obj() is None
    if isinstance(obj, _PROXY_TYPES):
        try:
            obj.__class__  # noqa: B018 - a dead proxy raises on any attribute
        except ReferenceError:
            return True
    return False


def _settled_count(objects: list[Any]) -> tuple[int, int]:
    """Return (objects counted, dead weak references left out) for a snapshot."""

    dead = sum(1 for obj in objects if _is_dead_weakref(obj))
    return len(objects) - dead, dead


def _settle_label_session() -> None:
    """Release the retired label session, as the next trace's start would.

    Between traces TorchLens keeps the last capture's label session (a weak
    inventory of stamped objects and a storage alias index) so the next
    capture can sweep stamps that outlived it; the next capture's start
    sweeps and drops it, whether or not the earlier trace is still held. The
    alias index is keyed by storage address and keeps one bucket per address
    it saw, so its size follows the allocator, not the model: whole
    ``WeakIdKeyDictionary`` buckets came and went between runs of one commit.
    Doing that sweep before each snapshot keeps the session out of both, so
    ``objects`` counts what the result keeps, not what the last trace left.
    """

    from torchlens._utils._tensor_label import sweep_retired_label_stamps

    sweep_retired_label_stamps()


#: Full collections after which a still-changing tracked count is an instrument error.
_MAX_SETTLE_PASSES = 16


def _collect_to_fixed_point() -> None:
    """Collect until a full collection leaves the tracked-object count unchanged.

    A pass untracks every tuple and dict whose items are all untracked when it
    reaches them, so a pass that changes nothing leaves nothing for the next
    one: the count is then a fixed point, the same whatever order allocation
    left the objects in.

    Raises
    ------
    RuntimeError
        If the count still changes after ``_MAX_SETTLE_PASSES`` collections
        (something allocates while the snapshot settles).
    """

    gc.collect()
    tracked = len(gc.get_objects())
    for _ in range(_MAX_SETTLE_PASSES):
        gc.collect()
        now = len(gc.get_objects())
        if now == tracked:
            return
        tracked = now
    raise RuntimeError(f"tracked objects still changing after {_MAX_SETTLE_PASSES} collections")


def retained_after(fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Run ``fn`` and report what its result keeps alive.

    Returns
    -------
    tuple
        ``fn``'s result (hold it; dropping it ends the measurement's meaning)
        and a dict with ``blocks`` (``sys.getallocatedblocks`` delta),
        ``objects`` (collector-tracked object delta, dead weak references left
        out), ``dead_weakrefs`` (the delta of those left out, for diagnosis)
        and ``kinds`` (live record objects per class name, delta), all after a
        full collection.

    Each side collects until a pass changes nothing (see
    :func:`_collect_to_fixed_point`). The collector stops tracking a tuple that
    holds only untracked items, but in one pass it can visit an outer tuple
    before its inner ones, so each pass settles one more level of nesting, in
    an order set by allocation history. A fixed count of passes leaves some
    nested key tuples tracked depending on that history (the call-site cache's
    keys are three levels deep); collecting to the fixed point settles all of
    them, so ``objects`` does not move with allocation timing.

    A weak reference whose referent has died is not counted on either side.
    The collector clears such a reference but the reference object lives on
    until whatever holds it lets go (a weak container's pending removals, an
    attribute nobody resets), and how many are waiting at each snapshot varies
    between runs of one commit. They keep nothing alive. A weak reference with a
    live referent is counted like any other object, so one the trace keeps
    still raises ``objects``.

    Each side first releases the retired label session (see
    :func:`_settle_label_session`), which TorchLens itself releases at the
    next trace's start; an object the result holds is still counted.
    """

    kinds = record_kinds()
    _settle_label_session()
    _collect_to_fixed_point()
    before = gc.get_objects()
    objects_before, dead_before = _settled_count(before)
    kinds_before = _kind_census(before, kinds)
    del before
    blocks_before = sys.getallocatedblocks()
    result = fn()
    _settle_label_session()
    _collect_to_fixed_point()
    blocks_after = sys.getallocatedblocks()
    after = gc.get_objects()
    objects_after, dead_after = _settled_count(after)
    kinds_after = _kind_census(after, kinds)
    del after
    census = {
        kind.__name__: kinds_after.get(kind.__name__, 0) - kinds_before.get(kind.__name__, 0)
        for kind in kinds
    }
    for name in sorted(set(kinds_after) - set(census)):
        census[name] = kinds_after[name] - kinds_before.get(name, 0)
    return result, {
        "blocks": blocks_after - blocks_before,
        "objects": objects_after - objects_before,
        "dead_weakrefs": dead_after - dead_before,
        "kinds": census,
    }


def _frame_args(frame: Any) -> list[Any]:
    code = frame.f_code
    local_values = frame.f_locals
    return [local_values.get(name) for name in code.co_varnames[: code.co_argcount]]


class StoreWriteObserver:
    """Count op-store cells written by capture, by ingest and by the passes (C3).

    The row writers assign into plain row lists, which no call counter sees,
    so each stage is measured as a difference of row snapshots, by identity:

    * capture: cells set in the capture journal's rows when postprocessing
      takes them over (a default-omitted cell is not set);
    * ingest: cells of the trace's store that are set or replaced between that
      moment and the end of ``materialize_from_events``;
    * passes: cells set, replaced or deleted from there to the store's freeze
      (the compaction sweep's replacements included).

    ``cell_set`` and ``cell_del`` calls are also counted per stage; a cell
    written twice in one stage counts twice there and once in the snapshot.
    """

    def __init__(self) -> None:
        from torchlens._trace_core.cell_states import STATE_DEFAULT, LazyCell
        from torchlens._trace_core.op_store import _MISSING

        self._missing = _MISSING
        self._lazy = LazyCell
        self._default = STATE_DEFAULT
        self.stage = "capture"
        self.cells = {"capture": 0, "ingest": 0, "passes": 0}
        self.calls: dict[str, int] = {}
        self.capture_rows = 0
        self.store_rows = 0
        self._trace: Any = None
        self._store: Any = None
        self._snapshot: dict[int, tuple[list[Any], list[Any]]] = {}
        self._closed = False

    def hooks(self) -> tuple[dict[str, Hook], dict[str, Hook]]:
        """Return the ``(call_hooks, return_hooks)`` for :func:`count_phases`."""

        call_hooks = {
            "materialize": self._on_materialize_call,
            "store_freeze": self._on_freeze_call,
            "cell_set": lambda _frame: self._count_call("cell_set"),
            "cell_del": lambda _frame: self._count_call("cell_del"),
        }
        return_hooks = {
            "materialize": self._on_materialize_return,
            "postprocess_body": self._on_body_return,
        }
        return call_hooks, return_hooks

    def _count_call(self, name: str) -> None:
        key = f"{name}.{self.stage}"
        self.calls[key] = self.calls.get(key, 0) + 1

    def _set_cells(self, row: list[Any]) -> int:
        # A default-omitted cell holds the field's shared marker, not a value:
        # nothing was written for it, so it counts as unset. Its later write
        # (a read's write-back, the freeze-seam settle) is an identity change
        # and counts in the stage where it happens.
        missing = self._missing
        lazy = self._lazy
        default = self._default
        return sum(
            1
            for cell in row
            if cell is not missing and not (cell.__class__ is lazy and cell.state == default)
        )

    def _take_snapshot(self, rows: list[list[Any]]) -> None:
        self._snapshot = {id(row): (row, list(row)) for row in rows}

    def _changed_cells(self, rows: list[list[Any]]) -> int:
        snapshot = self._snapshot
        changed = 0
        for row in rows:
            entry = snapshot.get(id(row))
            if entry is None or entry[0] is not row:
                changed += self._set_cells(row)
            else:
                changed += sum(1 for new, old in zip(row, entry[1]) if new is not old)
        return changed

    def _on_materialize_call(self, frame: Any) -> None:
        args = _frame_args(frame)
        self._trace = args[0]
        capture_rows = args[2] if len(args) > 2 and args[2] else {}
        rows = list(capture_rows.values())
        self.capture_rows = len(rows)
        self.cells["capture"] = sum(self._set_cells(row) for row in rows)
        self._take_snapshot(rows)
        self.stage = "ingest"

    def _on_materialize_return(self, _frame: Any) -> None:
        core = self._trace.__dict__.get("_trace_core")
        store = getattr(core, "ops", None)
        rows = store.rows_building() if store is not None else None
        if rows is not None:
            self._store = store
            self.cells["ingest"] = self._changed_cells(rows)
            self._take_snapshot(rows)
        self.stage = "passes"

    def _close_passes(self) -> None:
        if self._closed or self._store is None:
            return
        self._closed = True
        rows = self._store.rows_building()
        if rows is not None:
            self.store_rows = len(rows)
            self.cells["passes"] = self._changed_cells(rows)
        self._snapshot = {}
        self.stage = "after_freeze"

    def _on_freeze_call(self, frame: Any) -> None:
        if frame.f_locals.get("self") is self._store:
            self._close_passes()

    def _on_body_return(self, _frame: Any) -> None:
        self._close_passes()
        self._trace = None
        self._store = None

    def report(self) -> dict[str, Any]:
        """Return the stage totals (not yet divided by the op count)."""

        return {
            "cells": dict(self.cells),
            "calls": dict(sorted(self.calls.items())),
            "capture_rows": self.capture_rows,
            "store_rows_at_freeze": self.store_rows,
        }


SAVED_OUTPUT, ARGUMENT, FORWARD_INPUT, OTHER, MODEL_OWNED = 0, 1, 2, 3, 4
CATEGORY_NAMES = ("saved_output", "argument", "forward_input", "other", "caller_owned")
#: Names that hold the forward's own input (the copy the forward runs on), not an op's arguments.
_FORWARD_INPUT_NAMES = frozenset({"root_forward_args"})
_ARGUMENT_TOKENS = frozenset({"arg", "args", "kwarg", "kwargs"})
_OUTPUT_TOKENS = frozenset(
    {"out", "outs", "output", "outputs", "activation", "activations", "payload", "payloads"}
)
_INPUT_TOKENS = frozenset({"input", "inputs"})
_ATOMIC = frozenset({int, float, str, bool, type(None), bytes, complex})
_STOP_TYPES = (
    type,
    nn.Module,
    types.ModuleType,
    types.CodeType,
    types.FrameType,
    types.BuiltinFunctionType,
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    weakref.ReferenceType,
    weakref.ProxyType,
    weakref.CallableProxyType,
    torch.dtype,
    torch.device,
)
_WALK_LIMIT = 5_000_000
_NAME_RANKS: dict[str, int] = {}


def _name_rank(name: str) -> int:
    """Classify a field, attribute or key name as argument, saved output or other."""

    rank = _NAME_RANKS.get(name)
    if rank is None:
        tokens = set(name.strip("_").split("_"))
        if name in _FORWARD_INPUT_NAMES:
            rank = FORWARD_INPUT
        elif name == "input_activations" or tokens & _ARGUMENT_TOKENS:
            rank = ARGUMENT
        elif tokens & _INPUT_TOKENS:
            rank = OTHER
        elif tokens & _OUTPUT_TOKENS:
            rank = SAVED_OUTPUT
        else:
            rank = OTHER
        _NAME_RANKS[name] = rank
    return rank


def _storage_key(tensor: torch.Tensor) -> tuple[Any, int]:
    try:
        storage = tensor.untyped_storage()
        return ("storage", storage._cdata), int(storage.nbytes())
    except Exception:  # noqa: BLE001 - sparse and other storage-less layouts
        try:
            return ("tensor", id(tensor)), int(tensor.numel() * tensor.element_size())
        except Exception:  # noqa: BLE001
            return ("tensor", id(tensor)), 0


def _input_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _input_tensors(item)]
    return []


def caller_owned_keys(model: nn.Module, inputs: Any = ()) -> tuple[set[int], set[Any]]:
    """Return the ids and storage keys of the tensors the caller owns.

    They are the model's parameters and buffers and the tensors of ``inputs``
    (the forward's own input, nested in lists, tuples or dicts): the caller
    holds them whether or not a trace does, so they are not capture cost.
    """

    ids: set[int] = set()
    storages: set[Any] = set()
    for tensor in [*model.parameters(), *model.buffers(), *_input_tensors(inputs)]:
        ids.add(id(tensor))
        storages.add(_storage_key(tensor)[0])
    return ids, storages


def held_tensor_bytes(
    root: Any, model_ids: set[int], model_storages: set[Any]
) -> tuple[dict[str, int], dict[str, dict[str, int]], bool]:
    """Sum the tensor bytes strongly reachable from ``root``, by category.

    The walk follows containers, instance dicts, slots and collector referents,
    and stops at modules, classes, functions' globals, ``nn.Module`` instances
    and weak references. A tensor's category is that of the outermost named
    hop on the path to it (an op-store field, an attribute, a string key):
    ``root_forward_args`` holds the forward's own input ("forward_input");
    a name with an ``arg`` or ``kwarg`` token (or ``input_activations``) is an
    argument, any other name with an ``input`` token "other", one with an
    ``out``, ``output``, ``activation`` or ``payload`` token a saved output,
    anything else "other". Bytes are counted once per storage, under the
    strongest claim (saved output, argument, forward input, other); the tensors the
    caller owns (see :func:`caller_owned_keys`) are reported apart.

    Returns
    -------
    tuple
        Bytes per category, bytes per classifying name within each category,
        and whether the walk hit its object limit.
    """

    from torchlens._trace_core.op_store import _MISSING, OpRowStore

    best: dict[int, int] = {}
    storages: dict[Any, tuple[int, int, str]] = {}
    stack: list[tuple[Any, int, str]] = [(root, OTHER, "")]
    push = stack.append
    visited = 0
    truncated = False
    while stack:
        obj, rank, holder = stack.pop()
        kind = type(obj)
        if kind in _ATOMIC or obj is _MISSING:
            continue
        oid = id(obj)
        seen = best.get(oid)
        if seen is not None and seen <= rank:
            continue
        best[oid] = rank
        visited += 1
        if visited > _WALK_LIMIT:
            truncated = True
            break
        if isinstance(obj, torch.Tensor):
            key, nbytes = _storage_key(obj)
            claim = MODEL_OWNED if oid in model_ids or key in model_storages else rank
            known = storages.get(key)
            if known is None or claim < known[0]:
                storages[key] = (claim, nbytes, holder)
            continue
        if isinstance(obj, _STOP_TYPES):
            continue
        if kind is list or kind is tuple or kind is set or kind is frozenset:
            for item in obj:
                if type(item) not in _ATOMIC:
                    push((item, rank, holder))
            continue
        if kind is dict:
            for key, value in obj.items():
                if type(value) in _ATOMIC:
                    continue
                if rank == OTHER and type(key) is str:
                    named = _name_rank(key)
                    push((value, named, key if named != OTHER else holder))
                else:
                    push((value, rank, holder))
                    if type(key) not in _ATOMIC:
                        push((key, rank, holder))
            continue
        if kind is types.FunctionType:
            for cell in obj.__closure__ or ():
                push((cell, rank, holder))
            for default in obj.__defaults__ or ():
                push((default, rank, holder))
            continue
        if isinstance(obj, OpRowStore):
            rows = obj._rows
            if rows is not None:
                names = obj.layout.names
                ranks = [_name_rank(name) for name in names]
                for row in rows:
                    for fid, cell in enumerate(row):
                        if type(cell) in _ATOMIC or cell is _MISSING:
                            continue
                        if rank == OTHER and ranks[fid] != OTHER:
                            push((cell, ranks[fid], names[fid]))
                        else:
                            push((cell, rank, holder))
            _push_slots(obj, rank, holder, push, skip="_rows")
            continue
        try:
            instance_dict = object.__getattribute__(obj, "__dict__")
        except AttributeError:
            instance_dict = None
        if type(instance_dict) is dict:
            push((instance_dict, rank, holder))
        _push_slots(obj, rank, holder, push)
        for referent in gc.get_referents(obj):
            if type(referent) not in _ATOMIC:
                push((referent, rank, holder))
    totals = dict.fromkeys(CATEGORY_NAMES, 0)
    holders: dict[str, dict[str, int]] = {name: {} for name in CATEGORY_NAMES}
    for claim, nbytes, holder in storages.values():
        category = CATEGORY_NAMES[claim]
        totals[category] += nbytes
        by_holder = holders[category]
        by_holder[holder or "(unnamed)"] = by_holder.get(holder or "(unnamed)", 0) + nbytes
    return totals, holders, truncated


def _push_slots(
    obj: Any, rank: int, holder: str, push: Callable[[Any], None], skip: str = ""
) -> None:
    """Push the values of ``obj``'s slots, classified by slot name."""

    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__")
        if not slots:
            continue
        for name in (slots,) if isinstance(slots, str) else slots:
            if name == skip or name in ("__dict__", "__weakref__"):
                continue
            descriptor = cls.__dict__.get(name)
            if type(descriptor) is not types.MemberDescriptorType:
                continue
            try:
                value = descriptor.__get__(obj, cls)
            except AttributeError:
                continue
            if type(value) in _ATOMIC:
                continue
            if rank == OTHER:
                named = _name_rank(name)
                push((value, named, name if named != OTHER else holder))
            else:
                push((value, rank, holder))


class TensorBytesObserver:
    """Sample the tensor bytes the trace holds at op boundaries of the forward (C9).

    A boundary is the return of one op commit. Every boundary is sampled up to
    ``dense``; after that every ``stride``-th; the end of the forward always.
    """

    def __init__(
        self, model: nn.Module, inputs: Any = (), *, dense: int = 64, stride: int = 16
    ) -> None:
        self._model_ids, self._model_storages = caller_owned_keys(model, inputs)
        self._dense = dense
        self._stride = stride
        self.boundaries = 0
        self.samples: list[tuple[int, dict[str, int]]] = []
        self.end_holders: dict[str, dict[str, int]] = {}
        self.truncated = False

    def hooks(self) -> tuple[dict[str, Hook], dict[str, Hook]]:
        """Return the ``(call_hooks, return_hooks)`` for :func:`count_phases`."""

        return {}, {"commit_ops": self._on_commit_return, "run_forward": self._on_forward_return}

    def _sample(self, trace: Any) -> dict[str, dict[str, int]]:
        totals, holders, truncated = held_tensor_bytes(trace, self._model_ids, self._model_storages)
        self.truncated = self.truncated or truncated
        self.samples.append((self.boundaries, totals))
        return holders

    def _on_commit_return(self, frame: Any) -> None:
        self.boundaries += 1
        if self.boundaries <= self._dense or self.boundaries % self._stride == 0:
            trace = frame.f_locals.get("trace")
            if trace is not None:
                self._sample(trace)

    def _on_forward_return(self, frame: Any) -> None:
        trace = frame.f_locals.get("self")
        if trace is not None:
            self.end_holders = self._sample(trace)

    def report(self) -> dict[str, Any]:
        """Return the peak, the per-category maxima and the end-of-forward holders."""

        def held(totals: dict[str, int]) -> int:
            return sum(totals[name] for name in CATEGORY_NAMES[:-1])

        if not self.samples:
            return {"boundaries": self.boundaries, "sampled": 0}
        peak_boundary, peak = max(self.samples, key=lambda sample: (held(sample[1]), -sample[0]))
        end = self.samples[-1][1]
        top = {
            category: dict(sorted(by_holder.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
            for category, by_holder in self.end_holders.items()
            if by_holder
        }
        return {
            "boundaries": self.boundaries,
            "sampled": len(self.samples),
            "walk_truncated": self.truncated,
            "peak_boundary": peak_boundary,
            "peak_bytes": held(peak),
            "peak": dict(peak),
            "argument_bytes_max": max(totals["argument"] for _b, totals in self.samples),
            "saved_output_bytes_max": max(totals["saved_output"] for _b, totals in self.samples),
            "forward_input_bytes_max": max(totals["forward_input"] for _b, totals in self.samples),
            "other_bytes_max": max(totals["other"] for _b, totals in self.samples),
            "end_of_forward": dict(end),
            "end_of_forward_holders": top,
        }
