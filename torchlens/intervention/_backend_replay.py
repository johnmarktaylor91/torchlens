"""Finished-trace replay for non-PyTorch backends, driven through a replay adapter.

The PyTorch replay engine (:mod:`torchlens.intervention._replay`) re-executes
each op from its captured argument template. Every other backend already
replays each recorded op from its parents for validation, and exposes that
per-op replay through a private adapter module
(``torchlens.backends._replay_adapters.<name>``: ``can_replay``, ``forward``,
``recipe``, ``to_host``, ``close``, ...). This module is the backend-neutral
driver over that adapter:

1. The edit lands at its site (a staged value, a callable written against the
   backend's arrays, or a built-in helper the adapter applies).
2. The downstream cone is walked in graph order. Each op whose parents changed
   is re-run once with ``adapter.forward`` on its parent occurrences (a parent
   used twice is passed twice), and every output slot of a multi-output op is
   rebuilt from that one call.
3. A recomputed branch or effect guard whose value changed refuses: replaying
   down the recorded branch would describe a run that never happens. Rerun the
   model with the edit staged instead.
4. Nothing is written until the whole cone succeeded; the commit then rewrites
   payloads and the replay evidence (fire records, the ``replay`` operation
   row, the replay-propagated state) the PyTorch engine writes.

:func:`verify_backend_replay` is the check of an edited trace: every recomputed
op must equal ``adapter.forward`` of its committed parents under the backend
validator's own comparison, except the sites that carry an edit record (the
user intervention).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._utils._record_group_table import record_member
from ._backend_adapter_lookup import backend_adapter, trace_backend
from ._replay_context import _replay_site_key
from ._replay_sites import _is_passthrough_replacement, _is_source_record
from .errors import ReplayPreconditionError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

#: Remedy shared by every refusal that a fresh capture with the edit staged avoids.
RERUN_REMEDY = (
    "stage the edit in a new run instead: tl.trace(model, inputs, backend=<backend>, intervene=...)"
)

#: Built-in helpers the driver applies with contract functions alone.
_NEUTRAL_HELPERS = frozenset({"zero_ablate"})


def refuse(message: str, code: str, *, remedy: str = RERUN_REMEDY) -> ReplayPreconditionError:
    """Build the coded refusal every check in this module raises."""

    return ReplayPreconditionError(message, code=code, remedy=remedy)


@dataclass(frozen=True)
class ReplayStep:
    """One op of the cone: its output slots and parent occurrences, as replay keys.

    ``kind`` is ``"op"`` (re-run through the adapter), ``"source"`` (an input
    or buffer record: its value is captured, never recomputed) or
    ``"passthrough"`` (a module-edit replacement op: its value is its parent's).
    """

    op: Any
    slots: tuple[str, ...]
    occurrences: tuple[str, ...]
    kind: str
    guard: bool


def _resolve_key(trace: Trace, label: str, *, owner: str) -> str:
    """Return the replay key of a label an adapter recipe names, refusing an unknown one."""

    site = trace._layer_dict_all_keys.get(label)
    if site is None:
        raise refuse(
            f"the replay recipe of {owner!r} names {label!r}, which is not an op of this trace",
            "backend_replay_recipe_unknown_label",
            remedy="capture the trace again; the recipe does not match its graph",
        )
    return _replay_site_key(site)


def _is_guard(site: Op) -> bool:
    """Whether a site's value steers control flow (a recorded scalar bool)."""

    return bool(getattr(site, "is_scalar_bool", False) or getattr(site, "is_terminal_bool", False))


def _step_for_site(trace: Trace, adapter: Any, site: Op) -> ReplayStep:
    """Describe one cone site as a replay step, refusing an op the adapter cannot replay."""

    key = _replay_site_key(site)
    guard = _is_guard(site)
    if _is_passthrough_replacement(site):
        parent = _resolve_key(trace, tuple(site.parents)[0], owner=key)
        return ReplayStep(site, (key,), (parent,), "passthrough", guard)
    if _is_source_record(site) or not tuple(getattr(site, "parents", ()) or ()):
        return ReplayStep(site, (key,), (), "source", guard)
    reason = adapter.can_replay(trace, site)
    recipe = None if reason is not None else adapter.recipe(trace, site)
    if recipe is None:
        raise refuse(
            f"{key!r} cannot be replayed on the {trace_backend(trace)} backend: "
            f"{reason or 'the adapter has no replay recipe for it'}",
            "backend_replay_op_unreplayable",
        )
    slots = tuple(_resolve_key(trace, slot, owner=key) for slot in recipe.output_slots) or (key,)
    occurrences = tuple(_resolve_key(trace, label, owner=key) for label in recipe.parent_labels)
    return ReplayStep(site, slots, occurrences, "op", guard)


def trace_steps(trace: Trace, adapter: Any, cone: Sequence[Op]) -> list[ReplayStep]:
    """Return the cone as replay steps in graph order, one step per multi-output op."""

    steps: list[ReplayStep] = []
    covered: set[str] = set()
    for site in cone:
        if _replay_site_key(site) in covered:
            continue
        step = _step_for_site(trace, adapter, site)
        covered.update(step.slots)
        steps.append(step)
    return steps


@dataclass
class ReplayWalk:
    """One backend replay: fixed inputs and the values the walk accumulates.

    ``payload`` reads a slot's captured value; ``edit`` applies the edits aimed
    at a slot (returning the new value and its fire records); ``origins`` holds
    values that stand as given (a pre-mutated origin). ``overlay`` collects
    every value the walk changed, and ``records`` the fire records, both
    committed only when the whole cone succeeded.
    """

    trace: Any
    adapter: Any
    payload: Callable[[str], Any]
    edit: Callable[[ReplayStep, str, Any], tuple[Any, list[Any]]]
    edited_slots: frozenset[str]
    origins: Mapping[str, Any] = field(default_factory=dict)
    overlay: dict[str, Any] = field(default_factory=dict)
    records: dict[str, list[Any]] = field(default_factory=dict)

    def value(self, slot: str) -> Any:
        """The slot's current value: changed in this walk, else captured."""

        return self.overlay[slot] if slot in self.overlay else self.payload(slot)


def _needs_replay(walk: ReplayWalk, step: ReplayStep) -> bool:
    """Whether a step's value can differ from its captured value in this walk."""

    if any(slot in walk.edited_slots or slot in walk.origins for slot in step.slots):
        return True
    return any(parent in walk.overlay for parent in step.occurrences)


def _step_outputs(walk: ReplayWalk, step: ReplayStep) -> tuple[Any, ...]:
    """Recompute a step's output values (one per slot) from current parent values."""

    if all(slot in walk.origins for slot in step.slots):
        return tuple(walk.origins[slot] for slot in step.slots)
    if step.kind == "source":
        return tuple(walk.payload(slot) for slot in step.slots)
    if step.kind == "passthrough":
        return (walk.value(step.occurrences[0]),)
    if not any(parent in walk.overlay for parent in step.occurrences):
        # Nothing upstream changed: the captured values are the replay values
        # exactly, so an edit that changes nothing reproduces the trace bit for bit.
        return tuple(walk.payload(slot) for slot in step.slots)
    parents = tuple(walk.value(parent) for parent in step.occurrences)
    outputs = tuple(walk.adapter.forward(walk.trace, step.op, parents))
    if len(outputs) != len(step.slots):
        raise refuse(
            f"replaying {step.slots[0]!r} returned {len(outputs)} values for "
            f"{len(step.slots)} output slots",
            "backend_replay_output_arity",
            remedy="report this: the adapter and the recorded graph disagree",
        )
    return outputs


def host_layout(adapter: Any, value: Any) -> tuple[tuple[int, ...], str]:
    """Return a value's (shape, dtype name) read through the adapter's host copy."""

    host = adapter.to_host(value)
    return tuple(getattr(host, "shape", ())), str(getattr(host, "dtype", type(host).__name__))


def check_layout(walk: ReplayWalk, slot: str, value: Any) -> None:
    """Refuse a replay value whose shape or dtype differs from the captured one.

    Off PyTorch a replay keeps every op's recorded metadata, so a changed
    layout (a shape-changing edit, or shape-dependent code downstream of one)
    goes through a fresh capture instead.
    """

    captured = host_layout(walk.adapter, walk.payload(slot))
    replayed = host_layout(walk.adapter, value)
    if captured != replayed:
        raise refuse(
            f"replaying {slot!r} produced shape and dtype {replayed}, but the trace recorded "
            f"{captured}; a finished-trace edit keeps every recorded layout",
            "backend_replay_layout_changed",
        )


def check_guard(walk: ReplayWalk, step: ReplayStep, slot: str, value: Any) -> None:
    """Refuse when a recomputed branch guard no longer holds its recorded value."""

    if step.guard and not walk.adapter.close(walk.payload(slot), value):
        raise refuse(
            f"the edit changes {slot!r}, a recorded condition that steers control flow; "
            "replaying down the recorded branch would describe a run that does not happen",
            "backend_replay_guard_changed",
        )


def propagate(walk: ReplayWalk, steps: Sequence[ReplayStep]) -> ReplayWalk:
    """Walk the steps in graph order, recomputing each one whose inputs changed.

    Every output slot of a step comes from one adapter call; edits aimed at a
    slot apply after it is recomputed and before its consumers read it.
    """

    for step in steps:
        if not _needs_replay(walk, step):
            continue
        outputs = _step_outputs(walk, step)
        for slot, value in zip(step.slots, outputs):
            if slot in walk.origins:
                value = walk.origins[slot]
            value, records = walk.edit(step, slot, value)
            check_layout(walk, slot, value)
            check_guard(walk, step, slot, value)
            walk.overlay[slot] = value
            if records:
                walk.records.setdefault(slot, []).extend(records)
    return walk


def host_digest(adapter: Any, value: Any) -> str:
    """Return the sha256 of a backend value's host bytes (shape and dtype not included)."""

    host = np.ascontiguousarray(np.asarray(adapter.to_host(value)))
    return hashlib.sha256(host.reshape(-1).view(np.uint8).data).hexdigest()


def verify_backend_replay(trace: Trace, labels: Sequence[str] | None = None) -> list[str]:
    """Return the labels whose committed value is not the replay of their parents.

    Every op among ``labels`` (default: every op of the trace) is re-run with
    ``adapter.forward`` on its committed parent values and compared with the
    backend validator's own comparison (``adapter.close``). A site that carries
    an edit record is the user intervention and is exempt; any other mismatch
    (a corrupted payload, or an edited value whose record was removed) is
    returned.
    """

    adapter = backend_adapter(trace)
    if adapter is None:
        raise refuse(
            f"no replay adapter is built for the {trace_backend(trace)} backend",
            "backend_replay_no_adapter",
            remedy="validate with tl.validate on a fresh capture",
        )
    sites = [trace._layer_dict_all_keys[label] for label in labels or trace.op_labels]
    walk = ReplayWalk(trace, adapter, _payload_reader(trace), _no_edit, frozenset())
    failures: list[str] = []
    for step in trace_steps(trace, adapter, sites):
        if step.kind != "op" or all(_is_edit_marked(trace, slot) for slot in step.slots):
            continue
        outputs = _step_outputs(walk, step)
        for slot, value in zip(step.slots, outputs):
            if not _is_edit_marked(trace, slot) and not adapter.close(walk.payload(slot), value):
                failures.append(slot)
    return failures


def _no_edit(step: ReplayStep, slot: str, value: Any) -> tuple[Any, list[Any]]:
    """Edit function of a walk that applies no edit."""

    del step, slot
    return value, []


def _payload_reader(trace: Trace) -> Callable[[str], Any]:
    """Return the captured-value reader of a trace, keyed by replay key."""

    def payload(slot: str) -> Any:
        return trace._layer_dict_all_keys[slot].out

    return payload


def _is_edit_marked(trace: Trace, slot: str) -> bool:
    """Whether the site carries a record of a user edit (the intervention exemption)."""

    site = trace._layer_dict_all_keys[slot]
    return bool(record_member(site, "is_edited", False) or record_member(site, "edits", None))
