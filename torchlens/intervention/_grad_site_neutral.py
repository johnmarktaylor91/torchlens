"""``tl.grad`` edit sites on any backend with a replay adapter.

PyTorch applies a gradient edit inside autograd: the tensor hook that saves
``Op.grad`` runs the site's edits on the gradient of the op's output, in
declared order, saves the final value and hands it to the op's backward
function, so every gradient upstream of the site sees the edit
(``intervention._runtime.apply_live_gradient_hooks``). This module gives the
same values off torch without a backward tape. It walks the captured graph in
reverse with the framework's own per-op derivative (``adapter.vjp``, through
the reverse engine's node and contribution helpers). When an op is reached,
every consumer has already contributed, so the gradient of each output slot is
complete. The slot's edits run then, and the edited value is both that slot's
gradient and what the op's derivative receives.

The results are gradient values, not backward records. Backward records come
only from an observed backward pass (``backends/_backward_neutral.py``); where
a backend has one, :func:`disagreeing_labels` checks that the edited values
agree with it.

The backend's own doors call this module: they resolve each ``tl.grad`` site
to the graph labels it names (:func:`grad_site_targets` gives the inner
forward selectors and any ``tl.in_backward_pass(n)`` qualifier), wrap each
edit as a backend-native callable (:class:`GradSiteEdit`) and pass the graph
the reverse engine takes, or the trace's ops and saved values
(:func:`edited_gradients_for_backend`).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..backends._reverse_engine import (
    ReplayGraph,
    _accumulate,
    _check_forward_order,
    _node_contributions,
    graph_from_recipes,
)
from ..errors import RunCapabilityUnavailableError

__all__ = (
    "GradSiteCall",
    "GradSiteEdit",
    "GradSiteFire",
    "GradSiteResult",
    "Operands",
    "arm_grad_sites",
    "armed_grad_sites",
    "build_grad_site_edits",
    "clear_grad_sites",
    "disagreeing_labels",
    "do_signal_plan_off_torch",
    "edited_gradients",
    "edited_gradients_for_backend",
    "grad_site_targets",
    "run_armed_grad_sites",
)

_CODE = "run_capability_unavailable"  # RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE
_GRAD_KIND = "grad"
_DOOR_MODULE = "torchlens.backends.{backend}._grad_site_door"
"""Where a backend's ``-int`` lane puts ``do_signal_plan(trace, plan, request)``."""
_DETECTION_STAGE = "grad_site_neutral"


Operands = Callable[[Any], tuple[str | None, Sequence[Any]]]
"""Reader of a composite target: ``(connective, children)``, ``(None, ())`` for a leaf."""


@dataclass(frozen=True)
class GradSiteCall:
    """What one edit sees when it fires (torch's hook context fields).

    Attributes
    ----------
    label:
        Graph label of the output slot whose gradient is edited.
    tuple_index:
        Index of that slot among its op's outputs.
    call_index:
        Zero-based count of earlier fires of the same edit at the same label.
    backward_pass_number:
        One-based number of the pass being computed.
    """

    label: str
    tuple_index: int
    call_index: int
    backward_pass_number: int


@dataclass(frozen=True)
class GradSiteEdit:
    """One edit armed on the gradient of one output slot.

    Attributes
    ----------
    label:
        Graph label of the output slot (an op's output, or a leaf such as an
        input or parameter).
    apply:
        ``apply(grad, call) -> grad`` on the backend's own arrays.
    name:
        Display name for the fire record.
    pass_index:
        Fire only in this backward pass (``tl.in_backward_pass(n)``), or in
        every pass when ``None``.
    """

    label: str
    apply: Callable[[Any, GradSiteCall], Any]
    name: str = "edit"
    pass_index: int | None = None


@dataclass(frozen=True)
class GradSiteFire:
    """One fired edit, in firing order.

    ``replaced`` is ``False`` when the edit returned its input object unchanged.
    """

    label: str
    name: str
    order: int
    replaced: bool


@dataclass(frozen=True)
class GradSiteResult:
    """Gradient per reached label (edited at sites) and the fires that produced it."""

    grads: Mapping[str, Any]
    fires: tuple[GradSiteFire, ...]


def _refuse(message: str) -> RunCapabilityUnavailableError:
    """Build this module's typed refusal."""

    return RunCapabilityUnavailableError(
        f"tl.grad edit sites are unavailable: {message}",
        code=_CODE,
        detection_stage=_DETECTION_STAGE,
    )


def grad_site_targets(site: Any, operands: Operands) -> tuple[tuple[Any, int | None], ...]:
    """Return ``(forward selector, pass qualifier)`` per ``tl.grad`` leaf of ``site``.

    ``tl.grad(a) | tl.grad(b)`` gives two entries; ``tl.grad(a) &
    tl.in_backward_pass(2)`` gives ``(a, 2)``. Autograd-node selectors
    (``tl.grad_fn``, ``tl.grad_input``, ...) address backward records, which
    this module does not build, so they are refused.

    ``operands`` is the caller's reader of a composite target: it returns
    ``("and" | "or" | "not", children)`` for a composite and ``(None, ())``
    for a leaf (the torch side passes ``intervention.hooks._target_operands``).
    The caller supplies it so this module stays out of the selector and hook
    import cycle. A ``tl.grad`` leaf is read by its selector kind ``"grad"``
    (live selector or stored ``TargetSpec`` alike).

    Raises
    ------
    RunCapabilityUnavailableError
        When ``site`` has an operand that is neither a ``tl.grad`` address nor
        a backward-pass qualifier.
    """

    if getattr(site, "selector_kind", None) == _GRAD_KIND:
        return ((getattr(site, "selector_value", None), None),)
    operator, children = operands(site)
    if operator == "or":
        return tuple(found for child in children for found in grad_site_targets(child, operands))
    if operator == "and":
        return _qualified_targets(children, operands)
    raise _refuse(
        f"{site!r} is not a tl.grad(...) address; autograd-node selectors need a "
        "recorded backward pass"
    )


def _qualified_targets(
    children: Sequence[Any], operands: Operands
) -> tuple[tuple[Any, int | None], ...]:
    """Targets of an ``&`` of ``tl.grad`` addresses and one backward-pass qualifier."""

    passes = [c for c in children if getattr(c, "selector_kind", None) == "backward_pass"]
    rest = [c for c in children if getattr(c, "selector_kind", None) != "backward_pass"]
    if len(passes) > 1 or not rest:
        raise _refuse("a gradient site takes at most one tl.in_backward_pass(n) qualifier")
    pass_index = int(passes[0].selector_value) if passes else None
    targets = [found for child in rest for found in grad_site_targets(child, operands)]
    if any(qualifier is not None for _inner, qualifier in targets):
        raise _refuse("a gradient site takes at most one tl.in_backward_pass(n) qualifier")
    return tuple((inner, pass_index) for inner, _none in targets)


class _Firing:
    """Applies the armed edits slot by slot and keeps the fire log."""

    def __init__(self, edits: Sequence[GradSiteEdit], pass_index: int) -> None:
        self.pass_index = pass_index
        self.by_label: dict[str, list[GradSiteEdit]] = {}
        for edit in edits:
            if edit.pass_index is None or edit.pass_index == pass_index:
                self.by_label.setdefault(edit.label, []).append(edit)
        self.fires: list[GradSiteFire] = []
        self._calls: dict[tuple[int, str], int] = {}

    def fire(self, label: str, tuple_index: int, grad: Any) -> Any:
        """Run every edit armed on ``label`` over ``grad`` in declared order."""

        current = grad
        for edit in self.by_label.get(label, ()):
            key = (id(edit), label)
            call = GradSiteCall(label, tuple_index, self._calls.get(key, 0), self.pass_index)
            self._calls[key] = call.call_index + 1
            edited = edit.apply(current, call)
            if edited is None:
                raise _refuse(f"edit {edit.name!r} at {label!r} returned None")
            self.fires.append(
                GradSiteFire(label, edit.name, len(self.fires), edited is not current)
            )
            current = edited
        return current


def _check_sites_exist(graph: ReplayGraph, firing: _Firing) -> None:
    """Refuse an edit armed on a label the graph neither produces nor consumes."""

    known = {slot for node in graph.nodes for slot in (*node.outputs, *node.parents)}
    missing = sorted(label for label in firing.by_label if label not in known)
    if missing:
        raise _refuse(f"no recorded op has the output {missing[0]!r}")


def edited_gradients(
    trace: Any,
    adapter: Any,
    graph: ReplayGraph,
    seeds: Mapping[str, Any],
    edits: Sequence[GradSiteEdit],
    *,
    pass_index: int = 1,
) -> GradSiteResult:
    """Propagate ``seeds`` backward over ``graph`` with the gradient edits applied.

    Parameters
    ----------
    trace, adapter, graph, seeds:
        As :func:`torchlens.backends._reverse_engine.reverse_gradients` takes them.
    edits:
        Edits on output-slot gradients; several on one label fire in this order.
    pass_index:
        Number of the backward pass being computed (for ``pass_index``-qualified edits).

    Returns
    -------
    GradSiteResult
        Each reached label's gradient (the edited value at a site, as torch
        saves ``Op.grad``) and one fire record per fired edit. A site that no
        gradient reaches does not fire, as on torch.

    Raises
    ------
    RunCapabilityUnavailableError
        When an edit names a label outside the graph, an edit returns ``None``,
        or the reverse engine refuses (a reached op the adapter cannot replay,
        wrong derivative arity, nodes out of forward order).
    """

    _check_forward_order(graph)
    firing = _Firing(edits, pass_index)
    _check_sites_exist(graph, firing)
    cotangents: dict[str, Any] = {}
    for label, seed in seeds.items():
        _accumulate(adapter, cotangents, label, seed)
    produced: set[str] = set()
    for node in reversed(graph.nodes):
        produced.update(node.outputs)
        incoming = _fire_slots(firing, cotangents, node.outputs)
        if all(item is None for item in incoming) or not (node.parents or node.blocked):
            continue
        contributions = _node_contributions(trace, adapter, graph, node, incoming)
        for parent, contribution in zip(node.parents, contributions, strict=True):
            _accumulate(adapter, cotangents, parent, contribution)
    leaves = tuple(label for label in cotangents if label not in produced)
    _fire_slots(firing, cotangents, leaves, leaf=True)
    return GradSiteResult(grads=cotangents, fires=tuple(firing.fires))


def _fire_slots(
    firing: _Firing, cotangents: dict[str, Any], slots: Sequence[str], *, leaf: bool = False
) -> list[Any]:
    """Edit each reached slot's complete gradient in place; return the slots' gradients."""

    incoming = []
    for index, slot in enumerate(slots):
        grad = cotangents.get(slot)
        if grad is not None and slot in firing.by_label:
            grad = firing.fire(slot, 0 if leaf else index, grad)
            cotangents[slot] = grad
        incoming.append(grad)
    return incoming


def edited_gradients_for_backend(
    trace: Any,
    backend: str,
    ops: Iterable[Any],
    value: Callable[[str], Any],
    seeds: Mapping[str, Any],
    edits: Sequence[GradSiteEdit],
    *,
    pass_index: int = 1,
) -> GradSiteResult:
    """Build the graph from the backend adapter's recipes, then run :func:`edited_gradients`.

    Parameters
    ----------
    trace:
        The backend trace.
    backend:
        Registered backend name with a replay adapter.
    ops:
        The trace's ops in forward order, graph inputs left out so they are leaves;
        records repeating a multi-output op's recipe are folded into one node.
    value:
        Saved value of an output-slot label.
    seeds, edits, pass_index:
        As :func:`edited_gradients` takes them.

    Raises
    ------
    RunCapabilityUnavailableError
        When the backend has no replay adapter, or :func:`edited_gradients` refuses.
    """

    from ..backends._replay_adapters import ReplayRefusal, get_adapter

    try:
        adapter = get_adapter(backend)
    except ReplayRefusal as exc:
        raise _refuse(str(exc)) from exc
    graph = _one_node_per_op(graph_from_recipes(trace, adapter, ops, value))
    return edited_gradients(trace, adapter, graph, seeds, edits, pass_index=pass_index)


def _one_node_per_op(graph: ReplayGraph) -> ReplayGraph:
    """Keep the first node of each output-slot tuple.

    Backends that list each output of a multi-output op as its own record give
    every record the op's whole recipe; one node per op keeps the op's
    derivative from being counted once per output.
    """

    seen: set[tuple[str, ...]] = set()
    nodes = []
    for node in graph.nodes:
        if node.outputs in seen:
            continue
        seen.add(node.outputs)
        nodes.append(node)
    return ReplayGraph(nodes=tuple(nodes), value=graph.value)


def disagreeing_labels(
    adapter: Any, result: GradSiteResult, native: Mapping[str, Any]
) -> tuple[str, ...]:
    """Labels whose native recorded gradient differs from the edited value.

    ``native`` maps graph labels to the gradient a native backward record holds
    for the same pass with the same edits. The comparison is the backend
    validator's own (``adapter.close``); labels missing on either side are
    reported too.
    """

    labels = sorted(set(result.grads) | set(native))
    return tuple(
        label
        for label in labels
        if label not in result.grads
        or label not in native
        or not adapter.close(native[label], result.grads[label])
    )


_ARMED: weakref.WeakKeyDictionary[Any, tuple[GradSiteEdit, ...]] = weakref.WeakKeyDictionary()
"""Edits armed per trace object until a backward computation applies them."""


def build_grad_site_edits(
    site: Any,
    apply: Callable[[Any, GradSiteCall], Any],
    resolve_labels: Callable[[Any], Sequence[str]],
    *,
    operands: Operands,
    name: str = "edit",
) -> tuple[GradSiteEdit, ...]:
    """Lower one ``tl.when(site, edit)`` clause to one :class:`GradSiteEdit` per label.

    Parameters
    ----------
    site:
        The clause's gradient site (``tl.grad(s)``, unions, an
        ``tl.in_backward_pass(n)`` qualifier).
    apply:
        The edit as a backend-native ``apply(grad, call) -> grad``.
    resolve_labels:
        The backend's resolution of a forward selector to the graph labels
        (output slots) it names on this trace.
    operands:
        The caller's composite-target reader (see :func:`grad_site_targets`).
    name:
        Display name for fire records.

    Raises
    ------
    RunCapabilityUnavailableError
        When ``site`` is not a gradient address (see :func:`grad_site_targets`)
        or a forward selector names no op.
    """

    edits: list[GradSiteEdit] = []
    for inner, pass_index in grad_site_targets(site, operands):
        labels = tuple(resolve_labels(inner))
        if not labels:
            raise _refuse(f"tl.grad({inner!r}) names no recorded op")
        edits.extend(GradSiteEdit(lbl, apply, name, pass_index) for lbl in labels)
    return tuple(edits)


def arm_grad_sites(trace: Any, edits: Sequence[GradSiteEdit]) -> None:
    """Arm ``edits`` on ``trace`` after any already armed (declared order is kept)."""

    _ARMED[trace] = (*_ARMED.get(trace, ()), *edits)


def armed_grad_sites(trace: Any) -> tuple[GradSiteEdit, ...]:
    """The edits armed on ``trace``, in declared order."""

    return _ARMED.get(trace, ())


def clear_grad_sites(trace: Any) -> None:
    """Disarm every edit on ``trace``."""

    _ARMED.pop(trace, None)


def run_armed_grad_sites(
    trace: Any,
    backend: str,
    ops: Iterable[Any],
    value: Callable[[str], Any],
    seeds: Mapping[str, Any],
    *,
    pass_index: int = 1,
) -> GradSiteResult:
    """Compute one backward pass's gradients with the edits armed on ``trace`` applied.

    The arguments are those of :func:`edited_gradients_for_backend`; the edits
    stay armed, so a second pass applies them again, as torch's armed hooks do.
    """

    return edited_gradients_for_backend(
        trace, backend, ops, value, seeds, armed_grad_sites(trace), pass_index=pass_index
    )


def do_signal_plan_off_torch(trace: Any, plan: Any, request: Any, backend: str) -> Any:
    """Run ``Trace.do`` with a ``tl.grad`` site on a non-torch trace.

    ``intervention._grad_site_plan.do_signal_plan`` routes here before any
    torch-only step. The backend's door module
    (``torchlens/backends/<backend>/_grad_site_door.py``, owned by its
    ``-int`` lane) takes the whole plan: it lowers the activation half the way
    its finished-trace edits do, arms the gradient half with
    :func:`build_grad_site_edits` and :func:`arm_grad_sites`, runs
    ``request.loss`` when given, and returns the armed trace.

    Raises
    ------
    RunCapabilityUnavailableError
        When the backend has no door module yet.
    """

    from importlib import import_module

    name = _DOOR_MODULE.format(backend=backend)
    try:
        door = import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name not in (name, name.rpartition(".")[0]):
            raise
        raise _refuse(f"Trace.do with a tl.grad site is not open on the {backend} backend") from exc
    return door.do_signal_plan(trace, plan, request)
