"""Realize-once capture: every recorded tinygrad value comes from one evaluation.

The traced forward runs once to its outputs under observation (``backend``).
This module reads the untouched UOp graph the forward built, decides what the
capture records, and realizes every recorded value in a single
:meth:`TinygradProtocol.realize_once` call, so each payload is taken from that
one evaluation and random-number state is consumed exactly as one eager forward
consumes it. Nothing in the capture realizes a recorded value a second time:
the payload copies read realized buffers back, they do not evaluate the graph.

Random-init weights. A ``tinygrad.nn`` layer's default init leaves each weight
as a lazy Threefry graph until something realizes it. A UOp that a live tensor
already held before the forward ran, and whose graph carries RNG state, is such
a pre-forward random value: the capture records it as a parentless value
realized in the same single call and does not walk into its generator subgraph,
so the capture records the same ops it records for a model whose weights were
realized first. The forward's outputs are computed as the eager forward
computes them, so they equal the eager forward of the same seed.
Random state drawn inside the forward has no such boundary: its value would
change if anything evaluated it again, so the capture refuses
``rng_state_in_graph`` before anything is realized.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .._protocols.tinygrad import TinygradProtocol
from ._derived_grads import call_input_leaves
from ._rules import (
    CONSTANT_UNREALIZABLE_CODE,
    RNG_STATE_IN_GRAPH_CODE,
    RNG_UOP_NAMES,
    refuse_constant_unrealizable,
    refuse_rng_state_in_graph,
)
from ._uop_graph import _data_flow_uops, _unique_uops, _uop_name, edited_unique_uops

__tl_layer__ = "L2"


@dataclass(frozen=True)
class RealizeOncePlan:
    """What a capture records, decided on the graph before anything is realized.

    Parameters
    ----------
    recorded
        UOps recorded as ops, in topological order, sources excluded.
    random_values
        ``id()`` of the recorded UOps that are pre-forward random values
        (recorded parentless; their generator subgraph is not walked).
    pinned
        Every UOp the plan's ``id()`` keys refer to, held so no key is reused.
    edit_sites
        For an edited capture, ``id(replacement UOp) -> replaced result UOp``:
        each replacement is recorded as the op it replaced (``_edits``).
    """

    recorded: tuple[Any, ...]
    random_values: frozenset[int]
    pinned: tuple[Any, ...]
    edit_sites: Mapping[int, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealizedCapture:
    """A capture after its one realization: what to record and the realized values.

    Parameters
    ----------
    plan
        The capture plan (:func:`plan_capture`).
    realized
        Realized tensors keyed by the ``id()`` of the pre-realization UOp.
    owned
        Keys of ``realized`` whose tensor holds a buffer the capture created
        and nothing else references; such a tensor is the payload as is.
        Any other value (an input or weight that already had a buffer, a
        random-init weight the model now holds) shares its buffer and is
        copied for the payload.
    input_leaves
        ``(container_path, tensor, uop)`` per tensor input leaf, the UOp
        read before realization (it matches the forward graph's sources).
    output_uops
        Output UOps read before realization, parallel to the flat outputs
        (realization repoints each output tensor at its buffer).
    """

    plan: RealizeOncePlan
    realized: Mapping[int, Any]
    owned: frozenset[int]
    input_leaves: tuple[tuple[str, Any, Any], ...]
    output_uops: tuple[Any, ...]


def live_tensor_uops() -> tuple[Any, ...]:
    """The UOps every live tinygrad tensor holds right now.

    Read just before the forward runs: a UOp in this set existed before the
    forward, so it is model state (a weight, a closure constant), not a value
    the forward computed. tinygrad tracks its live tensors in
    ``tinygrad.tensor.all_tensors``; when that registry is absent the set is
    empty and every RNG-bearing graph refuses ``rng_state_in_graph``.

    Returns
    -------
    tuple[Any, ...]
        The held UOps (the tuple keeps them alive for ``id()`` lookups).
    """

    try:
        from tinygrad import tensor as tensor_module
    except ImportError:
        return ()
    registry = getattr(tensor_module, "all_tensors", None)
    if not isinstance(registry, Mapping):
        return ()
    uops: list[Any] = []
    for ref in list(registry):
        tensor = ref() if callable(ref) else None
        uop = getattr(tensor, "uop", None)
        if uop is not None:
            uops.append(uop)
    return tuple(uops)


def _rng_flags(uops: Sequence[Any], cut_ids: Collection[int]) -> dict[int, bool]:
    """Whether each UOp's value depends on RNG state, not looking past ``cut_ids``.

    Parameters
    ----------
    uops
        UOps in topological order (sources before their users).
    cut_ids
        ``id()`` of UOps whose own RNG dependence is not propagated (sources
        and pre-forward random values the capture records whole).

    Returns
    -------
    dict[int, bool]
        ``id(uop) -> depends on a generator UOp`` for every UOp in ``uops``.
    """

    flags: dict[int, bool] = {}
    for uop in uops:
        key = id(uop)
        if key in cut_ids:
            flags[key] = False
            continue
        flags[key] = _uop_name(uop) in RNG_UOP_NAMES or any(
            flags.get(id(src), False) for src in getattr(uop, "src", ()) or ()
        )
    return flags


def plan_capture(
    outputs: Sequence[Any],
    source_uops: Sequence[Any],
    preexisting_uops: Iterable[Any],
    edit_sites: Mapping[int, Any] | None = None,
) -> RealizeOncePlan:
    """Decide the recorded UOps of a capture without realizing anything.

    Parameters
    ----------
    outputs
        Flat tinygrad output tensors of the forward.
    source_uops
        UOps of the traced input leaves, read before any realization.
    preexisting_uops
        UOps live tensors held before the forward ran (:func:`live_tensor_uops`).
    edit_sites
        ``id(replacement UOp) -> replaced result UOp`` of an edited capture.

    Returns
    -------
    RealizeOncePlan
        The recorded UOps and which of them are pre-forward random values.

    Raises
    ------
    BackendUnsupportedError
        ``rng_state_in_graph`` when an output depends on RNG state the forward
        itself drew; nothing has been realized at that point.
    """

    edits = dict(edit_sites or {})
    uops = edited_unique_uops(outputs, edits) if edits else _unique_uops(outputs)
    source_ids = frozenset(id(uop) for uop in source_uops)
    preexisting_ids = frozenset(id(uop) for uop in preexisting_uops)
    rng = _rng_flags(uops, source_ids)
    candidates = frozenset(key for key, flag in rng.items() if flag and key in preexisting_ids)
    recorded = _data_flow_uops(uops, outputs, source_ids, boundary_ids=candidates, edit_sites=edits)
    random_values = candidates & frozenset(id(uop) for uop in recorded)
    remaining = _rng_flags(uops, source_ids | random_values)
    if any(remaining.get(id(cast(Any, output).uop), False) for output in outputs):
        culprit = next(
            uop for uop in uops if remaining[id(uop)] and _uop_name(uop) in RNG_UOP_NAMES
        )
        raise refuse_rng_state_in_graph(culprit, code=RNG_STATE_IN_GRAPH_CODE)
    return RealizeOncePlan(
        recorded=recorded,
        random_values=random_values,
        pinned=(*uops, *source_uops, *edits.values()),
        edit_sites=edits,
    )


def realize_recorded_values(
    plan: RealizeOncePlan, source_uops: Sequence[Any], outputs: Sequence[Any]
) -> tuple[dict[int, Any], frozenset[int]]:
    """Realize the sources, every recorded value and the outputs in one call.

    Each source and recorded forward UOp is listed as ``CONTIGUOUS(uop)`` and
    the whole list, plus the outputs, goes to a single ``realize_once``: one
    tinygrad schedule, so random state is drawn once. The ``CONTIGUOUS``
    wrapper gives each recorded value its own kernel and buffer without making
    the UOp itself a buffer boundary, so every consumer, the outputs included,
    is computed exactly as the eager forward computes it (tinygrad fuses
    elementwise and reduce chains, and storing an intermediate would change
    the rounding downstream). A pre-forward random value is listed the same
    way, so the forward's outputs equal the eager forward of the same seed.

    Parameters
    ----------
    plan
        The capture plan (:func:`plan_capture`).
    source_uops
        UOps of the traced input leaves, read before any realization.
    outputs
        Flat tinygrad output tensors.

    Returns
    -------
    tuple[dict[int, Any], frozenset[int]]
        Realized tensors keyed by the ``id()`` of the pre-realization UOp, and
        the keys whose buffer the capture owns (:class:`RealizedCapture`).

    Raises
    ------
    BackendUnsupportedError
        ``constant_unrealizable`` when tinygrad cannot realize a recorded value
        (``NotImplementedError``); the message names the first recorded
        ``BUFFER`` source, else the first recorded op.
    """

    realized, owned = _realization_targets(plan, source_uops)
    try:
        TinygradProtocol().realize_once([*realized.values(), *outputs])
    except NotImplementedError as error:
        candidates = (*plan.recorded, *source_uops) or (cast(Any, outputs[0]).uop,)
        culprit = next((uop for uop in candidates if _uop_name(uop) == "BUFFER"), candidates[0])
        raise refuse_constant_unrealizable(
            culprit, error, code=CONSTANT_UNREALIZABLE_CODE
        ) from error
    return realized, owned


def _realization_targets(
    plan: RealizeOncePlan, source_uops: Sequence[Any]
) -> tuple[dict[int, Any], frozenset[int]]:
    """The tensors :func:`realize_recorded_values` lists, before realizing.

    Every value is listed as ``CONTIGUOUS(uop)``; when tinygrad returns the
    UOp itself (it already has a buffer, is already contiguous, or has no
    device) the buffer is shared, so the key is not owned. A pre-forward
    random value is listed the same way: its consumers keep computing it
    inline as the eager forward does, and its random bits (``Tensor.rand`` is
    a buffer boundary) are drawn once in the same schedule, after which
    tinygrad points the weight's graph at that buffer.
    """

    from tinygrad import Tensor

    realized: dict[int, Any] = {}
    owned: set[int] = set()
    for uop in (*source_uops, *plan.recorded):
        if id(uop) in realized:
            continue
        target = uop.contiguous()
        realized[id(uop)] = Tensor(target)
        if target is not uop:
            owned.add(id(uop))
    return realized, frozenset(owned)


def realized_value_uops(realized_tensor: Any) -> tuple[Any, ...]:
    """The UOps a realized pre-forward random value is recorded as.

    After ``realize_once`` the value is a buffer (a ``BUFFER``, or a
    ``RESHAPE`` over one). The same data-flow rule that picks the forward's
    ops picks these, so the capture records exactly the ops it records for a
    weight realized before the forward.

    Parameters
    ----------
    realized_tensor
        The tensor :func:`realize_recorded_values` realized for the value.

    Returns
    -------
    tuple[Any, ...]
        The UOps to record, in topological order; the last is the value itself.
    """

    leaves = (realized_tensor,)
    return _data_flow_uops(_unique_uops(leaves), leaves, frozenset())


def realize_capture(
    args: Sequence[Any],
    outputs: Sequence[Any],
    preexisting_uops: Iterable[Any],
    is_tensor: Callable[[Any], bool],
    *,
    kwargs: Mapping[str, Any] | None = None,
    edit_sites: Mapping[int, Any] | None = None,
) -> RealizedCapture:
    """Read the forward's graph, plan the capture, and realize it once.

    Every UOp is read before the single realization, which repoints the
    output tensors at their buffers.

    Parameters
    ----------
    args
        Normalized positional call arguments.
    outputs
        Flat tinygrad output tensors of the forward.
    preexisting_uops
        UOps live tensors held before the forward ran (:func:`live_tensor_uops`).
    is_tensor
        Predicate selecting tinygrad tensor leaves of ``args`` and ``kwargs``.
    kwargs
        Keyword call arguments; their tensor leaves are inputs too.
    edit_sites
        ``id(replacement UOp) -> replaced result UOp`` of an edited capture.

    Returns
    -------
    RealizedCapture
        The plan, the realized values and the pre-realization UOps.
    """

    input_leaves = tuple(
        (path, value, cast(Any, value).uop)
        for path, value in call_input_leaves(args, kwargs)
        if is_tensor(value)
    )
    source_uops = tuple(uop for _path, _value, uop in input_leaves)
    output_uops = tuple(cast(Any, output).uop for output in outputs)
    plan = plan_capture(outputs, source_uops, preexisting_uops, edit_sites)
    realized, owned = realize_recorded_values(plan, source_uops, outputs)
    return RealizedCapture(plan, realized, owned, input_leaves, output_uops)


def recorded_values(capture: RealizedCapture) -> Iterator[tuple[Any, Any, Any, bool]]:
    """The ops a capture records, each with its realized value, in record order.

    A forward value is recorded as its own UOp; an edit's replacement is
    recorded as the result it replaced (parents from that result, value from
    the replacement, model UOp the replacement). A pre-forward random value is
    recorded as the buffer ops tinygrad realized it into
    (:func:`realized_value_uops`); the last of them stands for the weight, so
    its model UOp is the pre-realization weight UOp the model and the
    forward's consumers refer to.

    Parameters
    ----------
    capture
        The realized capture (:func:`realize_capture`).

    Yields
    ------
    tuple[Any, Any, Any, bool]
        ``(uop, realized_tensor, model_uop, owned)``: the recorded UOp (its
        ``src`` gives the parent edges), the tensor holding its value, the UOp
        the model, the observer and consumers know the value by, and whether
        the capture owns the tensor's buffer (else the payload is a copy).
    """

    from tinygrad import Tensor

    for uop in capture.plan.recorded:
        tensor = capture.realized[id(uop)]
        if id(uop) in capture.plan.edit_sites:
            yield capture.plan.edit_sites[id(uop)], tensor, uop, id(uop) in capture.owned
            continue
        if id(uop) not in capture.plan.random_values:
            yield uop, tensor, uop, id(uop) in capture.owned
            continue
        owned = id(uop) in capture.owned
        for value_uop in realized_value_uops(tensor):
            if value_uop is tensor.uop:
                yield value_uop, tensor, uop, owned
            else:
                yield value_uop, Tensor(value_uop), value_uop, owned
