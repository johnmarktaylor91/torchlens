"""Cone-of-effect membership and site labelling for saved-DAG replay.

Split from :mod:`torchlens.intervention.replay`: the cone walk, the label-to-replay-key
map every replay step resolves through, origin validation, the user-facing
disclosure spelling of a site and the dependency-ordered call-cone scheduler that
replay and sparse runnable execution share. Nothing here mutates a trace; the
scheduler only calls the per-call callback its caller passes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from .._errors import InvalidArgumentError
from ._replay_context import _func_call_groups, _replay_site_key
from .errors import ReplayPreconditionError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _disclosure_label(site: Op) -> str:
    """Return the user-facing label for replay disclosures and frontier keys.

    Bare layer labels are unambiguous only when the layer is single-pass;
    multi-pass ops disclose their pass-qualified spelling.
    """

    if int(getattr(site, "num_passes", 1) or 1) > 1:
        return _replay_site_key(site)
    return site.layer_label


def _label_key_map(trace: Trace) -> dict[str, tuple[str, ...]]:
    """Map every string label spelling to the replay keys it may denote.

    A spelling denoting exactly one op (pass-qualified labels, single-pass
    bare labels, historical position labels) maps to that op's replay key; a
    layer-wide spelling of a multi-pass layer (its bare ``layer_label``, its
    short label) maps to EVERY pass's key and is therefore pass-ambiguous.
    """

    mapping: dict[str, list[str]] = {}
    for op in trace._layer_list:
        key = _replay_site_key(op)
        spellings = {key, op.layer_label}
        for spelling in getattr(op, "lookup_keys", ()) or ():
            if isinstance(spelling, str):
                spellings.add(spelling)
        for spelling in spellings:
            keys = mapping.setdefault(spelling, [])
            if key not in keys:
                keys.append(key)
    return {spelling: tuple(keys) for spelling, keys in mapping.items()}


def _final_label_key_map(trace: Trace) -> dict[str, tuple[str, ...]]:
    """Map every FINAL label spelling to the replay keys it denotes.

    Relations (``children``, ``out_versions_by_child``) are spelled in final
    labels: the pass-qualified ``label`` of a multi-pass op, the bare
    ``layer_label`` of a single-pass one. A historical spelling kept in
    ``lookup_keys`` (the raw label an op carried before pass grouping
    renumbered later steps) can equal another op's final label, so a relation
    resolved through :func:`_label_key_map` would also name the op that once
    carried it and pull an unrelated branch into the cone. This map holds
    final spellings only; the bare label of a multi-pass layer still expands
    to every pass.
    """

    mapping: dict[str, list[str]] = {}
    for op in trace._layer_list:
        key = _replay_site_key(op)
        for spelling in {key, op.layer_label}:
            keys = mapping.setdefault(spelling, [])
            if key not in keys:
                keys.append(key)
    return {spelling: tuple(keys) for spelling, keys in mapping.items()}


def _validated_cone_origins(origins: Iterable[Op]) -> list[Op]:
    """Refuse non-op cone origins typed (leverage B3), returning them listed."""

    if isinstance(origins, str):
        raise InvalidArgumentError(
            f"cone_of_effect origins must be op records, not the string "
            f"{origins!r}: iterating a string walks its CHARACTERS and would "
            "silently return an empty cone. Resolve the label first — "
            "trace.layers[label].ops, or trace.find_sites(label).",
            code="cone_origin_invalid",
            remedy="pass op records (e.g. trace.layers[label].ops)",
        )
    listed = list(origins)
    for origin in listed:
        if isinstance(origin, str) or not hasattr(origin, "layer_label"):
            raise InvalidArgumentError(
                f"cone_of_effect origin {origin!r} is not an op record; "
                "resolve string addresses first (trace.layers[label].ops, or "
                "trace.find_sites(label)).",
                code="cone_origin_invalid",
                remedy="pass op records (e.g. trace.layers[label].ops)",
            )
    return listed


def _child_labels(site: Op) -> tuple[str, ...]:
    """Return child labels from edge and tensor-version metadata.

    Parameters
    ----------
    site:
        Layer pass whose children should be traversed.

    Returns
    -------
    tuple[str, ...]
        Child labels.
    """

    labels = list(getattr(site, "children", ()) or ())
    labels.extend((getattr(site, "out_versions_by_child", {}) or {}).keys())
    return tuple(dict.fromkeys(labels))


def cone_of_effect(trace: Trace, origins: Iterable[Op]) -> list[Op]:
    """Return downstream cone in topological order.

    Parameters
    ----------
    trace:
        Model log whose saved graph should be traversed.
    origins:
        Origin layer ops whose downstream dependents are affected.

    Returns
    -------
    list[Op]
        Origin and downstream sites in execution order, with call-group
        siblings included. Traversal is keyed by pass-qualified op labels, so
        edges crossing recurrence-grouped (multi-pass) layers are followed
        per-pass rather than silently dropped.
    """

    origins = _validated_cone_origins(origins)
    all_keys = trace._layer_dict_all_keys
    label_keys = _label_key_map(trace)
    final_keys = _final_label_key_map(trace)
    call_groups = _func_call_groups(trace)
    visited: set[str] = set()
    frontier: deque[str] = deque()
    for origin in origins:
        key = _replay_site_key(origin)
        if key in all_keys:
            frontier.append(key)

    def _enqueue_children(site: Op) -> None:
        """Push one op's unvisited children onto the cone frontier, per pass.

        Parameters
        ----------
        site:
            The op whose child relations are being expanded.
        """

        for child_label in _child_labels(site):
            # A final-label spelling names that op only (never one that carried
            # it raw); a pass-ambiguous spelling expands to every pass, a safe
            # conservative superset for cone traversal.
            for child_key in final_keys.get(child_label) or label_keys.get(child_label, ()):
                if child_key not in visited:
                    frontier.append(child_key)

    while frontier:
        key = frontier.popleft()
        if key in visited:
            continue
        visited.add(key)
        layer = all_keys.get(key)
        if layer is None:
            continue

        group = call_groups.get(layer.func_call_id, ()) if layer.func_call_id is not None else ()
        for sibling in group:
            visited.add(_replay_site_key(sibling))
            _enqueue_children(sibling)

        _enqueue_children(layer)

    return [layer for layer in trace._layer_list if _replay_site_key(layer) in visited]


def _resolve_single_origin(
    log: Trace,
    site: Any,
    *,
    strict: bool,
) -> Op:
    """Resolve one replay_from origin.

    Parameters
    ----------
    log:
        Model log to query.
    site:
        Layer pass or selector-like query.
    strict:
        Whether strict selector resolution is active.

    Returns
    -------
    Op
        Single origin site.
    """

    if hasattr(site, "layer_label") and hasattr(site, "out"):
        return cast("Op", site)
    return cast("Op", log._resolve_sites(site, strict=strict, max_fanout=1).first())


class _CallConeNode(Protocol):
    """Dependency fields required by the shared call-cone scheduler."""

    @property
    def call_id(self) -> str:
        """Return the stable call identifier."""

        ...

    @property
    def parent_call_ids(self) -> tuple[str, ...]:
        """Return stable identifiers for dependency-parent calls."""

        ...


def _walk_call_cone(
    calls: Sequence[_CallConeNode],
    execute_call: Callable[[_CallConeNode], None],
) -> None:
    """Execute a dependency-complete call cone in stable recorded order.

    Parameters
    ----------
    calls:
        Recorded call nodes whose parent IDs define the runnable cone.
    execute_call:
        Transaction-local callback that executes and stages one ready node.

    Raises
    ------
    ReplayPreconditionError
        If the recorded cone has a missing parent or cyclic dependency.

    Notes
    -----
    The caller owns transaction commit. Sparse runnable execution uses an
    unexposed Trace fork, while intervention replay uses pending update maps.
    """

    call_ids = {call.call_id for call in calls}
    missing = {
        parent_id
        for call in calls
        for parent_id in call.parent_call_ids
        if parent_id not in call_ids
    }
    if missing:
        raise ReplayPreconditionError(
            "Recorded call cone references missing parents: " + ", ".join(sorted(missing))
        )
    pending = list(calls)
    completed: set[str] = set()
    while pending:
        ready_index = next(
            (index for index, call in enumerate(pending) if set(call.parent_call_ids) <= completed),
            None,
        )
        if ready_index is None:
            blocked = ", ".join(call.call_id for call in pending)
            raise ReplayPreconditionError(
                f"Recorded call cone contains a dependency cycle among: {blocked}."
            )
        call = pending.pop(ready_index)
        execute_call(call)
        completed.add(call.call_id)
