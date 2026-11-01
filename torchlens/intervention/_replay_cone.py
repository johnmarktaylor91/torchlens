"""Cone-of-effect membership and site labelling for saved-DAG replay.

Split from :mod:`torchlens.intervention.replay`: the cone walk, the label-to-replay-key
map every replay step resolves through, origin validation and the user-facing
disclosure spelling of a site. Nothing here executes a callable or mutates a trace.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from .._errors import InvalidArgumentError
from ._replay_context import _func_call_groups, _replay_site_key

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


def _validated_cone_origins(origins: Iterable[Op]) -> list[Op]:
    """Refuse non-op cone origins typed (leverage B3), returning them listed."""

    if isinstance(origins, str):
        raise InvalidArgumentError(
            f"cone_of_effect origins must be op records, not the string "
            f"{origins!r}: iterating a string walks its CHARACTERS and would "
            "silently return an empty cone. Resolve the label first — "
            "trace[label].ops, or trace.find_sites(label).",
            code="cone_origin_invalid",
            remedy="pass op records (e.g. trace[label].ops)",
        )
    listed = list(origins)
    for origin in listed:
        if isinstance(origin, str) or not hasattr(origin, "layer_label"):
            raise InvalidArgumentError(
                f"cone_of_effect origin {origin!r} is not an op record; "
                "resolve string addresses first (trace[label].ops, or "
                "trace.find_sites(label)).",
                code="cone_origin_invalid",
                remedy="pass op records (e.g. trace[label].ops)",
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
            # A pass-ambiguous child spelling (not produced by finished-trace
            # relations, but guarded against) expands to every pass: a
            # conservative superset is safe for cone traversal, guessing one
            # pass is not.
            for child_key in label_keys.get(child_label, ()):
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
