"""Replay run-context store: pass-qualified site keys and the capture-truth digest ledger.

Session-time state the replay/push engine keeps on ``Trace.last_run`` (RUNTIME
storage, never persisted): the pass-qualified key every replay structure is
addressed by, the resolver from a template parent ref to the exact recorded
op, the run-context accessor that seeds it, and the ledger of
capture-time out digests recorded the first time replay overwrites a site,
and the rollback journal a multi-door transaction restores on failure.
Split out of ``replay.py`` so validation can read the ledger without importing
the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from .edge_substitution import tensor_content_digest

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _replay_site_key(site: Op) -> str:
    """Return the pass-qualified replay key for one op record.

    ``Op.label`` is the pass-qualified ``layer_label:pass`` spelling on every
    finished-trace op (single-pass ops carry ``:1``), and every such spelling
    is a ``layer_dict_all_keys`` lookup key, so replay state keyed by it can
    never collide across passes of a recurrence-grouped layer. Bare
    ``layer_label`` keys map to the LAST pass only — keying replay state by
    them is exactly the pass-blind corruption this key exists to prevent.
    """

    label = getattr(site, "label", None)
    if isinstance(label, str) and label:
        return label
    return site.layer_label


def _final_label_for_ref(log: Trace, label: str) -> str:
    """Resolve raw or final parent-ref label to a current lookup label.

    The exact recorded op comes first: a raw label through the pass-exact
    ``_raw_to_final_op_labels``, a final lookup label (what labeling renames a
    template ref to) back through ``_final_to_raw_layer_labels`` and then the
    pass-exact map. Both maps come from the same labeling loop, so the result
    names one pass of one layer. A layer-wide spelling can also be another op's
    public raw label (a fused multi-pass layer shifts later step indices), so
    resolving it as a lookup key could look ambiguous or name the wrong op.
    Labels the maps do not hold (traces without them) keep the layer-wide
    lookup, where an ambiguous spelling still refuses.

    Parameters
    ----------
    log:
        Model log to query.
    label:
        Raw or final label from a template.

    Returns
    -------
    str
        Lookup label.
    """

    op_labels = log._raw_to_final_op_labels or {}
    raw_label = label
    if raw_label not in op_labels:
        raw_label = (log._final_to_raw_layer_labels or {}).get(label, "")
    exact_label = op_labels.get(raw_label)
    if isinstance(exact_label, str) and exact_label in log.layer_dict_all_keys:
        return exact_label
    if label in log.layer_dict_all_keys:
        return label
    return cast(str, getattr(log, "_raw_to_final_layer_labels", {}).get(label, label))


def _ensure_replay_run_ctx(log: Trace) -> dict[str, Any]:
    """Return a mutable replay run context on ``log``.

    Parameters
    ----------
    log:
        Model log being replayed.

    Returns
    -------
    dict[str, Any]
        Run context dictionary.
    """

    if not isinstance(getattr(log, "last_run", None), dict):
        log.last_run = {}
    run_ctx = cast(dict[str, Any], log.last_run)
    # Seed law D5 (F02): thread the capture's recorded seed so seed='auto'
    # stochastic edits canonicalize their base seed on the replay door too.
    trace_seed = getattr(log, "random_seed", None)
    if trace_seed is not None:
        run_ctx.setdefault("trace_random_seed", trace_seed)
    return run_ctx


def _record_capture_digest(
    site: Op, known_digests: Mapping[str, str], new_digests: dict[str, str]
) -> None:
    """Record a site's capture-truth out digest the FIRST time replay overwrites it.

    The site's current out IS the capture-time value its children's
    ``saved_args`` snapshotted, so it is digested before being replaced;
    validation compares those snapshots against the digest once the value
    itself is gone (see :func:`replay_capture_digest`). Later overwrites of an
    already-recorded site never re-digest (the pushed value is not truth).
    """

    site_key = _replay_site_key(site)
    if site_key in known_digests or site_key in new_digests:
        return
    current = site.out
    if isinstance(current, torch.Tensor):
        new_digests[site_key] = tensor_content_digest(current)


REPLAY_CAPTURE_DIGESTS_KEY = "replay_capture_digests"
"""Run-context key holding ``{replay key: sha256}`` of every site's out as it
stood the FIRST time this trace's replay engine overwrote it.

A pushed trace no longer carries the capture-time value of any recomputed
site, but its children's ``saved_args`` snapshots still do; the digest is the
one piece of capture truth that lets validation keep comparing those
snapshots against the producer (:func:`replay_capture_digest`) instead of
against a pushed value that legitimately differs. Session-time only: it rides
``Trace.last_run`` (the replay run context, RUNTIME storage) and is never
persisted.
"""


def replay_capture_digest(log: Trace, site: Op) -> str | None:
    """Return the capture-truth digest of a replay-recomputed site's out.

    ``None`` means the replay engine has never overwritten this site on this
    trace (its current ``out`` IS capture truth, compare values directly) or
    that no digest was recordable (the pre-edit out was not a tensor).
    """

    run_ctx = getattr(log, "last_run", None)
    if not isinstance(run_ctx, dict):
        return None
    digests = run_ctx.get(REPLAY_CAPTURE_DIGESTS_KEY)
    if not isinstance(digests, dict):
        return None
    digest = digests.get(_replay_site_key(site))
    return digest if isinstance(digest, str) else None


#: Per-op fields replay commits and the PARAM/EDGE doors stage (saved out and
#: its metadata, fire records, tier-(ii) substitution stores).
_JOURNAL_OP_FIELDS = (
    "out",
    "transformed_out",
    "shape",
    "transformed_out_shape",
    "dtype",
    "dtype_ref",
    "device_ref",
    "transformed_out_dtype",
    "activation_memory",
    "transformed_activation_memory",
    "has_saved_activation",
    "interventions",
    "is_edited",
    "edge_substitutions",
    "edge_replacement_stamps",
)
#: Trace fields a replay rebinds or mutates (run context, state, revisions,
#: audit rows and operation history).
_JOURNAL_TRACE_FIELDS = (
    "last_run",
    "state",
    "_out_recipe_revision",
    "_has_direct_writes",
    "_replay_frontier",
    "state_history",
    "intervention_audit",
)


def _journal_copy(value: Any) -> Any:
    """Shallow-copy a mutable container so later in-place writes cannot reach it."""

    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _journal_same(current: Any, recorded: Any) -> bool:
    """Whether ``current`` still holds exactly the recorded objects (identity-wise)."""

    if isinstance(recorded, list) and isinstance(current, list):
        return len(current) == len(recorded) and all(
            a is b for a, b in zip(current, recorded, strict=True)
        )
    if isinstance(recorded, dict) and isinstance(current, dict):
        return current.keys() == recorded.keys() and all(
            current[key] is recorded[key] for key in recorded
        )
    return current is recorded


def journal_replay_state(trace: Trace) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, Any]]:
    """Record every value replay and the PARAM/EDGE doors may write on ``trace``.

    Tensors are held by reference (replay replaces outs, never writes them in
    place); containers are shallow-copied. ``restore_replay_state`` puts each
    value back, so a transaction spanning several doors can be undone whole.
    """

    ops = [
        (op, {name: _journal_copy(getattr(op, name, None)) for name in _JOURNAL_OP_FIELDS})
        for op in trace.layer_list
    ]
    fields = {
        name: _journal_copy(getattr(trace, name))
        for name in _JOURNAL_TRACE_FIELDS
        if hasattr(trace, name)
    }
    return ops, fields


def restore_replay_state(
    trace: Trace, journal: tuple[list[tuple[Any, dict[str, Any]]], dict[str, Any]]
) -> None:
    """Put back every value ``journal_replay_state`` recorded."""

    ops, fields = journal
    for op, values in ops:
        for name, value in values.items():
            if not _journal_same(getattr(op, name, None), value):
                op._internal_set(name, _journal_copy(value))
    for name, value in fields.items():
        current = getattr(trace, name)
        if isinstance(value, list) and isinstance(current, list):
            current[:] = value
        else:
            setattr(trace, name, _journal_copy(value))
