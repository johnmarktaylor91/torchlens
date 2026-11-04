"""Save-entry gate helpers for ``torchlens._io.bundle.save``.

Three entry-time guards that run before (or instead of) the per-trace save
pipeline: the two PERMANENT erasure-prevention invariants (edge-substitution
carriers and the shard-local disclosure, both keyed on the ACTIVE SCHEMA so a
regression re-fires them), and the Bundle container-door delegation that
routes ``bundle.save(path, ...)`` through ``Bundle.save`` (foldB D8/D9); plus
the blob-loop refusal of a tensor a strict (portable) save cannot write. Split
out of ``torchlens/_io/bundle.py`` to respect its size ceiling (R43
split-never-raise); ``bundle.py`` re-exports these names, so the monkeypatch
surface ``torchlens._io.bundle._refuse_*`` is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NoReturn

from .._data_classes._trace_missing_attrs import is_live_trace
from .._errors import InvalidArgumentError
from .format_errors import ArtifactError


def _refuse_edge_intervened_save(trace: Any) -> None:
    """PERMANENT erasure-prevention invariant for edge substitutions (L6 4.3).

    The tlspec v8 coordinated bump persists ``Op.edge_substitutions``
    (BLOB_RECURSIVE) and ``Op.edge_replacement_stamps`` (KEEP), so ordinary
    saves of edge-intervened traces now proceed. Like the shard-local guard
    below, the predicate keys on the ACTIVE SCHEMA rather than being deleted:
    it refuses typed IFF tier-(ii) entries are present AND the active policy
    would drop them (a schema regression re-dropping the carrier, with the S3
    switch inactive). Any such regression re-fires this refusal instead of
    silently presenting post-edit values with zero edge provenance.
    """

    from .._data_classes.trace import Trace

    if not isinstance(trace, Trace):
        # A Recording or other non-Trace save subject holds no Ops, so it
        # cannot carry edge substitutions; its own refusal fires later.
        return
    carriers = [
        op.label
        for op in (trace._layer_list if is_live_trace(trace) else ()) or ()
        if getattr(op, "edge_substitutions", None)
    ]
    if not carriers:
        return
    from .._data_classes.op import Op as _Op
    from . import FieldPolicy
    from .prerelease import prerelease_fields_active

    policy_entry = _Op._FIELD_POLICY.get("edge_substitutions")
    portable_policy = getattr(policy_entry, "portable_policy", policy_entry)
    if portable_policy is not None and portable_policy is not FieldPolicy.DROP:
        return
    if prerelease_fields_active():
        return
    raise InvalidArgumentError(
        "this trace carries edge-substitution interventions, and the active "
        "schema has no occurrence-granular carrier at ANY save level: the "
        "artifact would present post-edit values with zero edge provenance. "
        "Edge-intervened traces are session-only under such a schema.",
        code="edge_intervention_save_unsupported",
        remedy="analyze in-session, or re-capture without the edge edit before saving",
        carriers=tuple(carriers),
    )


def _refuse_shard_local_erasure(trace: Any) -> None:
    """PERMANENT erasure-prevention invariant for the shard-local disclosure.

    L8/F6 (census plan 3.2b): an ordinary bundle write must NEVER complete if
    it would silently drop the shard-local disclosure -- a saved shard-local
    trace reloading as a plain dense-looking trace is the marker-free-artifact
    class this invariant keeps EMPTY BY CONSTRUCTION. Refuses typed IFF the
    trace carries ``distributed_scope == "rank_local_shard"`` AND the marker
    is not persisted by the active schema (still ``FieldPolicy.DROP`` with the
    S3 pre-release switch inactive). The wave-3 coordinated bump changes the
    ENVIRONMENT, not this predicate: once the policy persists, the second
    conjunct goes false by construction and ordinary saves proceed; the code,
    predicate, and forced-DROP tamper red all REMAIN so any schema regression
    that would re-drop the disclosure re-fires the invariant. Never deleted,
    never a narrowing, no D-ruling owed.
    """

    if (
        trace._provenance_distributed_scope if is_live_trace(trace) else None
    ) != "rank_local_shard":
        return
    from .._data_classes.trace import Trace as _Trace
    from . import FieldPolicy
    from .prerelease import prerelease_fields_active

    policy_entry = _Trace._FIELD_POLICY.get("_provenance_distributed_scope")
    portable_policy = getattr(policy_entry, "portable_policy", policy_entry)
    if portable_policy is not None and portable_policy is not FieldPolicy.DROP:
        return
    if prerelease_fields_active():
        return
    raise InvalidArgumentError(
        "this trace is a shard-local capture (distributed_scope == "
        "'rank_local_shard'), and the active schema does not persist the "
        "shard-local disclosure: an ordinary save would reload as a plain "
        "trace with silently mis-stated parameter geometry. Shard-local "
        "traces are session-only until the coordinated schema bump persists "
        "the marker and its dual-geometry evidence.",
        code="shard_local_persistence_unsupported",
        remedy="analyze in-session; persistence lands with the coordinated schema bump",
    )


# trace.save() defaults for the per-trace payload options the Bundle container
# door does not offer (its surface is exactly path/level/overwrite). An
# EXPLICIT non-default value on a bundle save refuses typed below instead of
# being silently dropped.
_BUNDLE_TRACE_ONLY_SAVE_DEFAULTS: dict[str, bool | None] = {
    "include_outs": None,
    "include_grads": None,
    "include_saved_args": None,
    "include_rng_states": None,
    "include_weights": False,
    "include_activations": False,
    "include_source": True,
    "include_custom_attributes": True,
    "include_buffer_values": None,
    "strict": True,
}


def _save_bundle_via_container_door(
    bundle: Any,
    path: str | Path,
    *,
    level: str,
    overwrite: bool,
    trace_only_options: dict[str, bool | None],
) -> None:
    """Route ``bundle.save(path, ...)`` through the one container door.

    Both public save doors resolve to ``Bundle.save`` (the container door),
    so they agree on every settled bundle by construction: the N1 capture-
    outcome gate runs PER MEMBER inside each member's own save (foldB D8/D9),
    never against the container. Per-trace payload options the container door
    cannot honor refuse typed rather than being silently dropped.

    Parameters
    ----------
    bundle:
        The ``torchlens.bundle.Bundle`` instance passed to ``Trace.save``.
    path:
        Destination ``.tlspec`` directory path.
    level:
        Save level forwarded to the container door.
    overwrite:
        Whether an existing destination may be replaced.
    trace_only_options:
        The received values of every per-trace-only ``Trace.save`` option,
        keyed by parameter name.

    Raises
    ------
    InvalidArgumentError
        ``bundle_save_option_unsupported`` when a per-trace payload option
        was set away from its default.
    """

    unsupported = sorted(
        name
        for name, value in trace_only_options.items()
        if value != _BUNDLE_TRACE_ONLY_SAVE_DEFAULTS[name]
    )
    if unsupported:
        raise InvalidArgumentError(
            "bundle.save(path, ...) routes through the Bundle container door, "
            "which supports only 'level' and 'overwrite'; it cannot honor "
            f"per-trace save options: {', '.join(unsupported)}.",
            code="bundle_save_option_unsupported",
            remedy=(
                "Drop the per-trace options (or call bundle.save(path, "
                "level=..., overwrite=...)); for per-member payload control, "
                "save one member with bundle[name].save(path, ...)."
            ),
            options=tuple(unsupported),
        )
    bundle.save(path, level=level, overwrite=overwrite)


def _refuse_unsupported_tensor(error_text: str, blob_spec: Any, reason: str) -> NoReturn:
    """Refuse a torch tensor the strict (portable) save cannot write.

    Parameters
    ----------
    error_text:
        Refusal text naming the blob's label, kind and the codec's reason.
    blob_spec:
        The refused blob (its ``label`` and ``kind`` ride on the error).
    reason:
        The payload codec's refusal reason.

    Raises
    ------
    ArtifactError
        With ``fields["code"] == "artifact_tensor_unsupported"``.
    """

    raise ArtifactError(
        error_text,
        code="artifact_tensor_unsupported",
        remedy="save at level='audit', or make the tensor dense and strided",
        label=blob_spec.label,
        kind=blob_spec.kind,
        reason=reason,
    )


_PATCH_SOURCE_BOUND = (
    "This capture carries {count} op(s) patched from another run (tl.patch_from); the "
    "patched values belong to the donor run's inputs, not to a function of this run's "
    "inputs, so no runnable artifact can recompute them. Runnable save is refused; keep "
    "the patched fork with the analysis save, trace.save(path) at level 'portable' (the "
    "default) or 'audit', which stores the patched activations as data."
)
_USER_INTERVENTION_REPLACED = (
    "This capture carries {count} user-intervention-replaced op(s); the replacement value "
    "has no traceable function, so a sparse runnable replay would recompute the "
    "UN-intervened computation -- a different function than the captured provenance. "
    "Runnable save is refused; save the intervention spec separately and re-apply it at "
    "capture time, or use an analysis-level save for the intervened artifact."
)


def intervention_replay_diagnostics(
    ops: Iterable[Any], *, diagnostic: Callable[..., Any], code: Any
) -> list[Any]:
    """Producer diagnostics for captured values a runnable replay cannot recompute.

    Ops patched from another run (``tl.patch_from``) are refused first, naming the
    analysis save that keeps the patched fork as data; every other user-intervention
    replacement is refused as a value with no traceable function. ``diagnostic`` is
    the producer's structured-diagnostic constructor and ``code`` the error code both
    refusals carry.
    """

    ops = list(ops)
    patched = tuple(str(op.label) for op in ops if _op_carries_patch_from(op))
    replaced = tuple(
        str(op.label)
        for op in ops
        if bool(getattr(op, "is_edited", False)) and str(op.label) not in patched
    )
    found: list[Any] = []
    if patched:
        found.append(
            diagnostic(
                code,
                _PATCH_SOURCE_BOUND.format(count=len(patched)),
                affected_ops=patched,
                detection_stage="producer_patch_source_bound",
            )
        )
    if replaced:
        found.append(
            diagnostic(
                code,
                _USER_INTERVENTION_REPLACED.format(count=len(replaced)),
                affected_ops=replaced,
                detection_stage="producer_user_intervention",
            )
        )
    return found


def _op_carries_patch_from(op: Any) -> bool:
    """Return whether an edited op's fire records include a ``patch_from`` edit."""

    if not getattr(op, "is_edited", False):
        return False
    for record in getattr(op, "edits", ()) or ():
        names = str(getattr(record, "edit_name", None) or "").split(",")
        if "patch_from" in names:
            return True
        edit = getattr(record, "edit", None)
        if getattr(edit, "name", None) == "patch_from":
            return True
    return False
