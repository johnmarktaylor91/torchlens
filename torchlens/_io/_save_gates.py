"""Save-entry gate helpers for ``torchlens._io.bundle.save``.

Three entry-time guards that run before (or instead of) the per-trace save
pipeline: the two PERMANENT erasure-prevention invariants (edge-substitution
carriers and the shard-local disclosure, both keyed on the ACTIVE SCHEMA so a
regression re-fires them), and the Bundle container-door delegation that
routes ``tl.save(bundle, ...)`` through ``Bundle.save`` (foldB D8/D9). Split
out of ``torchlens/_io/bundle.py`` to respect its size ceiling (R43
split-never-raise); ``bundle.py`` re-exports these names, so the monkeypatch
surface ``torchlens._io.bundle._refuse_*`` is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._errors import InvalidArgumentError


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

    carriers = [
        op.label
        for op in getattr(trace, "layer_list", ()) or ()
        if getattr(op, "edge_substitutions", None)
    ]
    if not carriers:
        return
    from ..data_classes.op import Op as _Op
    from . import FieldPolicy
    from .prerelease import prerelease_fields_active

    policy_entry = _Op.FIELD_POLICY.get("edge_substitutions")
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

    if getattr(trace, "distributed_scope", None) != "rank_local_shard":
        return
    from ..data_classes.trace import Trace as _Trace
    from . import FieldPolicy
    from .prerelease import prerelease_fields_active

    policy_entry = _Trace.FIELD_POLICY.get("distributed_scope")
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


# tl.save() defaults for the per-trace payload options the Bundle container
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
    """Route ``tl.save(bundle, ...)`` through the one container door.

    Both public save doors resolve to ``Bundle.save`` (the container door),
    so they agree on every settled bundle by construction: the N1 capture-
    outcome gate runs PER MEMBER inside each member's own save (foldB D8/D9),
    never against the container. Per-trace payload options the container door
    cannot honor refuse typed rather than being silently dropped.

    Parameters
    ----------
    bundle:
        The ``torchlens.bundle.Bundle`` instance passed to ``tl.save``.
    path:
        Destination ``.tlspec`` directory path.
    level:
        Save level forwarded to the container door.
    overwrite:
        Whether an existing destination may be replaced.
    trace_only_options:
        The received values of every per-trace-only ``tl.save`` option,
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
            "tl.save(bundle, ...) routes through the Bundle container door, "
            "which supports only 'level' and 'overwrite'; it cannot honor "
            f"per-trace save options: {', '.join(unsupported)}.",
            code="bundle_save_option_unsupported",
            remedy=(
                "Drop the per-trace options (or call bundle.save(path, "
                "level=..., overwrite=...)); for per-member payload control, "
                "save one member with tl.save(bundle[name], path, ...)."
            ),
            options=tuple(unsupported),
        )
    bundle.save(path, level=level, overwrite=overwrite)
