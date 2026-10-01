"""``bundle.vary(mapping)`` — one explicit edit or identity PER member (F03 item 5).

The teachable law (ledger memo Decision 2): **do = one edit to all members;
vary = one explicit edit or identity per member; both take the same spec
objects.** Broadcast-versus-vary is the scientific content of the call, and
a silently unedited member is the single most expensive quiet asymmetry an
experiment API can produce — it reads as a null result.

Contract:

- mapping values are :class:`~torchlens.intervention.spec.InterventionSpec`
  objects, ``(selection, edit)`` sugar (normalized immediately to one-clause
  specs), or ``None`` meaning INTENTIONALLY unchanged (recorded as an
  explicit identity outcome).
- COMPLETE COVERAGE by default: unknown, missing, or
  duplicate-after-normalization members refuse typed BEFORE any mutation;
  subsets require explicit ``unmentioned="unchanged"``.
- Donor-plan normalization runs ONCE across the whole mapping before members
  diverge (the edits panel's D8 contract, honored via
  :func:`~torchlens.intervention.stochastic.assign_batch_donor_groups` —
  one reused SamplingPlan object keeps one persisted ``donor_group_id``
  across members; separately constructed plans never share).
- A mid-apply runtime failure claims NO rollback: the per-member
  completed / identity / failed / unstarted outcomes land in the operation
  chronology row and the typed ``vary_partial_failure`` refusal carries
  ``material_action_completed=True`` with the original failure chained.
- Returns the receiver (mutator law); ``bundle.fork().vary(...)`` is the
  preserving spelling; broadcast ``do`` is untouched forever.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..errors.episode import BundleExperimentError

if TYPE_CHECKING:
    from . import Bundle

__all__ = ["bundle_vary"]


def _normalize_mapping(bundle: Bundle, mapping: Mapping[Any, Any]) -> dict[str, Any]:
    """Coerce keys to member names and refuse duplicates, typed."""

    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        try:
            name = bundle._coerce_member_name(key)
        except (KeyError, ValueError, TypeError) as exc:
            raise BundleExperimentError(
                f"vary() mapping names a non-member: {key!r} ({exc})",
                code="vary_member_unknown",
                member=str(key),
            ) from exc
        if name not in bundle._members:
            raise BundleExperimentError(
                f"vary() mapping names a non-member: {name!r}; current members "
                f"are {list(bundle._members)}",
                code="vary_member_unknown",
                member=name,
            )
        if name in normalized:
            raise BundleExperimentError(
                f"vary() mapping names member {name!r} more than once after "
                "normalization (e.g. once by name and once by Trace reference); "
                "one explicit edit or identity per member.",
                code="vary_member_duplicate",
                member=name,
            )
        normalized[name] = value
    return normalized


def _normalize_values(assignments: dict[str, Any]) -> dict[str, Any]:
    """Lower sugar pairs to one-clause specs after ONE donor normalization pass.

    ``None`` stays the identity sentinel; specs pass through unchanged; a
    ``(selection, edit)`` pair lowers to ``when(selection, edit)`` AFTER the
    whole mapping's pairs ride one ``assign_batch_donor_groups`` call, so
    one reused SamplingPlan object yields one donor group across members.
    """

    from ..intervention.spec import InterventionSpec, when
    from ..intervention.stochastic import assign_batch_donor_groups

    pair_names = [
        name for name, value in assignments.items() if isinstance(value, tuple) and len(value) == 2
    ]
    normalized_pairs = assign_batch_donor_groups([assignments[name] for name in pair_names])
    lowered = dict(assignments)
    for name, (selection, edit) in zip(pair_names, normalized_pairs, strict=True):
        lowered[name] = when(selection, edit)
    for name, value in lowered.items():
        if value is None or isinstance(value, InterventionSpec):
            continue
        raise BundleExperimentError(
            f"vary() value for member {name!r} must be an InterventionSpec, a "
            f"(selection, edit) pair, or None (explicit identity); got "
            f"{type(value).__name__}. do() broadcasts one edit; vary() names "
            "one edit or identity per member.",
            code="vary_mapping_invalid",
            member=name,
            received_type=type(value).__name__,
        )
    return lowered


def _apply_member_specs(
    bundle: Bundle,
    lowered: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str | None], tuple[str, BaseException] | None]:
    """Apply one lowered spec (or identity) per member; stop at the first failure."""

    from ..intervention.audit import append_event_audit_row

    outcomes: dict[str, str] = {
        name: ("unstarted" if name in lowered else "unmentioned") for name in bundle._members
    }
    spec_digests: dict[str, str | None] = {}
    failure: tuple[str, BaseException] | None = None
    for name in bundle._members:
        value = lowered.get(name)
        if name not in lowered:
            continue
        if value is None:
            outcomes[name] = "identity"
            spec_digests[name] = None
            continue
        member = bundle._members[name]
        try:
            member.do(value)
            events = [
                row
                for row in getattr(member, "state_history", ())
                if isinstance(row, dict) and row.get("op") == "intervention_event"
            ]
            if events:
                # The experiment layer opts this transaction into the
                # canonical v9 EVENT audit row (hash-chained).
                append_event_audit_row(member, events[-1])
            outcomes[name] = "completed"
            spec_digests[name] = value.spec_digest
        except Exception as exc:  # noqa: BLE001 - disclosed partial outcome
            outcomes[name] = "failed"
            spec_digests[name] = getattr(value, "spec_digest", None)
            failure = (name, exc)
            break
    return outcomes, spec_digests, failure


def bundle_vary(
    bundle: Bundle,
    mapping: Mapping[Any, Any],
    *,
    unmentioned: str | None = None,
) -> Bundle:
    """Apply one explicit spec (or explicit identity) per member.

    Parameters
    ----------
    bundle:
        The receiver (mutating spelling; ``bundle.fork().vary(...)``
        preserves the source).
    mapping:
        ``{member: InterventionSpec | (selection, edit) | None}``.
    unmentioned:
        ``None`` (default) demands COMPLETE coverage — a member absent from
        the mapping refuses typed before any mutation. The explicit
        ``"unchanged"`` opts a subset call in; unmentioned members are
        recorded as unmentioned (distinguishable from explicit ``None``
        identity after load).

    Returns
    -------
    Bundle
        The receiver.

    Raises
    ------
    BundleExperimentError
        Typed preflight refusals before any mutation
        (``vary_mapping_invalid`` / ``vary_member_unknown`` /
        ``vary_member_duplicate`` / ``vary_coverage_incomplete``), or the
        disclosed mid-apply ``vary_partial_failure`` carrying
        ``material_action_completed=True`` and the per-member outcomes with
        the original failure chained.
    """

    if not isinstance(mapping, Mapping):
        raise BundleExperimentError(
            f"vary() takes a mapping of member -> spec/None, got {type(mapping).__name__}",
            code="vary_mapping_invalid",
            received_type=type(mapping).__name__,
        )
    if unmentioned not in (None, "unchanged"):
        raise BundleExperimentError(
            f"vary(unmentioned=...) accepts only 'unchanged', got {unmentioned!r}",
            code="vary_mapping_invalid",
        )
    assignments = _normalize_mapping(bundle, mapping)
    missing = [name for name in bundle._members if name not in assignments]
    if missing and unmentioned != "unchanged":
        raise BundleExperimentError(
            f"vary() mapping omits members {missing}: complete coverage is the "
            "default because a silently unedited member reads as a null result. "
            "Name every member (None = explicit identity), or pass "
            "unmentioned='unchanged' to opt the subset call in.",
            code="vary_coverage_incomplete",
            missing_members=missing,
        )
    lowered = _normalize_values(assignments)

    # Emission (item 9b): ONE material step per vary call, started BEFORE
    # material work; per-member outcomes live in the MATERIAL record (the
    # chronology row), never as per-member ledger steps. Unarmed = zero rows.
    from ..experiment._ledger import active_ledger

    armed = active_ledger()
    step_id = armed.material_step("vary") if armed is not None else None

    # ---- material phase (no rollback claimed) ----------------------------
    outcomes, spec_digests, failure = _apply_member_specs(bundle, lowered)

    operation = bundle._record_bundle_operation(
        "vary",
        member_names=tuple(lowered),
        params={
            "outcomes": dict(outcomes),
            "spec_digests": dict(spec_digests),
            "unmentioned": unmentioned,
        },
    )
    for name, outcome in outcomes.items():
        if outcome != "completed":
            continue
        anchor = dict(bundle._member_construction.get(name, {}))
        anchor["origin"] = "varied"
        anchor["operation_id"] = operation.operation_id
        bundle._member_construction[name] = anchor

    if armed is not None and step_id is not None:
        from ..experiment._ledger import EvidenceRef

        armed.finalize_step(
            step_id,
            outcome="completed" if failure is None else "partial_failure",
            refs=[
                EvidenceRef(
                    kind="bundle",
                    uri=f"live://{bundle.bundle_id}",
                    object_id=f"{bundle.bundle_id}:{operation.operation_id}",
                )
            ],
            material_result=bundle,
        )
    if failure is not None:
        failed_member, original = failure
        raise BundleExperimentError(
            f"vary() failed at member {failed_member!r} after completing "
            f"{sum(1 for value in outcomes.values() if value == 'completed')} member(s): "
            f"{type(original).__name__}: {original}. NO rollback is claimed — the "
            "per-member outcomes are recorded on the bundle's operation "
            "chronology (completed members keep their edits; later members are "
            "unstarted).",
            code="vary_partial_failure",
            failed_member=failed_member,
            outcomes=dict(outcomes),
            material_action_completed=True,
            operation_id=operation.operation_id,
        ) from original
    return bundle
