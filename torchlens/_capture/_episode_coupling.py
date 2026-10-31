"""Attested episode-intervention coupling (lane F42).

The trace-verb verdict's evidence bar for the ``episode=`` x ``intervene=``
COUPLED cell: the ledger is BOUND to its exact product (the capture digest,
consumed here), a perturbed run records an intervention digest and per-step
fire counts (zero and multiple fires are first-class facts), replay derives
a fresh ledger or refuses, and facts across a broken join scope PER SEGMENT
-- never one whole-episode claim spanning a break.

The live half is one funnel note: every live ``FireRecord`` lands through
``intervention.runtime._append_active_spec_records``, which notifies the one
armed coupling session; the session attributes each fire to the live step
position (the F40c join session's boundary hooks) or to the outside-step
bucket. The settlement half writes the reserved C07X slots (header
``intervention_digest``, row ``fire_count``, ``fidelity_basis="perturbed"``)
-- a behavior change on reserved slots, never a schema break.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session; the
refusal codes are the stable branch surface. Doc of record:
``docs/reference/episode_capture.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "INTERVENTION_DIGEST_SCHEMA",
    "CouplingAttestation",
    "CouplingFire",
    "CouplingSegment",
    "CouplingSession",
    "active_coupling_session",
    "attest_coupling",
    "coupling_segments",
    "mint_intervention_digest",
    "quarantine_episode_after_perturbed_replay",
    "refuse_coupled_replay",
]

#: Schema tag hashed into the intervention digest's canonical encoding.
INTERVENTION_DIGEST_SCHEMA = "episode_intervention_digest_v1"

#: The one armed coupling session of the running capture (single-threaded by
#: design; ``_episode_join.armed_capture`` arms and clears it beside the join
#: session).
_ACTIVE_COUPLING: CouplingSession | None = None


def active_coupling_session() -> CouplingSession | None:
    """Return the armed coupling session of the running episode capture."""

    return _ACTIVE_COUPLING


def _set_active_coupling(session: CouplingSession | None) -> None:
    """Install/clear the armed coupling session (armed_capture's seam)."""

    global _ACTIVE_COUPLING
    _ACTIVE_COUPLING = session


@dataclass(frozen=True)
class CouplingFire:
    """One live intervention firing, attributed to its episode position.

    ``step`` is the 0-based episode step the fire executed inside, or
    ``None`` for root-loop fires between/before/after steps (the
    outside-step bucket -- disclosed, never guessed into a row).
    """

    step: int | None
    site: str
    helper: str | None
    timing: str | None
    direction: str | None
    replaced: bool


@dataclass
class CouplingSession:
    """Live fire-attribution session for one intervened episode capture.

    Armed for exactly one capture beside the join session; the funnel note
    (:func:`note_fire_records` via ``_append_active_spec_records``) is the
    ONE ingestion point, so every engine that mints a live ``FireRecord``
    is counted without per-engine seams.

    Parameters
    ----------
    rule_identity:
        Deterministic identity rows of the armed intervention --
        ``(rule_id, where_repr)`` pairs from an ``InterventionSpec``, or
        ``()`` for opaque callable predicates (disclosed by absence).
    """

    rule_identity: tuple[tuple[str, str], ...] = ()
    fires: list[CouplingFire] = field(default_factory=list)

    def reset(self) -> None:
        """Drop recorded fires (a fresh capture pass is starting)."""

        self.fires = []

    def note_fire_records(self, records: Sequence[Any]) -> None:
        """Attribute freshly minted live fire records to the live step.

        Reads the join session's live step position ONCE per note; a
        measurement failure never raises (the fire stays counted, in the
        outside-step bucket).
        """

        if not records:
            return
        try:
            from ._episode_join import active_episode_step

            step = active_episode_step()
        except Exception:  # noqa: BLE001 - attribution degrades, never raises
            step = None
        for record in records:
            self.fires.append(
                CouplingFire(
                    step=step,
                    site=str(
                        getattr(record, "site_label", None) or getattr(record, "target_label", "")
                    ),
                    helper=getattr(record, "helper_name", None),
                    timing=getattr(record, "timing", None),
                    direction=getattr(record, "direction", None),
                    replaced=bool(getattr(record, "replaced", False)),
                )
            )

    # -- settlement reads ----------------------------------------------------

    def fire_counts(self) -> dict[int, int]:
        """Per-step fire counts (steps with zero fires are absent here)."""

        counts: dict[int, int] = {}
        for fire in self.fires:
            if fire.step is not None:
                counts[fire.step] = counts.get(fire.step, 0) + 1
        return counts

    @property
    def outside_step_fires(self) -> int:
        """Fires attributed to no step (root-loop before/between/after)."""

        return sum(1 for fire in self.fires if fire.step is None)

    @property
    def any_replaced(self) -> bool:
        """Whether any fire actually replaced a value (a perturbation)."""

        return any(fire.replaced for fire in self.fires)


def rule_identity_of(intervene: Any) -> tuple[tuple[str, str], ...]:
    """Extract the deterministic rule identity of an ``intervene=`` operand.

    ``tl.when``/``InterventionSpec`` rules carry ``rule_id`` and
    ``where_repr``; opaque callables yield ``()`` -- the digest then rests on
    the fire facts alone, and the attestation discloses the absent identity.
    """

    rules = getattr(intervene, "rules", None)
    if not rules:
        return ()
    identity: list[tuple[str, str]] = []
    for rule in rules:
        rule_id = getattr(rule, "rule_id", None)
        where_repr = getattr(rule, "where_repr", None)
        if rule_id is None and where_repr is None:
            continue
        identity.append((str(rule_id), str(where_repr)))
    return tuple(identity)


def mint_intervention_digest(session: CouplingSession) -> str:
    """Mint the intervention digest binding fire evidence to the product.

    Hex SHA-256 over the canonical-JSON encoding of the armed rule identity
    and the ORDERED fire facts (step, site, helper, timing, direction,
    replaced) plus the outside-step count. Every input is deterministic for
    a fixed capture (raw site labels follow capture order; no timestamps),
    so an identical re-run mints the identical digest -- the "assert digest
    identity, never output equality alone" test law.
    """

    canonical = json.dumps(
        {
            "schema": INTERVENTION_DIGEST_SCHEMA,
            "rules": [list(row) for row in session.rule_identity],
            "fires": [
                [fire.step, fire.site, fire.helper, fire.timing, fire.direction, fire.replaced]
                for fire in session.fires
            ],
            "outside_step_fires": session.outside_step_fires,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stamp_episode_steps(trace: Any, calls: list[Any]) -> None:
    """Write the ``Op.episode_step`` read axis (tlspec v9 entry-dark slot).

    Every op recorded inside stepped-module call ``k`` (nested submodule ops
    included -- ``ModuleCall.ops`` lists the call's whole pass-qualified op
    set) is stamped with 0-based step ``k``; root-loop ops between steps keep
    ``None``. The step-qualified selectors and per-segment facts read this
    axis post hoc; load validation (``_io/_forgery_identity_facts.py``)
    admits stamps only on episode products. Best-effort per op: a label that
    no longer resolves (orphan removal) is skipped, never a settlement
    failure.
    """

    ops_accessor = getattr(trace, "ops", None)
    if ops_accessor is None:
        return
    for step, call in enumerate(calls):
        for op_label in getattr(call, "ops", ()) or ():
            try:
                ops_accessor[op_label].episode_step = step
            except (LookupError, AttributeError, TypeError, ValueError):
                continue


def settlement_fire_counts(
    session: CouplingSession | None, *, started: int, n_total: int
) -> list[int | None] | None:
    """Derive the per-row ``fire_count`` column at settlement.

    ``None`` when no coupling session ran (uncoupled episodes keep the
    entry-dark ``None`` slots). Started rows carry MEASURED counts -- zero
    fires is a first-class fact (``0``, never ``None``); rows that never
    started carry ``None`` (no opportunity, not a measured zero).
    """

    if session is None:
        return None
    counts = session.fire_counts()
    return [counts.get(step, 0) if step < started else None for step in range(n_total)]


# ---------------------------------------------------------------------------
# Per-segment facts (pure derivation; nothing new persists)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouplingSegment:
    """One contiguous run of steps not spanning a measured break join.

    Derived from the persisted join envelope and rows -- never persisted
    itself. ``join_basis`` is ``"measured"`` when the envelope exists and
    ``"unmeasured"`` otherwise (an unmeasured episode yields ONE undivided
    segment that says so). ``unchecked_joins`` counts interior joins whose
    continuity could not be measured: a segment with a nonzero count is
    disclosed, not fully attested. ``fire_count_total`` is ``None`` unless
    every member row carries a measured count.
    """

    start_step: int
    end_step: int
    join_basis: str
    unchecked_joins: int
    fire_count_total: int | None


def _rows_and_envelope(payload: Mapping[str, Any]) -> tuple[list[Any], Any]:
    """Split one episode payload into its row list and join envelope."""

    rows = payload.get("rows")
    header = payload.get("header")
    envelope = header.get("step_join") if isinstance(header, Mapping) else None
    return (list(rows) if isinstance(rows, Sequence) else [], envelope)


def coupling_segments(subject: Any) -> tuple[CouplingSegment, ...]:
    """Derive the per-segment facts of one episode product (lane F42).

    Segments split at measured BREAK joins (``exogenous``/``declared``
    grades): a claim never spans a break. Rows that never started
    (``absent``) close the started prefix. Products without an episode
    ledger yield ``()``.
    """

    payload = _episode_payload(subject)
    if payload is None:
        return ()
    rows, envelope = _rows_and_envelope(payload)
    started_rows = [row for row in rows if row.get("status") != "absent"]
    if not started_rows:
        return ()
    grades: list[Any] = [None] * len(rows)
    basis = "unmeasured"
    if isinstance(envelope, Mapping):
        basis = "measured"
        raw_grades = envelope.get("grades")
        if isinstance(raw_grades, Sequence):
            for index, grade in enumerate(raw_grades):
                if index < len(grades):
                    grades[index] = grade
    segments: list[CouplingSegment] = []
    start = 0
    for boundary in range(1, len(started_rows) + 1):
        at_end = boundary == len(started_rows)
        breaks = (not at_end) and grades[boundary] in ("exogenous", "declared")
        if not (at_end or breaks):
            continue
        member_rows = started_rows[start:boundary]
        counts = [row.get("fire_count") for row in member_rows]
        total = None if any(count is None for count in counts) else int(sum(counts))
        unchecked = sum(1 for join in range(start + 1, boundary) if grades[join] == "unchecked") + (
            0 if basis == "measured" else max(len(member_rows) - 1, 0)
        )
        segments.append(
            CouplingSegment(
                start_step=start,
                end_step=boundary - 1,
                join_basis=basis,
                unchecked_joins=unchecked,
                fire_count_total=total,
            )
        )
        start = boundary
    return tuple(segments)


# ---------------------------------------------------------------------------
# Attestation (consumes the capture digest; the positive binding claim)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouplingAttestation:
    """The derived coupling verdict of one episode product.

    ``bound`` is the POSITIVE claim "this ledger belongs to this product":
    the capture digest recomputed from the product's own recomputable facts
    equals the persisted binding (an unbound product REFUSES before this
    object exists, so a held attestation is always bound). ``coupled`` is
    whether intervention evidence rides the ledger at all.
    """

    coupled: bool
    bound: bool
    capture_digest: str
    intervention_digest: str | None
    fidelity_basis: str | None
    fire_counts: tuple[int | None, ...]
    outside_step_fires: int | None
    segments: tuple[CouplingSegment, ...]
    #: W051 (audit 3.2): whether the evidence column was re-derived from the
    #: product's RETAINED root output and matched (``True``); ``None`` when
    #: nothing could be compared (kind ``none``, unretained payload) -- the
    #: binding then rests on identity + ledger content alone, disclosed.
    evidence_rederived: bool | None = None


def _episode_payload(subject: Any) -> Mapping[str, Any] | None:
    """Return the finalized episode payload of a Trace-like, or ``None``."""

    annotations = getattr(subject, "annotations", None)
    if not isinstance(annotations, dict):
        return None
    payload = annotations.get("episode")
    if isinstance(payload, Mapping) and "rows" in payload and "header" in payload:
        return payload
    return None


def _coupling_error(message: str, *, code: str, **payload: Any) -> Exception:
    """Build the typed coupling refusal (codes ride ``exc.fields``)."""

    from ..errors.episode import EpisodeCaptureError

    return EpisodeCaptureError(message, code=code, **payload)


def attest_coupling(subject: Any) -> CouplingAttestation | None:
    """Attest one episode product's ledger-to-product binding (lane F42).

    Recomputes the capture digest from the product's own persisted facts
    (ordered recorded op labels, stepped-module call count, entry seed,
    stepped address -- the F40b minting inputs) and compares it to the
    header's binding; equality is the positive "this ledger belongs to this
    product" claim. Returns ``None`` for products without a finalized
    episode ledger (mirrors ``Trace.episode``).

    Raises
    ------
    EpisodeCaptureError
        ``episode_coupling_unbound`` when the recomputed digest disagrees
        with the persisted one (a foreign/tampered ledger -- the exact
        wrongness the binding exists to catch), and
        ``episode_coupling_unmintable`` when the header predates the digest
        (no binding was ever minted; re-capture to bind).
    """

    payload = _episode_payload(subject)
    if payload is None:
        return None
    header = payload["header"]
    persisted = header.get("capture_digest")
    if not isinstance(persisted, str) or not persisted:
        raise _coupling_error(
            "this episode ledger carries no capture digest (a pre-binding "
            "artifact), so the ledger-to-product binding cannot be attested. "
            "Re-capture with this TorchLens version to mint the binding.",
            code="episode_coupling_unmintable",
        )
    from ._episode_derivation import recompute_capture_digests, rederived_evidence_matches

    recomputed, legacy = recompute_capture_digests(subject, payload)
    if recomputed != persisted:
        if legacy == persisted:
            raise _coupling_error(
                "this episode ledger carries a pre-content-binding capture "
                "digest (the v1 identity-only form), so the ledger content "
                "cannot be attested against the product. Re-capture with this "
                "TorchLens version to mint the content-binding digest.",
                code="episode_coupling_unmintable",
            )
        raise _coupling_error(
            "this episode ledger's capture digest does not match the digest "
            "recomputed from the product it rides and the ledger's own "
            "content: the ledger does not describe this execution (a foreign "
            "or rewritten ledger). Re-capture, or recover the original product.",
            code="episode_coupling_unbound",
            persisted_digest=persisted,
            recomputed_digest=recomputed,
        )
    # The digest binds content; the re-derivation binds that content to the
    # product's own retained values (a re-minted foreign ledger over an
    # identical program -- two equal-length prompts -- is caught here).
    evidence_rederived = rederived_evidence_matches(subject, payload)
    if evidence_rederived is False:
        raise _coupling_error(
            "this episode ledger's evidence column does not re-derive from the "
            "product's retained root output: the ledger does not describe this "
            "execution (a foreign ledger over an identical program). Re-capture, "
            "or recover the original product.",
            code="episode_coupling_unbound",
            persisted_digest=persisted,
            recomputed_digest=recomputed,
        )
    rows = payload.get("rows") or []
    fire_counts = tuple(row.get("fire_count") for row in rows)
    intervention_digest = header.get("intervention_digest")
    return CouplingAttestation(
        coupled=intervention_digest is not None,
        bound=True,
        capture_digest=persisted,
        intervention_digest=intervention_digest,
        fidelity_basis=header.get("fidelity_basis"),
        fire_counts=fire_counts,
        # Outside-step fires (before/between/after the stepped calls) have
        # no reserved persisted slot: they are folded into the intervention
        # digest's canonical form (identity-tested) and readable live on the
        # CouplingSession; post hoc this reads None -- NOT MEASURED-HERE, a
        # disclosed residual, never a zero claim.
        outside_step_fires=None,
        segments=coupling_segments(subject),
        evidence_rederived=evidence_rederived,
    )


# ---------------------------------------------------------------------------
# Replay doors (derive a fresh ledger or refuse; never a foreign ledger)
# ---------------------------------------------------------------------------


def refuse_coupled_replay(subject: Any) -> None:
    """Refuse ``run()`` on a COUPLED episode product (both engines).

    A coupled product's ledger attests ONE perturbed execution; the run
    providers re-execute WITHOUT re-arming the capture-time intervention,
    so no engine can both reapply the intervention and derive a fresh
    ledger. Per the verdict, replay derives a fresh ledger or refuses --
    this is the refusal arm, and re-capturing with
    ``tl.trace(model, x, episode=..., intervene=...)`` IS the fresh-ledger
    arm. Uncoupled episode products keep their shipped replay behavior
    (the travel policy quarantines the stale evidence on the product).
    """

    payload = _episode_payload(subject)
    if payload is None:
        return
    if payload["header"].get("intervention_digest") is None:
        return
    raise _coupling_error(
        "this is a COUPLED episode product (episode= x intervene=): its "
        "ledger attests one perturbed execution, and run() cannot reapply "
        "the capture-time intervention, so it cannot derive a fresh ledger "
        "for the re-execution. Re-capture with tl.trace(model, x, "
        "episode=..., intervene=...) to produce a fresh coupled product, or "
        "capture without intervene= for a replayable episode.",
        code="episode_coupled_replay_underivable",
    )


def quarantine_episode_after_perturbed_replay(fork: Any) -> None:
    """Quarantine episode evidence on a ``do()``-edited fork (lane F42).

    ``fork.do(...)`` replays a perturbed cone in place: the fork's inherited
    episode ledger then describes the ORIGINAL execution while values are
    perturbed -- exactly the foreign-evidence shape the travel policy closes
    on the run providers. The edit commits; the evidence quarantines to the
    same inert note grammar (a diagnostic token, never raised, outside the
    exception code contract).
    """

    payload = _episode_payload(fork)
    if payload is None:
        return
    annotations = fork.annotations
    if isinstance(payload, Mapping) and payload.get("quarantined") is True:
        return
    rows = payload.get("rows")
    n_rows = len(rows) if isinstance(rows, Sequence) else 0
    annotations["episode"] = {
        "quarantined": True,
        "code": "episode_evidence_dropped_perturbed_replay",
        "detail": (
            f"episode step evidence ({n_rows} rows) described the original "
            "execution; a do() edit replayed a perturbed cone on this fork, "
            "so the evidence was quarantined. Re-capture with episode= and "
            "intervene= for a coupled product."
        ),
    }
