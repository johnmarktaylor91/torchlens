"""Episode ledger: per-step status disclosure for ``capture_kind=episode``.

An EPISODE is one wrapped session capture product (the L2 SESSION ruling,
ratified at S2 2026-08-16): a wrapper ``nn.Module`` whose ``forward`` steps a
declared STEPPED MODULE N times is captured as ONE product carrying
``capture_kind=episode`` and this ledger ON the product, attached at
``trace.annotations["episode"]``. The ledger is a DISCLOSURE, never a
settlement authority: the product's one settled ``CaptureOutcome`` is written
by the existing authority (``torchlens.capture.outcome``) with its vocabulary
unchanged, and no derivation here may bless COMPLETE or revise a settled
record (the R06 discipline).

Pattern-of-record: :mod:`torchlens.distributed._ledger` (S7 instruction) —
closed ``Literal`` vocabularies, frozen payload key sets, fail-closed
``from_payload``, write-once container, derived views that refuse
out-of-vocabulary tokens. Parse boundaries raise ``ValueError``; callers
convert into the typed episode refusal family (stable codes, see
:mod:`torchlens.errors`).

Schema of record: the S6/S7 drafts in the L2 spike memo, BINDING as of the
S2 ratification. Every spelling here is DOCUMENTED-UNSTABLE pending the
rolling naming session (spike section 6.5); semantics are pinned.

PERSISTENCE: ``annotations["episode"]`` persists plainly as of the
coordinated tlspec v8 bump (the pre-release registrar retired empty); loads
validate the ledger payload fail-closed (illegal attachment refuses
``episode_ledger_without_declaration``, geometry violations quarantine
``episode_ledger_incoherent``), and a grammar-v1 payload quarantines typed
rather than normalizing (the C07X grammar-v2 amendment).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from ._episode_derivation import (
    _call_returned,
    _declaration_error,
    _derive_step_column,
    _derive_step_evidence,  # noqa: F401 -- the four-clause price probe imports it from here
    _pass_range_for_call,
    mint_capture_digest,
    validate_step_output_positions,
)
from ._episode_ledger_anchors import _anchor_loaded_ledger, quarantine_loaded_ledger

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ..options import EpisodeSpec

__all__ = [
    "CAPTURE_KIND_EPISODE",
    "EPISODE_ANNOTATIONS_KEY",
    "EPISODE_DECLARED_STEP_CEILING",
    "EPISODE_STEP_JOIN_MEASURED",
    "STEP_JOIN_UNMEASURED",
    "EpisodeFoldResult",
    "EpisodeLedger",
    "EpisodeLedgerHeader",
    "EpisodeLedgerRow",
    "ResolvedEpisode",
    "attach_episode_header",
    "capture_kind_for",
    "derive_episode_status",
    "episode_ledger_for",
    "episode_step_join_claim",
    "mint_capture_digest",
    "producer_digest",
    "resolve_episode_declaration",
    "row_status_for_member_outcome",
    "row_status_for_recording_status",
    "validate_loaded_episode_annotations",
    "write_episode_ledger",
]

#: Annotations key the ledger (header + rows) lives under on the Trace.
EPISODE_ANNOTATIONS_KEY = "episode"

#: Lane F40c landed the live + settlement ``step_join`` measurement, flipping
#: the F40a claim-off switch: NEW episode captures measure cross-step
#: continuity through the boundary hooks (:mod:`._episode_join`) and persist
#: the ``episode_step_join_v1`` envelope in the header's reserved slot. The
#: claim stays PER-ARTIFACT: a ledger without the envelope (pre-measurement
#: artifact, failed live measurement) still reads UNMEASURED on every surface
#: -- the build-level switch never upgrades an artifact's own evidence.
EPISODE_STEP_JOIN_MEASURED = True

#: The closed step-join claim vocabulary served while the switch is off.
STEP_JOIN_UNMEASURED = "unmeasured"


def episode_step_join_claim() -> str:
    """Return the BUILD-level step-join claim token (F40a claim-off door).

    ``"measured"`` as of lane F40c (the join measurement is behind it).
    Surfaces presenting one ARTIFACT's joins must read that artifact's own
    ``step_join`` envelope (absent = unmeasured), never this build token.
    """

    return "measured" if EPISODE_STEP_JOIN_MEASURED else STEP_JOIN_UNMEASURED


#: Declared-step cost ceiling for the wrapped episode tier (lane F40a).
#: The wrapped tier's cost is SUPERLINEAR in step count -- measured on
#: gpt2-124M (CPU): N=20 = 79 s / 1.9 GB peak RSS; N=100 = 657 s (323x
#: native) / 5.4 GB. A declaration beyond this ceiling refuses typed
#: (``episode_step_ceiling_exceeded``) at declaration time unless the
#: EpisodeSpec carries the explicit ``acknowledge_step_cost=True`` override,
#: so hundreds-of-steps wrapped captures are a decision, never an accident.
EPISODE_DECLARED_STEP_CEILING = 100

#: The capture-kind marker value stamped in the ledger header.
CAPTURE_KIND_EPISODE = "episode"

#: Episode-family grammar version (foldA s5 item 3(v), the C07X amendment).
#: Grammar v2 is the row grammar of record: declared step-output kind +
#: source, generic step-output/step-axis names, ``cache_len`` DELETED for
#: the measured carried-state witness slots, capture-digest and reserved
#: coupling slots. Future grammar changes bump THIS field (family-local),
#: never the tlspec version; a payload of any other version quarantines at
#: load (``episode_ledger_incoherent`` — the cross-grammar quarantine row).
EPISODE_LEDGER_VERSION = 2

StepOutputKind = Literal["tokens", "digest", "none"]
"""Declared step-output kind (foldA D8; default ``tokens``). The evidence
derivation is DECLARATION-DRIVEN (lane F40b): ``tokens`` reads per-step
emitted token ids from the declared source, ``digest`` reads per-step
content digests of the declared source (any dtype -- the float-root /
hidden-state shape), and ``none`` derives no evidence at all (a status-only
ledger; the diffusion shape whose root output has no per-step axis)."""

_STEP_OUTPUT_KINDS = frozenset({"tokens", "digest", "none"})

Role = Literal["prefill", "decode"]
"""Row role: row 0 is the prefill (the stepped module's call 1); later rows decode."""

RowStatus = Literal["complete", "interrupted", "absent"]
"""Closed row-status vocabulary (S7).

``complete`` is row-scoped FORWARD-STEP truth (did step k's forward return),
never a claim about the PRODUCT — product truth is the settled
``CaptureOutcome``. ``interrupted`` means the settled outcome's frontier lies
INSIDE step k. ``absent`` means the step never started.
"""

EscalationReason = Literal["step_failed", "divergence", "requested"]
"""Closed vocabulary for the escalation disclosure (E-A2)."""

FidelityBasis = Literal["tokens", "diverged", "none", "forced", "perturbed"]
"""Token-fidelity disclosure basis (E-A3; ``forced`` = teacher-forced feed,
an explicitly NON-VERIFYING disclosed mode — never a settlement input).
``perturbed`` is the coupling basis (foldA s5 item 3(iii), lane F42):
written when a fired intervention REPLACED a value -- the evidence column is
true of the perturbed execution and of nothing else."""

TokenFeed = Literal["free", "forced"]
"""Whether the episode's step inputs were model-emitted (``free``) or
teacher-forced from a declared token sequence (``forced``)."""

ProvenanceTier = Literal["exact", "ledger_only"]
"""Per-step provenance tier (S2 combination table). A session product is
uniformly ``exact``; mixed tiers arise only in the floor, where the episode
tier is the MINIMUM of its members' tiers, never the maximum."""

#: Derived episode-status terms for the floor fold (section 2.3). PROVISIONAL
#: spellings; these are derivations, never settled CaptureOutcome values.
EpisodeFoldStatus = Literal[
    "episode_complete",
    "episode_halted_at_step",
    "episode_aborted_at_step",
    "episode_failed_at_step",
    "episode_unknown",
]

_ROLES = frozenset({"prefill", "decode"})
_ROW_STATUSES = frozenset({"complete", "interrupted", "absent"})
_ESCALATION_REASONS = frozenset({"step_failed", "divergence", "requested"})
_FIDELITY_BASES = frozenset({"tokens", "diverged", "none", "forced", "perturbed"})
_TOKEN_FEEDS = frozenset({"free", "forced"})
_PROVENANCE_TIERS = frozenset({"exact", "ledger_only"})

# ``to_payload`` emits exactly these keys; an unknown key in a loaded payload
# is a forged or drifted artifact, never something to silently ignore.
_HEADER_PAYLOAD_KEYS = frozenset(
    {
        "episode_ledger_version",
        "episode_id",
        "capture_kind",
        "stepped_module",
        "n_steps_declared",
        "entry_seed",
        "token_feed",
        "provenance_tier",
        "structure_only",
        "escalated_from",
        "reason",
        "fidelity_basis",
        "step_output_kind",
        "step_output_from",
        "step_axis",
        "step_output_positions",
        "step_join",
        "capture_digest",
        "intervention_digest",
    }
)
# Optional at LOAD (W051 FIX2): pre-disclosure v2 artifacts lack the key (None = tail).
_OPTIONAL_HEADER_PAYLOAD_KEYS = frozenset({"step_output_positions"})
_ROW_PAYLOAD_KEYS = frozenset(
    {
        "episode_step",
        "role",
        "coord",
        "step_output",
        "status",
        "frontier",
        "entry_state_digest",
        "exit_state_digest",
        "fire_count",
        "rng_digest",
        "escalation",
    }
)
_FRONTIER_KEYS = frozenset({"boundary_kind", "boundary_label"})
_COORD_KEYS = frozenset({"member_call_index", "pass_range", "member"})


def _require_vocabulary(payload: Mapping[str, Any], key: str, vocabulary: frozenset[str]) -> str:
    """Return a required closed-vocabulary field, refusing anything else."""

    value = payload.get(key)
    if value not in vocabulary:
        raise ValueError(
            f"episode-ledger field {key!r} is {value!r}, outside the closed "
            f"vocabulary {sorted(vocabulary)}"
        )
    return str(value)


def _require_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    """Return a required non-negative integer field, refusing anything else."""

    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"episode-ledger field {key!r} must be a non-negative int")
    return value


def _optional_non_negative_int(payload: Mapping[str, Any], key: str) -> int | None:
    """Return an optional non-negative integer field, refusing other types."""

    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"episode-ledger field {key!r} must be a non-negative int or null")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    """Return an optional string field, refusing a non-string value."""

    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"episode-ledger field {key!r} must be a string or null")
    return value


def _require_string_keyed_mapping(value: Any, *, field_name: str) -> None:
    """Envelope-shape check: a mapping with non-empty string keys."""

    if not isinstance(value, Mapping):
        raise ValueError(f"episode-ledger field {field_name!r} must be a mapping or null")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ValueError(f"episode-ledger field {field_name!r} keys must be non-empty strings")


def _parse_state_digest(payload: Mapping[str, Any], key: str) -> dict[str, str] | None:
    """Parse one channel-keyed carried-state digest field, FAIL-CLOSED.

    The MEASURED GENERIC carried-state witness slots (SV-6 = generic, the
    JMT semantic ruling; persisted spellings ``entry_state_digest`` /
    ``exit_state_digest`` per foldB D16 — "witness" is triple-booked).
    ``None`` = NOT MEASURED (the only writer-free value; nothing may imply
    an unmeasured cache fact); a present value maps channel names to digest
    strings and is written only by a lane that actually measured the
    carried state at the step boundary (F-WITNESS, after the F20 seam).
    """

    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"episode-ledger field {key!r} must be a mapping or null")
    parsed: dict[str, str] = {}
    for channel, digest in value.items():
        if not isinstance(channel, str) or not channel:
            raise ValueError(f"episode-ledger field {key!r} channels must be non-empty strings")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"episode-ledger field {key!r} digests must be non-empty strings")
        parsed[channel] = digest
    return parsed


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    """Return a required non-empty string field, refusing anything else."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"episode-ledger field {key!r} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EpisodeLedgerHeader:
    """Episode-level ledger header (S7).

    Parameters
    ----------
    episode_id:
        Opaque identifier naming this episode entity; the S6 ``episode_member``
        relation rows reference it.
    stepped_module:
        Module identity (address within the episode root) whose successive
        top-level calls define step boundaries.
    entry_seed:
        The capture's effective ``random_seed`` (the managed RNG recipe: the
        drawn-or-passed seed is recorded, never inferred).
    n_steps_declared:
        Declared step count, when the declaration carries one.
    token_feed:
        ``free`` (model-emitted step inputs) or ``forced`` (teacher-forced
        feed; explicitly non-verifying, disclosed).
    provenance_tier:
        Uniform per-product tier for the session home (``exact`` /
        ``ledger_only``).
    structure_only:
        Mirror of the capture's structure-only marker. Episodes with
        ``structure_only=True`` TYPED-REFUSE at entry this sprint (S2 table);
        the field exists so a loaded ledger can validate the token presence
        rule in both directions.
    escalated_from:
        Producer digest of the cheap-tier product this capture escalates
        (:func:`producer_digest`); present iff this product is an escalation.
    reason:
        Escalation reason (closed vocabulary); present iff escalation.
    fidelity_basis:
        Token-fidelity disclosure (E-A3); present iff escalation, forced,
        or perturbed (lane F42's coupling: a fired edit replaced a value).
    episode_ledger_version:
        Episode-family grammar version (C07X item (v)); always
        :data:`EPISODE_LEDGER_VERSION` on live writes, validated at load.
    step_output_kind:
        Declared step-output kind (``tokens``/``digest``/``none``; foldA
        D8). The settlement derivation is driven by this declaration
        (lane F40b), never by root-output-shape guessing.
    step_output_from:
        The step-output SOURCE disclosure: which root-output slot the
        per-step evidence derives from. The declared
        ``EpisodeSpec.step_output_from`` path when one was declared,
        ``"output"`` for the undeclared single-output default, and ``None``
        exactly when ``step_output_kind="none"`` (no source is consumed).
    step_axis:
        Declared step axis of the step-output source (the generic spelling).
        ``None`` when ``step_output_kind="none"``.
    step_output_positions:
        The step-axis positions each row's ``step_output`` was read from
        (W051 FIX2): the tail on the one-emission-per-step shapes, the measured
        chain positions under the declared-crossing arm; ``None`` = no column.
    step_join:
        Measured cross-step continuity envelope (``episode_step_join_v1``,
        lane F40c): per-row join grades from the closed
        continuous/forced/transformed/declared/exogenous/unchecked vocabulary,
        the first break step, live-detection disclosure, exogenous position
        counts, and unchecked reasons. ABSENT (``None``) = UNMEASURED (a
        pre-measurement artifact or a failed live measurement); every
        surface discloses an absent envelope as unmeasured. Envelope-shape
        validated here; full v1 geometry validated at ledger assembly
        against the rows.
    capture_digest:
        Capture digest binding the ledger to THIS capture product (C07X
        item (ii), foldA D7): minted at settlement by
        :func:`mint_capture_digest` over the product's recomputable
        identity facts. Lane F42's attested coupling consumes it (the
        positive "this ledger belongs to this product" claim).
    intervention_digest:
        Coupling slot (C07X item (iii), lane F42): the deterministic digest
        over the armed rule identity and ordered fire facts
        (``episode_intervention_digest_v1``), written on intervened
        (COUPLED) captures; ``None`` on uncoupled episodes.
    """

    episode_id: str
    stepped_module: str
    entry_seed: int
    n_steps_declared: int | None = None
    token_feed: TokenFeed = "free"
    provenance_tier: ProvenanceTier = "exact"
    structure_only: bool = False
    escalated_from: str | None = None
    reason: EscalationReason | None = None
    fidelity_basis: FidelityBasis | None = None
    episode_ledger_version: int = EPISODE_LEDGER_VERSION
    step_output_kind: StepOutputKind = "tokens"
    step_output_from: str | None = None
    step_axis: int | None = None
    step_output_positions: tuple[int, ...] | None = None
    step_join: Mapping[str, Any] | None = None
    capture_digest: str | None = None
    intervention_digest: str | None = None

    def __post_init__(self) -> None:
        # Closed vocabularies, table-driven: (field, vocabulary, None admitted).
        vocabulary_fields: tuple[tuple[str, frozenset[str], bool], ...] = (
            ("token_feed", _TOKEN_FEEDS, False),
            ("provenance_tier", _PROVENANCE_TIERS, False),
            ("reason", _ESCALATION_REASONS, True),
            ("fidelity_basis", _FIDELITY_BASES, True),
            ("step_output_kind", _STEP_OUTPUT_KINDS, False),
        )
        for field_name, vocabulary, optional in vocabulary_fields:
            value = getattr(self, field_name)
            if optional and value is None:
                continue
            if value not in vocabulary:
                raise ValueError(f"episode-ledger {field_name} {value!r} is out of vocabulary")
        # E-A2: the escalation disclosure is present iff escalation. reason and
        # escalated_from travel together, always.
        if (self.escalated_from is None) != (self.reason is None):
            raise ValueError(
                "episode-ledger escalation disclosure is all-or-nothing: "
                "escalated_from and reason must be present together"
            )
        if self.episode_ledger_version != EPISODE_LEDGER_VERSION:
            raise ValueError(
                f"episode-ledger grammar version {self.episode_ledger_version!r} "
                f"is not the supported family version {EPISODE_LEDGER_VERSION} "
                "(episode grammar changes are family-local; C07X item (v))"
            )
        if self.step_axis is not None and (
            not isinstance(self.step_axis, int) or isinstance(self.step_axis, bool)
        ):
            raise ValueError("episode-ledger step_axis must be an int or None")
        validate_step_output_positions(
            self.step_output_positions, step_output_kind=self.step_output_kind
        )
        if self.step_join is not None:
            _require_string_keyed_mapping(self.step_join, field_name="step_join")
        # Optional string slots share one non-empty-or-None contract.
        for string_field in ("step_output_from", "capture_digest", "intervention_digest"):
            string_value = getattr(self, string_field)
            if string_value is not None and (not isinstance(string_value, str) or not string_value):
                raise ValueError(
                    f"episode-ledger {string_field} must be a non-empty string or None"
                )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        return {
            "episode_ledger_version": self.episode_ledger_version,
            "episode_id": self.episode_id,
            "capture_kind": CAPTURE_KIND_EPISODE,
            "stepped_module": self.stepped_module,
            "n_steps_declared": self.n_steps_declared,
            "entry_seed": self.entry_seed,
            "token_feed": self.token_feed,
            "provenance_tier": self.provenance_tier,
            "structure_only": self.structure_only,
            "escalated_from": self.escalated_from,
            "reason": self.reason,
            "fidelity_basis": self.fidelity_basis,
            "step_output_kind": self.step_output_kind,
            "step_output_from": self.step_output_from,
            "step_axis": self.step_axis,
            "step_output_positions": (
                list(self.step_output_positions) if self.step_output_positions is not None else None
            ),
            "step_join": dict(self.step_join) if self.step_join is not None else None,
            "capture_digest": self.capture_digest,
            "intervention_digest": self.intervention_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeLedgerHeader:
        """Rebuild a header from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On an unknown key, a missing/ill-typed field, a value outside a
            closed vocabulary, or a ``capture_kind`` that is not ``episode``
            (callers convert into ``episode_ledger_without_declaration``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("episode-ledger header payload must be a mapping")
        unknown = set(payload) - _HEADER_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"episode-ledger header has unknown keys {sorted(unknown)}")
        missing = _HEADER_PAYLOAD_KEYS - _OPTIONAL_HEADER_PAYLOAD_KEYS - set(payload)
        if missing:
            raise ValueError(f"episode-ledger header is missing keys {sorted(missing)}")
        if payload["capture_kind"] != CAPTURE_KIND_EPISODE:
            raise ValueError(
                f"episode-ledger capture_kind is {payload['capture_kind']!r}, "
                f"not {CAPTURE_KIND_EPISODE!r}"
            )
        structure_only = payload["structure_only"]
        if not isinstance(structure_only, bool):
            raise ValueError("episode-ledger field 'structure_only' must be a bool")
        entry_seed = payload["entry_seed"]
        if not isinstance(entry_seed, int) or isinstance(entry_seed, bool):
            raise ValueError("episode-ledger field 'entry_seed' must be an int")
        # reason/fidelity_basis vocabulary, step_join envelope shape, and the
        # family-version pin re-validate inside ``cls(...)`` (__post_init__)
        # with the same messages; only payload-typing runs here.
        reason = _optional_str(payload, "reason")
        fidelity_basis = _optional_str(payload, "fidelity_basis")
        version = payload["episode_ledger_version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("episode-ledger field 'episode_ledger_version' must be an int")
        step_axis = payload["step_axis"]
        if step_axis is not None and (
            not isinstance(step_axis, int) or isinstance(step_axis, bool)
        ):
            raise ValueError("episode-ledger field 'step_axis' must be an int or null")
        return cls(
            episode_id=_require_str(payload, "episode_id"),
            stepped_module=_require_str(payload, "stepped_module"),
            entry_seed=entry_seed,
            n_steps_declared=_optional_non_negative_int(payload, "n_steps_declared"),
            token_feed=cast("TokenFeed", _require_vocabulary(payload, "token_feed", _TOKEN_FEEDS)),
            provenance_tier=cast(
                "ProvenanceTier",
                _require_vocabulary(payload, "provenance_tier", _PROVENANCE_TIERS),
            ),
            structure_only=structure_only,
            escalated_from=_optional_str(payload, "escalated_from"),
            reason=cast("EscalationReason | None", reason),
            fidelity_basis=cast("FidelityBasis | None", fidelity_basis),
            episode_ledger_version=version,
            step_output_kind=cast(
                "StepOutputKind",
                _require_vocabulary(payload, "step_output_kind", _STEP_OUTPUT_KINDS),
            ),
            step_output_from=_optional_str(payload, "step_output_from"),
            step_axis=step_axis,
            step_output_positions=validate_step_output_positions(
                payload.get("step_output_positions"),
                step_output_kind=str(payload["step_output_kind"]),
            ),
            step_join=cast("Mapping[str, Any] | None", payload["step_join"]),
            capture_digest=_optional_str(payload, "capture_digest"),
            intervention_digest=_optional_str(payload, "intervention_digest"),
        )


@dataclass(frozen=True)
class EpisodeLedgerRow:
    """One per-step ledger row (S7).

    Parameters
    ----------
    episode_step:
        0-based step index; row 0 is the prefill.
    role:
        ``prefill`` for row 0, ``decode`` afterwards.
    status:
        Row status (closed vocabulary; see :data:`RowStatus`).
    coord:
        Addressing coordinates. Session home: ``{"member_call_index": int,
        "pass_range": [lo, hi] | None}``. Floor home: ``{"member": <bundle
        member name>}`` — the ONLY ``member`` spelling in the schema, always
        the Bundle-key sense.
    step_output:
        Per-step evidence under the header's declared ``step_output_kind``
        (grammar v2's generic name; C07X): emitted token ids
        (``kind="tokens"``), a ``sha256:``-prefixed content digest of the
        per-step source slice (``kind="digest"``), or ``None``
        (``kind="none"``). REQUIRED on complete rows of completed
        value-mode ``tokens``/``digest`` episodes; ABSENT (typed absence)
        on structure-only episodes. Load-validated in both directions. The arithmetic ``cache_len`` field is DELETED (it was a
        derived guess that is wrong on real KV-cache feeds, never a
        measured cache fact); the measured carried-state witness slots
        below replace it.
    entry_state_digest / exit_state_digest:
        The MEASURED GENERIC carried-state witness (SV-6 = generic; C07X
        item (i)): channel-keyed digests of the carried state at the step's
        entry/exit boundaries. ``None`` = NOT MEASURED — the only value any
        shipped writer emits; no field may imply an unmeasured cache fact.
        Lane F-WITNESS writes real digests after the F20 retention seam.
    fire_count:
        Coupling slot (C07X item (iii), lane F42): the step's MEASURED
        intervention fire count on coupled captures -- ``0`` on zero-fire
        started rows (a first-class fact, never ``None``); ``None`` on
        uncoupled episodes and rows that never started.
    frontier:
        ``{"boundary_kind": str, "boundary_label": str}`` — interrupted rows
        only, from the settled record's disclosure. Never promotes a ragged
        frontier to a step boundary.
    rng_digest:
        Optional declared-scope RNG snapshot digest (floor escalation E-B3).
    escalation:
        Optional reference to an S6 ``escalates`` relation row (floor home).
    """

    episode_step: int
    role: Role
    status: RowStatus
    coord: Mapping[str, Any]
    step_output: tuple[int, ...] | str | None = None
    entry_state_digest: Mapping[str, str] | None = None
    exit_state_digest: Mapping[str, str] | None = None
    fire_count: int | None = None
    frontier: Mapping[str, str] | None = None
    rng_digest: str | None = None
    escalation: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"episode-ledger row role {self.role!r} is out of vocabulary")
        if self.status not in _ROW_STATUSES:
            raise ValueError(f"episode-ledger row status {self.status!r} is out of vocabulary")
        if self.episode_step < 0:
            raise ValueError("episode-ledger row episode_step must be non-negative")
        if (self.role == "prefill") != (self.episode_step == 0):
            raise ValueError(
                "episode-ledger row 0 is the prefill and only row 0 may carry role='prefill'"
            )
        if self.frontier is not None and self.status != "interrupted":
            raise ValueError("episode-ledger frontier is disclosed on interrupted rows only")
        unknown_coord = set(self.coord) - _COORD_KEYS
        if unknown_coord:
            raise ValueError(f"episode-ledger row coord has unknown keys {sorted(unknown_coord)}")
        for digest_key in ("entry_state_digest", "exit_state_digest"):
            digest_value = getattr(self, digest_key)
            if digest_value is not None:
                _parse_state_digest({digest_key: digest_value}, digest_key)
        if self.fire_count is not None and (
            not isinstance(self.fire_count, int)
            or isinstance(self.fire_count, bool)
            or self.fire_count < 0
        ):
            raise ValueError("episode-ledger row fire_count must be a non-negative int or null")

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        step_output: Any = self.step_output
        if isinstance(step_output, tuple):
            step_output = list(step_output)
        return {
            "episode_step": self.episode_step,
            "role": self.role,
            "status": self.status,
            "coord": dict(self.coord),
            "step_output": step_output,
            "entry_state_digest": (
                dict(self.entry_state_digest) if self.entry_state_digest is not None else None
            ),
            "exit_state_digest": (
                dict(self.exit_state_digest) if self.exit_state_digest is not None else None
            ),
            "fire_count": self.fire_count,
            "frontier": dict(self.frontier) if self.frontier is not None else None,
            "rng_digest": self.rng_digest,
            "escalation": self.escalation,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeLedgerRow:
        """Rebuild a row from :meth:`to_payload` output, FAIL-CLOSED.

        Raises
        ------
        ValueError
            On an unknown key, a missing/ill-typed field, or a value outside
            a closed vocabulary (callers convert into
            ``episode_ledger_incoherent``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("episode-ledger row payload must be a mapping")
        unknown = set(payload) - _ROW_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"episode-ledger row has unknown keys {sorted(unknown)}")
        missing = _ROW_PAYLOAD_KEYS - set(payload)
        if missing:
            raise ValueError(f"episode-ledger row is missing keys {sorted(missing)}")
        coord = payload["coord"]
        if not isinstance(coord, Mapping):
            raise ValueError("episode-ledger row field 'coord' must be a mapping")
        step_output = _parse_row_step_output(payload["step_output"])
        frontier = _parse_row_frontier(payload["frontier"])
        fire_count = _optional_non_negative_int(payload, "fire_count")
        return cls(
            episode_step=_require_non_negative_int(payload, "episode_step"),
            role=cast("Role", _require_vocabulary(payload, "role", _ROLES)),
            status=cast("RowStatus", _require_vocabulary(payload, "status", _ROW_STATUSES)),
            coord=dict(coord),
            step_output=step_output,
            entry_state_digest=_parse_state_digest(payload, "entry_state_digest"),
            exit_state_digest=_parse_state_digest(payload, "exit_state_digest"),
            fire_count=fire_count,
            frontier=frontier,
            rng_digest=_optional_str(payload, "rng_digest"),
            escalation=_optional_str(payload, "escalation"),
        )


def _parse_row_step_output(value: Any) -> tuple[int, ...] | str | None:
    """Parse one row's ``step_output`` payload field, FAIL-CLOSED.

    The row-local check admits the union of per-kind shapes (token-id list,
    digest string, or null); the kind-conditioned coherence check runs at
    ledger assembly, where the header's declared ``step_output_kind`` is in
    scope (:func:`_check_step_output_kind_coherence`).
    """

    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise ValueError("episode-ledger row field 'step_output' digest must be non-empty")
        return value
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in value):
            raise ValueError("episode-ledger row field 'step_output' must hold ints")
        return tuple(int(token) for token in value)
    raise ValueError(
        "episode-ledger row field 'step_output' must be a list of ints, a digest string, or null"
    )


def _parse_row_frontier(frontier_value: Any) -> dict[str, str] | None:
    """Parse one row's ``frontier`` payload field, FAIL-CLOSED."""

    if frontier_value is None:
        return None
    if isinstance(frontier_value, Mapping):
        if set(frontier_value) != _FRONTIER_KEYS:
            raise ValueError(
                f"episode-ledger row frontier must carry exactly {sorted(_FRONTIER_KEYS)}"
            )
        if not all(isinstance(value, str) for value in frontier_value.values()):
            raise ValueError("episode-ledger row frontier values must be strings")
        return {key: str(value) for key, value in frontier_value.items()}
    raise ValueError("episode-ledger row field 'frontier' must be a mapping or null")


class EpisodeLedger:
    """Write-once episode ledger: one header plus ordered per-step rows.

    Rows are written ONCE at settlement/finalize (S7 L3); the finalized view
    is identity-stable and immutable. No post-hoc row edits: escalation
    annotates via S6 relation rows, never rewrites history, and demotion
    replaces the outcome record, never these rows.
    """

    def __init__(self, header: EpisodeLedgerHeader, rows: Sequence[EpisodeLedgerRow]) -> None:
        self._header = header
        self._rows: tuple[EpisodeLedgerRow, ...] = tuple(rows)
        _check_monotone_prefix_law(self._rows)
        _check_step_output_presence_rule(header, self._rows)
        validate_step_output_positions(
            header.step_output_positions,
            step_output_kind=header.step_output_kind,
            n_rows=len(self._rows),
            rows_with_output=sum(row.step_output is not None for row in self._rows),
        )
        if header.step_join is not None:
            from ._episode_join import validate_step_join_envelope

            validate_step_join_envelope(header.step_join, len(self._rows))

    @property
    def header(self) -> EpisodeLedgerHeader:
        """The episode-level header."""

        return self._header

    @property
    def rows(self) -> tuple[EpisodeLedgerRow, ...]:
        """Immutable, identity-stable view of the per-step rows in order."""

        return self._rows

    @property
    def steps_completed(self) -> int:
        """Derived disclosure: count of ``complete`` rows."""

        return sum(1 for row in self._rows if row.status == "complete")

    def step_output_series(self) -> tuple[Any, ...]:
        """The whole-episode step-output SERIES -- an episode-dependent claim.

        Returns the ordered per-step evidence column (one entry per row) as
        ONE series, which claims the rows form one closed feed. The claim
        door (lane F40c): a measured broken join refuses with the verbatim
        teaching text (``episode_feed_break_exogenous`` /
        ``episode_join_declared_crossing``); an unmeasured or unchecked join
        refuses ``episode_join_unmeasured``. Per-row ``rows[k].step_output``
        reads are per-segment facts and stay unaffected.

        Raises
        ------
        EpisodeJoinError
            Per the claim door above.
        EpisodeLedgerError
            ``episode_ledger_incoherent`` when the declared kind derives no
            evidence column (``step_output_kind="none"``).
        """

        if self._header.step_output_kind == "none":
            raise _ledger_error(
                "step_output_kind='none' episodes derive no evidence column; "
                "there is no step-output series to read. Declare "
                "step_output_kind='tokens' or 'digest' to derive one.",
                code="episode_ledger_incoherent",
            )
        from ._episode_join import check_series_join_claim

        check_series_join_claim(self._header)
        return tuple(row.step_output for row in self._rows)

    @property
    def truncated_at_step(self) -> int | None:
        """Derived disclosure: index of the interrupted or first absent row.

        ``None`` when every row is complete (nothing was truncated).
        """

        for row in self._rows:
            if row.status != "complete":
                return row.episode_step
        return None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for portable artifacts."""

        return {
            "header": self._header.to_payload(),
            "rows": [row.to_payload() for row in self._rows],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeLedger:
        """Rebuild a ledger from :meth:`to_payload` output, FAIL-CLOSED.

        The rebuild routes rows through the same monotone-prefix and
        token-presence checks the live writer enforces, so a forged or
        drifted payload can never load as a coherent ledger.

        Raises
        ------
        ValueError
            On any invalid header/row payload or a prefix-law violation
            (callers convert into ``episode_ledger_incoherent``).
        TypeError
            When ``payload`` is not a mapping.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("episode-ledger payload must be a mapping")
        if set(payload) != {"header", "rows"}:
            raise ValueError(
                "episode-ledger payload must carry exactly the keys {'header', 'rows'}"
            )
        rows_payload = payload["rows"]
        if not isinstance(rows_payload, list):
            raise ValueError("episode-ledger payload field 'rows' must be a list")
        header = EpisodeLedgerHeader.from_payload(payload["header"])
        rows = [EpisodeLedgerRow.from_payload(entry) for entry in rows_payload]
        return cls(header, rows)


def _check_monotone_prefix_law(rows: Sequence[EpisodeLedgerRow]) -> None:
    """Enforce the MONOTONE PREFIX LAW (S7 L1), refusing violations.

    Rows 0..k-1 complete, row k in {complete, interrupted}, rows k+1.. absent.
    At most one interrupted row; it is the last non-absent row. Steps must be
    the contiguous run 0..N-1.

    Raises
    ------
    ValueError
        On any geometry outside the law (callers convert into
        ``episode_ledger_incoherent`` and the outcome derivation degrades
        fail-closed).
    """

    seen_interrupted = False
    seen_absent = False
    for position, row in enumerate(rows):
        if row.episode_step != position:
            raise ValueError(
                "episode-ledger rows must be the contiguous 0-based step run; "
                f"row at position {position} carries episode_step {row.episode_step}"
            )
        if row.status == "complete":
            if seen_interrupted or seen_absent:
                raise ValueError(
                    "episode-ledger monotone prefix law violated: a complete row "
                    f"follows a non-complete row (step {row.episode_step})"
                )
        elif row.status == "interrupted":
            if seen_interrupted or seen_absent:
                raise ValueError(
                    "episode-ledger monotone prefix law violated: interrupted row "
                    f"at step {row.episode_step} is not the first non-complete row"
                )
            seen_interrupted = True
        else:  # absent
            seen_absent = True


def _check_step_output_presence_rule(
    header: EpisodeLedgerHeader, rows: Sequence[EpisodeLedgerRow]
) -> None:
    """Enforce the S7 step-output PRESENCE RULE in both directions (v2).

    Structure-only episodes must not carry step-output payloads
    (``episode_ledger_payload_in_structure_only``). Value-mode ``tokens``
    and ``digest`` episodes must carry step_output on every complete row
    WHEN the episode ran to completion (all rows complete): the evidence
    derives from the product's own root output, which a truncated
    (halted/aborted/failed) episode never produced, so absence on a
    truncated ledger is the disclosed consequence of the truncation, never
    incoherence. Interrupted/absent rows never carry step output (the step
    emitted none); the ``none`` kind never carries any.

    Raises
    ------
    ValueError
        With a message carrying the owning refusal code's semantics; callers
        map the structure-only direction onto
        ``episode_ledger_payload_in_structure_only`` and the value-mode
        direction onto ``episode_ledger_incoherent``.
    """

    _check_step_output_kind_coherence(header, rows)
    all_complete = bool(rows) and all(row.status == "complete" for row in rows)
    for row in rows:
        if header.structure_only and row.step_output is not None:
            raise ValueError(
                "episode_ledger_payload_in_structure_only: a structure-only "
                f"episode ledger carries step output at step {row.episode_step}"
            )
        if row.status in ("interrupted", "absent") and row.step_output is not None:
            raise ValueError(
                "episode-ledger rows that did not complete cannot carry step "
                f"output (step {row.episode_step}, status {row.status!r})"
            )
        if (
            not header.structure_only
            and header.step_output_kind in ("tokens", "digest")
            and all_complete
            and row.step_output is None
        ):
            raise ValueError(
                "episode_ledger_incoherent: a value-mode "
                f"{header.step_output_kind!r} episode ledger is missing "
                f"step_output at complete step {row.episode_step}"
            )


def _check_step_output_kind_coherence(
    header: EpisodeLedgerHeader, rows: Sequence[EpisodeLedgerRow]
) -> None:
    """Kind-conditioned step-output TYPE coherence (grammar v2).

    ``tokens`` rows carry token-id tuples, ``digest`` rows digest strings,
    ``none`` rows nothing — a shape outside the declared kind is a forged
    or drifted payload, never something to silently coerce.
    """

    kind = header.step_output_kind
    for row in rows:
        value = row.step_output
        if value is None:
            continue
        if kind == "none":
            raise ValueError(
                "episode-ledger step_output_kind='none' rows cannot carry step "
                f"output (step {row.episode_step})"
            )
        if kind == "tokens" and not isinstance(value, tuple):
            raise ValueError(
                "episode-ledger step_output_kind='tokens' rows carry token-id "
                f"lists, got {type(value).__name__} at step {row.episode_step}"
            )
        if kind == "digest" and not isinstance(value, str):
            raise ValueError(
                "episode-ledger step_output_kind='digest' rows carry digest "
                f"strings, got {type(value).__name__} at step {row.episode_step}"
            )


def row_status_for_member_outcome(status: str, phase: str | None) -> RowStatus:
    """Map a member's settled ``(CaptureStatus, CapturePhase)`` to a row status.

    The 2.1 mapping table (a), total over the shipped vocabulary INCLUDING the
    phase-less FAILED record (phase is optional on the shipped record; the row
    is NON-UPGRADING — with the failure phase unattributed, forward-step
    completion cannot be inferred, so the row never claims complete).

    Parameters
    ----------
    status:
        ``CaptureStatus`` value name (e.g. ``"COMPLETE"``) or value string.
    phase:
        ``CapturePhase`` value string, or ``None`` when the settled record
        carries no phase (legal only on FAILED).

    Returns
    -------
    RowStatus
        The derived row status. UNATTESTED/UNKNOWN members return
        ``"interrupted"`` here only as the non-upgrading floor for a
        driver-less derivation; the fold routes such members to
        ``episode_unknown`` regardless (mapping-table note: their rows load
        as UNVERIFIED disclosures and nothing gates on them).
    """

    normalized = status.upper()
    if normalized == "COMPLETE":
        return "complete"
    if normalized in ("HALTED", "ABORTED_NONFINITE"):
        return "interrupted"
    if normalized == "FAILED":
        if phase in ("finalize", "postprocess", "teardown"):
            # The step's FORWARD returned; the failure is post-forward product
            # processing. Nothing is blessed: product truth stays FAILED and
            # the fold consumes the member outcome, not the row.
            return "complete"
        # phase == "forward" or phase absent (unattributed): non-upgrading.
        return "interrupted"
    if normalized in ("UNATTESTED", "UNKNOWN"):
        return "interrupted"
    raise ValueError(f"unknown member CaptureStatus {status!r} for episode-ledger row mapping")


def row_status_for_recording_status(status: str) -> RowStatus:
    """Map a ``Recording.status`` to a row status (2.1 mapping table (b)).

    ``recovered`` maps to ``interrupted`` FAIL-CLOSED: a recovered pass is
    never blessed by derivation (the R06 rule).
    """

    mapping: dict[str, RowStatus] = {
        "complete": "complete",
        "halted": "interrupted",
        "partial_error": "interrupted",
        "recovered": "interrupted",
    }
    try:
        return mapping[status]
    except KeyError:
        raise ValueError(
            f"unknown Recording.status {status!r} for episode-ledger row mapping"
        ) from None


def producer_digest(outcome_payload: Mapping[str, Any], ledger: EpisodeLedger) -> str:
    """Digest identifying a cheap-tier producer for the E-A2 disclosure.

    Defined (per the spike's E-A2 declaration) over the producer's persisted
    ``_capture_outcome`` payload plus its ledger rows: hex SHA-256 over the
    canonical-JSON encoding of both. Spelling and construction are
    DOCUMENTED-UNSTABLE (naming session 2).
    """

    canonical = json.dumps(
        {
            "outcome": dict(outcome_payload),
            "ledger": ledger.to_payload(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Entry declaration, settlement-time writer, and load validation (session home)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedEpisode:
    """An entry-validated episode declaration bound to a concrete model.

    Produced by :func:`resolve_episode_declaration` at ``tl.trace`` entry,
    after the E-A4 preflight; consumed by the header attach and the
    settlement-time ledger writer.
    """

    episode_id: str
    address: str
    n_steps: int | None
    step_axis: int
    step_output_kind: StepOutputKind
    step_output_from: str | None
    forced_tokens: tuple[int, ...] | None
    escalated_from: str | None
    reason: EscalationReason | None
    expected_tokens: tuple[tuple[int, ...], ...] | None = None
    # Join arms (lane F40c). Defaults keep pre-existing keyword construction
    # (the four-clause price probe) valid: an undeclared resolution has open
    # feed, the FORK-4 2-1 disclose default, no crossings, and no live
    # session (settlement then writes NO envelope -- unmeasured, disclosed).
    feed: str = "open"
    on_feed_break: str = "disclose"
    crossings: tuple[int, ...] = ()
    join_session: Any = None
    # W051 (audit 3.3): the declared entry argument the join measures
    # (positional index or keyword name); None selects by the disclosed
    # preference rule. Read from ``EpisodeSpec.step_input_from`` when the
    # spec carries it (the spec field itself is the options owner's).
    step_input_from: str | int | None = None
    # Attested coupling (lane F42): the live fire-attribution session, set
    # by the trace entry exactly when intervene= rides an episode capture.
    # None = uncoupled (the reserved slots stay entry-dark None).
    coupling_session: Any = None


def _ledger_error(message: str, *, code: str) -> Exception:
    """Build the typed ledger refusal."""

    from ..errors.episode import EpisodeLedgerError

    return EpisodeLedgerError(message, code=code)


def _resolved_stepped_address(stepped: Any, model: Any) -> str:
    """Validate the stepped module and resolve its address inside the root."""

    import torch.nn as nn

    if stepped is None:
        # F41 bound-method roots: the stepped module DEFAULTS to the bound
        # method's owner (foldA MEMO s5 item 9); on a module root it stays
        # required -- an unowned None has no honest default.
        from ..backends.torch.bound_root import TLBoundMethodRoot

        if isinstance(model, TLBoundMethodRoot):
            stepped = model.tl_owner
        else:
            raise _declaration_error(
                "EpisodeSpec.stepped_module is required on a module-root "
                "capture (it defaults to the bound method's owner only on a "
                "bound-method root, e.g. tl.trace(model.generate, ids, "
                "episode=tl.options.EpisodeSpec(n_steps=N))). Remedy: pass "
                "EpisodeSpec(stepped_module=<the stepped nn.Module>).",
                code="episode_declaration_invalid",
            )
    if not isinstance(stepped, nn.Module):
        raise _declaration_error(
            "EpisodeSpec.stepped_module must be an nn.Module (the stepped model "
            f"whose calls define step boundaries), got {type(stepped).__name__}."
        )
    if stepped is model:
        raise _declaration_error(
            "EpisodeSpec.stepped_module is the episode root itself. The "
            "stepped model must be a PROPER submodule of the traced root. "
            "Remedy: trace the bound generation method "
            "(tl.trace(model.generate, ids, episode=...) -- stepped_module "
            "then defaults to the owner), or wrap the generation loop in an "
            "nn.Module whose forward steps this model, and trace the wrapper."
        )
    for name, candidate in model.named_modules():
        if candidate is stepped and name:
            return name
    raise _declaration_error(
        "EpisodeSpec.stepped_module is not a submodule of the traced episode "
        "root; the ledger's step boundaries are the stepped module's calls "
        "inside the root's forward."
    )


def _validate_step_output_declaration(spec: EpisodeSpec) -> None:
    """Validate the declared step-output evidence triple (foldA D8, F40b).

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` for an ill-typed step axis, a kind
        outside the closed vocabulary, an ill-typed source path, or the
        ``none``-kind x declared-source contradiction.
    """

    if not isinstance(spec.step_axis, int) or isinstance(spec.step_axis, bool):
        raise _declaration_error(
            f"EpisodeSpec.step_axis={spec.step_axis!r} must be an int axis index."
        )
    if spec.step_output_kind not in _STEP_OUTPUT_KINDS:
        raise _declaration_error(
            f"EpisodeSpec.step_output_kind={spec.step_output_kind!r} is outside "
            f"the closed vocabulary {sorted(_STEP_OUTPUT_KINDS)}: 'tokens' reads "
            "per-step emitted token ids, 'digest' reads per-step content "
            "digests (any dtype), 'none' derives no evidence (status-only "
            "ledger).",
            code="episode_declaration_invalid",
        )
    if spec.step_output_from is not None and (
        not isinstance(spec.step_output_from, str) or not spec.step_output_from
    ):
        raise _declaration_error(
            f"EpisodeSpec.step_output_from={spec.step_output_from!r} must be a "
            "non-empty root-output slot path (dot-separated dict keys, "
            "namedtuple/ModelOutput field names, or tuple indices, e.g. "
            "'sequences' or '0') or None.",
            code="episode_declaration_invalid",
        )
    if spec.step_output_kind == "none" and spec.step_output_from is not None:
        raise _declaration_error(
            "EpisodeSpec.step_output_from declares an evidence source but "
            "step_output_kind='none' declares that no per-step evidence is "
            "derived; the declaration contradicts itself. Remedy: drop "
            "step_output_from, or declare kind 'tokens'/'digest'.",
            code="episode_declaration_invalid",
        )


def _validated_forced_tokens(spec: EpisodeSpec) -> tuple[int, ...] | None:
    """Validate the teacher-forced feed against the declaration."""

    if spec.forced_tokens is None:
        return None
    try:
        forced = tuple(int(token) for token in spec.forced_tokens)
    except (TypeError, ValueError):
        raise _declaration_error(
            "EpisodeSpec.forced_tokens must be an iterable of ints (the "
            "teacher-forced feed), got "
            f"{type(spec.forced_tokens).__name__}."
        ) from None
    if not forced:
        raise _declaration_error("EpisodeSpec.forced_tokens must be non-empty when given.")
    if spec.n_steps is not None and len(forced) != spec.n_steps:
        raise _declaration_error(
            f"EpisodeSpec.forced_tokens has {len(forced)} tokens but n_steps="
            f"{spec.n_steps}; the forced feed must cover exactly the declared steps."
        )
    return forced


def _preflight_declared_state(spec: EpisodeSpec) -> None:
    """E-A4: declared-state preflight, unconditional, at declaration time."""

    import torch

    for index, item in enumerate(spec.state):
        if isinstance(item, torch.Tensor):
            continue  # tensors snapshot/restore by clone within declared scope
        try:
            copy.deepcopy(item)
        except Exception as exc:
            raise _declaration_error(
                f"EpisodeSpec.state[{index}] ({type(item).__name__}) is not "
                "snapshot/restorable within the declared checkpoint scope: "
                f"deepcopy failed with {type(exc).__name__}: {exc}. Remedy: "
                "declare only snapshotable state, or make the item deep-copyable.",
                code="episode_state_unsnapshotable",
            ) from exc


def resolve_episode_declaration(spec: EpisodeSpec, model: Any) -> ResolvedEpisode:
    """Validate an episode declaration at entry, BEFORE execution.

    Performs the structural entry checks and the E-A4 preflight: every
    declared episode-carried state item must be snapshot/restorable within
    the declared scope, unconditionally — whether or not any mid-episode
    checkpoint is ever used. Any failure refuses typed here, before the
    forward runs.

    Parameters
    ----------
    spec:
        The user's :class:`torchlens.options.EpisodeSpec`.
    model:
        The episode root about to be traced.

    Returns
    -------
    ResolvedEpisode
        The bound declaration (stepped-module address resolved).

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` for structural problems;
        ``episode_state_unsnapshotable`` for the E-A4 refusal.
    """

    from ..options import EpisodeSpec as _EpisodeSpec

    if not isinstance(spec, _EpisodeSpec):
        raise _declaration_error(
            f"episode= expects an EpisodeSpec, got {type(spec).__name__}. "
            "Remedy: pass tl.options.EpisodeSpec(stepped_module=...)"
        )
    address = _resolved_stepped_address(spec.stepped_module, model)
    if spec.rng != "managed":
        raise _declaration_error(
            f"EpisodeSpec.rng={spec.rng!r} is unsupported; only the 'managed' "
            "seeding discipline ships (the capture's effective random_seed is "
            "recorded as the ledger entry_seed)."
        )
    if spec.n_steps is not None and (not isinstance(spec.n_steps, int) or spec.n_steps <= 0):
        raise _declaration_error(
            f"EpisodeSpec.n_steps={spec.n_steps!r} must be a positive int or None."
        )
    _validate_step_output_declaration(spec)
    forced = _validated_forced_tokens(spec)
    declared_steps = (
        spec.n_steps if spec.n_steps is not None else (len(forced) if forced is not None else None)
    )
    if (
        declared_steps is not None
        and declared_steps > EPISODE_DECLARED_STEP_CEILING
        and not spec.acknowledge_step_cost
    ):
        raise _declaration_error(
            f"the declared episode is {declared_steps} steps, beyond the "
            f"wrapped-tier cost ceiling of {EPISODE_DECLARED_STEP_CEILING}. "
            "Wrapped episode capture cost is SUPERLINEAR in step count "
            "(measured gpt2-124M CPU: N=100 = 657 s / 5.4 GB peak RSS); the "
            "guarded-fast tier (trace.run(inputs=..., fast=True), which needs "
            "a functional save= such as save=tl.func(...) on the capture) is "
            "the engine for episode-scale re-runs. Remedy: declare fewer steps, "
            "or pass EpisodeSpec(acknowledge_step_cost=True) to accept the "
            "diagnostic-tier cost explicitly.",
            code="episode_step_ceiling_exceeded",
        )
    from ._episode_join import _validated_join_declaration

    crossings = _validated_join_declaration(spec, declared_steps)
    step_input_from = getattr(spec, "step_input_from", None)
    if step_input_from is not None and not (
        (isinstance(step_input_from, int) and not isinstance(step_input_from, bool))
        or (isinstance(step_input_from, str) and step_input_from)
    ):
        raise _declaration_error(
            f"EpisodeSpec.step_input_from={step_input_from!r} must be a positional "
            "argument index (int) or a keyword argument name (non-empty str) "
            "naming the stepped call's carried input, or None.",
            code="episode_declaration_invalid",
        )
    reason = spec.reason
    if (spec.escalated_from is None) != (reason is None):
        raise _declaration_error(
            "escalated_from and reason must be declared together (the escalation "
            "disclosure is all-or-nothing, E-A2)."
        )
    if reason is not None and reason not in _ESCALATION_REASONS:
        raise _declaration_error(
            f"EpisodeSpec.reason={reason!r} is outside the closed vocabulary "
            f"{sorted(_ESCALATION_REASONS)}."
        )
    _preflight_declared_state(spec)
    from ._episode_join import EpisodeJoinSession

    return ResolvedEpisode(
        episode_id=f"ep-{uuid.uuid4().hex[:16]}",
        address=address,
        n_steps=spec.n_steps,
        step_axis=spec.step_axis,
        step_output_kind=cast("StepOutputKind", spec.step_output_kind),
        step_output_from=spec.step_output_from,
        forced_tokens=forced,
        escalated_from=spec.escalated_from,
        reason=cast("EscalationReason | None", reason),
        expected_tokens=(
            tuple(tuple(int(t) for t in step) for step in spec.expected_tokens)
            if spec.expected_tokens is not None
            else None
        ),
        feed=spec.feed,
        on_feed_break=spec.on_feed_break,
        crossings=crossings,
        # The join session hooks the RESOLVED stepped module (the module at
        # the validated address), never the raw spec value -- on a
        # bound-method root the spec's stepped_module is None and defaults
        # to the owner (F41), which only the address resolution sees.
        join_session=EpisodeJoinSession(
            model.get_submodule(address),
            feed=spec.feed,
            on_feed_break=spec.on_feed_break,
            crossings=crossings,
            step_input_from=step_input_from,
        ),
        step_input_from=step_input_from,
    )


def attach_episode_header(trace: Any, resolved: ResolvedEpisode) -> None:
    """Stamp the episode declaration marker on the trace BEFORE the forward.

    The pre-capture marker makes the episode declaration visible to
    postprocess consumers (episode folding policy) and rides the partial
    product on a failed forward. The settlement-time writer replaces it with
    the finalized ``{"header", "rows"}`` payload.
    """

    trace.annotations[EPISODE_ANNOTATIONS_KEY] = {
        "declared": {
            "episode_id": resolved.episode_id,
            "capture_kind": CAPTURE_KIND_EPISODE,
            "stepped_module": resolved.address,
            "n_steps_declared": resolved.n_steps,
        }
    }


def _fidelity_basis(
    resolved: ResolvedEpisode,
    tokens_by_row: list[tuple[int, ...]] | list[str] | None,
) -> FidelityBasis | None:
    """Derive the E-A3 fidelity disclosure for the ledger header.

    A teacher-forced feed is ``forced`` (non-verifying by declaration). An
    escalation with BOTH token columns available compares the escalated
    column against the producer's up to the shorter prefix: equal records
    ``tokens``, a mismatch records ``diverged`` (the escalation FAILED its
    purpose — the product is still a valid capture of what it ran, it just
    is not an escalation of the original episode, and says so). Escalations
    without comparable columns record ``none`` — including any
    non-``tokens`` declared kind, whose evidence column is never a token
    column. Never a settlement input.
    """

    session = resolved.coupling_session
    if session is not None and getattr(session, "any_replaced", False):
        # Lane F42: a fired-and-replacing intervention PERTURBED the run --
        # the evidence column is true of the perturbed execution only.
        # Outranks forced/escalation (both keep their own slots).
        return "perturbed"
    if resolved.forced_tokens is not None:
        return "forced"
    if resolved.escalated_from is None:
        return None
    emitted = tokens_by_row if resolved.step_output_kind == "tokens" else None
    expected = resolved.expected_tokens
    prefix = 0 if emitted is None or expected is None else min(len(emitted), len(expected))
    if prefix == 0 or emitted is None or expected is None:
        return "none"
    for step in range(prefix):
        if emitted[step] != expected[step]:
            return "diverged"
    return "tokens"


def _build_header(
    trace: Any,
    resolved: ResolvedEpisode,
    *,
    fidelity: FidelityBasis | None = None,
    step_join: Mapping[str, Any] | None = None,
    positions: tuple[int, ...] | None = None,
) -> EpisodeLedgerHeader:
    """Build the finalized ledger header from the settled trace."""

    entry_seed = getattr(trace, "random_seed", None)
    if not isinstance(entry_seed, int) or isinstance(entry_seed, bool):
        raise _ledger_error(
            "episode ledger requires the capture's effective random_seed (the "
            "managed RNG recipe); the trace carries none.",
            code="episode_ledger_incoherent",
        )
    forced = resolved.forced_tokens is not None
    consumes_source = resolved.step_output_kind != "none"
    coupling = resolved.coupling_session
    intervention_digest = None
    if coupling is not None:
        # Lane F42: binds the ORDERED fire evidence + armed rule identity to
        # this product; minted iff a coupling session ran, zero fires included.
        from ._episode_coupling import mint_intervention_digest

        intervention_digest = mint_intervention_digest(coupling)
    # Grammar v2 (C07X): kind/source/axis are written from the declaration
    # (the F40b derivation consumes them); the capture digest is minted here
    # and binds the ledger to this product; step_join rides the F40c
    # settlement grade -- entry-dark slots, never guessed values.
    return EpisodeLedgerHeader(
        episode_id=resolved.episode_id,
        stepped_module=resolved.address,
        entry_seed=entry_seed,
        n_steps_declared=resolved.n_steps,
        token_feed="forced" if forced else "free",
        provenance_tier="exact",
        structure_only=False,
        escalated_from=resolved.escalated_from,
        reason=resolved.reason,
        fidelity_basis=fidelity,
        step_output_kind=resolved.step_output_kind,
        step_output_from=((resolved.step_output_from or "output") if consumes_source else None),
        step_axis=resolved.step_axis if consumes_source else None,
        step_output_positions=positions if consumes_source else None,
        step_join=step_join,
        # The content-binding digest (W051, audit 3.2) is minted over the
        # FINISHED ledger payload, so the header is built unbound here and
        # bound in ``_write_settled_episode_ledger`` once the rows exist.
        capture_digest=None,
        intervention_digest=intervention_digest,
    )


def _derive_row_statuses(
    status: str, phase: Any, calls: list[Any], started: int
) -> list[RowStatus]:
    """Derive per-step row statuses from the settled outcome and call records."""

    if status == "COMPLETE" or (
        status == "FAILED" and phase in ("finalize", "postprocess", "teardown")
    ):
        return ["complete"] * started
    if status in ("HALTED", "ABORTED_NONFINITE") or (
        status == "FAILED" and phase in ("forward", None)
    ):
        if started == 0:
            return []
        tail: RowStatus = "complete" if _call_returned(calls[-1]) else "interrupted"
        prefix: list[RowStatus] = ["complete"] * (started - 1)
        return [*prefix, tail]
    # UNATTESTED / UNKNOWN: structural, unverified disclosures.
    return [
        cast("RowStatus", "complete" if _call_returned(call) else "interrupted") for call in calls
    ]


@dataclass(frozen=True)
class _RowDecorations:
    """Per-row decoration facts shared across the ledger row build."""

    evidence_by_row: list[tuple[int, ...]] | list[str] | None
    frontier: dict[str, str] | None
    # Lane F42: the measured per-row fire counts of a coupled capture
    # (None = uncoupled; the rows keep their entry-dark None slots).
    fire_counts: list[int | None] | None = None


def _build_ledger_rows(
    n_total: int,
    statuses: list[RowStatus],
    calls: list[Any],
    decorations: _RowDecorations,
) -> list[EpisodeLedgerRow]:
    """Build the ordered per-step rows (absent rows past the started prefix)."""

    started = len(calls)
    evidence_by_row = decorations.evidence_by_row
    frontier = decorations.frontier
    rows: list[EpisodeLedgerRow] = []
    for step in range(n_total):
        if step < started:
            row_status = statuses[step]
            call = calls[step]
            coord: dict[str, Any] = {
                "member_call_index": getattr(call, "call_index", step + 1),
                "pass_range": list(_pass_range_for_call(call) or ()) or None,
            }
        else:
            row_status = "absent"
            coord = {"member_call_index": step + 1, "pass_range": None}
        rows.append(
            EpisodeLedgerRow(
                episode_step=step,
                role="prefill" if step == 0 else "decode",
                status=row_status,
                coord=coord,
                # Grammar v2: the arithmetic cache_len guess is DELETED; the
                # carried-state witness slots stay None (NOT MEASURED) until
                # a lane actually measures the step-boundary state.
                step_output=(evidence_by_row[step] if evidence_by_row is not None else None),
                frontier=(frontier if row_status == "interrupted" else None),
                fire_count=(
                    decorations.fire_counts[step] if decorations.fire_counts is not None else None
                ),
            )
        )
    return rows


def write_episode_ledger(trace: Any, resolved: ResolvedEpisode) -> EpisodeLedger:
    """Write the finalized episode ledger onto the settled product (S7 L3).

    Runs ONCE at settlement/finalize, after postprocess, reading only the
    settled ``CaptureOutcome`` (via the public accessor) and the finished
    module-call records. Rows are write-once: a second write refuses typed.

    Returns
    -------
    EpisodeLedger
        The finalized, validated ledger (also attached at
        ``trace.annotations["episode"]`` as its payload).

    Raises
    ------
    EpisodeCaptureError
        Every settlement refusal (ledger incoherence, token-derivation
        refusals) carries the settled product on ``exc.partial_log``
        (lane F40a): the capture already ran and postprocessed, so the
        refusal names the declaration mismatch WITHOUT discarding the
        product -- recover it with ``tl.partial.from_failed_capture(exc)``.
    """

    try:
        return _write_settled_episode_ledger(trace, resolved)
    except Exception as exc:
        _attach_settlement_partial_log(exc, trace)
        raise


def _attach_settlement_partial_log(exc: BaseException, trace: Any) -> None:
    """Attach the settled product to a settlement refusal as ``partial_log``.

    Best-effort and idempotent: an exception that already carries a partial
    product keeps it, and an exception that rejects attribute assignment
    propagates unmodified (the refusal itself is never masked).
    """

    from ..errors.episode import EpisodeCaptureError

    if not isinstance(exc, EpisodeCaptureError):
        return
    if getattr(exc, "partial_log", None) is not None:
        return
    from ..partial import PartialTrace

    try:
        exc.partial_log = PartialTrace(trace=trace, original_exception=exc)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - never mask the settlement refusal
        return


def _write_settled_episode_ledger(trace: Any, resolved: ResolvedEpisode) -> EpisodeLedger:
    """The settlement-time ledger write body (see :func:`write_episode_ledger`)."""

    existing = trace.annotations.get(EPISODE_ANNOTATIONS_KEY)
    if isinstance(existing, Mapping) and "rows" in existing:
        raise _ledger_error(
            "episode ledger rows are write-once at settlement; this product "
            "already carries a finalized ledger.",
            code="episode_ledger_incoherent",
        )
    outcome = trace.outcome
    status = outcome.status.name if outcome is not None else "UNKNOWN"
    phase_value = getattr(outcome, "phase", None) if outcome is not None else None
    phase = getattr(phase_value, "value", phase_value)

    try:
        module_record = trace.modules[resolved.address]
        # ModuleCallAccessor ITERATION yields int keys, not ModuleCall
        # records (F40c gotcha): values() is the record sequence. The bare
        # list(...) spelling silently degraded every row's pass_range to
        # None and served member_call_index only through the getattr
        # default coinciding with the key.
        calls = list(module_record.calls.values())
    except (LookupError, AttributeError, TypeError, ValueError):
        calls = []
    started = len(calls)

    if status == "COMPLETE" and resolved.n_steps is not None and started != resolved.n_steps:
        raise _ledger_error(
            f"episode declared n_steps={resolved.n_steps} but the COMPLETE "
            f"capture ran {started} stepped-module calls; the declaration does "
            "not match the executed episode.",
            code="episode_ledger_incoherent",
        )

    n_total = resolved.n_steps if resolved.n_steps is not None else started
    n_total = max(n_total, started)

    statuses = _derive_row_statuses(status, phase, calls, started)

    evidence_by_row: list[tuple[int, ...]] | list[str] | None = None
    positions: tuple[int, ...] | None = None
    if (
        statuses
        and all(row_status == "complete" for row_status in statuses)
        and (started == n_total)
    ):
        evidence_by_row, positions = _derive_step_column(trace, resolved, started)

    # Settlement join grading (lane F40c): re-grade every join exactly
    # against the evidence column; the envelope rides the header's reserved
    # slot. No session (a bare ResolvedEpisode) writes NO envelope --
    # unmeasured, disclosed as such on every surface.
    from ._episode_join import _raise_settled_break_arm, settle_step_join

    full_statuses: list[str] = [*statuses, *(["absent"] * (n_total - started))]
    join_envelope = settle_step_join(trace, resolved, full_statuses, evidence_by_row)

    header = _build_header(
        trace,
        resolved,
        fidelity=_fidelity_basis(resolved, evidence_by_row),
        step_join=join_envelope,
        positions=positions,
    )

    frontier: dict[str, str] | None = None
    halt_frontier = getattr(trace, "halt_frontier", None)
    if halt_frontier:
        first = halt_frontier[0] if isinstance(halt_frontier, (list, tuple)) else halt_frontier
        frontier = {"boundary_kind": "op", "boundary_label": str(first)}

    from ._episode_coupling import settlement_fire_counts, stamp_episode_steps

    rows = _build_ledger_rows(
        n_total,
        statuses,
        calls,
        _RowDecorations(
            evidence_by_row=evidence_by_row,
            frontier=frontier,
            fire_counts=settlement_fire_counts(
                resolved.coupling_session, started=started, n_total=n_total
            ),
        ),
    )

    try:
        unbound = EpisodeLedger(header, rows)
        # Bind LAST (W051, audit 3.2): the digest covers every persisted
        # ledger fact, so it is minted over the validated unbound payload
        # and written into the header the product carries.
        header = dataclasses.replace(
            header,
            capture_digest=mint_capture_digest(
                trace, resolved.address, started, unbound.to_payload()
            ),
        )
        ledger = EpisodeLedger(header, rows)
    except ValueError as exc:
        raise _ledger_error(str(exc), code="episode_ledger_incoherent") from exc
    trace.annotations[EPISODE_ANNOTATIONS_KEY] = ledger.to_payload()
    stamp_episode_steps(trace, calls)

    # Strict/refuse arms fire AFTER the ledger is attached, so the raised
    # refusal's exc.partial_log carries the product WITH its measured
    # envelope (recoverable partial evidence, nothing discarded).
    if join_envelope is not None:
        _raise_settled_break_arm(resolved, header, join_envelope)
    return ledger


def escalation_spec(
    producer: Any,
    *,
    stepped_module: Any,
    reason: EscalationReason,
    n_steps: int | None = None,
    step_axis: int | None = None,
) -> EpisodeSpec:
    """Build the E-A escalation declaration FROM a cheap-tier episode product.

    E-A1: the escalation product of an episode is a NEW whole-episode wrapped
    session capture — same declaration, same inputs, same recorded entry
    seed, re-run from t=0; there is no partial escalation product under the
    session ruling. This helper derives the disclosure fields from the
    producer: ``escalated_from`` (the producer digest over its persisted
    outcome payload + ledger rows), the declared step count, and the
    producer's token column so the write-time E-A3 fidelity comparison can
    discharge (``tokens`` / ``diverged`` / ``none``).

    Re-run the escalation with the producer's recorded entry seed
    (``producer.random_seed``) to satisfy E-A1's same-seed requirement.

    Raises
    ------
    EpisodeLedgerError
        ``episode_ledger_without_declaration`` when the producer carries no
        finalized episode ledger.
    """

    from ..options import EpisodeSpec as _EpisodeSpec

    ledger = episode_ledger_for(producer)
    if ledger is None:
        raise _ledger_error(
            "escalation requires a producer carrying a finalized episode "
            "ledger; this product has none.",
            code="episode_ledger_without_declaration",
        )
    # Lane F40c: an escalation re-runs the WHOLE episode -- an episode-
    # dependent claim across every join. A producer with a measured broken
    # join refuses typed (per-segment reads stay available on its rows).
    from ._episode_join import refuse_broken_join_claim

    refuse_broken_join_claim(producer)
    outcome = producer.outcome
    outcome_payload = outcome.to_payload() if outcome is not None else {}
    digest = producer_digest(outcome_payload, ledger)
    expected = tuple(row.step_output for row in ledger.rows if isinstance(row.step_output, tuple))
    header = ledger.header
    return _EpisodeSpec(
        stepped_module=stepped_module,
        n_steps=n_steps if n_steps is not None else header.n_steps_declared,
        step_axis=(
            step_axis
            if step_axis is not None
            else (header.step_axis if header.step_axis is not None else -1)
        ),
        step_output_kind=header.step_output_kind,
        step_output_from=(
            None if header.step_output_from in (None, "output") else header.step_output_from
        ),
        escalated_from=digest,
        reason=reason,
        expected_tokens=expected if expected else None,
    )


def episode_ledger_for(trace: Any) -> EpisodeLedger | None:
    """Parse and return the trace's episode ledger, or ``None``.

    Returns ``None`` for non-episode products, pre-finalize declarations,
    and quarantined (load-degraded) payloads. This is the implementation
    behind the public step-access spelling ``Trace.episode`` (lane F40a;
    DOCUMENTED-UNSTABLE pending the naming session).
    """

    payload = getattr(trace, "annotations", {}).get(EPISODE_ANNOTATIONS_KEY)
    if not isinstance(payload, Mapping) or set(payload) != {"header", "rows"}:
        return None
    try:
        return EpisodeLedger.from_payload(payload)
    except (TypeError, ValueError):
        return None


#: Travel-drop note code minted by ``capture/_annotations_travel.py``: the one
#: episode-key payload shape that legally rides a NON-episode product (a fresh
#: re-execution whose evidence was dropped), so ``capture_kind_for`` reads it
#: as plain.
_TRAVEL_DROP_NOTE_CODE = "episode_evidence_dropped_fresh_execution"


def capture_kind_for(trace: Any) -> str:
    """Return the capture-kind marker for one capture product.

    ``"episode"`` for a product carrying an episode declaration (the
    pre-finalize marker, a finalized ledger, or a load-quarantined ledger
    record -- the declaration itself is not in doubt there); ``"plain"``
    otherwise, including fresh re-execution products whose episode evidence
    the annotations travel policy dropped (those are not episode products;
    the in-key note is the disclosure). Implementation behind the public
    ``Trace.capture_kind`` spelling (lane F40a; DOCUMENTED-UNSTABLE).
    """

    annotations = getattr(trace, "annotations", None)
    payload = annotations.get(EPISODE_ANNOTATIONS_KEY) if isinstance(annotations, dict) else None
    if not isinstance(payload, Mapping):
        return "plain"
    if payload.get("quarantined") is True and payload.get("code") == _TRAVEL_DROP_NOTE_CODE:
        return "plain"
    return CAPTURE_KIND_EPISODE


#: Loaded episode-key payload shapes that carry no ledger to validate: an
#: already-quarantined record round-tripping and the pre-finalize declaration
#: marker (session-time shape). Both load as-is.
_INERT_EPISODE_PAYLOAD_KEYS: tuple[frozenset[str], ...] = (
    frozenset({"quarantined", "code", "detail"}),
    frozenset({"declared"}),
)


def _inert_loaded_episode_shape(payload: Any) -> bool:
    """``True`` when the loaded episode-key payload needs no validation."""

    if payload is None:
        return True
    return isinstance(payload, Mapping) and frozenset(payload) in _INERT_EPISODE_PAYLOAD_KEYS


def _loaded_ledger_version(payload: Mapping[str, Any]) -> Any:
    """The family version the loaded header spells (``None`` when unreadable)."""

    header_payload = payload.get("header")
    if not isinstance(header_payload, Mapping):
        return None
    return header_payload.get("episode_ledger_version")


def validate_loaded_episode_annotations(trace: Trace) -> None:
    """Validate a loaded ``annotations["episode"]`` payload, FAIL-CLOSED.

    Load semantics (S7 L1/L2/L4 + the S2 combination table):

    - A payload whose header does not declare ``capture_kind=episode``
      refuses typed (``episode_ledger_without_declaration``): an episode
      ledger on a non-episode capture is ILLEGAL.
    - A structure-only ledger carrying tokens refuses typed
      (``episode_ledger_payload_in_structure_only``).
    - Any other geometry/parse violation QUARANTINES the payload (replaced
      by a diagnostic record; rows stop being claims) with ONE warning, and
      records the fail-closed degrade signal the outcome derivation consumes
      at the coordinated schema bump (the ledger never upgrades an outcome;
      the degrade wiring in the settlement authority is the S2 author's bump
      change, tracked on the record written here).
    - On UNATTESTED/UNKNOWN products a valid ledger loads as an UNVERIFIED
      disclosure — kept as-is; nothing gates on it.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        return
    payload = annotations.get(EPISODE_ANNOTATIONS_KEY)
    if _inert_loaded_episode_shape(payload):
        return
    parse_error: str | None = None
    if isinstance(payload, Mapping) and set(payload) == {"header", "rows"}:
        header_payload = payload.get("header")
        if (
            isinstance(header_payload, Mapping)
            and header_payload.get("capture_kind") != CAPTURE_KIND_EPISODE
        ):
            raise _ledger_error(
                "an episode ledger is attached to a capture whose header does "
                "not declare capture_kind=episode; an episode ledger without "
                "the declaration is illegal.",
                code="episode_ledger_without_declaration",
            )
        # Cross-grammar quarantine (C07X item (v), lane F40b): the family
        # version is consumed BEFORE the full parse so a foreign-grammar
        # payload quarantines with the grammar named, never as an incidental
        # unknown-key finding -- and it NEVER normalizes.
        loaded_version = _loaded_ledger_version(payload)
        if loaded_version != EPISODE_LEDGER_VERSION:
            parse_error = (
                f"episode ledger grammar version {loaded_version!r} is not the "
                f"supported family version {EPISODE_LEDGER_VERSION} (a missing "
                "version is the pre-v2 grammar, whose tokens/cache_len rows "
                "never normalize); episode grammar changes are family-local. "
                "Re-capture with this TorchLens to produce a current-grammar "
                "ledger."
            )
        else:
            try:
                EpisodeLedger.from_payload(payload)
            except (TypeError, ValueError) as exc:
                message = str(exc)
                if "episode_ledger_payload_in_structure_only" in message:
                    raise _ledger_error(
                        message, code="episode_ledger_payload_in_structure_only"
                    ) from exc
                parse_error = message
            else:
                # Structural anchors (W051, audit 2.19/3.2): a grammatically
                # valid ledger must also describe THIS product.
                parse_error = _anchor_loaded_ledger(trace, payload)
                if parse_error is None:
                    return  # valid — keep as claims (or unverified disclosure)
    else:
        parse_error = "episode annotations payload does not carry the {'header', 'rows'} shape"
    quarantine_loaded_ledger(annotations, parse_error, stacklevel=3)


# Re-export (import at the BOTTOM: the fold module imports this module's
# vocabularies, so a top placement would be a circular import): the S6-floor
# fold split out to ``_episode_fold`` at the C07X amendment.
from ._episode_fold import (  # noqa: E402
    EpisodeFoldResult,
    derive_episode_status,
)
