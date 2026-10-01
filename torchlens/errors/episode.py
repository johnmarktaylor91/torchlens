"""Episode-capture and bundle-relation exception families (L2 core).

Stable refusal codes ride ``exc.fields["code"]``; public code branches on the
code value or the structured report, never on exception text. The exhaustive
episode code vocabulary is maintained in
``docs/reference/error_refusal_contract.md``.

Every spelling is DOCUMENTED-UNSTABLE pending the rolling naming session;
the codes themselves are the stable branch surface.
"""

from __future__ import annotations

from enum import Enum

from ._base import ConfigurationError, TorchLensError, ValidationError

__all__ = [
    "BundleRelationError",
    "CheckpointSeriesLiveParamsError",
    "EpisodeCaptureError",
    "EpisodeDeclarationError",
    "EpisodeErrorCode",
    "EpisodeJoinError",
    "EpisodeLedgerError",
]


class EpisodeErrorCode(str, Enum):
    """Frozen episode/bundle-relation refusal codes (S6/S7 contracts)."""

    #: Ledger geometry violates the monotone prefix law, a coherence arm, or
    #: the value-mode token presence rule; the outcome derivation degrades
    #: fail-closed to UNKNOWN.
    EPISODE_LEDGER_INCOHERENT = "episode_ledger_incoherent"

    #: Declared episode-carried state without snapshot/restore support inside
    #: the declared checkpoint scope (E-A4/E-B4); refuses at DECLARATION time,
    #: before execution.
    EPISODE_STATE_UNSNAPSHOTABLE = "episode_state_unsnapshotable"

    #: An episode ledger is present on a capture that carries no episode
    #: declaration (S2 combination table: ILLEGAL, refuse at load).
    EPISODE_LEDGER_WITHOUT_DECLARATION = "episode_ledger_without_declaration"

    #: A structure-only episode ledger carries token payloads (S7 presence
    #: rule, structure-only direction).
    EPISODE_LEDGER_PAYLOAD_IN_STRUCTURE_ONLY = "episode_ledger_payload_in_structure_only"

    #: The episode declaration itself is unusable (stepped module not a
    #: submodule of the episode root, token axis unreadable, forced-token
    #: declaration malformed, or a structure-only episode was declared —
    #: TYPED REFUSE per the S2 combination table this sprint).
    EPISODE_DECLARATION_INVALID = "episode_declaration_invalid"

    #: The declared step count exceeds the wrapped-tier cost ceiling without
    #: the explicit acknowledgement (the measured cost is SUPERLINEAR:
    #: gpt2-124M CPU N=100 = 657 s / 5.4 GB peak RSS); refused at declaration
    #: time, before execution.
    EPISODE_STEP_CEILING_EXCEEDED = "episode_step_ceiling_exceeded"

    #: A measured cross-step join is broken by UNDECLARED exogenous content
    #: (lane F40c): the trace is intact and chain-shaped at the join, and
    #: episode-dependent claims across it (step-series reads, whole-episode
    #: replay, the blessing fold, escalation) refuse. Also the settlement
    #: refusal under the FORK-4 ``on_feed_break="refuse"`` arm (partial
    #: evidence on ``exc.partial_log``).
    EPISODE_FEED_BREAK_EXOGENOUS = "episode_feed_break_exogenous"

    #: ``feed="closed"`` was declared and an UNDECLARED crossing was
    #: measured: the strict arm halts capture at the next step entry (the
    #: first observable point; one-step detection latency), or settles typed
    #: when the break is only measurable against the settlement evidence
    #: column.
    EPISODE_FEED_CLOSED_VIOLATION = "episode_feed_closed_violation"

    #: ``feed="closed"`` with a DECLARED crossing: capture stops before
    #: entering the declared crossing step (no observation needed -- the
    #: crossing is declared).
    EPISODE_DECLARED_CROSSING_STOP = "episode_declared_crossing_stop"

    #: An episode-SERIES claim was made over unmeasured or unchecked joins
    #: (absent ``step_join`` envelope, or an ``unchecked`` grade): an
    #: unmeasured cross-step join can never support a series claim. Per-row
    #: reads are unaffected.
    EPISODE_JOIN_UNMEASURED = "episode_join_unmeasured"

    #: An episode-dependent claim crosses a DECLARED crossing (lane F40c):
    #: disclosed, chain-shaped by declaration, still not one closed feed.
    EPISODE_JOIN_DECLARED_CROSSING = "episode_join_declared_crossing"

    #: The ledger's persisted capture digest disagrees with the digest
    #: recomputed from the product it rides (lane F42): a foreign or
    #: tampered ledger -- the exact wrongness the binding exists to catch.
    EPISODE_COUPLING_UNBOUND = "episode_coupling_unbound"

    #: The ledger predates capture-digest minting (no binding exists), so
    #: the coupling attestation cannot run (lane F42); re-capture to bind.
    EPISODE_COUPLING_UNMINTABLE = "episode_coupling_unmintable"

    #: ``run()`` on a COUPLED episode product (lane F42): the ledger attests
    #: one perturbed execution and no run provider re-arms the capture-time
    #: intervention, so no engine can derive a fresh ledger -- the verdict's
    #: refusal arm (re-capturing with episode= x intervene= is the
    #: fresh-ledger arm).
    EPISODE_COUPLED_REPLAY_UNDERIVABLE = "episode_coupled_replay_underivable"

    #: A bundle relation row names a member absent from the Bundle (S6 R1:
    #: no dangling edges, ever).
    BUNDLE_RELATION_MEMBER_MISSING = "bundle_relation_member_missing"

    #: A bundle mutator would orphan relation rows (S6 R5: cascade explicitly
    #: or refuse typed; silent orphaning is forbidden).
    BUNDLE_MEMBER_HAS_RELATIONS = "bundle_member_has_relations"

    #: A relation row is outside the closed S6 schema (unknown kind, wrong
    #: row shape for the kind, or undeclared/missing-required param keys —
    #: S6 R2, grammar v2's required/optional split included).
    BUNDLE_RELATION_SCHEMA_INVALID = "bundle_relation_schema_invalid"

    #: A relation row's evidence envelope crosses the per-row canonical-JSON
    #: byte budget (C07X item 7; the sidecar family default of 1 MiB).
    #: Evidence envelopes are metadata-sized graded claims — bulk payloads
    #: belong in a sidecar family, cited by digest.
    BUNDLE_RELATION_EVIDENCE_OVER_BUDGET = "bundle_relation_evidence_over_budget"

    #: A cross-member parameter value/difference/trajectory read with no
    #: immutable capture-time parameter evidence on every member (A-CKPT).
    #: Parameter reads resolve through a live model handle (or nothing at
    #: all after deserialization), so per-member "capture value" claims are
    #: unprovable; the read refuses BEFORE tensor lookup, keyed on the
    #: claim, never on Python object identity.
    CHECKPOINT_SERIES_LIVE_PARAMS = "checkpoint_series_live_params"


class EpisodeCaptureError(TorchLensError):
    """Base class for episode-capture (``capture_kind=episode``) failures."""


class EpisodeDeclarationError(EpisodeCaptureError, ConfigurationError, ValueError):
    """Episode declaration refused at entry, before execution.

    ``fields["code"]`` carries ``episode_state_unsnapshotable``,
    ``episode_declaration_invalid``, or
    ``episode_step_ceiling_exceeded``.
    """


class EpisodeLedgerError(EpisodeCaptureError, ValidationError, ValueError):
    """Episode ledger validation refusal.

    ``fields["code"]`` carries ``episode_ledger_incoherent``,
    ``episode_ledger_without_declaration``, or
    ``episode_ledger_payload_in_structure_only``.
    """


class EpisodeJoinError(EpisodeCaptureError, ValidationError, RuntimeError):
    """Cross-step join (``step_join``) refusal (lane F40c).

    ``fields["code"]`` carries ``episode_feed_break_exogenous``,
    ``episode_feed_closed_violation``, ``episode_declared_crossing_stop``,
    ``episode_join_unmeasured``, or ``episode_join_declared_crossing``;
    ``fields["break_step"]`` names the broken join's step where one exists.
    Strict-arm instances raised mid-capture carry the settled partial
    product on ``exc.partial_log`` (recover with
    ``tl.partial.from_failed_capture(exc)``).
    """


class BundleRelationError(ValidationError, ValueError):
    """Bundle member-relation table refusal (S6, grammar v2).

    ``fields["code"]`` carries ``bundle_relation_member_missing``,
    ``bundle_member_has_relations``, ``bundle_relation_schema_invalid``, or
    ``bundle_relation_evidence_over_budget``.
    """


class BundleExperimentError(ValidationError, RuntimeError):
    """Experiment-layer bundle verb refusal or disclosed partial outcome (F03).

    ``fields["code"]`` carries ``vary_mapping_invalid``,
    ``vary_member_unknown``, ``vary_member_duplicate``,
    ``vary_coverage_incomplete``, or ``vary_partial_failure`` (whose fields
    additionally carry ``material_action_completed`` and the per-member
    ``outcomes`` record — a mid-apply failure claims NO rollback).
    """


class CheckpointSeriesLiveParamsError(ValidationError, RuntimeError):
    """Cross-member parameter read refused: no immutable capture-time evidence.

    TorchLens records which parameter a run used, not its bytes, so a
    parameter read resolves through the live model and returns TODAY'S
    weights, not the weights at capture; after deserialization the bytes are
    simply gone. Any cross-member parameter value/difference/trajectory read
    is therefore an unprovable historical claim until every member carries
    immutable parameter evidence (capture-time snapshots via R8(b), a
    runnable artifact bound to immutable weights, or a digest tied to
    immutable checkpoint bytes).

    ``fields["code"]`` carries ``checkpoint_series_live_params``;
    ``fields["members"]`` the member names, ``fields["param_address"]`` the
    parameter address, and ``fields["bases"]`` the per-member derived value
    bases. The class name is DOCUMENTED-UNSTABLE pending the naming session;
    branch on the code.
    """
