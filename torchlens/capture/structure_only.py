"""Structure-only capture: typed refusals, capability authority, discharge.

L7a wave-0 module (D8-default branch). One in-code capability authority
(:data:`STRUCTURE_ONLY_CAPABILITIES`) with ONE chokepoint
(:func:`require_structure_only_capability`), mirrored by the human table in
``docs/reference/structure_only_capabilities.md``; the teaching-refusal error
types raised by the torch mode belt and forward-boundary backstop; the
hypothesis claim vocabulary (:class:`StructureClaimStatus`); and the real-run
discharge machinery (:func:`discharge_against`, surfaced as
``Trace.discharge_against``).

NAMING: every public spelling in this module is DOCUMENTED-UNSTABLE pending
the rolling naming session, and every refusal code is additionally S2-gated
(the slate ratifies SPELLING, S2 ratifies EXISTENCE and regime). No
deprecation shim is owed on rename. S2 SEAM (labeled): registration of these
error classes in ``torchlens.errors`` and the settlement-side marker stamp in
``torchlens/capture/outcome.py`` ride the S2 author's ratification PR — this
module never touches the S2 fence files.
"""

from __future__ import annotations

import enum
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclasses_field
from typing import TYPE_CHECKING, Any, Final

from ..errors._base import CaptureError

if TYPE_CHECKING:
    pass

__all__ = [
    "STRUCTURE_ONLY_CAPABILITIES",
    "CapabilityRow",
    "MetaKernelUnavailableError",
    "StructureClaimStatus",
    "StructureDischarge",
    "StructureOnlyCapabilityError",
    "ValueDependentBranchError",
    "claim_status_for",
    "discharge_against",
    "registered_discharge",
    "require_structure_only_capability",
]


# ---------------------------------------------------------------------------
# Teaching-refusal error types (memo sec 2.3/2.4)
# ---------------------------------------------------------------------------


class ValueDependentBranchError(CaptureError, RuntimeError):
    """A value escape reached the Python host from USER code under
    ``structure_only=True`` (memo sec 2.2 Layer 1 / Layer 2).

    Device-neutral by contract: under structure-only the recorded graph must
    never be VALUE-SELECTED, so enumerated escapes from user frames refuse on
    meta AND real tensors alike. ``fields["code"]`` is
    ``value_dependent_branch_unsupported``; ``fields["consumer_kind"]`` uses
    the closed ``ast_branches`` vocabulary plus ``scalar_escape`` and
    ``unclassified_escape``; ``fields["offenses"]`` carries
    ``{file, line, consumer_kind, tensor_label, escape_method, substrate}``
    records.
    """


class MetaKernelUnavailableError(CaptureError, RuntimeError):
    """An op with no meta kernel died inside torch dispatch under
    ``structure_only=True`` (memo sec 2.4).

    Classification is by raising-frame provenance and exception family, never
    message text; the original ``NotImplementedError`` is chained via
    ``raise ... from``. ``fields["code"]`` is ``meta_kernel_unavailable``.
    """


class StructureOnlyCapabilityError(CaptureError, RuntimeError):
    """A value-requiring consumer refused on a structure-only trace.

    Raised only by :func:`require_structure_only_capability`;
    ``fields["code"]`` carries the row's stable refusal code and
    ``fields["capability"]`` the row key.
    """


# ---------------------------------------------------------------------------
# Claim vocabulary (memo sec 3.2; S2-gated, documented-unstable)
# ---------------------------------------------------------------------------


class StructureClaimStatus(str, enum.Enum):
    """Tri-state evidence class of a structure-only trace's value-bearing
    claims. Never silently promoted (G4): ``CORROBORATED`` is written only by
    the discharge authority, and a registered refuted discharge flips the
    in-session state to ``REFUTED`` (G5)."""

    HYPOTHESIS = "hypothesis"
    CORROBORATED = "corroborated"
    REFUTED = "refuted"


# ---------------------------------------------------------------------------
# Chokepoint refusal codes raised only through the capability rows below
# (constant-spelled so the error-contract lockstep sees them; enrolled in
# tests/test_error_contract_lockstep.py::_CONSTANT_SPELLED_CODES). All
# S2-gated, DOCUMENTED-UNSTABLE.
# ---------------------------------------------------------------------------

STRUCTURE_ONLY_MEASUREMENTS_UNSUPPORTED = "structure_only_measurements_unsupported"
STRUCTURE_ONLY_RUNNABLE_UNSUPPORTED = "structure_only_runnable_unsupported"
STRUCTURE_ONLY_REPLAY_UNSUPPORTED = "structure_only_replay_unsupported"
STRUCTURE_ONLY_VALIDATION_UNSUPPORTED = "structure_only_validation_unsupported"
STRUCTURE_ONLY_BACKWARD_UNSUPPORTED = "structure_only_backward_unsupported"
STRUCTURE_ONLY_EPISODE_UNSUPPORTED = "structure_only_episode_unsupported"


# ---------------------------------------------------------------------------
# Capability table (memo sec 6) — the IN-CODE authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRow:
    """One frozen row of the structure-only capability contract.

    ``status_v1`` uses the CLOSED five-member grammar
    ``supported_structural | supported_hypothesis | refuse:<code> |
    verify:<Vn> | out_of_scope:<contract-ref>``; ``refusal_code`` is populated
    exactly when the status is ``refuse:<code>``. The ``claim`` column's
    conditional wording is the contract: an amender may strike a condition it
    has FULFILLED, never widen the claim.
    """

    key: str
    claim: str
    status_v1: str
    flip_event: str
    evidence: str
    amend_owner: str
    refusal_code: str | None = None


_ROWS: Final[tuple[CapabilityRow, ...]] = (
    CapabilityRow(
        key="graph_structure",
        claim=(
            "The op graph, edges, order, and module nesting of THIS meta "
            "execution are recorded exactly."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py (E-1 structural pins)",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="param_geometry",
        claim=(
            "Parameter/buffer names, shapes, dtypes are recorded as declared; "
            "the persistence partition is recorded IF V8 verifies "
            "meta-compatibility."
        ),
        status_v1="supported_structural",
        flip_event="V8 verdict",
        evidence="pending: V8 verification test",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="shapes_dtypes",
        claim=(
            "Per-op shapes/dtypes are HYPOTHESES: valid under meta "
            "propagation, unproven until discharged by a real capture of the "
            "same graph."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_structure_only_discharge.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="flops_estimates",
        claim=(
            "FLOPs/MACs are derived from hypothesis shapes; they inherit "
            "hypothesis status and are labelled estimated, never measured."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_structure_only_honesty.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="memory_estimates",
        claim=(
            "Memory figures are geometry estimates; measured-memory columns "
            "render unknown, never zero."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_structure_only_honesty.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="taken_path_conditionals",
        claim=(
            "Conditional structure of the taken path is recorded; any "
            "VALUE-dependent branch through the enumerated escape surface "
            "refuses typed at the user's source line REGARDLESS of the "
            "tensor's device; unenumerated meta deaths refuse typed via the "
            "backstop; unenumerated REAL-value escapes in form (b) are "
            "undetectable and are priced by hypothesis status (coverage claim "
            "exactly per memo sec 2.1 C-ENUM/C-BACKSTOP/C-RESIDUAL)."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_teaching.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="meta_admission",
        claim=(
            "Meta-materialized models (form (a)) are ADMITTED under the D8 "
            "grant (JMT 2026-08-26), if and only if structure-only is in "
            "force (scoped admission, W2): the graph, module nesting, "
            "parameter geometry, and shape/dtype HYPOTHESES are recorded "
            "with no tensor values. Without structure_only the entry gate "
            "refuses meta unchanged."
        ),
        status_v1="supported_structural",
        flip_event="D8 granted 2026-08-26 (this row IS the flip; last merge)",
        evidence=(
            "tests/test_weightsfree_admission.py; parity gate: "
            "tests/test_weightsfree_parity.py (real digest == meta digest "
            "AND discharge CORROBORATED on every fixture)"
        ),
        amend_owner="S2-amendment",
    ),
    CapabilityRow(
        key="value_payloads",
        claim=(
            "Activations, argument values, output values are never recorded; requests refuse typed."
        ),
        status_v1="refuse:structure_only_values_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_values_unsupported",
    ),
    CapabilityRow(
        key="previews",
        claim="Value previews/thumbnails require values; refused.",
        status_v1="refuse:structure_only_values_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_values_unsupported",
    ),
    CapabilityRow(
        key="nonfinite_predicates",
        claim=(
            "raise_on_nan and nonfinite halt predicates have no values to "
            "test; the combination refuses typed at entry."
        ),
        status_v1="refuse:structure_only_option_conflict",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_option_conflict",
    ),
    CapabilityRow(
        key="runnable_ready_composition",
        claim=(
            "structure_only + runnable_ready is refused at entry: runnable "
            "eligibility and the structure substrate are incompatible in v1."
        ),
        status_v1="refuse:structure_only_option_conflict",
        flip_event="L7b amendment lands",
        evidence=(
            "tests/test_structure_only_entry.py; conflict lift rides the S2 "
            "StateSource amendment (request R-L7B-1) with the belt-coverage "
            "pin re-authored in the same change"
        ),
        amend_owner="L7b",
        refusal_code="structure_only_option_conflict",
    ),
    CapabilityRow(
        key="substrate_uniformity",
        claim=(
            "Admission requires a UNIFORM substrate: every input tensor leaf "
            "meta AND every registered parameter/buffer meta (tied objects "
            "deduplicated by identity; parameterless models judged by "
            "inputs). Mixed cells refuse typed in BOTH directions at entry "
            "with structure_only_substrate_mismatch, naming which side is "
            "which; a REAL tensor discovered mid-forward (stale pre-wrap "
            "factory reference, device='cpu' literal) refuses through the "
            "same family at the user's source line (W1-CLS)."
        ),
        status_v1="supported_structural",
        flip_event="D8 granted 2026-08-26",
        evidence="tests/test_weightsfree_admission.py (mixed cells, tamper rows)",
        amend_owner="S2-amendment",
    ),
    CapabilityRow(
        key="plan_shape_check",
        claim=(
            "The audit-only plan checker (Trace.check_plan, D14) resolves "
            "selectors, checks multiplicity, and compares declared "
            "replacement geometry against hypothesis shapes/dtypes; the "
            "report is executable=false ALWAYS and never arms, replays, or "
            "lifts the late-bind refusal. REFUTED sources refuse through "
            "this row (G5); callables and value-derived selection refuse "
            "typed (plan_check_unsupported)."
        ),
        status_v1="supported_hypothesis",
        flip_event="discharge",
        evidence="tests/test_weightsfree_plan_check.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="measurement_exports",
        claim=(
            "Measurement-shaped exports (chrome_trace, speedscope, "
            "flamegraph, memory_timeline) refuse typed: timings and "
            "allocator peaks are MEASUREMENTS, not payload values, and a "
            "value-free capture has none — meta-dispatch overhead rendered "
            "as 'measured' inverts real cost rankings (weightsfree memo "
            "D15; the existing values code must not silently widen)."
        ),
        status_v1="refuse:structure_only_measurements_unsupported",
        flip_event="never",
        evidence="tests/test_weightsfree_disclosure.py",
        amend_owner="S2-amendment",
        refusal_code=STRUCTURE_ONLY_MEASUREMENTS_UNSUPPORTED,
    ),
    CapabilityRow(
        key="viz_graph_render",
        claim=(
            "Graph rendering (incl. size_by consuming hypothesis shapes) "
            "works, carrying the structure-only banner."
        ),
        status_v1="supported_hypothesis",
        flip_event="never",
        evidence="tests/test_structure_only_honesty.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="viz_payload_visualizers",
        claim=(
            "Payload-consuming visualizers (activation heatmaps, custom value "
            "visualizers) require values; refused typed."
        ),
        status_v1="refuse:structure_only_values_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_entry.py",
        amend_owner="L7a",
        refusal_code="structure_only_values_unsupported",
    ),
    CapabilityRow(
        key="structure_digests",
        claim=(
            "Graph-shape and meta-domain content digests are always computed; "
            "they can never collide with value-bearing digests."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_honesty.py (G2 domain pin)",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="discharge",
        claim=(
            "A real capture of the same graph upgrades hypothesis rows to "
            "corroborated or refutes them; upgrades happen ONLY via the "
            "discharge authority."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_discharge.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="refuted_rows",
        claim=("Consumers that tolerate hypothesis rows refuse REFUTED rows typed (G5)."),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_discharge.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="teaching_refusals",
        claim=(
            "Enumerated value escapes (device-neutral, both forms) and "
            "missing meta kernels refuse typed with the user source line; "
            "other meta-mechanism deaths are typed via the backstop without "
            "branch classification; unenumerated real-value escapes are "
            "outside the detectable surface (C-RESIDUAL); user exceptions "
            "propagate unchanged."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_teaching.py",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="save_analysis_artifact",
        claim=(
            "Analysis-level artifacts persist the structure_only marker "
            "plainly (tlspec v8); loads validate marker coherence (M-C2/M-C3 "
            "in torchlens/_io/forgery_validation.py) and every value-claim "
            "on the loaded trace stays a HYPOTHESIS."
        ),
        status_v1="supported_structural",
        flip_event="never",
        evidence="tests/test_structure_only_capabilities.py",
        amend_owner="P1",
    ),
    CapabilityRow(
        key="save_runnable",
        claim=(
            "Runnable save is refused in v1; IF the L7b late-bind posture "
            "lands (wave 1, post-L4, S1-serialized), declared late-bind slots "
            "replace this refusal for eligible models."
        ),
        status_v1="refuse:structure_only_runnable_unsupported",
        flip_event="L7b amendment lands",
        evidence=(
            "tests/test_structure_only_capabilities.py; entry-dark bridge + "
            "mandatory bind-digest authority shipped: "
            "tests/test_structure_only_bridge.py (flip blocked on the S2 "
            "StateSource amendment, request R-L7B-1)"
        ),
        amend_owner="L7b",
        refusal_code=STRUCTURE_ONLY_RUNNABLE_UNSUPPORTED,
    ),
    CapabilityRow(
        key="live_replay",
        claim=(
            "Replay requires values; refused unless and until late-bind (see "
            "save_runnable) provides them at bind time."
        ),
        status_v1="refuse:structure_only_replay_unsupported",
        flip_event="L7b amendment lands",
        evidence=(
            "tests/test_structure_only_capabilities.py; entry-dark bridge + "
            "S1 validator-reuse binding path shipped: "
            "tests/test_structure_only_bridge.py (flip blocked on the S2 "
            "StateSource amendment, request R-L7B-1)"
        ),
        amend_owner="L7b",
        refusal_code=STRUCTURE_ONLY_REPLAY_UNSUPPORTED,
    ),
    CapabilityRow(
        key="validation_entry",
        claim=(
            "There is nothing to validate against; refused permanently by "
            "design (discharge is the verification story)."
        ),
        status_v1="refuse:structure_only_validation_unsupported",
        flip_event="never",
        evidence="tests/test_structure_only_capabilities.py",
        amend_owner="L7a",
        refusal_code=STRUCTURE_ONLY_VALIDATION_UNSUPPORTED,
    ),
    CapabilityRow(
        key="backward_grads",
        claim=(
            "Backward/gradient capture is refused in v1; any future support "
            "is an L9-adjacent S2 amendment, not implied here."
        ),
        status_v1="refuse:structure_only_backward_unsupported",
        flip_event="S2 amendment",
        evidence="tests/test_structure_only_capabilities.py",
        amend_owner="S2-amendment",
        refusal_code=STRUCTURE_ONLY_BACKWARD_UNSUPPORTED,
    ),
    CapabilityRow(
        key="episode_composition",
        claim=(
            "Episode capture composes with structure-only ONLY if a later S2 "
            "amendment rules it; refused in v1."
        ),
        status_v1="refuse:structure_only_episode_unsupported",
        flip_event="S2 amendment",
        evidence="reserved: no episode surface exists on this branch yet",
        amend_owner="S2-amendment",
        refusal_code=STRUCTURE_ONLY_EPISODE_UNSUPPORTED,
    ),
    CapabilityRow(
        key="distributed",
        claim=(
            "Distributed structure-only capture is out of scope this sprint; "
            "the existing distributed refusal contract governs."
        ),
        status_v1="out_of_scope:distributed-contract",
        flip_event="S2 amendment",
        evidence="torchlens/_distributed.py refusal surface",
        amend_owner="S2-amendment",
    ),
    CapabilityRow(
        key="fake_tensor_substrate",
        claim=(
            "Capturing a REAL model structure-only via FakeTensorMode is a "
            "VERIFY item, not a capability."
        ),
        status_v1="verify:V1",
        flip_event="V1 verdict",
        evidence="pending: V1 spike",
        amend_owner="L7a",
    ),
    CapabilityRow(
        key="symbolic_shapes",
        claim=(
            "Symbolic/dynamic shapes remain refused at the variant gate; a "
            "ShapeEnv-backed range hypothesis is a VERIFY item."
        ),
        status_v1="verify:V2",
        flip_event="V2 verdict",
        evidence="pending: V2 verification",
        amend_owner="L7a",
    ),
)

STRUCTURE_ONLY_CAPABILITIES: Final[Mapping[str, CapabilityRow]] = {row.key: row for row in _ROWS}
"""Frozen structure-only capability rows; consumers branch ONLY through
:func:`require_structure_only_capability` (lockstep-tested both directions)."""


_VALID_AMEND_OWNERS: Final[frozenset[str]] = frozenset({"L7a", "L7b", "S2-amendment", "P1"})

_L7B_ROW_TEACHING: Final[Mapping[str, str]] = {
    "save_runnable": (
        "This is the L7b v1 floor: runnable save flips to the declared "
        "late-bind posture (state slots declared at capture time, values "
        "bound at run time with mandatory byte digests) when its S2 "
        "StateSource amendment lands — this row's named flip event. Until "
        "then, capture the real model with intervention_ready=True to "
        "produce a runnable artifact."
    ),
    "live_replay": (
        "A structure-only capture records no values to replay. Until the "
        "L7b declared late-bind posture lands (this row's named flip "
        "event), run the real model directly, or corroborate this trace "
        "against a real capture via trace.discharge_against(real_trace)."
    ),
}
"""Row-scoped teaching sentences for the L7b-owned rows (house rule: every
refusal names the boundary and what to do instead). Amends refusal TEACHING
only — claims, statuses, and codes are untouched; the entries are keyed to
rows whose ``amend_owner`` is L7b."""


def _validate_row_grammar(row: CapabilityRow) -> None:
    """Enforce the closed 6.2 grammar at import; a bad row is a bug."""

    status = row.status_v1
    valid = (
        status in ("supported_structural", "supported_hypothesis")
        or (status.startswith("refuse:") and len(status) > len("refuse:"))
        or (status.startswith("verify:V") and status[len("verify:V") :].isdigit())
        or (status.startswith("out_of_scope:") and len(status) > len("out_of_scope:"))
    )
    if not valid:
        raise AssertionError(f"capability row {row.key!r} violates the closed grammar: {status!r}")
    if status.startswith("refuse:"):
        if row.refusal_code != status.split(":", 1)[1]:
            raise AssertionError(
                f"capability row {row.key!r}: refusal_code must equal the refuse:<code> cell"
            )
    elif row.refusal_code is not None:
        raise AssertionError(f"capability row {row.key!r}: refusal_code without refuse status")
    if row.amend_owner not in _VALID_AMEND_OWNERS:
        raise AssertionError(f"capability row {row.key!r}: unknown amend_owner {row.amend_owner!r}")
    if not row.claim.strip() or not row.flip_event.strip() or not row.evidence.strip():
        raise AssertionError(f"capability row {row.key!r} has an empty contract column")


for _row in _ROWS:
    _validate_row_grammar(_row)
del _row


# ---------------------------------------------------------------------------
# Discharge registry (weak-keyed side table; the trace object never mutates)
# ---------------------------------------------------------------------------

_DISCHARGE_REGISTRY: weakref.WeakKeyDictionary[Any, StructureDischarge] = (
    weakref.WeakKeyDictionary()
)
"""Per-structure-trace registered discharge (the ledger pattern of
``completeness_witness._HOST_ESCAPE`` tables). G4: only
:func:`discharge_against` writes here; G5: a refuted entry flips the
in-session claim state consulted by the chokepoint."""


def registered_discharge(trace: Any) -> StructureDischarge | None:
    """Return the registered discharge for ``trace``, if any."""

    return _DISCHARGE_REGISTRY.get(trace)


def claim_status_for(trace: Any) -> StructureClaimStatus:
    """Return the in-session claim status of a structure-only trace.

    Born ``HYPOTHESIS``; flipped only by a registered discharge (G4/G5).
    """

    discharge = _DISCHARGE_REGISTRY.get(trace)
    if discharge is None:
        return StructureClaimStatus.HYPOTHESIS
    if discharge.verdict is StructureClaimStatus.REFUTED:
        return StructureClaimStatus.REFUTED
    return StructureClaimStatus.CORROBORATED


# ---------------------------------------------------------------------------
# THE chokepoint (memo sec 6.1; G1)
# ---------------------------------------------------------------------------


def meta_admission_open() -> bool:
    """Whether the ``meta_admission`` row is D8-flipped (weightsfree W2).

    The capability table is the ONE code authority for the flip (memo build
    item 16): the admission plumbing in ``torchlens._robustness`` reads the
    row state through this accessor — never by subscripting the table — so
    the flip stays a one-row status change here, mirrored in
    ``docs/reference/structure_only_capabilities.md`` in the same commit.
    """

    return not STRUCTURE_ONLY_CAPABILITIES["meta_admission"].status_v1.startswith("refuse:")


def require_structure_only_capability(
    trace: Any,
    capability: str,
    *,
    detail: str | None = None,
) -> CapabilityRow | None:
    """Enforce one structure-only capability row for ``trace``.

    No-op (returns ``None``) when ``trace`` is not a structure-only capture:
    the default path pays one attribute read. For structure-only traces a
    ``refuse:<code>`` row raises :class:`StructureOnlyCapabilityError` with
    the row's stable code, a ``supported_hypothesis`` row additionally
    refuses when a registered REFUTED discharge has flipped the trace's claim
    state (G5), and supported/verify rows return the row.

    The pre-bump ``save_analysis_artifact`` switch bypass is retired: the
    marker persists plainly as of tlspec v8 and the row is supported.
    """

    if not bool(getattr(trace, "structure_only", False)):
        return None
    row = STRUCTURE_ONLY_CAPABILITIES[capability]
    if row.status_v1.startswith("refuse:"):
        code = row.refusal_code or row.status_v1.split(":", 1)[1]
        message = (
            f"TorchLens refuses {capability!r} for a structure-only capture "
            f"(code {code}). {row.claim}"
        )
        if detail:
            message += f" {detail}"
        teaching = _L7B_ROW_TEACHING.get(capability)
        if teaching is not None:
            message += f" {teaching}"
        message += (
            " Remedy: run a real capture (tl.trace without structure_only) "
            "for value-bearing surfaces, or see "
            "docs/reference/structure_only_capabilities.md."
        )
        raise StructureOnlyCapabilityError(
            message,
            code=code,
            capability=capability,
            status=row.status_v1,
            flip_event=row.flip_event,
        )
    if (
        row.status_v1 == "supported_hypothesis"
        and claim_status_for(trace) is StructureClaimStatus.REFUTED
    ):
        discharge = _DISCHARGE_REGISTRY.get(trace)
        first = discharge.first_contradiction if discharge is not None else None
        raise StructureOnlyCapabilityError(
            f"TorchLens refuses {capability!r}: this structure-only "
            "capture's hypotheses were REFUTED by a registered real-run "
            f"discharge (first contradiction: {first}). A refuted "
            "hypothesis is strictly worse than no capture. Remedy: "
            "re-capture after fixing the model/meta divergence, or "
            "consume the discharge record's contradiction table directly.",
            code="structure_only_refuted_hypothesis",
            capability=capability,
            status=row.status_v1,
        )
    return row


# ---------------------------------------------------------------------------
# Discharge machinery (memo sec 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimComparison:
    """One per-claim discharge row."""

    claim_kind: str
    site: str
    hypothesis_value: Any
    observed_value: Any
    verdict: StructureClaimStatus


#: The versioned comparison vocabulary (D5): IDENTITY in v1 — comparison
#: digests equal the public digests and unknown spelling pairs fail closed as
#: refutations. A populated vocabulary is the named fallback only if the
#: owned-context/absorption mechanism fails feature detection somewhere.
COMPARISON_VOCABULARY_V1: Final[str] = "identity-v0"

#: Reserved comparison row kinds (D9): the one-sided-knowledge join rule is
#: endorsed policy, guarded, and DEFERRED — nothing emits these in v1.
RESERVED_COMPARISON_ROW_KINDS: Final[tuple[str, ...]] = ("op_identity_unverified",)


@dataclass(frozen=True)
class StructureDischarge:
    """Frozen result of discharging a structure-only trace against a real
    capture. ``verdict`` is CORROBORATED iff EVERY compared claim matched
    (first contradiction wins the overall floor); ``claims`` is the per-claim
    table; the digest pairs witness which graphs were joined.

    The comparison envelope (D5, identity-v0): ``structure_digest`` /
    ``real_digest`` are the RAW PUBLIC digests (``tl.hash.trace``, untouched
    byte-for-byte by this wave); the ``comparison_*`` digests are the
    vocabulary-normalized pair — equal to the public pair at identity-v0.
    ``claim_counts`` breaks the per-claim table down by kind;
    ``unavailable_evidence`` counts claims neither side could make (empty at
    identity-v0; the D9 row kinds are reserved, never emitted).
    """

    verdict: StructureClaimStatus
    claims: tuple[ClaimComparison, ...]
    structure_digest: str
    real_digest: str
    graph_matched: bool
    first_contradiction: str | None
    comparison_vocabulary: str = COMPARISON_VOCABULARY_V1
    comparison_structure_digest: str = ""
    comparison_real_digest: str = ""
    claim_counts: Mapping[str, int] = dataclasses_field(default_factory=dict)
    unavailable_evidence: Mapping[str, int] = dataclasses_field(default_factory=dict)
    reserved_row_kinds: tuple[str, ...] = RESERVED_COMPARISON_ROW_KINDS

    def report(self) -> str:
        """One-screen human projection (W1-RPT): verdict, vocabulary, both
        digest pairs, claim counts by kind, unavailable-evidence counts, and
        the first contradiction — measured to be the difference between a
        20-minute diff hunt and a one-line answer (3 phantom records once
        produced 300 apparent diffs and zero per-claim rows)."""

        lines = [
            f"discharge verdict: {self.verdict.value.upper()}",
            f"comparison vocabulary: {self.comparison_vocabulary}",
            f"public digests:     structure={self.structure_digest[:16]}... "
            f"real={self.real_digest[:16]}... "
            f"({'EQUAL' if self.structure_digest == self.real_digest else 'DIFFER'})",
            f"comparison digests: structure={self.comparison_structure_digest[:16]}... "
            f"real={self.comparison_real_digest[:16]}...",
            "claims by kind: "
            + (
                ", ".join(f"{kind}={count}" for kind, count in sorted(self.claim_counts.items()))
                or "none compared"
            ),
            "unavailable evidence: "
            + (
                ", ".join(
                    f"{kind}={count}" for kind, count in sorted(self.unavailable_evidence.items())
                )
                or "none"
            ),
        ]
        if self.first_contradiction is not None:
            lines.append(f"first contradiction: {self.first_contradiction}")
        return "\n".join(lines)


def _refuse_incomparable(reason: str, detail: str) -> None:
    """Raise the typed comparable-twins preflight refusal (D10).

    REFUSE IS NOT REFUTE: the discharge registry is untouched, the structure
    trace keeps its HYPOTHESIS status, and the remedy names the comparability
    condition to fix rather than declaring the hypotheses wrong.
    """

    raise StructureOnlyCapabilityError(
        f"discharge_against refuses: the twins are not comparable ({reason}). "
        f"{detail} Refuse is not refute: no verdict was registered and the "
        "structure trace's claims remain HYPOTHESES. Remedy: re-capture with "
        "matched twins (same class/config, same versions, eval on both, both "
        "constructed before the first capture) and discharge again.",
        code="structure_only_discharge_incomparable",
        capability="discharge",
        reason=reason,
    )


def _input_geometry(trace: Any) -> tuple[tuple[tuple[int, ...] | None, str], ...]:
    """The (shape, dtype) tuple of each input layer, for the preflight."""

    facts: list[tuple[tuple[int, ...] | None, str]] = []
    for label in getattr(trace, "input_layers", ()) or ():
        layer = trace[label]
        shape = getattr(layer, "shape", None)
        facts.append((tuple(shape) if shape else None, str(getattr(layer, "dtype", None))))
    return tuple(facts)


def _discharge_preflight(structure_trace: Any, real_trace: Any) -> None:
    """The comparable-twins preflight (D10/D8): refuses typed, never refutes.

    Conditions checked against session and trace facts: same model class,
    same input geometry, same backend runtime version, same wrap generation
    (W1-ORD — both twins captured under the same torch patch state; honest
    twins constructed on opposite sides of a wrap flip refute each other on
    saved-reference bindings alone), no rescue-flag asymmetry, and a real
    oracle carrying no opaque host-write completeness witness (an oracle
    whose own value truth is UNVERIFIABLE cannot corroborate anything).
    Training-mode asymmetry has no dedicated trace fact and is caught
    structurally by the W1-RPT alignment (dropout/BN-stat records diverge).
    """

    structure_class = getattr(structure_trace, "model_class_name", None)
    real_class = getattr(real_trace, "model_class_name", None)
    if structure_class and real_class and structure_class != real_class:
        _refuse_incomparable(
            "model_class",
            f"structure side captured {structure_class!r}, real side {real_class!r}.",
        )
    structure_inputs = _input_geometry(structure_trace)
    real_inputs = _input_geometry(real_trace)
    if structure_inputs != real_inputs:
        _refuse_incomparable(
            "input_geometry",
            f"structure side inputs {structure_inputs!r} vs real side {real_inputs!r}.",
        )
    structure_rt = getattr(structure_trace, "backend_runtime_version", None)
    real_rt = getattr(real_trace, "backend_runtime_version", None)
    if structure_rt and real_rt and structure_rt != real_rt:
        _refuse_incomparable(
            "backend_runtime_version",
            f"structure side ran {structure_rt!r}, real side {real_rt!r}.",
        )
    from ._weightsfree_admission import wrap_generation_of

    structure_generation = wrap_generation_of(structure_trace)
    real_generation = wrap_generation_of(real_trace)
    if (
        structure_generation is not None
        and real_generation is not None
        and structure_generation != real_generation
    ):
        _refuse_incomparable(
            "wrap_generation",
            f"the twins were captured under different torch wrap generations "
            f"({structure_generation} vs {real_generation}); construct both "
            "twins before the first capture, or both after (W1-ORD).",
        )
    structure_rescued = bool(getattr(structure_trace, "rescue_rerun", None))
    real_rescued = bool(getattr(real_trace, "rescue_rerun", None))
    if structure_rescued != real_rescued:
        _refuse_incomparable(
            "rescue_asymmetry",
            "exactly one side was produced by a rescue re-run; the rescued "
            "side's graph provenance is not comparable to the primary's.",
        )
    from ..backends.torch.completeness_witness import _HOST_ESCAPE_MUTABLE_WRITEBACK

    if real_trace in _HOST_ESCAPE_MUTABLE_WRITEBACK:
        _refuse_incomparable(
            "real_oracle_opaque_witness",
            "the real oracle carries an opaque host-write completeness "
            "witness: its own captured values are UNVERIFIABLE, so it cannot "
            "corroborate hypotheses.",
        )


_ADOPTION_MARKER: Final[str] = "internalsource"


def _structural_alignment_contradiction(structure_trace: Any, real_trace: Any) -> str:
    """W1-RPT: name the count/position delta structurally, never bare digests.

    Also the D8 discriminant: when the first divergence pairs an ADOPTION
    record against a NAMED op, the refutation is a construction-order
    artifact (a saved pre-wrap function reference on exactly one twin) and
    the preflight refusal fires instead of a verdict.
    """

    structure_funcs = [
        str(getattr(layer, "func_name", "?")) for layer in structure_trace.layer_list
    ]
    real_funcs = [str(getattr(layer, "func_name", "?")) for layer in real_trace.layer_list]
    for index, (structure_func, real_func) in enumerate(
        zip(structure_funcs, real_funcs, strict=False)
    ):
        if structure_func != real_func:
            adoption_pair = (_ADOPTION_MARKER in structure_func.lower()) != (
                _ADOPTION_MARKER in real_func.lower()
            )
            if adoption_pair:
                _refuse_incomparable(
                    "construction_order",
                    "an adoption record sits opposite a named op at position "
                    f"{index} ({structure_func!r} vs {real_func!r}): whichever "
                    "twin was constructed before TorchLens's first wrap holds "
                    "pre-wrap function references. Construct both twins before "
                    "the first capture, or both after (W1-ORD).",
                )
            return (
                f"structural alignment: first divergence at record {index}: "
                f"structure side has {structure_func!r}, real side has {real_func!r} "
                f"(record counts: structure {len(structure_funcs)}, real {len(real_funcs)})"
            )
    if len(structure_funcs) != len(real_funcs):
        return (
            f"structural alignment: record count delta — structure side has "
            f"{len(structure_funcs)} records, real side {len(real_funcs)}; the shorter "
            "stream is a prefix of the longer (extra records start at position "
            f"{min(len(structure_funcs), len(real_funcs))})"
        )
    # Same func stream and count: the digest difference lies in per-record
    # facts (shapes/dtypes) or topology — name the first such divergence.
    for index, (structure_layer, real_layer) in enumerate(
        zip(structure_trace.layer_list, real_trace.layer_list, strict=False)
    ):
        for fact in ("shape", "dtype"):
            structure_fact = getattr(structure_layer, fact, None)
            real_fact = getattr(real_layer, fact, None)
            if str(structure_fact) != str(real_fact):
                return (
                    f"structural alignment: record {index} "
                    f"({structure_funcs[index]!r}) diverges on {fact}: "
                    f"structure side {structure_fact!r} vs real side {real_fact!r}"
                )
    return (
        "structural alignment: the (func, shape, dtype) streams agree; the "
        "digest difference lies in graph topology (parent wiring)"
    )


def _require_discharge_preconditions(structure_trace: Any, real_trace: Any) -> None:
    """Typed precondition refusals (never silent Nones)."""

    if not bool(getattr(structure_trace, "structure_only", False)):
        raise StructureOnlyCapabilityError(
            "discharge_against is only defined on a structure-only capture; "
            "this trace is an ordinary capture. Remedy: call it on the "
            "structure-only trace, passing the real capture as the argument.",
            code="structure_only_discharge_precondition",
            capability="discharge",
        )
    if bool(getattr(real_trace, "structure_only", False)):
        raise StructureOnlyCapabilityError(
            "discharge_against requires an ORDINARY (non-structure-only) real "
            "capture as the oracle; received another structure-only trace. "
            "Remedy: capture the model without structure_only and discharge "
            "against that trace.",
            code="structure_only_discharge_precondition",
            capability="discharge",
        )
    real_outcome = getattr(real_trace, "outcome", None)
    status_value = getattr(getattr(real_outcome, "status", None), "value", None)
    if status_value != "complete":
        raise StructureOnlyCapabilityError(
            "discharge_against requires a settled COMPLETE real capture as "
            f"the oracle; received outcome status {status_value!r}. Remedy: "
            "re-run the real capture to completion.",
            code="structure_only_discharge_precondition",
            capability="discharge",
        )


def _layer_claims(layer: Any) -> tuple[tuple[str, Any], ...]:
    """Extract the compared claim kinds from one layer record."""

    shape = getattr(layer, "shape", None)
    param_shapes = getattr(layer, "param_shapes", None) or ()
    return (
        ("shape", tuple(shape) if shape is not None else None),
        ("dtype", str(getattr(layer, "dtype", None))),
        # Layer.param_shapes is an ordered sequence of parameter shapes
        # (weight, bias, ...); order is part of the geometry claim.
        ("param_geometry", tuple(tuple(entry) for entry in param_shapes)),
    )


def discharge_against(structure_trace: Any, real_trace: Any) -> StructureDischarge:
    """Discharge a structure-only trace's hypotheses against a real capture.

    The join is POSITIONAL over ``layer_list``, LICENSED BY DIGEST EQUALITY
    of the address-free graph-shape hash (``tl.hash.trace``): the digest
    covers each record's index and parent_indices, so equal digests guarantee
    identically-ordered record sequences and record *i* corresponds to record
    *i* — repeated ops, recurrent passes, multi-output layers, and symmetric
    subgraphs are covered by construction. Never joins by label string.

    Structurally different graphs settle ``REFUTED`` at the graph level
    without any per-claim comparison. The result registers in the weak-keyed
    side table (G5) and NEVER mutates either trace.
    """

    from .. import hash as tl_hash

    _require_discharge_preconditions(structure_trace, real_trace)
    _discharge_preflight(structure_trace, real_trace)
    structure_digest = tl_hash.trace(structure_trace)
    real_digest = tl_hash.trace(real_trace)
    if structure_digest != real_digest:
        # W1-RPT: align the two record sequences structurally and name the
        # count/position delta BEFORE any "digests differ" fallback. The
        # construction-order discriminant inside may refuse typed instead
        # (refuse is not refute — nothing registers below in that case).
        first = _structural_alignment_contradiction(structure_trace, real_trace)
        discharge = StructureDischarge(
            verdict=StructureClaimStatus.REFUTED,
            claims=(),
            structure_digest=structure_digest,
            real_digest=real_digest,
            graph_matched=False,
            first_contradiction=first,
            comparison_structure_digest=structure_digest,
            comparison_real_digest=real_digest,
        )
        _DISCHARGE_REGISTRY[structure_trace] = discharge
        return discharge

    claims: list[ClaimComparison] = []
    first_contradiction: str | None = None
    for index, (hyp_layer, real_layer) in enumerate(
        zip(structure_trace.layer_list, real_trace.layer_list, strict=False)
    ):
        site = (
            getattr(hyp_layer, "label", None)
            or getattr(hyp_layer, "layer_label", None)
            or f"layer[{index}]"
        )
        for claim_kind, hyp_value in _layer_claims(hyp_layer):
            observed = dict(_layer_claims(real_layer))[claim_kind]
            matched = hyp_value == observed
            claims.append(
                ClaimComparison(
                    claim_kind=claim_kind,
                    site=str(site),
                    hypothesis_value=hyp_value,
                    observed_value=observed,
                    verdict=(
                        StructureClaimStatus.CORROBORATED
                        if matched
                        else StructureClaimStatus.REFUTED
                    ),
                )
            )
            if not matched and first_contradiction is None:
                first_contradiction = (
                    f"{site}: {claim_kind} hypothesis {hyp_value!r} vs observed {observed!r}"
                )
    claim_counts: dict[str, int] = {}
    for claim in claims:
        claim_counts[claim.claim_kind] = claim_counts.get(claim.claim_kind, 0) + 1
    discharge = StructureDischarge(
        verdict=(
            StructureClaimStatus.REFUTED
            if first_contradiction is not None
            else StructureClaimStatus.CORROBORATED
        ),
        claims=tuple(claims),
        structure_digest=structure_digest,
        real_digest=real_digest,
        graph_matched=True,
        first_contradiction=first_contradiction,
        comparison_structure_digest=structure_digest,
        comparison_real_digest=real_digest,
        claim_counts=claim_counts,
    )
    _DISCHARGE_REGISTRY[structure_trace] = discharge
    return discharge
