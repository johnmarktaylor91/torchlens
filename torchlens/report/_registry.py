"""The report-family SurfaceRegistry (F09; sumfam D22-D24, item 16).

ONE registry; everything is a projection of it: the "which do I use"
docs page, the More: cross-links, agent guidance, and the capability
card are all GENERATED from these rows -- the hand-written map had
already rotted (seven unconditional entries, eight omissions, one entry
unexecutable by its own reader).

Row vocabulary (D12): subjects are MODEL / RUN-RESOURCES / RUN-VALUES /
RUN-HEALTH / CAPTURE / ENVIRONMENT; registers are the artifact's KIND
(SHAPE conserving table, RANK, VERDICT findings, NARRATIVE, ADDRESS
machine map, LOG); the honesty obligation is an invariant SET
(``conserves_totals``, ``truncation_disclosed``,
``checks_and_skips_complete``, ``mints_no_facts``, ``stable_addressing``,
``instrumented_time_labeled``).

Cross-links are TYPED records (D23): trigger reason, the unanswered
question, a copyable command, one-line reason, cost class, requires --
rendered only when THIS report holds the triggering evidence, never a
standing menu.

The capability card (D24) is the registry filtered by the object in
hand: which members work, which refuse and why, and what each costs on
THIS object. Metadata-only -- building the card never scans, captures,
or executes anything.

Every spelling here executes in CI against a real fixture in its
declared state (tests/test_report_family_registry.py). Spellings
DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Closed subject vocabulary (D12).
SUBJECTS: tuple[str, ...] = (
    "MODEL",
    "RUN-RESOURCES",
    "RUN-VALUES",
    "RUN-HEALTH",
    "CAPTURE",
    "ENVIRONMENT",
)

#: Closed register vocabulary (D12): the artifact's KIND, never its
#: rendering medium.
REGISTERS: tuple[str, ...] = ("SHAPE", "RANK", "VERDICT", "NARRATIVE", "ADDRESS", "LOG")

#: Closed invariant vocabulary (D12).
INVARIANTS: tuple[str, ...] = (
    "conserves_totals",
    "truncation_disclosed",
    "checks_and_skips_complete",
    "mints_no_facts",
    "stable_addressing",
    "instrumented_time_labeled",
)

#: Closed cost-class vocabulary (D4).
COST_CLASSES: tuple[str, ...] = (
    "metadata_only",
    "payload_scan",
    "env_probe",
    "new_forward",
    "external_tool",
)

#: Closed requirement tokens the capability card can evaluate on an
#: object WITHOUT running anything (D24: metadata-only).
REQUIREMENT_TOKENS: tuple[str, ...] = (
    "finished_trace",
    "live_model",
    "grad_observed",
)


@dataclass(frozen=True)
class NextStep:
    """One typed cross-link record (D23).

    Rendered ONLY when the triggering evidence is in hand; at most two
    human-facing links per render; no link invokes its target.
    """

    trigger_code: str
    question: str
    command: str
    reason: str
    cost_class: str
    requires: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceEntry:
    """One registry row (D22)."""

    key: str
    spelling: str
    subject: str
    register: str
    invariants: tuple[str, ...]
    answers: str
    reads_sections: tuple[str, ...]
    requires: tuple[str, ...]
    refuses_when: str
    cost_class: str
    example: str


SURFACE_REGISTRY: tuple[SurfaceEntry, ...] = (
    SurfaceEntry(
        key="summary",
        spelling="trace.summary()",
        subject="MODEL",
        register="SHAPE",
        invariants=("conserves_totals", "mints_no_facts"),
        answers="Orient me: what model ran, its layers, parameters, and totals.",
        reads_sections=("counts", "params", "compute"),
        requires=("finished_trace",),
        refuses_when="the capture failed before finalization (use explain on the partial)",
        cost_class="metadata_only",
        example="trace.summary()",
    ),
    SurfaceEntry(
        key="explain",
        spelling="tl.report.explain(trace)",
        subject="CAPTURE",
        register="NARRATIVE",
        invariants=("mints_no_facts", "truncation_disclosed"),
        answers="Narrate this capture: what ran, its health states, and anything unusual.",
        reads_sections=("counts", "params", "compute", "health"),
        requires=(),
        refuses_when="never (degrades to a partial-capture diagnosis)",
        cost_class="metadata_only",
        example="tl.report.explain(trace)",
    ),
    SurfaceEntry(
        key="profile",
        spelling="trace.profile(sort_by=..., top_k=...)",
        subject="RUN-RESOURCES",
        register="RANK",
        invariants=("truncation_disclosed", "instrumented_time_labeled", "conserves_totals"),
        answers="Rank me: which ops/modules cost the most time, FLOPs, or memory.",
        reads_sections=("compute",),
        requires=("finished_trace",),
        refuses_when="the capture failed before finalization",
        cost_class="metadata_only",
        example="trace.profile()",
    ),
    SurfaceEntry(
        key="cost_tree",
        spelling="tl.report.cost_tree(trace)",
        subject="RUN-RESOURCES",
        register="SHAPE",
        invariants=("conserves_totals", "truncation_disclosed"),
        answers="Show compute as the module tree with exact self/subtree conservation.",
        reads_sections=("compute",),
        requires=("finished_trace",),
        refuses_when="the capture failed before finalization",
        cost_class="metadata_only",
        example="tl.report.cost_tree(trace)",
    ),
    SurfaceEntry(
        key="flops_report",
        spelling="tl.report.flops_report(trace_or_model, ...)",
        subject="RUN-RESOURCES",
        register="SHAPE",
        invariants=("conserves_totals", "mints_no_facts"),
        answers="The one-call paper number: analytic forward FLOPs with coverage and params.",
        reads_sections=("compute", "params"),
        requires=("finished_trace",),
        refuses_when="the trace door is passed model-door inputs",
        cost_class="metadata_only",
        example="tl.report.flops_report(trace)",
    ),
    SurfaceEntry(
        key="stats_table",
        spelling="trace.stats_table()",
        subject="RUN-VALUES",
        register="SHAPE",
        invariants=("mints_no_facts", "truncation_disclosed"),
        answers="Observations of ONE captured batch per site: mean/std/zero/NaN fractions.",
        reads_sections=("identity",),
        requires=("finished_trace",),
        refuses_when="payloads were not retained (typed row states, never fabrication)",
        cost_class="payload_scan",
        example="trace.stats_table()",
    ),
    SurfaceEntry(
        key="audit",
        spelling="trace.audit()",
        subject="RUN-HEALTH",
        register="VERDICT",
        invariants=("checks_and_skips_complete",),
        answers="Judge me: findings with severities, checks run, and skips disclosed.",
        reads_sections=("health",),
        requires=("finished_trace",),
        refuses_when="never (partial captures get the degraded audit)",
        cost_class="payload_scan",
        example="trace.audit()",
    ),
    SurfaceEntry(
        key="health_facts",
        spelling="tl.report.health_facts(trace)",
        subject="RUN-HEALTH",
        register="SHAPE",
        invariants=("mints_no_facts",),
        answers="The three-state nonfinite record (found / clean / not-checked) with basis.",
        reads_sections=("health",),
        requires=("finished_trace",),
        refuses_when="never (an underivable basis is the NOT-CHECKED shape)",
        cost_class="payload_scan",
        example="tl.report.health_facts(trace)",
    ),
    SurfaceEntry(
        key="bill_of_materials",
        spelling="trace.bill_of_materials()",
        subject="CAPTURE",
        register="SHAPE",
        invariants=("mints_no_facts",),
        answers="What does THIS object retain: payload bytes now, counts, annotations.",
        reads_sections=("counts", "params", "memory"),
        requires=("finished_trace",),
        refuses_when="the capture failed before finalization",
        cost_class="metadata_only",
        example="trace.bill_of_materials()",
    ),
    SurfaceEntry(
        key="agent_json",
        spelling="trace.to_agent_json()",
        subject="CAPTURE",
        register="ADDRESS",
        invariants=("mints_no_facts", "stable_addressing", "truncation_disclosed"),
        answers="The machine navigation map: op rows, edges, hierarchy, guide.",
        reads_sections=("counts", "identity", "compute", "health"),
        requires=("finished_trace",),
        refuses_when="the capture failed before finalization",
        cost_class="metadata_only",
        example="trace.to_agent_json(max_ops=32)",
    ),
    SurfaceEntry(
        key="backward_status",
        spelling="tl.report.backward_status(trace)",
        subject="RUN-RESOURCES",
        register="SHAPE",
        invariants=("mints_no_facts",),
        answers="The three-state executed-backward answer (0 / unknown / observed).",
        reads_sections=("compute",),
        requires=("finished_trace",),
        refuses_when="never",
        cost_class="metadata_only",
        example="tl.report.backward_status(trace)",
    ),
    SurfaceEntry(
        key="backward_estimate",
        spelling="tl.report.backward_estimate(trace)",
        subject="RUN-RESOURCES",
        register="SHAPE",
        invariants=("mints_no_facts",),
        answers="The HYPOTHETICAL training-backward counterfactual behind its named door.",
        reads_sections=("compute",),
        requires=("finished_trace",),
        refuses_when="never (always labeled hypothetical; never an actual-cost slot)",
        cost_class="metadata_only",
        example="tl.report.backward_estimate(trace)",
    ),
    SurfaceEntry(
        key="roofline",
        spelling="tl.report.roofline(trace, ridge_intensity=...)",
        subject="RUN-RESOURCES",
        register="SHAPE",
        invariants=("mints_no_facts", "truncation_disclosed"),
        answers="Theoretical intensity/work map; bound verdicts are hypotheses.",
        reads_sections=("compute",),
        requires=("finished_trace",),
        refuses_when="never (uncoverable ops land in the by-reason split)",
        cost_class="metadata_only",
        example="tl.report.roofline(trace)",
    ),
    SurfaceEntry(
        key="instrumented_rate",
        spelling="tl.report.instrumented_rate(trace)",
        subject="RUN-RESOURCES",
        register="RANK",
        invariants=("instrumented_time_labeled",),
        answers="The boxed CPU-only FLOPs-per-instrumented-second triage diagnostic.",
        reads_sections=("compute",),
        requires=("finished_trace",),
        refuses_when="every timed op executed on CUDA (host time is not a CUDA rate)",
        cost_class="metadata_only",
        example="tl.report.instrumented_rate(trace)",
    ),
    SurfaceEntry(
        key="cost_report",
        spelling="tl.report.cost_report(model, x)",
        subject="RUN-RESOURCES",
        register="SHAPE",
        invariants=("mints_no_facts",),
        answers="Measured capture cost on YOUR model and host (never an estimate).",
        reads_sections=(),
        requires=("live_model",),
        refuses_when="an unknown tier is requested",
        cost_class="new_forward",
        example="tl.report.cost_report(model, x, repeats=1)",
    ),
    SurfaceEntry(
        key="doctor",
        spelling="tl.utils.doctor()",
        subject="ENVIRONMENT",
        register="VERDICT",
        invariants=("checks_and_skips_complete",),
        answers="Is my environment healthy: versions, capabilities, degradations.",
        reads_sections=(),
        requires=(),
        refuses_when="never",
        cost_class="env_probe",
        example="tl.utils.doctor()",
    ),
)


def registry_entry(key: str) -> SurfaceEntry:
    """Look up one registry row by key, refusing unknowns typed."""

    from .._errors import InvalidArgumentError

    for entry in SURFACE_REGISTRY:
        if entry.key == key:
            return entry
    raise InvalidArgumentError(
        f"unknown report-family surface {key!r}.",
        code="surface_registry_unknown",
        remedy=f"Use one of: {', '.join(entry.key for entry in SURFACE_REGISTRY)}.",
    )


@dataclass(frozen=True)
class CapabilityRow:
    """One capability-card row: THIS object's answer for one surface."""

    key: str
    spelling: str
    available: bool
    reason: str
    cost_class: str


@dataclass(frozen=True)
class CapabilityCard:
    """The registry filtered by the object in hand (D24): metadata-only."""

    object_kind: str
    rows: tuple[CapabilityRow, ...]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): rows point, never dump."""

        available = sum(1 for row in self.rows if row.available)
        return (
            f"CapabilityCard({self.object_kind}: {available}/{len(self.rows)} "
            f"available; print() for the card, read .rows)"
        )

    def __str__(self) -> str:
        """Render the card: available members first, refusals with reasons."""

        lines = [f"capability card ({self.object_kind}):"]
        for row in sorted(self.rows, key=lambda item: (not item.available, item.key)):
            marker = "ok " if row.available else "NO "
            lines.append(f"  {marker} {row.spelling:<44} [{row.cost_class}] {row.reason}")
        return "\n".join(lines)


def _object_kind(trace_like: Any) -> str:
    """Classify the object in hand without executing anything."""

    name = type(trace_like).__name__
    if name == "PartialTrace":
        return "partial_trace"
    if name == "MergedTrace":
        return "merged_trace"
    return "trace"


def capability_card(trace_like: Any) -> CapabilityCard:
    """Build the capability card for one trace-like object (D24).

    Metadata-only by construction: requirement tokens are evaluated
    against facts the object already holds; nothing scans, captures, or
    runs a forward.
    """

    kind = _object_kind(trace_like)
    rows: list[CapabilityRow] = []
    for entry in SURFACE_REGISTRY:
        available = True
        reason = entry.answers
        if "finished_trace" in entry.requires and kind == "partial_trace":
            available = False
            reason = (
                "partial capture: use tl.report.explain(partial) for the diagnosis or "
                "partial.audit() for the degraded findings"
            )
        if "live_model" in entry.requires:
            available = False
            reason = "needs the live model (a trace-like object is not enough)"
        rows.append(
            CapabilityRow(
                key=entry.key,
                spelling=entry.spelling,
                available=available,
                reason=reason,
                cost_class=entry.cost_class,
            )
        )
    return CapabilityCard(object_kind=kind, rows=tuple(rows))


def which_do_i_use() -> str:
    """Render the "which do I use" docs page from the registry (item 20).

    The docs page file (docs/reference/report_family.md) is this
    function's output, pinned by a lockstep test -- the page can never
    rot away from the registry again.
    """

    lines = [
        "# Which report surface do I use?",
        "",
        "<!-- GENERATED from torchlens/report/_registry.py::which_do_i_use();",
        "     edit the registry, then regenerate. The lockstep test",
        "     tests/test_report_family_registry.py pins this file to the",
        "     generator output. -->",
        "",
        "Every row is one question. Every spelling below executes in CI against a",
        "real fixture in its declared state.",
        "",
        "| surface | one question it answers | subject | register | cost |",
        "|---|---|---|---|---|",
    ]
    for entry in SURFACE_REGISTRY:
        lines.append(
            f"| `{entry.spelling}` | {entry.answers} | {entry.subject} | "
            f"{entry.register} | {entry.cost_class} |"
        )
    lines.extend(
        [
            "",
            "## Refusal conditions",
            "",
        ]
    )
    for entry in SURFACE_REGISTRY:
        lines.append(f"- `{entry.key}`: refuses when {entry.refuses_when}.")
    lines.extend(
        [
            "",
            "## Honesty invariants per surface",
            "",
        ]
    )
    for entry in SURFACE_REGISTRY:
        lines.append(f"- `{entry.key}`: {', '.join(entry.invariants) or '(none declared)'}")
    lines.append("")
    return "\n".join(lines)
