"""The composition ledger: one row schema, closed vocabularies, seed rows (row 0.8).

Compo memo 3.1: tests own verification policy through this importable ledger;
production owns product truth (operation ids, operation-grain backend
capabilities) which rows REFERENCE by id and never restate. Every declared
cell carries one of four shippable states or the transitional KNOWN-GAP with
teeth (D3/D11); oracle kinds are mandated by risk tags (D4); "it ran" is not
an oracle. The generated projection renders state, cost, and gaps TOGETHER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

EXPECTED_STATES = (
    "SUPPORTED",
    "SUPPORTED-WITH-DISCLOSURE",
    "REFUSES-TYPED-TEACHING",
    "N-A",
    "KNOWN-GAP",
)

ORACLE_KINDS = (
    "EXACT",
    "DIFFERENTIAL",
    "INVARIANT",
    "REFUSAL",
    "DISCLOSURE",
    "TRANSACTION",
    "CONTENT-FLOOR",
    "BUDGET",
    "ADMISSION",
)

GENERATOR_KINDS = ("hand", "matrix", "sweep", "gallery", "array-proposed", "generated-probe")

RISK_TAGS = (
    "accepted_then_discarded",
    "persistence_hop",
    "pass_resolution",
    "silent_noop",
    "cost_cliff",
    "untrusted_input",
    "version_shape",
    "container_shape",
    "shared_storage",
    "remedy_currency",
    "lane_asymmetry",
    "facade_teaching",
)

#: Risk tag -> oracle kinds that discharge it (memo 3.4: unclassified
#: effectful rows fail safe to the strictest applicable set).
RISK_TAG_ORACLE_LAW: dict[str, tuple[str, ...]] = {
    "accepted_then_discarded": ("DIFFERENTIAL",),
    "persistence_hop": ("INVARIANT",),
    "pass_resolution": ("EXACT",),
    "silent_noop": ("DIFFERENTIAL", "REFUSAL"),
    "cost_cliff": ("BUDGET",),
}

TIERS = ("smoke", "heavy", "slow", "nightly")


@dataclass(frozen=True)
class CompositionRow:
    """One composition cell under test (memo 3.1 row schema).

    KNOWN-GAP rows carry the D11 teeth (owner, issue, deadline, reproducer,
    auto-probe); REFUSES rows carry the code or a remedy probe; N-A rows
    carry reason + reviewer + review trigger ("not built yet" is never N/A).
    """

    row_id: str
    family: str
    generator_kind: str
    operation_ids: tuple[str, ...]
    axis_values: tuple[str, ...]
    expected_state: str
    risk_tags: tuple[str, ...]
    oracle_kind: str
    evidence_node: str = ""
    fixture_recipe: str = ""
    tier: str = "smoke"
    ci_job: str = "U"
    real_model_id: str = ""
    source_gap: str = ""
    requested_effective_receipt: bool = False
    refusal_code: str = ""
    remedy_probe: str = ""
    remedy_is_prose: bool = False
    n_a_reason: str = ""
    n_a_reviewer: str = ""
    n_a_review_trigger: str = ""
    constraint_id: str = ""
    cost_profile_id: str = ""
    gap_owner: str = ""
    gap_issue: str = ""
    gap_deadline: str = ""
    gap_reproducer: str = ""
    gap_auto_probe: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def row_violations(row: CompositionRow) -> list[str]:
    """Validate one ledger row against the closed contract.

    Returns human-readable violations (empty when compliant); the lockstep
    test red-capability-proves every rule with planted rows.
    """

    violations: list[str] = []
    if row.expected_state not in EXPECTED_STATES:
        violations.append(f"{row.row_id}: unknown state {row.expected_state!r}")
    if row.oracle_kind not in ORACLE_KINDS:
        violations.append(f"{row.row_id}: unknown oracle kind {row.oracle_kind!r}")
    if row.generator_kind not in GENERATOR_KINDS:
        violations.append(f"{row.row_id}: unknown generator kind {row.generator_kind!r}")
    if row.tier not in TIERS:
        violations.append(f"{row.row_id}: unknown tier {row.tier!r}")
    for tag in row.risk_tags:
        if tag not in RISK_TAGS:
            violations.append(f"{row.row_id}: unknown risk tag {tag!r}")
    if not row.operation_ids:
        violations.append(f"{row.row_id}: no operation ids")
    if not row.axis_values:
        violations.append(f"{row.row_id}: no axis values (a cell IS a composition)")

    state = row.expected_state
    # Oracle-kind law by risk tag applies to rows that CLAIM a verified state.
    if state in {"SUPPORTED", "SUPPORTED-WITH-DISCLOSURE"}:
        if not row.evidence_node:
            violations.append(f"{row.row_id}: {state} without an evidence node")
        for tag in row.risk_tags:
            mandated = RISK_TAG_ORACLE_LAW.get(tag)
            if mandated and row.oracle_kind not in mandated:
                violations.append(
                    f"{row.row_id}: risk tag {tag!r} mandates oracle {mandated}, "
                    f"got {row.oracle_kind!r}"
                )
        if state == "SUPPORTED-WITH-DISCLOSURE" and row.oracle_kind == "ADMISSION":
            violations.append(f"{row.row_id}: ADMISSION cannot discharge a supported state")
    if state == "SUPPORTED" and row.oracle_kind in {"ADMISSION", "DISCLOSURE"}:
        violations.append(f"{row.row_id}: SUPPORTED requires a semantic witness, not admission")
    if row.oracle_kind == "ADMISSION" and not (
        state == "REFUSES-TYPED-TEACHING" or row.generator_kind == "generated-probe"
    ):
        violations.append(
            f"{row.row_id}: ADMISSION is legal only on REFUSES rows and generated probes"
        )
    if state == "REFUSES-TYPED-TEACHING" and not (
        row.refusal_code or row.remedy_probe or row.remedy_is_prose
    ):
        violations.append(
            f"{row.row_id}: REFUSES row needs refusal_code, remedy_probe, or "
            "remedy_is_prose (counted)"
        )
    if state == "N-A":
        if not (row.n_a_reason and row.n_a_reviewer and row.n_a_review_trigger):
            violations.append(f"{row.row_id}: N-A needs reason + reviewer + review trigger")
        if "not built" in row.n_a_reason.lower():
            violations.append(f"{row.row_id}: 'not built yet' is never N/A (memo 3.4)")
    if state == "KNOWN-GAP":
        teeth = {
            "gap_owner": row.gap_owner,
            "gap_issue": row.gap_issue,
            "gap_deadline": row.gap_deadline,
            "gap_reproducer": row.gap_reproducer,
            "gap_auto_probe": row.gap_auto_probe,
        }
        for name, value in teeth.items():
            if not value:
                violations.append(f"{row.row_id}: KNOWN-GAP without {name} (D11 teeth)")
    return violations


#: Named first-wave cells from the panel's own live findings (memo section 4
#: tail). Fix lanes re-verify before acting; every row is transitional until
#: its lane lands the fix + evidence. Deadlines are megasprint phase gates.
LEDGER: tuple[CompositionRow, ...] = (
    CompositionRow(
        row_id="CELL-FLAGSHIP-RERUN-NOOP",
        family="intervention",
        generator_kind="hand",
        operation_ids=("trace_option:intervene:torch", "fork.do"),
        axis_values=("selector=postprocess-label", "engine=rerun"),
        expected_state="KNOWN-GAP",
        risk_tags=("silent_noop", "lane_asymmetry"),
        oracle_kind="DIFFERENTIAL",
        source_gap="M(compo) 5.4 worked example",
        gap_owner="A04",
        gap_issue="memo 5.4: taught spelling completes, warns, returns un-ablated trace",
        gap_deadline="PB1",
        gap_reproducer="M(compo) 5.4 six-line chain (12-layer MLP)",
        gap_auto_probe="S-18 rerun_zero_fire contracted warning (this wave)",
        notes=("the founding defect class live on the front door",),
    ),
    CompositionRow(
        row_id="CELL-SAVE-ALL-REMEDY-CURRENCY",
        family="capture-options",
        generator_kind="hand",
        operation_ids=("tl.trace",),
        axis_values=("save=all", "remedy=currency"),
        expected_state="KNOWN-GAP",
        risk_tags=("remedy_currency",),
        oracle_kind="REFUSAL",
        source_gap="M(compo) 4 named cells",
        gap_owner="A06",
        gap_issue="remedy steers into a deprecated spelling",
        gap_deadline="PB1",
        gap_reproducer="M(compo) section 4 named-cell list",
        gap_auto_probe="A.6 refusal-contract driver (currency clause)",
    ),
    CompositionRow(
        row_id="CELL-REPLAY-PRECONDITION-CODELESS",
        family="diagnostics",
        generator_kind="hand",
        operation_ids=("fork.do",),
        axis_values=("error=ReplayPreconditionError", "capture=default"),
        expected_state="KNOWN-GAP",
        risk_tags=("facade_teaching",),
        oracle_kind="REFUSAL",
        source_gap="M(compo) 3.5 witness",
        gap_owner="compo(S-17)",
        gap_issue="codeless on the default-capture path",
        gap_deadline="S-17 ratchet burn-down",
        gap_reproducer="tests/composition_expectations/data/refusal_site_classification.tsv",
        gap_auto_probe="S-17 unclassified/uncoded ratchets (this wave)",
    ),
    CompositionRow(
        row_id="CELL-SITE-RESOLUTION-ZERO-MATCH-CODELESS",
        family="diagnostics",
        generator_kind="hand",
        operation_ids=("fork.do",),
        axis_values=("error=SiteResolutionError", "match=zero"),
        expected_state="KNOWN-GAP",
        risk_tags=("facade_teaching",),
        oracle_kind="REFUSAL",
        source_gap="M(compo) 3.5 witness",
        gap_owner="compo(S-17)",
        gap_issue="typed and teaching in prose, code=None",
        gap_deadline="S-17 ratchet burn-down",
        gap_reproducer="tests/composition_expectations/data/refusal_site_classification.tsv",
        gap_auto_probe="S-17 unclassified/uncoded ratchets (this wave)",
    ),
    CompositionRow(
        row_id="CELL-ZERO-MATCH-LANE-ASYMMETRY",
        family="lane-parity",
        generator_kind="hand",
        operation_ids=("tl.trace", "fork.do"),
        axis_values=("lane=selector-vs-plan", "match=zero"),
        expected_state="KNOWN-GAP",
        risk_tags=("lane_asymmetry", "silent_noop"),
        oracle_kind="DIFFERENTIAL",
        source_gap="M(compo) 4 named cells (M4 witness)",
        gap_owner="A04",
        gap_issue="zero-match refuses typed via selector forms, silently warns via plan stage",
        gap_deadline="PB1",
        gap_reproducer="M(compo) section 4; S-18 first contracted row is the warn half",
        gap_auto_probe="M4 lane-parity matrix (wave A.3)",
    ),
    CompositionRow(
        row_id="CELL-DRAW-WRONG-KNOB-REFUSAL",
        family="render",
        generator_kind="hand",
        operation_ids=("Trace.draw",),
        axis_values=("kwarg=vis_mode", "view=False"),
        expected_state="KNOWN-GAP",
        risk_tags=("facade_teaching",),
        oracle_kind="REFUSAL",
        source_gap="M(compo) 4 named cells",
        gap_owner="C05",
        gap_issue="refusal names the wrong knob",
        gap_deadline="PB2a",
        gap_reproducer="M(compo) section 4 named-cell list",
        gap_auto_probe="A.6 refusal-contract driver (names-the-right-knob clause)",
    ),
    CompositionRow(
        row_id="CELL-FACADE-NO-DID-YOU-MEAN",
        family="entry",
        generator_kind="hand",
        operation_ids=("tl.__getattr__",),
        axis_values=("attr=explain|utils|bridge", "surface=module-facade"),
        expected_state="KNOWN-GAP",
        risk_tags=("facade_teaching",),
        oracle_kind="REFUSAL",
        source_gap="M(compo) 4 named cells (two advertised by the MCP api_map)",
        gap_owner="A10",
        gap_issue="no did-you-mean on tl.explain -> tl.report.explain",
        gap_deadline="PB1",
        gap_reproducer="M(compo) section 4 named-cell list",
        gap_auto_probe="A10 five-step package-wide __getattr__ suite",
    ),
    CompositionRow(
        row_id="CELL-STORAGE-STREAMING-SILENT-CONFLICT",
        family="capture-options",
        generator_kind="hand",
        operation_ids=("tl.trace", "trace_option:storage:torch"),
        axis_values=("storage=to_disk", "streaming=explicit"),
        expected_state="KNOWN-GAP",
        risk_tags=("accepted_then_discarded",),
        oracle_kind="DIFFERENTIAL",
        source_gap="M(compo) 4 named cells",
        gap_owner="A06",
        gap_issue="explicit-request conflict resolved silently",
        gap_deadline="PB1",
        gap_reproducer="M(compo) section 4 named-cell list",
        gap_auto_probe="M3 option receipt (requested vs effective; this wave's substrate)",
        requested_effective_receipt=True,
    ),
    CompositionRow(
        row_id="CELL-MCP-CACHE-KEY-SYMLINK",
        family="agent",
        generator_kind="hand",
        operation_ids=("bridge.mcp.load_overview",),
        axis_values=("path=symlinked", "cache=keyed-by-path"),
        expected_state="KNOWN-GAP",
        risk_tags=("untrusted_input",),
        oracle_kind="INVARIANT",
        source_gap="M(compo) 4 named cells",
        gap_owner="A09",
        gap_issue="cache key vs load path diverge across symlinks",
        gap_deadline="PB2a",
        gap_reproducer="M(compo) section 4 named-cell list",
        gap_auto_probe="C.5 MCP sweep first rows",
    ),
    CompositionRow(
        row_id="CELL-ENV-PARSER-BYPASS",
        family="environment",
        generator_kind="hand",
        operation_ids=("closed_bool_env",),
        axis_values=("vars=3-bypassing", "parser=closed_bool_env"),
        expected_state="KNOWN-GAP",
        risk_tags=("facade_teaching",),
        oracle_kind="INVARIANT",
        source_gap="M(compo) 3.2 (three env vars bypass the closed-bool parser)",
        gap_owner="A10",
        gap_issue='"=true" silently means OFF on the bypassing reads',
        gap_deadline="PB1",
        gap_reproducer="U-ENV-VARS census (this wave)",
        gap_auto_probe="ENV-VAR contract sweep (A.8)",
    ),
    CompositionRow(
        row_id="CELL-PARTIALTRACE-AGENT-VERBS",
        family="product-surface",
        generator_kind="hand",
        operation_ids=("Trace.to_agent_json", "tl.report.explain"),
        axis_values=("product=PartialTrace", "verb=agent-doors"),
        expected_state="KNOWN-GAP",
        risk_tags=("container_shape",),
        oracle_kind="CONTENT-FLOOR",
        source_gap="M(compo) 4 named cells (an M1 cell by construction)",
        gap_owner="A09",
        gap_issue="PartialTrace lacks agent verbs the docs imply",
        gap_deadline="PB1",
        gap_reproducer="M(compo) section 4 named-cell list",
        gap_auto_probe="M1 product x verb matrix (wave A.1)",
    ),
    CompositionRow(
        row_id="CELL-MULTIPASS-BARE-LABEL-REFUSAL",
        family="intervention",
        generator_kind="hand",
        operation_ids=("fork.do",),
        axis_values=("label=bare", "layer=multi-pass"),
        expected_state="REFUSES-TYPED-TEACHING",
        risk_tags=("pass_resolution",),
        oracle_kind="REFUSAL",
        evidence_node="tests/test_replay_pass_qualified.py",
        refusal_code="multipass_bare_label_ambiguous",
        source_gap="M(compo) 4 (the shipping-today positive: proof R0-shape fixtures exercise real machinery)",
        notes=("teaching message names every pass-qualified spelling",),
    ),
    CompositionRow(
        row_id="CELL-FORK-RAW-WRITE-SHARED-STORAGE",
        family="product-surface",
        generator_kind="gallery",
        operation_ids=("Trace.fork",),
        axis_values=("write=raw-inplace", "payload=shared-storage"),
        expected_state="SUPPORTED-WITH-DISCLOSURE",
        risk_tags=("shared_storage",),
        oracle_kind="DISCLOSURE",
        evidence_node=(
            "tests/composition_expectations/test_galleries.py::"
            "test_raw_fork_writes_reach_the_shared_storage"
        ),
        source_gap="found by this wave's gallery fingerprint tripwire",
        notes=(
            "fork payloads alias the parent (the 75ms fork economics); raw "
            "in-place writes through a fork poison the parent -- edit verbs only",
        ),
    ),
    CompositionRow(
        row_id="CELL-SUMMARY-CHARSET-PASSCOUNT",
        family="report-surface",
        generator_kind="hand",
        operation_ids=("Trace.summary", "tl.summary"),
        axis_values=("charset=ascii-vs-unicode", "layer=multi-pass", "view=ladder"),
        expected_state="SUPPORTED",
        risk_tags=("pass_resolution",),
        oracle_kind="EXACT",
        evidence_node=(
            "F08: tests/test_summary_rebuild_charset.py::test_dual_charset_goldens + "
            "tests/test_summary_rebuild_ladder.py::test_multipass_one_row_at_module_grain "
            "(byte-exact dual-charset goldens; one row owns all pass events; "
            "ascii == degrade(unicode))"
        ),
        source_gap="F08 summary rebuild (charset contract x multi-pass identity partition)",
        notes=(
            "the identity partition holds across charsets: a multi-pass layer is "
            "ONE ladder row owning every pass event, byte-stable in both styles",
        ),
    ),
)


def ledger_violations() -> list[str]:
    """Validate the whole ledger (unique ids + per-row contract)."""

    violations: list[str] = []
    seen: set[str] = set()
    for row in LEDGER:
        if row.row_id in seen:
            violations.append(f"duplicate row id {row.row_id}")
        seen.add(row.row_id)
        violations.extend(row_violations(row))
    return violations


def render_projection() -> str:
    """Render the generated Markdown projection (state + gaps TOGETHER).

    Regenerate-and-diff: the committed copy is
    ``data/composition_ledger.md``; the lockstep test fails on drift so the
    diff IS the ledger-change review.
    """

    lines = [
        "# Composition ledger (generated -- do not edit)",
        "",
        "Regenerate: update `tests/composition_expectations/ledger.py`, then",
        "copy the output of `render_projection()` over this file (the",
        "lockstep test prints the drift). State, oracle, risk, and gap teeth",
        "render TOGETHER so SUPPORTED cannot hide an unusable cell.",
        "",
        "| Row | Family | State | Oracle | Risk tags | Axes | Owner/Evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in LEDGER:
        owner_or_evidence = (
            row.evidence_node
            if row.expected_state not in {"KNOWN-GAP"}
            else f"{row.gap_owner} (due {row.gap_deadline})"
        )
        lines.append(
            f"| {row.row_id} | {row.family} | {row.expected_state} | {row.oracle_kind} "
            f"| {', '.join(row.risk_tags)} | {'; '.join(row.axis_values)} "
            f"| {owner_or_evidence} |"
        )
    gap_rows = [row for row in LEDGER if row.expected_state == "KNOWN-GAP"]
    lines += [
        "",
        f"KNOWN-GAP rows: {len(gap_rows)} of {len(LEDGER)} "
        "(transitional; each carries owner, issue, deadline, reproducer, auto-probe).",
        "",
    ]
    return "\n".join(lines)
