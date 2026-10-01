"""Surgery visuals (lane F43): ONE mark family over C03's audit entries.

The surgery memo's rendering half (foldA s5 item 11 + D17: F01 writes, F38
lints, F43 renders): every visual claim here is READ from the canonical
intervention record C03 landed -- per-fire :class:`FireRecord` rows on ops,
the ``intervention_event_v2`` transaction envelopes and ``region_do`` fact
rows on the free-form ``state_history`` stream, the closed ACT/EVENT rows in
``trace.intervention_audit``, and F01's stage-1 injected-op records. This
module never writes an engine file and never mints a fact: a mark either
cites a recorded audit entry (FACT, solid border) or a declared-but-unfired
target attribution (HEURISTIC, dashed border), and the two are never drawn
in the same style.

Shape rules (normative, from the fold):

- **One mark family, never a new view kind**: the marks ride the standard
  ``Trace.draw`` node-spec funnel through C05's lens registry (the
  ``"surgery"`` :class:`~torchlens.visualization.theme_registry.LensPreset`
  row registered below); there is no separate surgery picture type.
- **Per-lane splice-box disclosure**: each transaction's disclosure box
  carries the wording of ITS recorded lane -- a recorded
  ``execution_effect`` verbatim when the engine wrote one, else the closed
  D17 lane law (replay substitutes exits without re-executing the interior;
  live/bound lanes run the original op and replace values after execution).
  The two texts are mutually false across lanes by design; neither is ever
  emitted for the other lane.
- **Never ghosted-as-deleted**: execution removal exists in no lane, so no
  wording here (marks, boxes, census) ever uses the banned
  execution-removal verbs -- that vocabulary is unrepresentable by construction
  (:data:`torchlens.intervention.audit.EXECUTION_EFFECTS`).
- **The census travels with exported figures**: the per-lane census renders
  into the figure's disclosure caption (C05's rendered-artifact rule) and
  :func:`render_surgery` writes the full text beside every saved figure.
- **CREDIT rows** ride the census (D01's acknowledgment list).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from .._vocab.node_spec import INTERVENTION_SITE_COLOR, NodeSpec, NodeSpecFn
from ..utils._multipass_access import get_multipass_attr
from .theme_registry import GRAPHVIZ_DRAW_SURFACE, LensPreset, register_lens

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.trace import Trace

__all__ = [
    "CREDIT_ROWS",
    "MARK_BASES",
    "MARK_KINDS",
    "SURGERY_LENS",
    "RegionFact",
    "SurgeryCensus",
    "SurgeryFacts",
    "SurgeryMark",
    "SurgeryTransaction",
    "effect_wording",
    "make_surgery_mark_spec_fn",
    "render_surgery",
    "splice_box_lines",
    "surgery_census",
    "surgery_facts",
    "surgery_stage",
]

#: Closed mark-kind vocabulary (the ONE mark family). Every kind names the
#: audit entry class it cites; there is no free-form kind.
MARK_KINDS: frozenset[str] = frozenset(
    {"fire", "region_member", "region_exit", "injection_host", "declared_target"}
)

#: Closed mark-basis vocabulary: FACT marks cite a recorded audit entry at
#: the node; HEURISTIC marks are renderer-side attributions (a declared
#: target with no fire evidence at that node). The line-style law keys on
#: this: fact = solid border, heuristic = dashed border.
MARK_BASES: frozenset[str] = frozenset({"fact", "heuristic"})

#: The two closed execution-effect glosses (D17 vocabulary, rendered form).
_EFFECT_WORDING: Mapping[str, str] = {
    "values_replaced_after_execution": (
        "the original op ran; edited values replaced its output after execution"
    ),
    "exits_substituted_interior_not_replayed": (
        "exit values were substituted; the interior was not replayed"
    ),
}

#: The D17 lane law: which effect gloss each recorded lane implies when the
#: engine did not write ``execution_effect`` itself. ``set_only`` stages
#: without executing, so it carries its own (third) truthful text.
_LANE_LAW: Mapping[str, str] = {
    "replay": _EFFECT_WORDING["exits_substituted_interior_not_replayed"],
    "capture": _EFFECT_WORDING["values_replaced_after_execution"],
    "live_hook": _EFFECT_WORDING["values_replaced_after_execution"],
    "rerun": _EFFECT_WORDING["values_replaced_after_execution"],
    "bind": _EFFECT_WORDING["values_replaced_after_execution"],
    "set_only": "staged only; nothing has executed",
}

#: D01 acknowledgment rows; they ride every census verbatim.
CREDIT_ROWS: tuple[str, ...] = (
    "CREDIT: pyvene -- serializable intervention vocabulary",
    "CREDIT: nnsight -- persistent model.edit / scan experience",
    "CREDIT: the signed surgery panel -- anchored provenance; no-execution-removal doctrine",
)

#: Cap on per-node mark rows before the "+N more" fold (label legibility).
_MAX_MARK_ROWS_PER_NODE = 3

#: Cap on caption census lines before deferring to the sidecar/text form.
_MAX_CAPTION_LINES = 12


@dataclass(frozen=True)
class SurgeryMark:
    """One mark in the family: an audit-cited claim about one rendered node.

    Attributes
    ----------
    layer_label:
        Aggregate layer label (matches rolled nodes).
    call_label:
        Pass-qualified label (matches unrolled per-pass nodes); ``None``
        when the citing entry recorded no pass-qualified address.
    site_key:
        The structural site key the citing entry recorded, if any.
    kind:
        Closed :data:`MARK_KINDS` value.
    basis:
        Closed :data:`MARK_BASES` value -- the line-style law's input.
    row:
        The rendered node-label row for this mark (banned verbs
        unrepresentable: rows are built from the closed wording tables).
    """

    layer_label: str
    call_label: str | None
    site_key: str | None
    kind: str
    basis: str
    row: str


@dataclass(frozen=True)
class SurgeryTransaction:
    """One recorded intervention transaction (an envelope row, read-only)."""

    lane: str
    door: str
    status: str
    fire_count: int
    edit_names: tuple[str, ...]
    site_keys: tuple[str, ...]
    zero_fire_rule_ids: tuple[str, ...]
    execution_effect: str | None
    staged_only: bool


@dataclass(frozen=True)
class RegionFact:
    """One recorded ``region_do`` fact row (renderer-neutral, read-only)."""

    region_digest: str
    members: tuple[str, ...]
    exit_parents: tuple[str, ...]
    exit_children: tuple[str, ...]
    edit: str
    execution_effect: str | None


@dataclass(frozen=True)
class SurgeryFacts:
    """Everything the mark family may cite, derived from C03's record."""

    transactions: tuple[SurgeryTransaction, ...]
    regions: tuple[RegionFact, ...]
    marks: tuple[SurgeryMark, ...]
    injected_host_counts: Mapping[str, int]

    @property
    def has_evidence(self) -> bool:
        """Whether ANY audit entry exists for the marks to cite."""

        return bool(self.transactions or self.regions or self.marks)

    def marks_for(self, label: str) -> tuple[SurgeryMark, ...]:
        """Return the marks whose call or layer label equals ``label``."""

        return tuple(mark for mark in self.marks if label in (mark.call_label, mark.layer_label))


def effect_wording(lane: str, execution_effect: str | None) -> tuple[str, str]:
    """Return ``(wording, source)`` for one transaction's disclosure box.

    ``source`` is ``"recorded"`` when the engine wrote a closed
    ``execution_effect`` value (used verbatim), ``"lane_law"`` when the
    wording follows the D17 lane law from the recorded lane, and
    ``"unrecorded"`` when neither authority applies (the honest fallback --
    never a guess borrowed from the other lane).
    """

    if execution_effect is not None and execution_effect in _EFFECT_WORDING:
        return _EFFECT_WORDING[execution_effect], "recorded"
    law = _LANE_LAW.get(lane)
    if law is not None:
        return law, "lane_law"
    return "execution effect not recorded for this lane", "unrecorded"


def _envelope_rows(trace: Any) -> list[dict[str, Any]]:
    """The transaction envelopes on the free-form persisted stream."""

    return [
        row
        for row in getattr(trace, "state_history", ()) or ()
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]


def _region_rows(trace: Any) -> list[dict[str, Any]]:
    """The renderer-neutral region fact rows (F01 writes, F43 reads)."""

    return [
        row
        for row in getattr(trace, "state_history", ()) or ()
        if isinstance(row, dict) and row.get("op") == "region_do"
    ]


def _transaction_from_envelope(row: Mapping[str, Any]) -> SurgeryTransaction:
    """Project one envelope row onto the read-only transaction record."""

    return SurgeryTransaction(
        lane=str(row.get("lane", "")),
        door=str(row.get("door", "")),
        status=str(row.get("status", "")),
        fire_count=int(row.get("fire_count", 0) or 0),
        edit_names=tuple(str(name) for name in row.get("edit_names", ()) or ()),
        site_keys=tuple(str(key) for key in row.get("site_keys", ()) or ()),
        zero_fire_rule_ids=tuple(
            str(rule_id) for rule_id in row.get("zero_fire_rule_ids", ()) or ()
        ),
        execution_effect=(
            str(row["execution_effect"]) if row.get("execution_effect") is not None else None
        ),
        staged_only=bool(row.get("staged_only", False)),
    )


def _region_fact_from_row(row: Mapping[str, Any]) -> RegionFact:
    """Project one ``region_do`` row onto the read-only region record."""

    exits = [exit_row for exit_row in row.get("exits", ()) or () if isinstance(exit_row, dict)]
    return RegionFact(
        region_digest=str(row.get("region_digest", "")),
        members=tuple(str(member) for member in row.get("members", ()) or ()),
        exit_parents=tuple(str(exit_row.get("parent", "")) for exit_row in exits),
        exit_children=tuple(str(exit_row.get("child", "")) for exit_row in exits),
        edit=str(row.get("edit", "")),
        execution_effect=(
            str(row["execution_effect"]) if row.get("execution_effect") is not None else None
        ),
    )


def _fire_marks(trace: Any) -> list[SurgeryMark]:
    """FACT marks: one row per recorded FireRecord, at its recorded node."""

    marks: list[SurgeryMark] = []
    for label in getattr(trace, "op_labels", ()) or ():
        op = trace.ops[label]
        for record in getattr(op, "interventions", None) or ():
            helper = getattr(record, "helper_name", None) or "user_hook"
            engine = getattr(record, "engine", None) or "unknown"
            verb = "spliced" if helper == "splice_module" else "edited"
            marks.append(
                SurgeryMark(
                    layer_label=str(getattr(record, "target_label", None) or op.layer_label),
                    call_label=str(getattr(record, "call_label", None) or label),
                    site_key=(str(op.site_key) if getattr(op, "site_key", None) else None),
                    kind="fire",
                    basis="fact",
                    row=f"{verb}: {helper} ({engine} engine)",
                )
            )
    return marks


def _region_marks(trace: Any, regions: tuple[RegionFact, ...]) -> list[SurgeryMark]:
    """FACT marks for region members and exit parents (row-cited)."""

    marks: list[SurgeryMark] = []
    for region_fact in regions:
        wording, _source = effect_wording("replay", region_fact.execution_effect)
        for member in region_fact.members:
            marks.append(
                _mark_for_recorded_label(
                    trace,
                    member,
                    kind="region_member",
                    row=f"region member ({region_fact.edit}): {wording}",
                )
            )
        for parent, child in zip(region_fact.exit_parents, region_fact.exit_children, strict=True):
            marks.append(
                _mark_for_recorded_label(
                    trace, parent, kind="region_exit", row=f"region exit -> {child}"
                )
            )
    return marks


def _injection_marks(trace: Any) -> tuple[list[SurgeryMark], dict[str, int]]:
    """FACT marks for hosts of F01 stage-1 injected ops, plus host counts."""

    host_counts: dict[str, int] = {}
    for injected in getattr(trace, "injected_ops", None) or ():
        host = str(getattr(injected, "host_label", ""))
        if host:
            host_counts[host] = host_counts.get(host, 0) + 1
    marks = [
        _mark_for_recorded_label(
            trace,
            host,
            kind="injection_host",
            row=f"hosts {count} injected op(s) (recorded, from the intervention)",
        )
        for host, count in host_counts.items()
    ]
    return marks, host_counts


def _mark_for_recorded_label(trace: Any, label: str, *, kind: str, row: str) -> SurgeryMark:
    """Build a FACT mark for a pass-qualified label an audit entry recorded."""

    layer_label = label.split(":", 1)[0]
    site_key: str | None = None
    try:
        op = trace.ops[label]
    except (KeyError, AttributeError, TypeError):
        op = None
    if op is not None:
        layer_label = str(getattr(op, "layer_label", layer_label))
        raw_key = getattr(op, "site_key", None)
        site_key = str(raw_key) if raw_key else None
    return SurgeryMark(
        layer_label=layer_label,
        call_label=label,
        site_key=site_key,
        kind=kind,
        basis="fact",
        row=row,
    )


def _declared_target_marks(
    trace: Any,
    transactions: tuple[SurgeryTransaction, ...],
    fact_labels: set[str],
) -> list[SurgeryMark]:
    """HEURISTIC marks: targeted-site cohort members with no fire recorded.

    Envelope ``site_keys`` are site-key-first target refs. Attribution back
    to rendered nodes goes through the ops' own site keys, and a structural
    site key is shared by every pass instance of a reused call site -- so a
    cohort member WITHOUT its own FireRecord gets a dashed heuristic mark
    ("this site was targeted; no fire is recorded at this instance"), never
    a fact mark. Transactions that recorded no site refs (today's staged
    ``set_only`` envelopes) attribute nothing: no ref, no mark, no
    fabrication -- they still appear in the census and splice-box lines.
    """

    by_site_key: dict[str, list[Any]] = {}
    for label in getattr(trace, "op_labels", ()) or ():
        op = trace.ops[label]
        raw_key = getattr(op, "site_key", None)
        if raw_key:
            by_site_key.setdefault(str(raw_key), []).append(op)
    marks: list[SurgeryMark] = []
    seen: set[str] = set()
    for transaction in transactions:
        state = "staged" if transaction.staged_only else "targeted"
        for ref in transaction.site_keys:
            for op in by_site_key.get(ref, []):
                call_label = get_multipass_attr(op, "label", None, multipass=None)
                node_key = str(call_label or op.layer_label)
                if node_key in fact_labels or node_key in seen:
                    continue
                seen.add(node_key)
                marks.append(
                    SurgeryMark(
                        layer_label=str(op.layer_label),
                        call_label=str(call_label) if call_label else None,
                        site_key=ref,
                        kind="declared_target",
                        basis="heuristic",
                        row=f"{state} site cohort: "
                        f"{', '.join(transaction.edit_names) or 'edit'} "
                        "(no fire recorded at this instance)",
                    )
                )
    return marks


def surgery_facts(trace: Trace) -> SurgeryFacts:
    """Derive everything the mark family may cite from C03's audit record.

    Pure read of the canonical substrate: transaction envelopes and
    ``region_do`` rows on ``state_history``, per-op FireRecords, and F01
    stage-1 injected-op records. Nothing here resolves selectors or re-runs
    engines; a claim with no recorded entry simply does not appear.
    """

    transactions = tuple(_transaction_from_envelope(row) for row in _envelope_rows(trace))
    regions = tuple(_region_fact_from_row(row) for row in _region_rows(trace))
    marks: list[SurgeryMark] = _fire_marks(trace)
    marks.extend(_region_marks(trace, regions))
    injection_marks, host_counts = _injection_marks(trace)
    marks.extend(injection_marks)
    fact_labels = {
        label
        for mark in marks
        for label in (mark.call_label, mark.layer_label)
        if label is not None
    }
    marks.extend(_declared_target_marks(trace, transactions, fact_labels))
    return SurgeryFacts(
        transactions=transactions,
        regions=regions,
        marks=tuple(marks),
        injected_host_counts=host_counts,
    )


def splice_box_lines(facts: SurgeryFacts) -> tuple[str, ...]:
    """Render the per-lane splice-box disclosure lines.

    One box line per recorded transaction and per region row, each carrying
    ITS OWN lane's wording (recorded ``execution_effect`` verbatim, else the
    D17 lane law). The replay text and the live text are mutually false
    across lanes; a line never borrows the other lane's wording.
    """

    lines: list[str] = []
    for transaction in facts.transactions:
        wording, source = effect_wording(transaction.lane, transaction.execution_effect)
        edits = ", ".join(transaction.edit_names) or "edit"
        sites = ", ".join(transaction.site_keys) or "no recorded sites"
        lines.append(
            f"[surgery lane={transaction.lane} door={transaction.door}] {edits} at {sites}: "
            f"{wording} ({source}; {transaction.fire_count} fire(s), "
            f"status={transaction.status})"
        )
        for rule_id in transaction.zero_fire_rule_ids:
            lines.append(
                f"[surgery no-fire] rule {rule_id} declared and never fired: "
                "a misfire is data, not silence"
            )
    for region_fact in facts.regions:
        wording, source = effect_wording("replay", region_fact.execution_effect)
        lines.append(
            f"[surgery region lane=replay] edit={region_fact.edit} "
            f"members={len(region_fact.members)} exits={len(region_fact.exit_parents)}: "
            f"{wording} ({source})"
        )
    return tuple(lines)


@dataclass(frozen=True)
class SurgeryCensus:
    """The per-lane census that travels with every exported surgery figure."""

    lane_rows: tuple[str, ...]
    box_lines: tuple[str, ...]
    mark_totals: tuple[str, ...]
    credit_rows: tuple[str, ...]

    def lines(self) -> tuple[str, ...]:
        """Every census line, in render order."""

        return (
            ("surgery census (facts cited from the intervention audit)",)
            + self.lane_rows
            + self.box_lines
            + self.mark_totals
            + self.credit_rows
        )

    def to_text(self) -> str:
        """The full census as one plain-ASCII text block."""

        return "\n".join(self.lines()) + "\n"


def surgery_census(trace: Trace) -> SurgeryCensus:
    """Build the per-lane census for one trace's recorded surgery.

    An un-operated trace yields an honest empty census (the LENS refuses on
    missing headline evidence; the census itself is a disclosure and never
    refuses).
    """

    facts = surgery_facts(trace)
    per_lane: dict[str, tuple[int, int]] = {}
    for transaction in facts.transactions:
        count, fires = per_lane.get(transaction.lane, (0, 0))
        per_lane[transaction.lane] = (count + 1, fires + transaction.fire_count)
    lane_rows = tuple(
        f"lane {lane}: {count} transaction(s), {fires} fire(s)"
        for lane, (count, fires) in sorted(per_lane.items())
    ) or ("no intervention transactions recorded on this trace",)
    fact_count = sum(1 for mark in facts.marks if mark.basis == "fact")
    heuristic_count = sum(1 for mark in facts.marks if mark.basis == "heuristic")
    mark_totals = (
        f"marks: {fact_count} fact (solid border), {heuristic_count} heuristic (dashed border)",
        f"injected ops recorded: {sum(facts.injected_host_counts.values())}",
    )
    return SurgeryCensus(
        lane_rows=lane_rows,
        box_lines=splice_box_lines(facts),
        mark_totals=mark_totals,
        credit_rows=CREDIT_ROWS,
    )


def make_surgery_mark_spec_fn(trace: Trace, facts: SurgeryFacts | None = None) -> NodeSpecFn:
    """Build the ONE mark family's node-spec callback.

    Line-style law: FACT marks paint a solid border at penwidth 3.0 in the
    intervention site color; HEURISTIC marks paint a DASHED border at
    penwidth 2.25 in the same hue. Each mark appends its citation row to the
    node label (capped, with an honest "+N more" fold).
    """

    resolved = surgery_facts(trace) if facts is None else facts
    by_call: dict[str, list[SurgeryMark]] = {}
    by_layer: dict[str, list[SurgeryMark]] = {}
    for mark in resolved.marks:
        if mark.call_label:
            by_call.setdefault(mark.call_label, []).append(mark)
        by_layer.setdefault(mark.layer_label, []).append(mark)

    def surgery_mark_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec:
        """Apply the mark family to one rendered node (pre-user slot)."""

        pass_label = get_multipass_attr(layer_log, "label", None, multipass=None)
        node_marks: list[SurgeryMark] = []
        if isinstance(pass_label, str) and pass_label in by_call:
            node_marks = by_call[pass_label]
        else:
            node_marks = by_layer.get(str(getattr(layer_log, "layer_label", "")), [])
        if not node_marks:
            return default_spec
        any_fact = any(mark.basis == "fact" for mark in node_marks)
        # replace() copies shallowly; give the copy its OWN lines list so the
        # appended mark rows never leak into the caller's default spec.
        spec = default_spec.replace(
            color=INTERVENTION_SITE_COLOR,
            penwidth=3.0 if any_fact else 2.25,
            lines=list(default_spec.lines),
        )
        if not any_fact:
            spec.style = f"{spec.style},dashed" if spec.style else "dashed"
        shown = node_marks[:_MAX_MARK_ROWS_PER_NODE]
        for mark in shown:
            spec.lines.append(mark.row)
        if len(node_marks) > len(shown):
            spec.lines.append(f"+{len(node_marks) - len(shown)} more surgery mark(s)")
        return spec

    return surgery_mark_spec_fn


def surgery_stage(trace: Trace) -> tuple[NodeSpecFn, tuple[str, ...]]:
    """Resolve the surgery lens stage: marks callback + census disclosure.

    Refuses typed when the headline evidence (C03 audit entries) is absent:
    with nothing recorded, the picture would answer a different question.
    """

    facts = surgery_facts(trace)
    if not facts.has_evidence:
        raise InvalidArgumentError(
            "the surgery lens found no intervention audit entries on this "
            "trace: no transaction envelopes, no region rows, no fire "
            "records, no injected ops -- there is no recorded surgery to "
            "render",
            code="surgery_evidence_missing",
            remedy=(
                "run an edit first (fork.do(...), tl.trace(..., intervene=...), "
                "spec.bind(model), or a region do), or use theme='overview' "
                "for an un-operated picture"
            ),
            argument="lens",
        )
    census = surgery_census(trace)
    lines = census.lines()
    if len(lines) > _MAX_CAPTION_LINES:
        lines = lines[: _MAX_CAPTION_LINES - 1] + (
            f"+{len(lines) - (_MAX_CAPTION_LINES - 1)} more census line(s): "
            "see the .census.txt sidecar or surgery_census(trace).to_text()",
        )
    return make_surgery_mark_spec_fn(trace, facts), lines


def render_surgery(trace: Trace, **user_kwargs: Any) -> Any:
    """Draw ``trace`` under the surgery lens; the census travels with it.

    A thin door over the standard lens resolution (never a new view kind):
    the marks and the caption census render through ``Trace.draw``, and when
    a figure lands on disk the full census text is written beside it as
    ``<outpath>.census.txt`` -- the per-lane record cannot be separated from
    the exported picture.
    """

    from .lenses._resolve import draw_with_lens

    result = draw_with_lens(trace, "surgery", **user_kwargs)
    if not user_kwargs.get("return_graph", False):
        outpath = str(user_kwargs.get("vis_outpath", "modelgraph"))
        census_path = f"{outpath}.census.txt"
        with open(census_path, "w", encoding="ascii", errors="replace") as census_file:
            census_file.write(surgery_census(trace).to_text())
    return result


#: The C05 registry row: the surgery mark family IS a lens over the standard
#: draw surface. Headline evidence = C03's audit entries (checked by
#: :func:`surgery_stage`; absence refuses ``surgery_evidence_missing``).
SURGERY_LENS = register_lens(
    LensPreset(
        name="surgery",
        question="what did surgery change on this capture, and on what recorded authority?",
        members={"show_legend": True},
        surface=GRAPHVIZ_DRAW_SURFACE,
        headline_evidence="intervention_audit",
        disclosure=(
            "marks cite audit entries: solid border = recorded fact, "
            "dashed border = heuristic attribution",
            "edits replace values or substitute exits; execution removal exists in no lane",
        ),
    )
)
