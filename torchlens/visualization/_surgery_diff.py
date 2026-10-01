"""Site-key-first surgery diff (lane F43): two captures, joined on identity.

The diff answers "what is structurally different between this capture and
that one, and on what authority is each pairing claimed?" It joins the two
op populations SITE-KEY-FIRST: a pairing on the L1 structural site key is a
FACT join (the keys are minted by the engines, portable, and
policy-independent) and renders as a SOLID alignment line; a pairing that
falls back to pass-qualified label equality or positional order within a
shared site key is a HEURISTIC join and renders DASHED. The two styles are
never mixed: the line-style law is the same one the surgery mark family
uses (fact = solid, heuristic = dashed).

Never ghosted-as-deleted (the fold's CONFIRM-3 rule, backed by the round-3
execution counter): an op recorded on only one side renders at FULL
strength with the disclosure "not recorded in the <other> capture". No
wording here uses the banned execution-removal verbs -- execution
removal exists in no lane, and absence from a capture is a fact about the
RECORD, not about execution.

Surgery marks travel: subject-side nodes whose audit record carries fire /
region / injection evidence render with the mark family's solid border, so
the diff shows WHERE the recorded surgery sits relative to the structural
drift. The per-lane census (with its CREDIT rows) is written beside every
exported diff figure, and the figure's caption carries the join counts.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import graphviz

from .._errors import InvalidArgumentError
from .._vocab.node_spec import INTERVENTION_SITE_COLOR
from ._render_utils import html_escape, render_dot_to_file, strip_known_extension
from .surgery_visuals import SurgeryFacts, surgery_facts
from .themes import resolve_theme, theme_graph_attrs, theme_node_attrs

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

__all__ = [
    "DIFF_JOIN_KINDS",
    "SurgeryDiff",
    "SurgeryDiffRow",
    "surgery_diff",
]

#: Closed join vocabulary. ``site_key`` is the FACT join; ``label`` and
#: ``site_key_positional`` are HEURISTIC joins (dashed); the two only-rows
#: carry no join at all.
DIFF_JOIN_KINDS: frozenset[str] = frozenset({"site_key", "label", "site_key_positional"})

#: Drawn-row ceiling for the paired figure; overflow is disclosed in the
#: caption, never silently dropped.
_MAX_DRAWN_ROWS = 60


@dataclass(frozen=True)
class SurgeryDiffRow:
    """One diff row: a pairing claim or a single-side disclosure.

    Attributes
    ----------
    kind:
        ``"matched"`` / ``"only_in_subject"`` / ``"only_in_reference"``.
    join:
        Closed :data:`DIFF_JOIN_KINDS` value for matched rows; ``None``
        on only-rows.
    basis:
        ``"fact"`` (site-key join) or ``"heuristic"`` (label or positional
        join); ``None`` on only-rows. Drives the line-style law.
    subject_label / reference_label:
        Pass-qualified labels on each side (``None`` where absent).
    site_key:
        The shared structural site key, when the join recorded one.
    notes:
        Disclosed per-row facts: shape drift, subject-side surgery marks,
        and the "not recorded in the <other> capture" disclosure.
    """

    kind: str
    join: str | None
    basis: str | None
    subject_label: str | None
    reference_label: str | None
    site_key: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SurgeryDiff:
    """The site-key-first diff between one subject and one reference trace."""

    subject_name: str
    reference_name: str
    rows: tuple[SurgeryDiffRow, ...]
    subject_facts: SurgeryFacts

    def counts(self) -> dict[str, int]:
        """Row counts by claim class (fact/heuristic joins, only-rows)."""

        result = {"fact": 0, "heuristic": 0, "only_in_subject": 0, "only_in_reference": 0}
        for row in self.rows:
            if row.kind == "matched" and row.basis is not None:
                result[row.basis] += 1
            elif row.kind in ("only_in_subject", "only_in_reference"):
                result[row.kind] += 1
        return result

    def lines(self) -> tuple[str, ...]:
        """The diff as plain-ASCII rows (join authority always named)."""

        counts = self.counts()
        header = (
            f"surgery diff: {self.subject_name} vs {self.reference_name}",
            f"joins: {counts['fact']} site-key (fact, solid), "
            f"{counts['heuristic']} heuristic (dashed); "
            f"{counts['only_in_subject']} only in {self.subject_name}, "
            f"{counts['only_in_reference']} only in {self.reference_name}",
        )
        body = []
        for row in self.rows:
            if row.kind == "matched":
                body.append(
                    f"  {row.subject_label} <-> {row.reference_label} "
                    f"[{row.join}/{row.basis}]" + (f" {'; '.join(row.notes)}" if row.notes else "")
                )
            else:
                label = row.subject_label or row.reference_label
                body.append(f"  {label} [{row.kind}] {'; '.join(row.notes)}")
        return header + tuple(body)

    def to_text(self) -> str:
        """The full diff as one plain-ASCII text block."""

        return "\n".join(self.lines()) + "\n"

    def draw(
        self,
        vis_outpath: str = "surgery_diff",
        *,
        vis_fileformat: str = "svg",
        vis_save_only: bool = False,
        theme: str = "torchlens",
    ) -> str:
        """Render the paired two-column diff figure; the census travels.

        Solid alignment lines are site-key (fact) joins, dashed lines are
        heuristic joins; single-side ops render at full strength with their
        disclosure -- never ghosted. The subject's per-lane census (CREDIT
        rows included) is written beside the figure as
        ``<outpath>.census.txt``.
        """

        outpath = strip_known_extension(vis_outpath)
        dot = _build_diff_dot(self, theme=theme)
        render_dot_to_file(dot, outpath, vis_fileformat, vis_save_only)
        census_path = f"{outpath}.census.txt"
        with open(census_path, "w", encoding="ascii", errors="replace") as census_file:
            census_file.write(surgery_census_text_for_diff(self))
        return f"{outpath}.{vis_fileformat}"


def surgery_census_text_for_diff(diff: SurgeryDiff) -> str:
    """The sidecar text for one diff export: diff rows + subject census."""

    return diff.to_text() + "\n" + _subject_census_text(diff)


def _subject_census_text(diff: SurgeryDiff) -> str:
    """Rebuild the subject-side census block from the derived facts."""

    from .surgery_visuals import CREDIT_ROWS, splice_box_lines

    lines = ["subject surgery census:"]
    lines.extend(splice_box_lines(diff.subject_facts))
    lines.extend(CREDIT_ROWS)
    return "\n".join(lines) + "\n"


def _op_rows(trace: Any) -> list[tuple[str, Any]]:
    """(pass-qualified label, op) rows in execution order."""

    return [(str(label), trace.ops[label]) for label in getattr(trace, "op_labels", ()) or ()]


def _site_key_of(op: Any) -> str | None:
    """The op's structural site key, if the capture minted one."""

    raw_key = getattr(op, "site_key", None)
    return str(raw_key) if raw_key else None


def _shape_note(subject_op: Any, reference_op: Any) -> str | None:
    """A shape-drift note when both sides recorded output shapes."""

    subject_shape = getattr(subject_op, "shape", None)
    reference_shape = getattr(reference_op, "shape", None)
    if subject_shape and reference_shape and tuple(subject_shape) != tuple(reference_shape):
        return f"shape {tuple(reference_shape)} -> {tuple(subject_shape)}"
    return None


def _matched_row(
    subject: tuple[str, Any],
    reference: tuple[str, Any],
    *,
    join: str,
    facts: SurgeryFacts,
) -> SurgeryDiffRow:
    """Build one matched row with its authority and disclosed notes.

    ``subject`` and ``reference`` are the joined ``(label, op)`` pairs.
    """

    subject_label, subject_op = subject
    reference_label, reference_op = reference
    notes: list[str] = []
    shape_note = _shape_note(subject_op, reference_op)
    if shape_note is not None:
        notes.append(shape_note)
    notes.extend(mark.row for mark in facts.marks_for(subject_label))
    return SurgeryDiffRow(
        kind="matched",
        join=join,
        basis="fact" if join == "site_key" else "heuristic",
        subject_label=subject_label,
        reference_label=reference_label,
        site_key=_site_key_of(subject_op),
        notes=tuple(notes),
    )


def _join_by_site_key(
    subject_rows: list[tuple[str, Any]],
    reference_rows: list[tuple[str, Any]],
    facts: SurgeryFacts,
) -> tuple[list[SurgeryDiffRow], set[str], set[str]]:
    """Fact joins (unique keys) + positional heuristic joins (shared keys)."""

    def index(rows: list[tuple[str, Any]]) -> dict[str, list[tuple[str, Any]]]:
        """Group rows by structural site key, dropping keyless ops."""

        keyed: dict[str, list[tuple[str, Any]]] = {}
        for label, op in rows:
            key = _site_key_of(op)
            if key is not None:
                keyed.setdefault(key, []).append((label, op))
        return keyed

    subject_keyed = index(subject_rows)
    reference_keyed = index(reference_rows)
    rows: list[SurgeryDiffRow] = []
    used_subject: set[str] = set()
    used_reference: set[str] = set()
    for key, subject_group in subject_keyed.items():
        reference_group = reference_keyed.get(key)
        if reference_group is None:
            continue
        if len(subject_group) == 1 and len(reference_group) == 1:
            join = "site_key"
            pairs = [(subject_group[0], reference_group[0])]
        elif len(subject_group) == len(reference_group):
            # Same cohort size under one shared key: pair by pass order.
            # Positional pairing is an ATTRIBUTION, not a recorded identity,
            # so these rows are heuristic (dashed) by construction.
            join = "site_key_positional"
            pairs = list(zip(subject_group, reference_group, strict=True))
        else:
            continue  # ragged cohort: leave both sides to the label join
        for (subject_label, subject_op), (reference_label, reference_op) in pairs:
            rows.append(
                _matched_row(
                    (subject_label, subject_op),
                    (reference_label, reference_op),
                    join=join,
                    facts=facts,
                )
            )
            used_subject.add(subject_label)
            used_reference.add(reference_label)
    return rows, used_subject, used_reference


def surgery_diff(subject: Trace, reference: Trace) -> SurgeryDiff:
    """Diff ``subject`` against ``reference``, site-key-first.

    Join ladder: (1) unique structural site keys on both sides -- FACT;
    (2) equal-size cohorts under one shared key, paired by pass order --
    HEURISTIC; (3) pass-qualified label equality -- HEURISTIC; (4) the
    remainder becomes only-rows disclosed as "not recorded in the other
    capture", never ghosted and never worded as execution removal.

    Raises
    ------
    InvalidArgumentError
        ``surgery_diff_incomparable`` when the two operands are the same
        object (a self-diff is vacuous) or either side records no ops.
    """

    if subject is reference:
        raise InvalidArgumentError(
            "surgery_diff received the SAME trace object on both sides; a "
            "self-diff is vacuous and would render every row as a trivial "
            "fact join",
            code="surgery_diff_incomparable",
            remedy="pass the edited capture as subject and the baseline capture as reference",
            argument="reference",
        )
    subject_rows = _op_rows(subject)
    reference_rows = _op_rows(reference)
    if not subject_rows or not reference_rows:
        raise InvalidArgumentError(
            "surgery_diff needs recorded ops on BOTH sides; one operand records none",
            code="surgery_diff_incomparable",
            remedy="pass two finished captures of the same model family",
            argument="subject" if not subject_rows else "reference",
        )
    facts = surgery_facts(subject)
    rows, used_subject, used_reference = _join_by_site_key(subject_rows, reference_rows, facts)

    reference_by_label = dict(reference_rows)
    for subject_label, subject_op in subject_rows:
        if subject_label in used_subject:
            continue
        reference_op = reference_by_label.get(subject_label)
        if reference_op is not None and subject_label not in used_reference:
            rows.append(
                _matched_row(
                    (subject_label, subject_op),
                    (subject_label, reference_op),
                    join="label",
                    facts=facts,
                )
            )
            used_subject.add(subject_label)
            used_reference.add(subject_label)

    subject_name = "subject"
    reference_name = "reference"
    for subject_label, subject_op in subject_rows:
        if subject_label not in used_subject:
            rows.append(
                SurgeryDiffRow(
                    kind="only_in_subject",
                    join=None,
                    basis=None,
                    subject_label=subject_label,
                    reference_label=None,
                    site_key=_site_key_of(subject_op),
                    notes=(f"not recorded in the {reference_name} capture",),
                )
            )
    for reference_label, reference_op in reference_rows:
        if reference_label not in used_reference:
            rows.append(
                SurgeryDiffRow(
                    kind="only_in_reference",
                    join=None,
                    basis=None,
                    subject_label=None,
                    reference_label=reference_label,
                    site_key=_site_key_of(reference_op),
                    notes=(f"not recorded in the {subject_name} capture",),
                )
            )
    order = {label: position for position, (label, _op) in enumerate(subject_rows)}
    reference_order = {label: position for position, (label, _op) in enumerate(reference_rows)}
    rows.sort(
        key=lambda row: (
            order.get(row.subject_label or "", len(order))
            if row.subject_label
            else reference_order.get(row.reference_label or "", len(reference_order)),
            row.kind,
        )
    )
    return SurgeryDiff(
        subject_name=subject_name,
        reference_name=reference_name,
        rows=tuple(rows),
        subject_facts=facts,
    )


def _diff_node(
    dot: graphviz.Digraph,
    node_id: str,
    label: str,
    notes: tuple[str, ...],
    *,
    node_attrs: dict[str, str],
) -> None:
    """Emit one diff node with the caller-resolved per-node attributes."""

    text_rows = [f"<B>{html_escape(label)}</B>"]
    text_rows.extend(html_escape(note) for note in notes[:2])
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        + "".join(f'<TR><TD ALIGN="CENTER">{row}</TD></TR>' for row in text_rows)
        + "</TABLE>>"
    )
    dot.node(node_id, label=html, **node_attrs)


def _emit_side(
    dot: graphviz.Digraph,
    side: str,
    drawn: list[SurgeryDiffRow],
    facts: SurgeryFacts,
    node_attrs: dict[str, str],
) -> None:
    """Emit one column cluster (subject or reference) with ordering edges."""

    with dot.subgraph(name=f"cluster_{side}") as cluster:
        cluster.attr(label=side, style="rounded")
        previous: str | None = None
        for position, row in enumerate(drawn):
            label = row.subject_label if side == "subject" else row.reference_label
            if label is None:
                continue
            node_id = f"{side}_{position}"
            marked = side == "subject" and bool(facts.marks_for(label))
            # Per-lane honesty: a matched row's notes (shape drift, subject
            # surgery marks) are SUBJECT-side claims; the reference node
            # carries only its own single-side disclosure.
            if side == "subject":
                notes = row.notes if row.kind != "only_in_reference" else ()
            else:
                notes = row.notes if row.kind == "only_in_reference" else ()
            # Surgery-marked nodes carry the solid mark border.
            attrs = dict(node_attrs)
            if marked:
                attrs.update({"color": INTERVENTION_SITE_COLOR, "penwidth": "3.0"})
            _diff_node(cluster, node_id, label, notes, node_attrs=attrs)
            if previous is not None:
                cluster.edge(previous, node_id, style="invis")
            previous = node_id


def _build_diff_dot(diff: SurgeryDiff, *, theme: str) -> graphviz.Digraph:
    """Build the paired two-column DOT for one surgery diff."""

    resolved_theme = resolve_theme(theme)
    dot = graphviz.Digraph(name="surgery_diff")
    dot.attr(rankdir="TB", newrank="true", **theme_graph_attrs(resolved_theme))
    node_attrs = dict(theme_node_attrs(resolved_theme))
    node_attrs.setdefault("shape", "box")
    node_attrs.setdefault("style", "filled,rounded")

    drawn = list(diff.rows[:_MAX_DRAWN_ROWS])
    _emit_side(dot, "subject", drawn, diff.subject_facts, node_attrs)
    _emit_side(dot, "reference", drawn, diff.subject_facts, node_attrs)
    for position, row in enumerate(drawn):
        if row.kind != "matched":
            continue
        style = "solid" if row.basis == "fact" else "dashed"
        dot.edge(
            f"subject_{position}",
            f"reference_{position}",
            style=style,
            color="#808080",
            constraint="false",
            dir="none",
        )
    counts = diff.counts()
    caption_lines = [
        f"surgery diff: {counts['fact']} site-key joins (solid) / "
        f"{counts['heuristic']} heuristic joins (dashed)",
        f"{counts['only_in_subject']} only in subject, "
        f"{counts['only_in_reference']} only in reference "
        "(drawn at full strength: absence is a fact about the record, "
        "not execution)",
    ]
    if len(diff.rows) > len(drawn):
        caption_lines.append(f"+{len(diff.rows) - len(drawn)} more rows in the census sidecar")
    dot.attr(label="\\n".join(caption_lines), labelloc="b", fontsize="10")
    return dot
