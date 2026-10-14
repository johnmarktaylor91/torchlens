"""Automatic ``fed_by`` span finding and the strict token-pair read.

Two halves, both over integer token ids only (no tokenizer in core):

MATCHER. ``b.relate("fed_by", later, earlier, token_spans="auto")`` asks
TorchLens to find the earlier member's output inside the later member's
prompt. The source's eligible output (its bound ``completion`` range minus
``excluded_ids``) splits into maximal runs; each run is searched as an EXACT
run of ids in the reader's prompt (the reader's bound stream before its own
completion, when it names one). A run that is not found is retried with up to
``_MAX_TRIM`` boundary tokens trimmed per side (a quoted output often
retokenizes at its edges), fewest trimmed tokens first; a trimmed core keeps
at least two tokens and at least half its run. At the first trim level with
any hit, exactly one placement must exist: several placements (a repeated
quote, or a tie between two trims) refuse ``bundle_relation_auto_spans_ambiguous``
and the remedy is to declare the spans. No hit for any run refuses
``bundle_relation_auto_spans_unmatched``; missing evidence (no binding, no
completion range, ids not retained) refuses
``bundle_relation_auto_spans_unavailable``. The row is written with
``basis="matched"``.

ATTESTATION. A ``matched`` row is attested by re-derivation, at every door
(relate, the constructor, fork, load): its spans must re-verify exactly like
``verified`` spans (ids equal span by span inside the source completion), and
the matcher, re-run over the members' retained ids, must return the same
spans. Spans that verify but are not what the matcher finds settle as
``verified`` (checked, not found by TorchLens); contradicted spans refuse
``bundle_relation_span_unverified``; absent evidence clamps to ``declared``.

TOKEN PAIRS. ``Bundle.token_pairs(reader, source)`` is the one verified
alignment primitive: index pairs ``(source_pos, reader_pos)`` for positions
established one-to-one. It pairs only spans of ``verified``/``matched`` rows
that re-verify at call time. Declared spans (retokenized many-to-many text,
mixed-origin tokens) stay span correspondences and are reported in
``excluded`` with a reason, never converted to pairs; so are excluded ids,
positions claimed twice, and eligible source tokens no verified span covers.
Equal lengths are never evidence of matching meaning: only equal ids at
TorchLens-checked positions are.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..errors.episode import BundleRelationError
from ._fed_by import (
    EARNED_BASES,
    FED_BY_KIND,
    assess_spans,
    bind_member,
    member_binding,
    retained_ids,
    stream_ids,
)
from ._relations import MemberRelationRow

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from . import Bundle

#: Boundary tokens the matcher may trim per side of a run.
_MAX_TRIM = 2
#: A trimmed core keeps at least this many tokens (and at least half its run).
_MIN_TRIMMED_CORE = 2
_ITEM_BYTES = array("q").itemsize


def _eligible_runs(
    source_ids: Sequence[int], completion: Sequence[int], excluded: frozenset[int]
) -> list[tuple[int, int]]:
    """Return the maximal half-open runs of eligible source positions."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position in range(int(completion[0]), int(completion[1])):
        if source_ids[position] in excluded:
            if start is not None:
                runs.append((start, position))
            start = None
        elif start is None:
            start = position
    if start is not None:
        runs.append((start, int(completion[1])))
    return runs


def _placements(core: Sequence[int], haystack: bytes, *, limit: int = 2) -> list[int]:
    """Return up to ``limit`` token positions where ``core`` occurs exactly."""

    needle = array("q", core).tobytes()
    found: list[int] = []
    offset = haystack.find(needle)
    while offset >= 0 and len(found) < limit:
        if offset % _ITEM_BYTES == 0:
            found.append(offset // _ITEM_BYTES)
        offset = haystack.find(needle, offset + 1)
    return found


def _trims(length: int) -> list[list[tuple[int, int]]]:
    """Return the allowed ``(left, right)`` trims grouped by total, fewest first."""

    levels: list[list[tuple[int, int]]] = [[(0, 0)]]
    for total in range(1, 2 * _MAX_TRIM + 1):
        level = []
        for left in range(max(0, total - _MAX_TRIM), min(total, _MAX_TRIM) + 1):
            core = length - total
            if core >= _MIN_TRIMMED_CORE and 2 * core >= length:
                level.append((left, total - left))
        if level:
            levels.append(level)
    return levels


def _match_run(run: tuple[int, int], source_ids: Sequence[int], haystack: bytes) -> list[list[int]]:
    """Return every placement at the first trim level with a hit (empty: no hit)."""

    lo, hi = run
    for level in _trims(hi - lo):
        hits: list[list[int]] = []
        for left, right in level:
            core = source_ids[lo + left : hi - right]
            for position in _placements(core, haystack):
                hits.append([lo + left, hi - right, position, position + len(core)])
        if hits:
            return hits
    return []


def find_token_spans(
    source_ids: Sequence[int],
    completion: Sequence[int],
    reader_ids: Sequence[int],
    *,
    excluded_ids: Sequence[int] = (),
    search_end: int | None = None,
) -> list[list[int]]:
    """Find the source's eligible output inside ``reader_ids[:search_end]``.

    Returns
    -------
    list[list[int]]
        One ``[src_start, src_end, dst_start, dst_end]`` row per matched run,
        in source order. Runs with no placement are left out (they lower the
        row's coverage).

    Raises
    ------
    BundleRelationError
        ``bundle_relation_auto_spans_ambiguous`` when a run has several
        placements at its first hit level or two runs claim overlapping
        destination tokens; ``bundle_relation_auto_spans_unmatched`` when no
        run is found at all.
    """

    end = len(reader_ids) if search_end is None else min(search_end, len(reader_ids))
    haystack = array("q", reader_ids[:end]).tobytes()
    runs = _eligible_runs(source_ids, completion, frozenset(excluded_ids))
    spans: list[list[int]] = []
    for run in runs:
        hits = _match_run(run, source_ids, haystack)
        if len(hits) > 1:
            raise BundleRelationError(
                f"token_spans='auto' cannot place source run {list(run)} uniquely: it has "
                f"{len(hits)} or more placements in the reader's prompt (a repeated quote, "
                "or a tie between two trims).",
                code="bundle_relation_auto_spans_ambiguous",
                remedy="declare the spans as [src_start, src_end, dst_start, dst_end] rows",
                row_kind=FED_BY_KIND,
                source_run=list(run),
            )
        spans.extend(hits)
    if not spans:
        raise BundleRelationError(
            "token_spans='auto' found no run of the source output in the reader's prompt, "
            f"even with up to {_MAX_TRIM} boundary tokens trimmed per side.",
            code="bundle_relation_auto_spans_unmatched",
            remedy=(
                "check the source's bound completion range; declare the spans for "
                "retokenized or paraphrased text"
            ),
            row_kind=FED_BY_KIND,
            source_runs=[list(run) for run in runs],
        )
    claimed: set[int] = set()
    for span in spans:
        destination = set(range(span[2], span[3]))
        if claimed & destination:
            raise BundleRelationError(
                "token_spans='auto' cannot place the source output uniquely: two source "
                f"runs land on the same reader tokens (span {span}).",
                code="bundle_relation_auto_spans_ambiguous",
                remedy="declare the spans as [src_start, src_end, dst_start, dst_end] rows",
                row_kind=FED_BY_KIND,
                span=span,
            )
        claimed |= destination
    return spans


def _match_inputs(
    reader: Trace, source: Trace
) -> tuple[list[int], Mapping[str, Any], list[int], int] | str:
    """Return ``(source_ids, source_binding, reader_ids, search_end)`` or a reason."""

    source_binding, reader_binding = member_binding(source), member_binding(reader)
    if source_binding is None or source_binding.get("completion") is None:
        return "the source binding names no completion range, so its output is unknown"
    if reader_binding is None:
        return "the reader carries no token_boundary_v1 binding"
    source_ids = stream_ids(source, source_binding)
    reader_ids = stream_ids(reader, reader_binding)
    if source_ids is None or reader_ids is None:
        who = "source" if source_ids is None else "reader"
        return f"the {who}'s token ids are not retained (or its binding disagrees with them)"
    reader_completion = reader_binding.get("completion")
    search_end = int(reader_completion[0]) if reader_completion is not None else len(reader_ids)
    return source_ids, source_binding, reader_ids, search_end


def auto_token_spans(reader: Trace, source: Trace) -> list[list[int]]:
    """Bind both members and return the matched spans for ``token_spans='auto'``.

    Raises
    ------
    BundleRelationError
        ``bundle_relation_auto_spans_unavailable``, ``_ambiguous`` or
        ``_unmatched`` (see :func:`find_token_spans`).
    """

    bind_member(source)
    bind_member(reader)
    inputs = _match_inputs(reader, source)
    if isinstance(inputs, str):
        raise BundleRelationError(
            f"token_spans='auto' needs evidence TorchLens does not have: {inputs}.",
            code="bundle_relation_auto_spans_unavailable",
            remedy=(
                "attach a token_boundary_v1 descriptor naming the source's completion "
                "range (torchlens.io.attach_sidecar), keep the token-id inputs saved, "
                "or declare token_spans rows"
            ),
            row_kind=FED_BY_KIND,
            reason=inputs,
        )
    source_ids, source_binding, reader_ids, search_end = inputs
    return find_token_spans(
        source_ids,
        source_binding["completion"],
        reader_ids,
        excluded_ids=source_binding.get("excluded_ids", []),
        search_end=search_end,
    )


def rederives_as_matched(spans: Sequence[Sequence[int]], reader: Trace, source: Trace) -> bool:
    """Return whether the matcher, re-run on the members, finds exactly ``spans``."""

    inputs = _match_inputs(reader, source)
    if isinstance(inputs, str):
        return False
    source_ids, source_binding, reader_ids, search_end = inputs
    try:
        found = find_token_spans(
            source_ids,
            source_binding["completion"],
            reader_ids,
            excluded_ids=source_binding.get("excluded_ids", []),
            search_end=search_end,
        )
    except BundleRelationError:
        return False
    return found == [list(span) for span in spans]


# ---------------------------------------------------------------------------
# The strict token-pair read.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenPairs:
    """Result of ``Bundle.token_pairs(reader, source)`` (documented-unstable).

    Attributes
    ----------
    reader, source:
        Member names, in ``relate``'s order (later, then earlier).
    pairs:
        ``(source_pos, reader_pos)`` index pairs, each position at most once,
        sorted by source position. Positions index each member's bound token
        stream (the first integer input by default).
    excluded:
        Segments TorchLens did not pair, each a dict with ``side``
        (``"source"`` or ``"reader"``), ``start``/``end`` (half-open, ``None``
        when the row has no spans), ``reason`` and ``row_index`` (the relation
        row, ``None`` for uncovered source tokens).
    coverage:
        Paired share of the source's eligible tokens (bound completion minus
        ``excluded_ids``), or ``None`` without a bound completion range.
    """

    reader: str
    source: str
    pairs: tuple[tuple[int, int], ...]
    excluded: tuple[dict[str, Any], ...]
    coverage: float | None

    @property
    def source_positions(self) -> list[int]:
        """Source-side positions of :attr:`pairs`, for indexing the source's activations."""

        return [source_pos for source_pos, _ in self.pairs]

    @property
    def reader_positions(self) -> list[int]:
        """Reader-side positions of :attr:`pairs`, aligned with :attr:`source_positions`."""

        return [reader_pos for _, reader_pos in self.pairs]


def _segment(
    side: str, start: int | None, end: int | None, reason: str, row_index: int | None
) -> dict[str, Any]:
    """Return one ``excluded`` entry."""

    return {"side": side, "start": start, "end": end, "reason": reason, "row_index": row_index}


def _declared_reason(span: Sequence[int]) -> str:
    """Name why a declared span is a correspondence, not index pairs."""

    source_len, reader_len = span[1] - span[0], span[3] - span[2]
    if source_len != reader_len:
        return (
            f"declared span: {source_len} source tokens against {reader_len} reader "
            "tokens (retokenized text is a span correspondence, never index pairs)"
        )
    return "declared span: TorchLens has not verified that these tokens correspond"


def _tokenizers_conflict(reader: Trace, source: Trace) -> bool:
    """Return whether both bindings name a tokenizer and the names differ."""

    names = [(member_binding(trace) or {}).get("tokenizer") for trace in (reader, source)]
    return None not in names and names[0] != names[1]


def _row_candidates(
    index: int,
    row: MemberRelationRow,
    reader: Trace,
    source: Trace,
    excluded: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return the checked one-step candidates of one row; record what it excludes."""

    spans = row.params.get("token_spans") or []
    if not spans:
        excluded.append(_segment("source", None, None, "fed_by row declares no spans", index))
        return []
    if row.params.get("basis") not in EARNED_BASES:
        for span in spans:
            excluded.append(_segment("source", span[0], span[1], _declared_reason(span), index))
            excluded.append(_segment("reader", span[2], span[3], _declared_reason(span), index))
        return []
    candidates: list[tuple[int, int]] = []
    for span in spans:
        state, _, reason = assess_spans([span], reader, source)
        if state == "contradicted":
            raise BundleRelationError(
                f"fed_by row {index} reads basis {row.params['basis']!r} but its span "
                f"{list(span)} no longer re-checks: {reason}.",
                code="bundle_relation_span_unverified",
                remedy="re-relate the members from their captures, or drop the row",
                row_index=index,
                row_kind=FED_BY_KIND,
                reason=reason,
            )
        if state != "verified":
            excluded.append(_segment("source", span[0], span[1], f"unverifiable: {reason}", index))
            continue
        candidates.extend(zip(range(span[0], span[1]), range(span[2], span[3])))
    return candidates


def _one_to_one(
    candidates: Sequence[tuple[int, int]], excluded: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Keep only positions claimed exactly once on both sides."""

    by_source: dict[int, set[int]] = {}
    by_reader: dict[int, set[int]] = {}
    for source_pos, reader_pos in candidates:
        by_source.setdefault(source_pos, set()).add(reader_pos)
        by_reader.setdefault(reader_pos, set()).add(source_pos)
    kept: list[tuple[int, int]] = []
    for source_pos, reader_pos in sorted(set(candidates)):
        if len(by_source[source_pos]) > 1:
            reason = "source token maps to several reader tokens"
            excluded.append(_segment("source", source_pos, source_pos + 1, reason, None))
        elif len(by_reader[reader_pos]) > 1:
            reason = "reader token is claimed by several source tokens"
            excluded.append(_segment("reader", reader_pos, reader_pos + 1, reason, None))
        else:
            kept.append((source_pos, reader_pos))
    return kept


def _uncovered_runs(eligible: Sequence[int], paired: set[int]) -> list[tuple[int, int]]:
    """Return maximal runs of eligible source positions with no pair."""

    runs: list[tuple[int, int]] = []
    for position in eligible:
        if position in paired:
            continue
        if runs and runs[-1][1] == position:
            runs[-1] = (runs[-1][0], position + 1)
        else:
            runs.append((position, position + 1))
    return runs


def _eligible_positions(source: Trace) -> list[int] | None:
    """Return the source's eligible positions, or ``None`` without a completion range."""

    binding = member_binding(source)
    completion = binding.get("completion") if binding is not None else None
    ids = retained_ids(source, binding) if binding is not None else None
    if completion is None or ids is None:
        return None
    excluded_ids = set(binding.get("excluded_ids", [])) if binding is not None else set()
    return [p for p in range(int(completion[0]), int(completion[1])) if ids[p] not in excluded_ids]


def _bundle_token_pairs(self: Bundle, reader: str | Trace, source: str | Trace) -> TokenPairs:
    """Return the verified one-to-one token positions of "reader fed_by source".

    Arguments follow ``relate("fed_by", later, earlier)``: the READER (the
    later call, whose prompt holds the text) first, then the SOURCE (the
    earlier call that produced it). Exposed as the dynamic method
    ``Bundle.token_pairs`` (spelling documented-unstable).

    Only spans of ``verified``/``matched`` rows from ``reader`` to ``source``
    that re-verify now are paired; everything else is reported in
    ``excluded`` with a reason (see :class:`TokenPairs` and the module
    docstring). Nothing is reduced or averaged: any many-to-one reduction is
    the caller's descriptive choice.

    Raises
    ------
    KeyError
        When a name is not a member.
    BundleRelationError
        ``bundle_relation_span_unverified`` when a verified row's evidence
        now contradicts it.
    """

    names = [self._coerce_member_name(name) for name in (reader, source)]
    for name in names:
        if name not in self._members:
            raise KeyError(f"{name!r} is not a member of this Bundle; members: {self.names}")
    reader_trace, source_trace = self._members[names[0]], self._members[names[1]]
    excluded: list[dict[str, Any]] = []
    candidates: list[tuple[int, int]] = []
    conflict = _tokenizers_conflict(reader_trace, source_trace)
    for index, row in enumerate(self._member_relations.rows):
        if not (
            isinstance(row, MemberRelationRow)
            and row.kind == FED_BY_KIND
            and (row.from_member, row.to_member) == (names[0], names[1])
        ):
            continue
        if conflict:
            reason = "incompatible token streams: the bindings name different tokenizers"
            excluded.append(_segment("source", None, None, reason, index))
            continue
        candidates.extend(_row_candidates(index, row, reader_trace, source_trace, excluded))
    excluded_ids = set((member_binding(source_trace) or {}).get("excluded_ids", []))
    source_ids = _source_ids_or_none(source_trace)
    eligible_candidates = []
    for source_pos, reader_pos in candidates:
        if source_ids is not None and source_ids[source_pos] in excluded_ids:
            reason = "excluded id (padding or special token)"
            excluded.append(_segment("source", source_pos, source_pos + 1, reason, None))
        else:
            eligible_candidates.append((source_pos, reader_pos))
    pairs = _one_to_one(eligible_candidates, excluded)
    eligible = _eligible_positions(source_trace)
    coverage = None
    if eligible is not None:
        paired = {source_pos for source_pos, _ in pairs}
        reported = {
            position
            for segment in excluded
            if segment["side"] == "source" and segment["start"] is not None
            for position in range(segment["start"], segment["end"])
        }
        for start, end in _uncovered_runs(eligible, paired | reported):
            reason = "no verified span covers these source tokens"
            excluded.append(_segment("source", start, end, reason, None))
        coverage = len(paired.intersection(eligible)) / len(eligible) if eligible else None
    return TokenPairs(
        reader=names[0],
        source=names[1],
        pairs=tuple(pairs),
        excluded=tuple(excluded),
        coverage=coverage,
    )


def _source_ids_or_none(source: Trace) -> list[int] | None:
    """Return the source's bound token ids, or ``None`` when unbound or unretained."""

    binding = member_binding(source)
    return retained_ids(source, binding) if binding is not None else None


def register_token_pairs(table: dict[str, Callable[..., Any]]) -> None:
    """Install ``token_pairs`` on the Bundle dynamic-method table."""

    table["token_pairs"] = cast("Callable[..., Any]", _bundle_token_pairs)
