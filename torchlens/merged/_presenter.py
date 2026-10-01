"""``MergedTrace``: a presenter over lazy rank handles + the join table.

Composition, never inheritance (design-merge-ranks-c v5, 2.1): a merged trace
is NOT a ``Trace`` and NOT a ``Bundle``. Rank cores stay the single truth
(P1); the presenter only references them. Structural access is authority --
``merged.ranks[r][label]`` and ``merged.super_op(...)``; the ``r{rank}/label``
string is validated sugar. Refused surfaces raise typed
(:class:`MergedSurfaceUnsupportedError`); merged replay does not exist.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._engine import JoinKey, JoinRecord, MergeDerivation, derive_merge
from ._enums import BoundaryConsistency, MergeAlignment, MergedErrorCode, MergeValueStatus
from ._errors import (
    MergeConflictError,
    MergedFinding,
    MergedSurfaceUnsupportedError,
    MergeInputError,
)
from ._evidence import resolve_rank_inputs

__all__ = [
    "CollectiveJoin",
    "MergeReport",
    "MergedTrace",
    "merge_ranks",
    "merge_report",
]


def _rank_raw_to_final_op_labels(rank: int, trace: Any) -> Mapping[str, str]:
    """Read one rank core's DECLARED raw-to-final op-label seam, fail-closed.

    ``Trace._raw_to_final_op_labels`` is a declared, ``FieldPolicy.KEEP``
    field (``data_classes/_trace_components.py`` owns it under the ``graph``
    component), so every finished live or loaded rank core has it. Reaching
    for it through a string ``getattr(trace, "_raw_to_final_op_labels", {})``
    default made a renamed or absent field degrade SILENTLY into raw-label
    resolution -- fail-OPEN, contradicting merged/'s fail-closed ethos, and
    invisible to SLF001 (b5 R45-2 / SF-41). The read is now a direct private
    access: a rename breaks loudly here, and absence refuses typed.

    Parameters
    ----------
    rank:
        Global rank whose core is being read (diagnostic + payload field).
    trace:
        The rank core.

    Returns
    -------
    Mapping[str, str]
        The rank core's raw-to-final op-label mapping (possibly empty).

    Raises
    ------
    MergeInputError
        If the declared field is absent, or is not a mapping.
    """

    try:
        mapping = trace._raw_to_final_op_labels  # noqa: SLF001 -- declared rank-core seam
    except AttributeError as exc:
        raise MergeInputError(
            f"rank {rank}'s core does not carry the declared "
            "`_raw_to_final_op_labels` seam, so its boundary ops cannot be "
            "resolved. Merge inputs must be finished torch rank captures "
            "(live or loaded), not partial or foreign objects.",
            code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            rank=rank,
        ) from exc
    if not isinstance(mapping, Mapping):
        raise MergeInputError(
            f"rank {rank}'s `_raw_to_final_op_labels` seam is "
            f"{type(mapping).__name__}, not a mapping.",
            code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            rank=rank,
        )
    return mapping


class _RankHandle:
    """Lazy handle on one rank core: a live trace or an on-disk bundle."""

    def __init__(self, rank: int, trace: Any = None, path: str | None = None) -> None:
        self.rank = rank
        self.path = path
        self._trace = trace

    @property
    def trace(self) -> Any:
        """The rank's ``Trace``, loading from disk on first access."""

        if self._trace is None:
            from .._io.bundle import load as load_bundle

            assert self.path is not None, "a rank handle has a trace or a path"
            self._trace = load_bundle(self.path)
        return self._trace

    @property
    def is_loaded(self) -> bool:
        """Whether the trace is currently materialized in memory."""

        return self._trace is not None


class _RankMapping(Mapping[int, Any]):
    """Read-only ``rank -> Trace`` mapping with lazy loading."""

    def __init__(self, handles: Mapping[int, _RankHandle]) -> None:
        self._handles = dict(handles)

    def __getitem__(self, rank: int) -> Any:
        return self._handles[int(rank)].trace

    def __iter__(self) -> Iterator[int]:
        return iter(self._handles)

    def __len__(self) -> int:
        return len(self._handles)


@dataclass(frozen=True)
class CollectiveJoin:
    """Presenter view of one cross-rank join (references, never copies)."""

    record: JoinRecord

    @property
    def key(self) -> JoinKey:
        return self.record.key

    @property
    def kind(self) -> str:
        return self.record.kind

    @property
    def membership(self) -> tuple[int, ...]:
        return self.record.membership

    @property
    def presence(self) -> tuple[int, ...]:
        return self.record.presence

    @property
    def missing(self) -> tuple[int, ...]:
        return self.record.missing

    @property
    def consistency(self) -> BoundaryConsistency:
        return self.record.consistency

    def op_labels_raw(self, rank: int) -> tuple[str, ...]:
        """The rank core's raw op-label back-references for this join."""

        return self.record.per_rank[int(rank)].op_labels_raw

    def __repr__(self) -> str:
        """Semantic join line (F10; lovely bug 19 + ASCII rule).

        The membership digest is a correlation-key INTERNAL: the repr names
        the join semantically (kind, ordinal, channel, seq delta, presence,
        consistency) and leaves the digest to ``.record.key`` -- and the
        line is pure ASCII (bug 10: no unconditional unicode in returned
        strings).
        """

        _digest, ordinal, channel, delta = self.record.key
        return (
            f"CollectiveJoin({self.record.kind} group-ordinal {ordinal} "
            f"channel {channel} seq+{delta}, presence={list(self.record.presence)}, "
            f"consistency={self.record.consistency.value})"
        )


@dataclass(frozen=True)
class MergeReport:
    """Graph-free diagnostic summary of a merge derivation (3.2)."""

    ranks: tuple[int, ...]
    expected_ranks: tuple[int, ...] | None
    stored_alignment: MergeAlignment
    alignment: MergeAlignment
    value_status: MergeValueStatus
    n_joins: int
    findings: tuple[MergedFinding, ...]
    load_degradations: tuple[str, ...] = ()

    @property
    def gaps(self) -> tuple[MergedFinding, ...]:
        """The ``presence_gap`` findings: boundaries a rank never recorded."""

        return tuple(f for f in self.findings if f.kind == "presence_gap")

    @property
    def divergences(self) -> tuple[MergedFinding, ...]:
        """The ``value_divergence`` findings: ranks disagreeing on a witnessed value."""

        return tuple(f for f in self.findings if f.kind == "value_divergence")

    @property
    def conflicts(self) -> tuple[MergedFinding, ...]:
        """The structural conflict findings, which never join and never become gaps."""

        structural = {
            "group_lifetime_evidence_conflict",
            "relation_violation",
            "order_contradiction",
            "correlation_delta_mismatch",
        }
        return tuple(f for f in self.findings if f.kind in structural)

    def to_markdown(self) -> str:
        """Render a compact human-readable report."""

        lines = [
            "# Cross-rank merge report",
            f"- ranks: {list(self.ranks)}"
            + ("" if self.expected_ranks is None else f" (declared: {list(self.expected_ranks)})"),
            f"- alignment: {self.alignment.value}"
            + (
                f" (stored: {self.stored_alignment.value}; "
                f"{len(self.load_degradations)} load degradation(s))"
                if self.alignment is not self.stored_alignment
                else ""
            ),
            f"- value status: {self.value_status.value}",
            f"- joins: {self.n_joins}",
        ]
        for finding in self.findings:
            lines.append(f"- [{finding.kind}] {finding.detail}")
        return "\n".join(lines)


def _report_from(
    derivation: MergeDerivation, load_degradations: tuple[str, ...] = ()
) -> MergeReport:
    """Build the report projection of a derivation."""

    findings = derivation.findings
    if load_degradations:
        findings = findings + tuple(
            MergedFinding(kind="load_degradation", detail=detail) for detail in load_degradations
        )
    effective = derivation.stored_alignment
    if load_degradations and effective is MergeAlignment.ALIGNED:
        effective = MergeAlignment.PARTIAL
    effective_value_status = derivation.stored_value_status
    if load_degradations and effective_value_status is MergeValueStatus.ATTESTED_COMPLETE:
        # An unparseable member can never support a COMPLETE attestation claim
        # (R18-5 presenter half; same demote-only cap as the alignment above).
        effective_value_status = MergeValueStatus.ATTESTED_PARTIAL
    return MergeReport(
        ranks=derivation.ranks,
        expected_ranks=derivation.expected_ranks,
        stored_alignment=derivation.stored_alignment,
        alignment=effective,
        value_status=effective_value_status,
        n_joins=len(derivation.joins),
        findings=findings,
        load_degradations=load_degradations,
    )


class MergedTrace:
    """N rank-local captures stitched at their collective boundaries.

    Never construct directly: use :func:`merge_ranks` (live/loaded inputs)
    or :func:`torchlens.merged.load` (a saved ``merged-directory`` artifact).

    Lifetime contract (R37): the presenter holds its rank traces STRONGLY for
    its own lifetime -- ``merge_ranks([live_a, live_b])`` transitively pins
    both rank traces (and their retained activations) until the presenter is
    dropped or :meth:`release` is called. Call :meth:`release` when finished
    with a presenter built over live traces you do not otherwise retain.
    """

    def __init__(
        self,
        derivation: MergeDerivation,
        handles: Mapping[int, _RankHandle],
        load_degradations: tuple[str, ...] = (),
    ) -> None:
        self._derivation = derivation
        self._handles = dict(handles)
        self._load_degradations = tuple(load_degradations)
        self._source_path: str | None = None
        self._released = False
        self._member_outcomes = self._settle_member_outcomes(self._handles)

    @staticmethod
    def _settle_member_outcomes(handles: Mapping[int, _RankHandle]) -> dict[int, str]:
        """Read each loaded member's settled capture-outcome status once.

        Computed at construction (statuses are settled and frozen) so the
        disclosure stays readable after :meth:`release` and never forces a
        lazy member load. Members whose object carries no settled outcome
        sidecar (hand-built evidence carriers) make no claim and are absent.
        """

        from ..capture.outcome import outcome_for

        statuses: dict[int, str] = {}
        for rank, handle in sorted(handles.items()):
            if not handle.is_loaded:
                continue
            outcome = outcome_for(handle.trace)
            if outcome is not None:
                statuses[rank] = outcome.status.value
        return statuses

    def release(self) -> None:
        """Drop every rank-trace handle so the member traces can be reclaimed.

        The counterpart of ``Trace.cleanup()`` for the presenter (R37 /
        b2:B20, contract section 8): the derivation record, verdict
        properties, findings, and ``load_degradations`` stay readable, but
        every surface that resolves a member trace (``ranks``,
        ``__getitem__``, ``super_op()``, ``join_ops()``, ``save()``) refuses
        typed with ``fields["code"] = "merged_member_released"`` afterwards
        -- never a bare ``KeyError``, and never a zero-member presence lie.
        Idempotent.
        """

        self._handles.clear()
        self._released = True

    def _require_members(self, surface: str) -> None:
        """Refuse typed when ``surface`` needs member traces after release()."""

        if self._released:
            raise MergedSurfaceUnsupportedError(
                f"MergedTrace.{surface} needs the member rank traces, but "
                "release() dropped them. Re-merge (or re-load) the rank cores "
                "to access members again; the derivation verdicts remain "
                "readable on this presenter.",
                code=MergedErrorCode.MERGED_MEMBER_RELEASED,
            )

    # ------------------------------------------------------------------
    # Verdicts (3.2): stored vs effective are DISTINCT properties.
    # ------------------------------------------------------------------

    @property
    def stored_alignment(self) -> MergeAlignment:
        """The structural verdict derived at merge time (frozen in the descriptor)."""

        return self._derivation.stored_alignment

    @property
    def alignment(self) -> MergeAlignment:
        """The EFFECTIVE verdict: stored, lowered by load degradations."""

        if self._load_degradations and self.stored_alignment is MergeAlignment.ALIGNED:
            return MergeAlignment.PARTIAL
        return self.stored_alignment

    @property
    def stored_value_status(self) -> MergeValueStatus:
        """The value verdict derived at merge time (frozen in the descriptor)."""

        return self._derivation.stored_value_status

    @property
    def value_status(self) -> MergeValueStatus:
        """Merge-level value verdict from the join ledger (3.3).

        EFFECTIVE like ``alignment``: a rank core that no longer parses on
        this runtime cannot support a COMPLETE attestation claim, so load
        degradations cap ``attested_complete`` at ``attested_partial``
        (R18-5 presenter half; demote-only, the stored value stays visible
        through ``stored_value_status``).
        """

        stored = self._derivation.stored_value_status
        if self._load_degradations and stored is MergeValueStatus.ATTESTED_COMPLETE:
            return MergeValueStatus.ATTESTED_PARTIAL
        return stored

    @property
    def load_degradations(self) -> tuple[str, ...]:
        """Environment degradations recorded at load (empty on a live merge)."""

        return self._load_degradations

    @property
    def member_outcomes(self) -> Mapping[int, str]:
        """Read-only ``rank -> settled capture-outcome status`` disclosure.

        R06c: FAILED / ABORTED_NONFINITE / UNKNOWN member cores refuse at
        merge input resolution; HALTED (and legacy UNATTESTED) members merge
        but are disclosed here and in :meth:`summary` -- the merged verdicts
        then describe only what those members captured, and the presenter
        never claims ``attested_complete`` silently over a halted prefix.
        Ranks whose member object carries no settled outcome sidecar are
        absent. Settled at construction; stays readable after
        :meth:`release`.
        """

        return dict(self._member_outcomes)

    def _member_outcome_disclosure(self) -> str | None:
        """The non-complete member-outcome line, or ``None`` when all settle complete."""

        flagged = {
            rank: status
            for rank, status in sorted(self._member_outcomes.items())
            if status != "complete"
        }
        if not flagged:
            return None
        return (
            f"member capture outcomes: {flagged} -- non-complete member "
            "core(s); merged verdicts cover only what those members captured"
        )

    # ------------------------------------------------------------------
    # Structure.
    # ------------------------------------------------------------------

    @property
    def ranks(self) -> Mapping[int, Any]:
        """Lazy ``rank -> Trace`` mapping (structural access authority)."""

        self._require_members("ranks")
        return _RankMapping(self._handles)

    @property
    def rank_ids(self) -> tuple[int, ...]:
        """Global ranks present in the merge."""

        return self._derivation.ranks

    @property
    def expected_ranks(self) -> tuple[int, ...] | None:
        """Declared world, when one was given (can only widen expectations)."""

        return self._derivation.expected_ranks

    @property
    def joins(self) -> tuple[CollectiveJoin, ...]:
        """The cross-rank collective joins, in deterministic key order."""

        return tuple(CollectiveJoin(record) for record in self._derivation.joins)

    @property
    def findings(self) -> tuple[MergedFinding, ...]:
        """All merge findings (gaps, divergences, load degradations)."""

        return _report_from(self._derivation, self._load_degradations).findings

    @property
    def gaps(self) -> tuple[MergedFinding, ...]:
        """The typed presence-gap ledger."""

        return self._derivation.gap_findings

    @property
    def report(self) -> MergeReport:
        """The structured merge report."""

        return _report_from(self._derivation, self._load_degradations)

    # ------------------------------------------------------------------
    # Access surface (2.5).
    # ------------------------------------------------------------------

    def __getitem__(self, item: Any) -> Any:
        self._require_members("__getitem__")
        if not isinstance(item, str):
            raise MergedSurfaceUnsupportedError(
                "Merged-level selectors are not supported in this release; "
                "index a single rank core (merged.ranks[r][selector]) instead.",
                code=MergedErrorCode.MERGED_SELECTOR_UNSUPPORTED,
            )
        rank, label = self._split_sugar(item)
        if rank is not None:
            return self.ranks[rank][label]
        hits = {}
        for rank_id in self.rank_ids:
            try:
                hits[rank_id] = self.ranks[rank_id][item]
            except (KeyError, ValueError):
                # Narrowed like super_op (b5 R45-2): a lookup miss is a miss,
                # but any OTHER failure is a defect in the rank core and must
                # surface -- a broad catch read a corrupt core as a miss and
                # presented another rank's hit as an unambiguous single-rank
                # result.
                continue
        if not hits:
            raise KeyError(
                f"{item!r} resolves on none of ranks {list(self.rank_ids)}; "
                f"use a rank-qualified spelling (r{{rank}}/label, e.g. "
                f"r{self.rank_ids[0] if self.rank_ids else 0}/{item}) or query "
                "one rank core directly (merged.ranks[r][label])."
            )
        if len(hits) > 1:
            from .._errors import AmbiguousOpLookupError

            spellings = ", ".join(f"r{rank_id}/{item}" for rank_id in sorted(hits))
            raise AmbiguousOpLookupError(
                f"Label {item!r} resolves on {len(hits)} ranks; disambiguate "
                f"with a rank-qualified label ({spellings}) or use "
                "merged.super_op() for the cross-rank fan."
            )
        return next(iter(hits.values()))

    def _split_sugar(self, item: str) -> tuple[int | None, str]:
        """Parse the validated ``r{rank}/label`` sugar; real names win."""

        if "/" in item and item.startswith("r"):
            prefix, _, rest = item.partition("/")
            if prefix[1:].isdigit() and rest:
                rank = int(prefix[1:])
                if rank in self._handles:
                    return rank, rest
        return None, item

    def super_op(self, label: str) -> dict[int, Any]:
        """Return the SPMD fan: the ops matching ``label`` on every rank.

        Parameters
        ----------
        label:
            A rank-core op/layer label (unqualified).

        Returns
        -------
        dict[int, Any]
            Mapping from global rank to that rank's matching record; ranks
            where the label does not resolve are absent.
        """

        self._require_members("super_op()")
        fan: dict[int, Any] = {}
        for rank_id in self.rank_ids:
            try:
                fan[rank_id] = self.ranks[rank_id][label]
            except (KeyError, ValueError):
                # Documented SPMD contract: a rank where the label does not
                # resolve (miss) or resolves ambiguously is simply absent from
                # the fan. Narrow on purpose (b5 R45-2) -- any OTHER failure is
                # a defect in the rank core and must surface, not shrink the fan.
                continue
        if not fan:
            raise KeyError(
                f"{label!r} resolves on none of ranks {list(self.rank_ids)}; "
                "super_op takes an unqualified rank-core op/layer label."
            )
        return fan

    def join_ops(self, join: CollectiveJoin) -> dict[int, tuple[Any, ...]]:
        """Resolve a join's boundary ops on every presenting rank core.

        Uses each rank core's DECLARED raw-to-final label seam; a tensorless
        boundary (barrier, object collectives) records no op-label
        back-references and yields an empty tuple for that rank.

        Fail-closed (b5 R45-2): a rank core without the declared seam, and a
        recorded back-reference that the core cannot resolve to a node, both
        refuse typed. Silently dropping an unresolvable boundary op would
        present a SHORTER fan than the evidence recorded -- a presence claim
        the merge never made.

        Raises
        ------
        MergeInputError
            If a presenting rank core lacks the declared label seam, or does
            not resolve one of the boundary op labels its own journal
            recorded.
        """

        self._require_members("join_ops()")
        resolved: dict[int, tuple[Any, ...]] = {}
        for rank in join.presence:
            trace = self.ranks[rank]
            mapping = _rank_raw_to_final_op_labels(rank, trace)
            ops = []
            for raw in join.op_labels_raw(rank):
                final = mapping.get(raw, raw)
                try:
                    ops.append(trace[final])
                except (KeyError, ValueError) as exc:
                    raise MergeInputError(
                        f"rank {rank}'s core does not resolve boundary op "
                        f"{final!r} (raw label {raw!r}) recorded for join "
                        f"{join.key}. The rank core and its collective "
                        "journal disagree; re-capture the rank rather than "
                        "presenting a partial fan.",
                        code=MergedErrorCode.MERGED_SCHEMA_INVALID,
                        rank=rank,
                        raw_label=raw,
                        final_label=final,
                    ) from exc
            resolved[rank] = tuple(ops)
        return resolved

    def happens_before(self, left: JoinKey, right: JoinKey) -> bool | None:
        """Partial-order query over join keys (2.2).

        Returns ``True``/``False`` for provable order, ``None`` for
        genuinely concurrent joins (no path either way). Order flows only
        through rank-local issue order and the joins themselves; there is no
        global step numbering.
        """

        joins = {join.key: join for join in self.joins}
        if left not in joins or right not in joins:
            missing = [key for key in (left, right) if key not in joins]
            raise KeyError(
                f"happens_before takes keys of existing joins; {missing!r} "
                f"not among the {len(joins)} join key(s) (see merged.joins)."
            )
        successors: dict[JoinKey, set[JoinKey]] = {key: set() for key in joins}
        per_rank_sequence: dict[int, list[tuple[int, JoinKey]]] = {}
        for key, join in joins.items():
            for rank in join.presence:
                index = join.record.per_rank[rank].boundary_index
                per_rank_sequence.setdefault(rank, []).append((index, key))
        for sequence in per_rank_sequence.values():
            sequence.sort()
            for (_, earlier), (_, later) in zip(sequence, sequence[1:]):
                successors[earlier].add(later)

        def reachable(source: JoinKey, target: JoinKey) -> bool:
            """Whether ``target`` is reachable from ``source`` in the merged successor graph."""

            frontier, seen = [source], {source}
            while frontier:
                node = frontier.pop()
                for successor in successors[node]:
                    if successor == target:
                        return True
                    if successor not in seen:
                        seen.add(successor)
                        frontier.append(successor)
            return False

        if left == right:
            return False
        if reachable(left, right):
            return True
        if reachable(right, left):
            return False
        return None

    def to_pandas(self) -> Any:
        """Join table as a DataFrame with a ``(rank, rank_local_index)`` MultiIndex."""

        import pandas as pd

        rows = []
        index = []
        for join in self.joins:
            digest, ordinal, channel, delta = join.key
            for rank in join.presence:
                ref = join.record.per_rank[rank]
                index.append((rank, ref.boundary_index))
                rows.append(
                    {
                        "kind": join.kind,
                        "membership_digest": digest,
                        "lifetime_ordinal": ordinal,
                        "channel": channel,
                        "seq_delta": delta,
                        "seq_abs": ref.seq_abs,
                        "group_size": len(join.membership),
                        "presence": len(join.presence),
                        "consistency": join.consistency.value,
                    }
                )
        frame = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(index, names=["rank", "rank_local_index"]),
        )
        return frame.sort_index()

    # ------------------------------------------------------------------
    # Presentation.
    # ------------------------------------------------------------------

    def _witness_coverage_line(self) -> str:
        """The witness-coverage disclosure every ``aligned`` presentation carries."""

        joins = self._derivation.joins
        attested = sum(1 for j in joins if j.consistency is BoundaryConsistency.ATTESTED)
        applicable = sum(
            1 for j in joins if j.consistency is not BoundaryConsistency.NOT_APPLICABLE
        )
        return (
            f"witness coverage: {attested}/{applicable} applicable join(s) "
            f"attested, {len(joins) - applicable} not applicable "
            f"(value status: {self.value_status.value})"
        )

    def summary(self) -> str:
        """Human-readable merge summary."""

        alignment = self.alignment.value
        if self.alignment is not self.stored_alignment:
            alignment = (
                f"{alignment} (stored: {self.stored_alignment.value}; "
                f"{len(self._load_degradations)} load degradation(s))"
            )
        lines = [
            f"MergedTrace over ranks {list(self.rank_ids)}: alignment={alignment}",
        ]
        # The halted/unattested-member disclosure precedes the witness line so
        # an attested_complete value status is never presented without it.
        disclosure = self._member_outcome_disclosure()
        if disclosure is not None:
            lines.append(disclosure)
        lines.extend(
            [
                self._witness_coverage_line(),
                f"{len(self._derivation.joins)} collective join(s), "
                f"{len(self.gaps)} presence gap(s), "
                f"{len(self._derivation.divergence_findings)} value divergence(s)",
            ]
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MergedTrace(ranks={list(self.rank_ids)}, "
            f"alignment={self.alignment.value}, "
            f"value_status={self.value_status.value}, "
            f"joins={len(self._derivation.joins)})"
        )

    # ------------------------------------------------------------------
    # Refused surfaces (2.5 / 3.4): merged replay does not exist.
    # ------------------------------------------------------------------

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: merged replay does not exist (typed)."""

        raise MergedSurfaceUnsupportedError(
            "Merged replay does not exist: re-issuing collectives outside "
            "their communicator hangs or fabricates peer-dependent values. "
            "Run a single rank core instead (merged.ranks[r]).",
            code=MergedErrorCode.MERGE_RUN_UNSUPPORTED,
        )

    def validate(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: per-rank validation belongs to the rank cores (typed)."""

        raise MergedSurfaceUnsupportedError(
            "Merged validation does not exist; validate each rank core "
            "(noting that collective-crossing cores refuse forward replay "
            "typed while metadata invariants run in full).",
            code=MergedErrorCode.MERGED_SURFACE_UNSUPPORTED,
        )

    def _refuse_surface(self, surface: str) -> Any:
        """Raise the typed refusal for a contract-promised unsupported surface.

        The contract declares merged runnable export, receptive/projective
        fields, and intervention chaining "refused typed", but these surfaces
        previously raised bare AttributeError (R18-9 presenter half).
        """

        raise MergedSurfaceUnsupportedError(
            f"MergedTrace does not support {surface}; use a single rank core "
            "(merged.ranks[r]) instead.",
            code=MergedErrorCode.MERGED_SURFACE_UNSUPPORTED,
        )

    def fork(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: merged forking/intervention chaining does not exist (typed)."""

        self._refuse_surface("fork()")

    def intervene(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: merged intervention chaining does not exist (typed)."""

        self._refuse_surface("intervene()")

    def log_backward(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: backward/gradient merging is deferred (fork F1, typed)."""

        self._refuse_surface("log_backward()")

    def receptive_fields(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: merged influence geometry does not exist (typed)."""

        self._refuse_surface("receptive_fields()")

    def projective_fields(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: merged influence geometry does not exist (typed)."""

        self._refuse_surface("projective_fields()")

    # Artifact save lives in _artifact.py; bound late to avoid an import cycle.
    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the merged artifact as a ``merged-directory`` bundle."""

        self._require_members("save()")
        from ._artifact import save_merged

        save_merged(self, path, overwrite=overwrite)
        self._source_path = str(path)


def merge_ranks(
    inputs: Sequence[Any],
    *,
    expected_ranks: Iterable[int] | None = None,
) -> MergedTrace:
    """Merge N rank-local captures at their collective boundaries (rung C1).

    Parameters
    ----------
    inputs:
        Live/loaded ``Trace`` objects and/or rank-core ``.tlspec`` paths, one
        per rank, in any order. Every input must have been captured under the
        distributed opt-in.
    expected_ranks:
        Optional declared world. Widens presence expectations only: a
        declared rank without a core is a presence gap; declaring fewer
        ranks than the recorded group memberships never narrows anything.

    Returns
    -------
    MergedTrace
        The merged presenter over the rank cores and the join table.

    Raises
    ------
    MergeInputError
        On invalid inputs or out-of-C1-scope boundaries (p2p/pipeline: C3;
        DTensor topologies: C2).
    MergeConflictError
        When the presented cores structurally contradict each other
        (pre-join lineage-audit conflicts, relation violations, order
        contradictions, correlation cross-check disagreements). The
        graph-free diagnostic escape hatch is :func:`merge_report`.
    """

    resolved = resolve_rank_inputs(inputs)
    derivation = derive_merge(
        {rank: evidence for rank, (evidence, _trace) in resolved.items()},
        expected_ranks,
    )
    structural = derivation.structural_findings
    if structural:
        raise MergeConflictError(
            f"{len(structural)} structural conflict(s) between the presented "
            "rank cores; see fields['findings']. Diagnose without merging via "
            "torchlens.merged.merge_report().",
            code=MergedErrorCode.MERGE_CONFLICT
            if not all(f.kind == "group_lifetime_evidence_conflict" for f in structural)
            else MergedErrorCode.GROUP_LIFETIME_EVIDENCE_CONFLICT,
            findings=structural,
        )
    handles = {}
    for rank, (evidence, trace) in resolved.items():
        path = evidence.source if not evidence.source.startswith("live[") else None
        handles[rank] = _RankHandle(rank, trace=trace, path=path)
    return MergedTrace(derivation, handles)


def merge_report(
    inputs: Sequence[Any],
    *,
    expected_ranks: Iterable[int] | None = None,
) -> MergeReport:
    """Graph-free merge diagnostic: derive verdicts without constructing.

    Unlike :func:`merge_ranks` this never raises on structural conflicts --
    it is the escape hatch for diagnosing exactly those. Input validation
    (unloadable path, malformed core, out-of-scope boundary) still refuses
    typed.
    """

    resolved = resolve_rank_inputs(inputs)
    derivation = derive_merge(
        {rank: evidence for rank, (evidence, _trace) in resolved.items()},
        expected_ranks,
    )
    return _report_from(derivation)
