"""Comparative selection producers: differential and cross-pass (L6 producer wave).

Selects by HOW VALUES DIFFER, across runs or across passes. Two producer
families, both returning :class:`~torchlens.selection.Selection` queries that
compose with the full ``| & - ~`` algebra and resolve explicitly against one
trace:

- DIFFERENTIAL producers compare the resolution trace (the SUBJECT — the
  trace passed to ``resolve()``, where the masks land) against ONE explicit
  REFERENCE trace: ``changed`` (elementwise bounds on the delta — "units the
  intervention actually moved", "what changed when I swapped the input") and
  ``top_changed`` (global deterministic ranking of the delta — "units that
  changed most"). The delta is DIRECTIONAL: ``subject - reference``,
  elementwise in float64. Comparison is deliberately pairwise against one
  reference (multi-sample dispersion is the different spelling
  ``low_variance(samples=...)``).

- CROSS-PASS producers read ONE capture's multi-pass layers, pass-qualified
  throughout: ``stable_across_passes`` (per-element range across the pass
  window within ``tol``) and ``pass_variance`` (bounds on the per-element
  variance across the window). Evidence windows are explicit (``passes=``);
  a layer contributing fewer than TWO passes refuses typed — a cross-pass
  claim about a single-pass layer is vacuous, and vacuous truths are
  refusals here, never silent empty masks.

STRUCTURE MATCHING (differential): the population comes from the SUBJECT;
the reference must hold a retained, shape-identical activation at the same
pass-qualified ``(layer_label, pass_index)`` address — missing sites, unsaved
payloads, and shape drift refuse typed with the reference named, exactly the
multi-sample evidence contract. Structures are NEVER silently intersected.
When both sides carry L1 structural site keys, a key disagreement also
refuses (label coincidence across different architectures is caught, not
compared); either side keyless proceeds on address+shape.

PROVENANCE HONESTY (pinned by tests): every producer here names a statistic
of complete retained evidence — the pairwise delta between THESE two
captures, or the dispersion across THIS capture's passes — so entries declare
``relation="exact"`` (the ``low_variance`` precedent). None of these names
makes an open-world dispositional claim ("input-sensitive" would be one; it
is deliberately not spelled here). Population restriction composes through
the normative JOIN table, never by overwriting a relation.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification (provisional-name protocol), with the interface
flagged for the UI-sprint review. Producers are ACT-kind only: PARAM
populations refuse ``selection_kind_incompatible`` — ``Param`` records hold
only a LIVE parameter reference, never capture-time payloads, so a
"weights that shifted between checkpoints" claim cannot be made honestly
from Trace records (compare runnable-save ``state_dict_v1`` blobs instead;
a capture-time weight differential is a named possibility, not a promise).

Elements whose compared quantity is NaN never satisfy any criterion; rank
producers exclude them from the candidate population.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .errors._base import TorchLensWarning
from .selection import (
    ResolvedSelection,
    Selection,
    SelectionProvenance,
    SiteEntry,
    _join_relation,
    _mask_from_dense,
    _unresolvable,
    register_term_resolver,
)
from .selection_values import (
    _entry_with_mask,
    _lift_within,
    _op_for_entry,
    _read_saved_value,
    _resolve_population,
    _validate_real_number,
)

__all__ = [
    "changed",
    "pass_variance",
    "stable_across_passes",
    "top_changed",
]


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _reference_display(reference: Any) -> str:
    """Return a compact reference identity for reprs/sources (never the trace)."""

    label = getattr(reference, "trace_label", None)
    return str(label) if label else "<reference trace>"


def _validate_reference(reference: Any, producer: str) -> Any:
    """Validate the reference operand: exactly ONE Trace, refused with teaching."""

    from .data_classes.trace import Trace

    if isinstance(reference, Trace):
        return reference
    if isinstance(reference, (list, tuple, set)) or type(reference).__name__ == "Bundle":
        raise ValueError(
            f"{producer} compares against ONE reference trace (pairwise delta), "
            "not an evidence set. For dispersion across many samples use "
            "low_variance(samples=...)."
        )
    raise ValueError(f"{producer} `reference` must be a Trace (got {type(reference).__name__}).")


def _validate_by(by: str, producer: str) -> str:
    """Validate the delta-comparison mode (closed vocabulary)."""

    if by not in ("abs", "signed"):
        raise ValueError(f"{producer} `by` must be 'abs' or 'signed'; got {by!r}.")
    return by


def _validate_passes(passes: Any, producer: str) -> tuple[int, ...] | None:
    """Validate an explicit pass window: >= 2 distinct 1-based pass indices."""

    if passes is None:
        return None
    collected: list[int] = []
    for index in passes:
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError(
                f"{producer} `passes` must contain 1-based pass indices (ints >= 1); got {index!r}."
            )
        collected.append(index)
    window = tuple(sorted(set(collected)))
    if len(window) < 2:
        raise ValueError(
            f"{producer} makes a CROSS-PASS claim and needs a window of at least 2 "
            f"passes (got {len(window)}). For single-pass value claims use the "
            "value producers (threshold / sign / top_k)."
        )
    return window


#: Machinery layer types the intervention engine itself inserts. Excluded
#: from delta populations by disclosure (leverage D-7: painting TorchLens's
#: own replacement op as a changed model site misleads), never refused.
_MACHINERY_LAYER_TYPES = frozenset({"interventionreplacement"})


class _JoinGuard:
    """Guarded cross-trace pairing authority for the differential producers.

    Delegation to the shipped site join (leverage B4): when BOTH captures
    carry L1 structural site keys, subject entries pair to reference ops by
    SITE KEY under :func:`~torchlens.postprocess._site_join.join_site_profiles`
    verdicts — labels leave the join key entirely, so a live edit's label
    renumbering can no longer break comparison of structurally intact sites.
    ``corroborated`` and ``positional`` verdicts admit comparison (D-2);
    refused verdicts and one-sided keys refuse typed with the verdict
    disclosed, never guessed. Either side keyless degrades to the historical
    address+shape pairing (with its site-key-disagreement belt intact).
    """

    __slots__ = ("rows", "reference_by_coord", "subject_coords")

    def __init__(
        self,
        rows: dict[str, Any],
        reference_by_coord: dict[tuple[str, str, int], str],
        subject_coords: dict[str, tuple[str, str, int]],
    ) -> None:
        self.rows = rows
        self.reference_by_coord = reference_by_coord
        self.subject_coords = subject_coords

    @classmethod
    def build(cls, subject: Any, reference: Any) -> _JoinGuard | None:
        """Build the guard, or ``None`` when either side is keyless."""

        from .errors._base import TorchLensError
        from .postprocess._site_join import (
            join_site_profiles,
            occurrence_coordinates,
            site_profile,
        )

        try:
            subject_profile = site_profile(subject)
            reference_profile = site_profile(reference)
        except TorchLensError:
            # ``site_key_unavailable`` (legacy keyless artifact): the memo's
            # keyless rule — proceed on address+shape, never refuse the whole
            # comparison for a missing optional identity layer.
            return None
        rows = join_site_profiles(subject_profile, reference_profile)
        reference_by_coord = {
            coord: label
            for label, coord in occurrence_coordinates(reference, reference_profile.keys).items()
        }
        return cls(
            rows=dict(rows),
            reference_by_coord=reference_by_coord,
            subject_coords=occurrence_coordinates(subject, subject_profile.keys),
        )

    def reference_op(self, node: _CompareTerm, trace: Any, entry: SiteEntry) -> Any:
        """Resolve one subject entry's reference op through the join verdict."""

        key = entry.structural_site_key
        row = self.rows.get(key)
        if row is None:
            raise _unresolvable(
                "site_join_refused",
                f"site {entry.site_key!r} (structural key {key!r}) is present "
                "on the SUBJECT capture only: under the guarded site join it "
                "is a DECLARED addition relative to the reference, and deltas "
                "for one-sided sites are never guessed. Restrict `within=` to "
                "shared structure, or read the addition from the differential "
                "report.",
                code="selection_unresolvable",
                site=repr(entry.site_key),
                join_verdict="one_sided_subject",
            )
        if not row.joined:
            raise _unresolvable(
                "site_join_refused",
                f"the guarded site join REFUSED to pair site {entry.site_key!r} "
                f"(structural key {key!r}) across these two captures "
                f"(verdict: {row.verdict.value}): its structural cohort does "
                "not correspond one-to-one between the runs (an insertion, "
                "removal, or call-site change touched it), so any pairing "
                "would silently compare different graph positions. Compare "
                "the unaffected cohorts, or re-capture without the structural "
                "change.",
                code="selection_unresolvable",
                site=repr(entry.site_key),
                join_verdict=row.verdict.value,
            )
        subject_op = _op_for_entry(trace, entry)
        coord = self.subject_coords.get(subject_op.label)
        reference_label = None if coord is None else self.reference_by_coord.get(coord)
        if reference_label is None:
            raise _unresolvable(
                "site_join_refused",
                f"the joined structural key {key!r} has no occurrence at the "
                f"subject's call-instance coordinate {coord!r} on the "
                "reference; occurrence pairing is exact under the cardinality "
                "guard and never guessed.",
                code="selection_unresolvable",
                site=repr(entry.site_key),
                join_verdict="occurrence_absent_on_reference",
            )
        return node.reference.ops[reference_label]


def _reference_op_for_entry(node: _CompareTerm, trace: Any, entry: SiteEntry, guard: Any) -> Any:
    """Resolve the reference-side op: join-guarded where keyed, address belt otherwise."""

    if guard is not None and entry.structural_site_key is not None:
        return guard.reference_op(node, trace, entry)
    reference_op = _op_for_entry(node.reference, entry, sample_name="reference")
    subject_key = entry.structural_site_key
    reference_key = getattr(reference_op, "site_key", None)
    if subject_key is not None and reference_key is not None and subject_key != reference_key:
        raise _unresolvable(
            "site_not_in_trace",
            f"the reference trace's op at address {entry.site_key!r} is a "
            f"structurally DIFFERENT site (structural site key {reference_key!r} "
            f"vs subject {subject_key!r}): the traces do not share this "
            "position, and a delta between structurally different sites would "
            "be a false comparison. Restrict `within=` to genuinely shared "
            "structure.",
            site=repr(entry.site_key),
            sample="reference",
        )
    return reference_op


def _delta_for_entry(
    node: _CompareTerm, trace: Any, entry: SiteEntry, guard: Any = None
) -> torch.Tensor:
    """Compute one entry's subject-minus-reference delta with typed refusals.

    Reference addressing is JOIN-GUARDED (leverage B4): when both captures
    carry structural site keys the pairing authority is the shipped guarded
    join's verdict (labels never enter the key); keyless captures keep the
    historical pass-qualified address pairing with its site-key belt. Reads
    both sides through the shared evidence reader (missing site / unsaved
    payload / shape drift refuse typed with the reference named).
    """

    reference_op = _reference_op_for_entry(node, trace, entry, guard)
    subject_value = _read_saved_value(trace, entry)
    reference_value = _read_reference_value(node, entry, reference_op)
    if subject_value.is_complex() or reference_value.is_complex():
        return subject_value.to(torch.complex128) - reference_value.to(torch.complex128)
    # float64 exactly represents every float32/16/bfloat16/int32 value; the
    # int64 tail beyond 2**53 is a documented precision residual.
    return subject_value.to(torch.float64) - reference_value.to(torch.float64)


def _read_reference_value(node: _CompareTerm, entry: SiteEntry, reference_op: Any) -> torch.Tensor:
    """Read the reference op's retained activation with the shared typed refusals."""

    from .errors._base import TorchLensError

    if not getattr(reference_op, "has_saved_activation", False):
        raise _unresolvable(
            "value_not_saved",
            f"value criteria need the saved activation at {reference_op.label!r}, "
            "which reference did not retain. Re-capture with a `save=` "
            "predicate covering this site.",
            code="selection_unresolvable",
            site=reference_op.label,
            sample="reference",
        )
    try:
        value = reference_op.out
    except TorchLensError as exc:
        raise _unresolvable(
            "value_not_saved",
            f"value criteria need the saved activation at {reference_op.label!r}, "
            f"which reference cannot serve: {exc}",
            code="selection_unresolvable",
            site=reference_op.label,
            sample="reference",
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise _unresolvable(
            "non_tensor_site",
            f"site {reference_op.label!r} has a non-tensor output; value "
            "criteria address single-tensor outputs only.",
            code="selection_unresolvable",
            site=reference_op.label,
        )
    if tuple(value.shape) != entry.shape:
        raise _unresolvable(
            "mask_shape_mismatch",
            f"saved activation at {reference_op.label!r} on reference has shape "
            f"{tuple(value.shape)!r}, which does not match the population's "
            f"index space {entry.shape!r}.",
            code="selection_unresolvable",
            site=reference_op.label,
            sample="reference",
        )
    return value.detach().cpu()


def _machinery_entry(entry: SiteEntry, source: str) -> SiteEntry:
    """Return one machinery site as a touched-but-unselected (zero-mask) entry."""

    dense = torch.zeros(entry.shape, dtype=torch.bool)
    return _entry_with_mask(entry, dense, "exact", source + " [machinery_excluded]")


def _is_machinery_entry(trace: Any, entry: SiteEntry) -> bool:
    """Whether one subject entry is an engine-inserted machinery op (D-7)."""

    op = _op_for_entry(trace, entry)
    return getattr(op, "layer_type", None) in _MACHINERY_LAYER_TYPES


def _compared_delta(delta: torch.Tensor, by: str, producer: str, site: Any) -> torch.Tensor:
    """Return the compared quantity (|delta| or signed delta; complex refuses ordered)."""

    if by == "abs":
        return delta.abs()
    if delta.is_complex():
        raise _unresolvable(
            "value_criterion_invalid",
            f"{producer} with by='signed' orders raw deltas, and complex deltas "
            f"have no total order (site {site!r}). Use by='abs' (delta magnitude).",
            site=site,
        )
    return delta


# ---------------------------------------------------------------------------
# Differential producers (subject vs one explicit reference trace).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompareTerm:
    """AST leaf for the differential producers (changed / top_changed)."""

    criterion: str
    reference: Any
    within: Any
    by: str = "abs"
    above: float | None = None
    below: float | None = None
    k: int | None = None
    fraction: float | None = None
    largest: bool = True

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure (never the trace)."""

        population = "saved_sites" if self.within is None else repr(self.within)
        vs = _reference_display(self.reference)
        if self.criterion == "changed":
            head = f"changed(vs={vs!r}, above={self.above}, below={self.below}, by={self.by!r}"
        elif self.k is not None:
            head = f"top_changed(vs={vs!r}, k={self.k}, by={self.by!r}, largest={self.largest}"
        else:
            head = (
                f"top_changed(vs={vs!r}, fraction={self.fraction}, by={self.by!r}, "
                f"largest={self.largest}"
            )
        return f"{head}, within={population})"


def _refuse_self_comparison(node: _CompareTerm, trace: Any) -> None:
    """Refuse resolving a differential producer against its own reference."""

    if node.reference is trace:
        raise _unresolvable(
            "value_criterion_invalid",
            f"{node.criterion} resolved against its own reference trace: the "
            "delta of a capture with itself is identically zero, so the "
            "criterion is vacuous by construction. Resolve on the SUBJECT run "
            "and pass the OTHER run as the reference (a fork is a different "
            "trace object).",
        )


def _resolve_changed(node: _CompareTerm, trace: Any) -> ResolvedSelection:
    """Resolve the elementwise delta-bound criterion exactly."""

    _refuse_self_comparison(node, trace)
    population = _resolve_population(node.within, trace)
    guard = _JoinGuard.build(trace, node.reference)
    source = (
        f"changed(above={node.above}, below={node.below}, by={node.by!r}, "
        f"vs={_reference_display(node.reference)!r})"
    )
    entries: list[SiteEntry] = []
    for entry in population:
        if _is_machinery_entry(trace, entry):
            entries.append(_machinery_entry(entry, source))
            continue
        delta = _delta_for_entry(node, trace, entry, guard)
        compared = _compared_delta(delta, node.by, "changed", entry.site_key)
        dense = torch.ones(entry.shape, dtype=torch.bool)
        if node.above is not None:
            dense &= compared > node.above
        if node.below is not None:
            dense &= compared < node.below
        dense &= entry._mask._dense_ro()
        entries.append(_entry_with_mask(entry, dense, "exact", source))
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_top_changed(node: _CompareTerm, trace: Any) -> ResolvedSelection:
    """Resolve the global delta-rank criterion exactly.

    Ranking is GLOBAL across the population with a deterministic tie-break:
    stable sort over the concatenation of sites in canonical order, so ties
    resolve by (site order, flat index). NaN deltas never enter the candidate
    population.
    """

    _refuse_self_comparison(node, trace)
    population = _resolve_population(node.within, trace)
    guard = _JoinGuard.build(trace, node.reference)
    per_entry: list[tuple[SiteEntry, torch.Tensor, torch.Tensor]] = []
    for entry in population:
        if _is_machinery_entry(trace, entry):
            span = math.prod(entry.shape)
            zeros = torch.zeros(span, dtype=torch.float64)
            per_entry.append((entry, zeros, torch.zeros(span, dtype=torch.bool)))
            continue
        delta = _delta_for_entry(node, trace, entry, guard)
        keys = _compared_delta(delta, node.by, "top_changed", entry.site_key)
        keys = keys.to(torch.float64)
        valid = entry._mask._dense_ro() & ~torch.isnan(keys)
        per_entry.append((entry, keys.reshape(-1), valid.reshape(-1)))
    total_valid = int(sum(valid.sum().item() for _, _, valid in per_entry))
    vs = _reference_display(node.reference)
    if node.fraction is not None:
        k = math.ceil(node.fraction * total_valid)
        source = (
            f"top_changed(fraction={node.fraction}, by={node.by!r}, "
            f"largest={node.largest}, vs={vs!r})"
        )
    else:
        if node.k is None:
            raise RuntimeError("top_changed node lost its k")
        k = node.k
        source = f"top_changed(k={k}, by={node.by!r}, largest={node.largest}, vs={vs!r})"
        if k > total_valid:
            raise _unresolvable(
                "population_too_small",
                f"top_changed needs {k} elements but the population has only "
                f"{total_valid} rankable (non-NaN, in-population) elements.",
                requested=k,
                available=total_valid,
            )
    sentinel = float("-inf") if node.largest else float("inf")
    flat_keys = (
        torch.cat(
            [
                torch.where(valid, keys, torch.tensor(sentinel, dtype=torch.float64))
                for _, keys, valid in per_entry
            ]
        )
        if per_entry
        else torch.zeros(0, dtype=torch.float64)
    )
    order = torch.argsort(flat_keys, descending=node.largest, stable=True)[:k]
    selected_flat = torch.zeros(flat_keys.shape[0], dtype=torch.bool)
    selected_flat[order] = True
    entries: list[SiteEntry] = []
    offset = 0
    for entry, keys, _ in per_entry:
        span = keys.shape[0]
        dense = selected_flat[offset : offset + span].reshape(entry.shape)
        offset += span
        entries.append(_entry_with_mask(entry, dense, "exact", source))
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_compare_term(node: _CompareTerm, trace: Any) -> ResolvedSelection:
    """Resolve one differential term against the subject trace's payloads."""

    if node.criterion == "changed":
        return _resolve_changed(node, trace)
    return _resolve_top_changed(node, trace)


def changed(
    reference: Any,
    within: Any = None,
    *,
    above: float | None = None,
    below: float | None = None,
    by: str = "abs",
) -> Selection:
    """Select elements whose value differs from a reference run's ("what moved?").

    The delta is DIRECTIONAL — ``subject - reference``, where the SUBJECT is
    the trace the selection resolves against and ``reference`` is one other
    Trace (a fork after ``do()``, a capture on another input, ...). ``by``
    picks the compared quantity: ``'abs'`` (default) compares ``|delta|``,
    ``'signed'`` the signed delta (so ``above=`` selects INCREASED elements,
    ``below=`` with a negative bound the decreased). ``above=`` / ``below=``
    are strict bounds (both = open band); with NEITHER given the default is
    ``above=0.0`` — bare ``changed(ref)`` selects every element that moved at
    all, the intervention-effect mask. Both runs must retain the compared
    payloads at the same pass-qualified site with the same shape: missing
    sites, unsaved payloads, shape drift, and structural-site-key
    disagreement refuse typed (structures never silently intersect).
    Resolving against the reference itself refuses (vacuous by construction).
    NaN deltas never satisfy a bound; complex deltas refuse ``by='signed'``.
    ``provenance.relation`` is ``exact``: the claim is the pairwise delta
    between THESE two captures. PARAM populations refuse
    ``selection_kind_incompatible`` (capture-time weights are not retained on
    Trace records). DOCUMENTED-UNSTABLE spelling.
    """

    reference = _validate_reference(reference, "changed")
    _validate_by(by, "changed")
    if above is None and below is None:
        above = 0.0
    if above is not None:
        above = _validate_real_number(above, "above", "changed")
    if below is not None:
        below = _validate_real_number(below, "below", "changed")
    return Selection(
        _CompareTerm(
            criterion="changed",
            reference=reference,
            within=_lift_within(within, "changed"),
            by=by,
            above=above,
            below=below,
        ),
        kind="ACT",
    )


def top_changed(
    reference: Any,
    within: Any = None,
    k: int | None = None,
    *,
    fraction: float | None = None,
    by: str = "abs",
    largest: bool = True,
) -> Selection:
    """Select the elements that changed most (or least) versus a reference run.

    Globally ranks ``subject - reference`` deltas across the whole population
    (``within=None`` means every retained tensor site) and selects exactly
    ``k`` elements — or ``ceil(fraction * population)`` with ``fraction=``;
    exactly ONE of the two must be given — without replacement, with a
    deterministic tie-break (stable sort; canonical site order, then flat
    index). ``by='abs'`` (default) ranks ``|delta|``; ``by='signed'`` ranks
    the signed delta (``largest=True`` = most increased). ``largest=False``
    selects the LEAST-moved elements (a real control: what the intervention
    did NOT touch). NaN deltas never enter the candidate population; a
    population with fewer than ``k`` rankable elements refuses
    ``population_too_small``. Structure matching, self-comparison, and
    provenance follow :func:`changed` exactly. DOCUMENTED-UNSTABLE spelling.
    """

    reference = _validate_reference(reference, "top_changed")
    _validate_by(by, "top_changed")
    if not isinstance(largest, bool):
        raise ValueError(f"top_changed `largest` must be a bool; got {largest!r}.")
    if (k is None) == (fraction is None):
        raise ValueError(
            "top_changed requires exactly one of `k=` (element count) or "
            "`fraction=` (share of the rankable population)."
        )
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 0):
        raise ValueError(f"top_changed `k` must be a non-negative int; got {k!r}.")
    if fraction is not None:
        fraction = _validate_real_number(fraction, "fraction", "top_changed")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"top_changed `fraction` must be in [0, 1]; got {fraction!r}.")
    return Selection(
        _CompareTerm(
            criterion="top_changed",
            reference=reference,
            within=_lift_within(within, "top_changed"),
            by=by,
            k=k,
            fraction=fraction,
            largest=largest,
        ),
        kind="ACT",
    )


# ---------------------------------------------------------------------------
# Cross-pass producers (one capture; evidence = a multi-pass layer's passes).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PassTerm:
    """AST leaf for the cross-pass producers (stable_across_passes / pass_variance)."""

    stat: str
    within: Any
    tol: float = 0.0
    above: float | None = None
    below: float | None = None
    passes: tuple[int, ...] | None = None

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure."""

        population = "saved_sites" if self.within is None else repr(self.within)
        window = "all" if self.passes is None else repr(list(self.passes))
        if self.stat == "stable":
            head = f"stable_across_passes(tol={self.tol}, passes={window}"
        else:
            head = f"pass_variance(above={self.above}, below={self.below}, passes={window}"
        return f"{head}, within={population})"


def _pass_groups(
    node: _PassTerm, population: ResolvedSelection, producer: str
) -> list[tuple[str, list[SiteEntry]]]:
    """Group population entries per layer and select each layer's pass window.

    Enforces the cross-pass honesty floor (>= 2 window passes per layer, else
    ``population_too_small`` with the teaching message), explicit-window
    presence (a requested pass missing from the population refuses
    ``site_not_in_trace``), and a constant index space across the window
    (shape drift refuses ``mask_shape_mismatch`` — restrict ``passes=`` to a
    constant-shape window).
    """

    by_layer: dict[str, dict[int, SiteEntry]] = {}
    layer_order: list[str] = []
    for entry in population:
        layer_label, pass_index = entry.site_key
        if layer_label not in by_layer:
            by_layer[layer_label] = {}
            layer_order.append(layer_label)
        by_layer[layer_label][int(pass_index)] = entry
    groups: list[tuple[str, list[SiteEntry]]] = []
    for layer_label in layer_order:
        passes_present = by_layer[layer_label]
        if node.passes is None:
            window = sorted(passes_present)
        else:
            window = list(node.passes)
            for pass_index in window:
                if pass_index not in passes_present:
                    raise _unresolvable(
                        "site_not_in_trace",
                        f"{producer} window names pass {pass_index} of layer "
                        f"{layer_label!r}, which is not in the population "
                        "(absent from the trace, or excluded by `within=`).",
                        site=f"{layer_label}:{pass_index}",
                    )
        if len(window) < 2:
            raise _unresolvable(
                "population_too_small",
                f"{producer} makes a CROSS-PASS claim, and layer {layer_label!r} "
                f"contributes only {len(window)} pass to the window — the claim "
                "would be vacuously true. Restrict `within=` to the recurrent "
                "(multi-pass) layers, or use the single-capture value producers "
                "(threshold / sign / top_k) for one-pass claims.",
                site=layer_label,
                requested=2,
                available=len(window),
            )
        window_entries = [passes_present[pass_index] for pass_index in window]
        first_shape = window_entries[0].shape
        for entry in window_entries[1:]:
            if entry.shape != first_shape:
                raise _unresolvable(
                    "mask_shape_mismatch",
                    f"layer {layer_label!r} changes shape across the pass window "
                    f"({first_shape!r} vs {entry.shape!r} at pass "
                    f"{entry.site_key[1]}): elementwise cross-pass statistics "
                    "need one index space. Restrict `passes=` to a "
                    "constant-shape window.",
                    site=layer_label,
                )
        groups.append((layer_label, window_entries))
    return groups


def _disclose_episode_pass_axis(trace: Any, producer: str, layer_labels: list[str]) -> None:
    """Disclose the pass-vs-step axis when a pass window resolves on an episode.

    Pass indices are per-site OCCURRENCE counters (pass k of a layer is the
    k-th time THAT layer ran), never episode steps: a layer absent from a
    step has no pass there, so on any model whose layer is not called at
    every step (a routed mixture-of-experts expert) pass k need not lie at
    episode step k (foldB D11). Both counts exist today — each layer's
    ``num_passes`` and the episode ledger's step rows — so the read
    discloses them at the point of use, without new arguments; the typed
    refusal for an explicit step axis is a named future surface (F-EPISODE
    ``axis=``), not this warning. Fires once per resolve, only on episode
    captures with a settled ledger; the numbers are disclosures, never a
    settlement input.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, Mapping):
        return
    # Downward (L4 -> L2) import, deferred to keep this module's import-time
    # footprint free of the capture package (the file-local Trace precedent).
    from .capture._episode_ledger import EPISODE_ANNOTATIONS_KEY

    payload = annotations.get(EPISODE_ANNOTATIONS_KEY)
    if not isinstance(payload, Mapping):
        return
    rows = payload.get("rows")
    if not isinstance(rows, (list, tuple)) or not rows:
        # Declared-only marker (pre-settlement / partial product): no settled
        # step rows exist, so there is nothing provable to disclose.
        return
    n_steps = len(rows)
    site_passes = {label: int(trace[label].num_passes) for label in layer_labels}
    ran = ", ".join(f"{label} ran {count}x" for label, count in site_passes.items())
    warnings.warn(
        TorchLensWarning(
            f"{producer} resolved a pass window on an EPISODE capture: pass "
            "indices are per-site occurrence counters (pass k of a layer is "
            "the k-th time THAT layer ran), never episode steps. This episode "
            f"ran {n_steps} steps; resolved layers: {ran}. A layer absent "
            "from a step has no pass there, so pass k need not lie at episode "
            "step k. Remedy: window by per-site occurrence deliberately, "
            "mapping occurrences to steps through "
            "trace.annotations['episode']['rows']; an explicit step axis is "
            "a named future surface, not a selector argument today",
            code="episode_pass_window_occurrence_axis",
            producer=producer,
            episode_steps=n_steps,
            site_passes=site_passes,
            affected_sites=sorted(site_passes),
        ),
        stacklevel=3,
    )


def _resolve_pass_term(node: _PassTerm, trace: Any) -> ResolvedSelection:
    """Resolve one cross-pass statistic against a capture's per-pass payloads.

    The element population per layer is the INTERSECTION of the window
    entries' masks (a cross-pass claim needs the element in evidence at EVERY
    window pass); the resulting mask lands on every window pass-site, since
    the claim is about the unit across the whole window. On episode captures
    the resolve additionally discloses the pass-vs-step axis
    (``episode_pass_window_occurrence_axis``).
    """

    producer = "stable_across_passes" if node.stat == "stable" else "pass_variance"
    population = _resolve_population(node.within, trace)
    entries: list[SiteEntry] = []
    resolved_layers: list[str] = []
    for layer_label, window_entries in _pass_groups(node, population, producer):
        values = []
        for entry in window_entries:
            value = _read_saved_value(trace, entry)
            if value.is_complex():
                raise _unresolvable(
                    "value_criterion_invalid",
                    f"{producer} orders/spreads raw values across passes, and "
                    f"complex payloads have no total order (layer "
                    f"{layer_label!r}).",
                    site=layer_label,
                )
            values.append(value.to(torch.float64))
        stacked = torch.stack(values)
        shared = window_entries[0]._mask._dense_ro().clone()
        relation = window_entries[0].provenance.relation
        for entry in window_entries[1:]:
            shared &= entry._mask._dense_ro()
            relation = _join_relation(relation, entry.provenance.relation)
        n_passes = len(window_entries)
        if node.stat == "stable":
            spread = stacked.max(dim=0).values - stacked.min(dim=0).values
            dense = spread <= node.tol
            source = f"stable_across_passes(n_passes={n_passes}, tol={node.tol})"
        else:
            variance = torch.var(stacked, dim=0)
            dense = torch.ones(window_entries[0].shape, dtype=torch.bool)
            if node.above is not None:
                dense &= variance > node.above
            if node.below is not None:
                dense &= variance < node.below
            source = f"pass_variance(n_passes={n_passes}, above={node.above}, below={node.below})"
        dense &= shared
        resolved_layers.append(layer_label)
        for entry in window_entries:
            entries.append(
                SiteEntry(
                    kind=entry.kind,
                    site_key=entry.site_key,
                    provenance=SelectionProvenance(
                        relation=_join_relation(relation, "exact"), source=source
                    ),
                    _mask=_mask_from_dense(entry.shape, dense),
                    structural_site_key=entry.structural_site_key,
                )
            )
    if resolved_layers:
        _disclose_episode_pass_axis(trace, producer, resolved_layers)
    return ResolvedSelection(trace, "ACT", entries)


def stable_across_passes(
    within: Any = None,
    *,
    tol: float = 0.0,
    passes: Any = None,
) -> Selection:
    """Select units stable across a recurrent layer's passes (its occurrences).

    Per element of each multi-pass layer in the population, computes the
    RANGE (max - min, in float64) of the retained activation across the pass
    window and selects elements whose range is ``<= tol``. ``passes=None``
    (default) uses every population pass of each layer; an explicit iterable
    of 1-based pass indices restricts the window (every named pass must be in
    the population). Pass indices are per-site OCCURRENCE counters (pass
    ``k`` = the k-th time THAT layer ran), not steps of an outer generation
    loop; on episode captures the resolve discloses both counts
    (``episode_pass_window_occurrence_axis``). Addressing is pass-qualified throughout; a bare layer
    label in ``within=`` is the all-passes Layer spelling, never one silent
    pass. A layer contributing fewer than two window passes refuses
    ``population_too_small`` (a single-pass "stability" claim is vacuous);
    cross-pass shape drift refuses ``mask_shape_mismatch``. The mask lands on
    EVERY window pass-site (the claim is about the unit across the window,
    so ``do()`` on it edits every window pass); the element population is the
    intersection of the window entries' masks. An element with a NaN at any
    window pass has a NaN range and is never selected. Complex payloads
    refuse (no total order). ``provenance.relation`` is ``exact`` — the name
    scopes the claim to THIS capture's passes, complete evidence (the
    ``low_variance`` precedent); the unstable selection is the touched-family
    complement ``~stable_across_passes(...)``. DOCUMENTED-UNSTABLE spelling.
    """

    tol = _validate_real_number(tol, "tol", "stable_across_passes")
    if tol < 0:
        raise ValueError(f"stable_across_passes `tol` must be non-negative; got {tol!r}.")
    return Selection(
        _PassTerm(
            stat="stable",
            within=_lift_within(within, "stable_across_passes"),
            tol=tol,
            passes=_validate_passes(passes, "stable_across_passes"),
        ),
        kind="ACT",
    )


def pass_variance(
    within: Any = None,
    *,
    above: float | None = None,
    below: float | None = None,
    passes: Any = None,
) -> Selection:
    """Select units by their variance across a recurrent layer's passes.

    Per element, computes the unbiased (n-1) variance in float64 of the
    retained activation across the pass window and applies strict bounds:
    ``below=`` selects low-variance (steady) units, ``above=`` high-variance
    (swinging) units, both together the open band; at least one bound is
    required. Pass indices are per-site OCCURRENCE counters — pass ``k`` of
    a layer is the k-th time THAT layer ran, never step ``k`` of the model's
    generation loop — so ``pass_variance(above=t, passes=[8, 9, 10])`` reads
    "variance explodes across the layer's LATE OCCURRENCES". Only on a
    layer that runs exactly once per step do occurrences and steps
    coincide; on any model whose layer is not called at every step (a
    routed mixture-of-experts expert, a conditional branch), occurrence 8
    need not lie at episode step 8, and resolving a pass window on an
    episode capture discloses both counts
    (``episode_pass_window_occurrence_axis``). Window semantics, the
    two-pass honesty floor, shape-drift and complex refusals, NaN exclusion,
    mask placement (every window pass-site), and the ``exact`` provenance
    relation all follow :func:`stable_across_passes`. DOCUMENTED-UNSTABLE
    spelling.
    """

    if above is None and below is None:
        raise ValueError(
            "pass_variance requires at least one bound: `above=` (high-variance) "
            "and/or `below=` (low-variance)."
        )
    if above is not None:
        above = _validate_real_number(above, "above", "pass_variance")
    if below is not None:
        below = _validate_real_number(below, "below", "pass_variance")
    return Selection(
        _PassTerm(
            stat="variance",
            within=_lift_within(within, "pass_variance"),
            above=above,
            below=below,
            passes=_validate_passes(passes, "pass_variance"),
        ),
        kind="ACT",
    )


register_term_resolver(_CompareTerm, _resolve_compare_term)
register_term_resolver(_PassTerm, _resolve_pass_term)
