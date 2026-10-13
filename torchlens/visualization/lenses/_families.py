"""View-aware source-family resolution for performance lenses (knob N16).

A performance lens declares a source FAMILY (``"time"`` / ``"bytes"`` /
``"flops"``) rather than a concrete record field. Resolution binds the
per-pass member on unrolled views (``func_duration``) and the summed member
on rolled views (``total_func_duration``), with the aggregation line
("total across N passes") mandatory in the rolled case and a typed refusal
when the resolved member has zero coverage (themes memo section 2 item 7).

Measured necessity (themes memo M11): the naive pairing ships a 0%-coverage
graph under a populated legend in one direction and a silent unit drop in
the other; two of the four view x member cells were wrong before this
module existed.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..._errors import InvalidArgumentError

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "SOURCE_FAMILIES",
    "ResolvedSource",
    "SourceCoverage",
    "SourceFamily",
    "resolve_source_family",
    "source_coverage",
]


@dataclass(frozen=True)
class SourceFamily:
    """One performance source family and its per-view record members.

    Attributes
    ----------
    name:
        Family token (``"time"``, ``"bytes"``, ``"flops"``).
    per_pass_member:
        Record field bound on unrolled views (one value per op call).
    summed_member:
        Record field bound on rolled views (exact cross-pass total).
    unit_wording:
        Human wording for legend lines ("wall-clock seconds", ...).
    capture_remedy:
        The remedy naming how to capture the missing evidence, used verbatim
        by the headline-evidence refusal.
    """

    name: str
    per_pass_member: str
    summed_member: str
    unit_wording: str
    capture_remedy: str


#: The closed family table (themes memo section 3: speed / memory / compute).
SOURCE_FAMILIES: dict[str, SourceFamily] = {
    "time": SourceFamily(
        name="time",
        per_pass_member="func_duration",
        summed_member="total_func_duration",
        unit_wording="wall-clock seconds per op call (instrumented capture)",
        capture_remedy=(
            "re-capture with a plain tl.trace(model, x) forward; op timings are "
            "recorded on every eager capture -- a loaded artifact saved without "
            "timing fields cannot serve the speed lens"
        ),
    ),
    "bytes": SourceFamily(
        name="bytes",
        per_pass_member="activation_memory",
        summed_member="total_activation_memory",
        unit_wording="output-storage bytes, not live/allocator/peak memory",
        capture_remedy=(
            "re-capture with a plain tl.trace(model, x) forward; activation "
            "memory is recorded per op on every eager capture"
        ),
    ),
    "flops": SourceFamily(
        name="flops",
        per_pass_member="flops_forward",
        summed_member="total_flops_forward",
        unit_wording="forward FLOPs (2 FLOPs per multiply-accumulate)",
        capture_remedy=(
            "re-capture with a plain tl.trace(model, x) forward; FLOP counts "
            "derive from captured op metadata where a counting rule exists"
        ),
    ),
}


@dataclass(frozen=True)
class ResolvedSource:
    """A family bound to one concrete member for one view.

    Attributes
    ----------
    family:
        The declaring :class:`SourceFamily`.
    member:
        The resolved record field name for the requested view.
    view:
        ``"rolled"`` or ``"unrolled"``.
    aggregation_line:
        Mandatory rendered wording in the rolled case, ``None`` unrolled.
    """

    family: SourceFamily
    member: str
    view: str
    aggregation_line: str | None


@dataclass(frozen=True)
class SourceCoverage:
    """Coverage of one member over a trace's op population.

    ``encoded`` counts ops whose member coerces to a finite scalar;
    ``total`` is the op population examined. ``fraction`` is 0.0 on an
    empty population (disclosed, never a division error).
    """

    member: str
    encoded: int
    total: int

    @property
    def fraction(self) -> float:
        """Return encoded/total, 0.0 on an empty population."""

        return self.encoded / self.total if self.total else 0.0


def resolve_source_family(family: str, view: str) -> ResolvedSource:
    """Bind ``family`` to its per-view member (N16 resolution step).

    Parameters
    ----------
    family:
        Family token from :data:`SOURCE_FAMILIES`.
    view:
        Effective render view: ``"rolled"`` binds the summed member and
        carries the mandatory aggregation line; ``"unrolled"`` binds the
        per-pass member.

    Raises
    ------
    InvalidArgumentError
        ``lens_source_family_unknown`` for a token outside the closed table;
        ``lens_view_invalid`` for a view outside {rolled, unrolled}.
    """

    row = SOURCE_FAMILIES.get(family)
    if row is None:
        raise InvalidArgumentError(
            f"unknown performance source family {family!r}; "
            f"known families: {', '.join(sorted(SOURCE_FAMILIES))}",
            code="lens_source_family_unknown",
            remedy="declare one of the closed family tokens (time, bytes, flops)",
            argument="family",
        )
    if view not in ("rolled", "unrolled"):
        raise InvalidArgumentError(
            f"view must be 'rolled' or 'unrolled'; received {view!r}",
            code="lens_view_invalid",
            remedy="pass the effective vis_mode ('rolled' or 'unrolled')",
            argument="view",
        )
    if view == "rolled":
        return ResolvedSource(
            family=row,
            member=row.summed_member,
            view=view,
            aggregation_line=f"{row.summed_member}: total across passes on rolled nodes",
        )
    return ResolvedSource(family=row, member=row.per_pass_member, view=view, aggregation_line=None)


def scalar_or_none(value: Any) -> float | None:
    """Coerce a record value to a finite float, or ``None``.

    Quantity-valued record fields (``"98.5 us"`` reprs) support ``float()``;
    anything unconvertible or non-finite reads as absent evidence.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if as_float != as_float or as_float in (float("inf"), float("-inf")):
        return None
    return as_float


def records_for_member(trace: Trace, member: str) -> list[Any]:
    """Return the record population a member reads on.

    Summed (``total_*``) members are LAYER aggregates -- the rolled view
    renders layers, so coverage and values read the layer population; every
    other member reads per-op.
    """

    if member.startswith("total_"):
        return list(trace.layers)
    return list(trace.ops)


def safe_label(record: Any) -> str:
    """Return a record's pass-qualified label, falling back to layer_label.

    A multi-pass LAYER refuses bare ``.label`` typed (pass-ambiguous); the
    layer label is the honest spelling for the aggregate record.
    """

    try:
        label = record.label
    except (AttributeError, ValueError):  # multi-pass Layer.label raises ValueError
        label = None
    return str(label) if label else str(record.layer_label)


def source_coverage(trace: Trace, member: str) -> SourceCoverage:
    """Measure ``member`` coverage over its record population.

    Boundary pseudo-records (inputs/outputs) are part of the examined
    population -- the coverage line's denominator is the population a
    rendered picture actually shows.
    """

    total = 0
    encoded = 0
    for record in records_for_member(trace, member):
        total += 1
        if scalar_or_none(getattr(record, member, None)) is not None:
            encoded += 1
    return SourceCoverage(member=member, encoded=encoded, total=total)
