"""Canonical forward-compute aggregation (the ONE numbers source).

Lane A07 (megasprint 2026-08-27); costreport memo D1-D3 seed. Every reporting
surface (summary footer, profile, flop_count) derives its forward compute
totals from this ONE walk over the identity partition -- never from a private
re-sum -- so two surfaces can no longer agree by sharing a bug. The C02
FactCore substrate absorbs this module; spellings are DOCUMENTED-UNSTABLE.

The aggregation returns rows' classification counts alongside the totals:
every layer-list row lands in EXACTLY ONE of four classes (totality is
pinned):

- ``known``          -- a compute op with a nonzero-capable counted value
- ``zero_by_rule``   -- a compute op whose value is 0 by a named table rule
                        (memory-layout ops, zero-arithmetic constructions)
- ``not_applicable`` -- boundary pseudo-rows and buffer state rows (own no
                        compute by the identity partition)
- ``unknown``        -- a compute op outside every cost table (named; a work
                        queue, never silently dropped)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from ..capture.flops import ZERO_ARITHMETIC_CONSTRUCTIONS, ZERO_FLOPS_OPS
from ..quantities import Flops, Macs

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


@dataclass(frozen=True)
class ComputeTotals:
    """Aggregated forward-compute truth over one trace's identity partition.

    Parameters
    ----------
    flops_fma2:
        Total forward FLOPs under the stored fma=2 convention (one MAC = 2
        FLOPs). Sums exactly the real op rows; alias rows own nothing.
    macs:
        Total TRUE multiply-accumulate count over ops with a known MAC split.
    macs_unknown_split:
        Labels of compute ops whose MAC split could not be derived; when
        nonempty ``macs`` is a lower bound.
    known / zero_by_rule / not_applicable / unknown:
        The four-way row classification counts (exactly one class per row).
    unknown_ops:
        Labels of the ``unknown`` rows -- the named work queue
        (``register_op_rule`` is the remedy).
    """

    flops_fma2: Flops
    macs: Macs
    macs_unknown_split: tuple[str, ...]
    known: int
    zero_by_rule: int
    not_applicable: int
    unknown: int
    unknown_ops: tuple[str, ...]

    @property
    def total_rows(self) -> int:
        """Total classified rows (the totality denominator)."""

        return self.known + self.zero_by_rule + self.not_applicable + self.unknown


def classify_row(op: object) -> str:
    """Classify one layer-list row into the four-way coverage vocabulary."""

    if getattr(op, "is_input", False) or getattr(op, "is_output", False):
        return "not_applicable"
    if getattr(op, "is_buffer", False):
        return "not_applicable"
    flops = getattr(op, "flops_forward", None)
    if flops is None:
        return "unknown"
    func_name = getattr(op, "func_name", None)
    if int(flops) == 0 and (
        func_name in ZERO_FLOPS_OPS or func_name in ZERO_ARITHMETIC_CONSTRUCTIONS
    ):
        return "zero_by_rule"
    return "known"


def aggregate_forward_compute(trace: Trace) -> ComputeTotals:
    """Run the ONE canonical aggregation over a finished trace."""

    flops_total = 0
    macs_total = 0
    macs_unknown: list[str] = []
    counts = {"known": 0, "zero_by_rule": 0, "not_applicable": 0, "unknown": 0}
    unknown_ops: list[str] = []
    for op in trace.layer_list:
        row_class = classify_row(op)
        counts[row_class] += 1
        if row_class == "unknown":
            unknown_ops.append(str(op.layer_label))
            continue
        if row_class == "not_applicable":
            continue
        flops_total += int(op.flops_forward or 0)
        macs = op.macs_forward
        if macs is not None:
            macs_total += int(macs)
        elif row_class == "known":
            macs_unknown.append(str(op.layer_label))
    return ComputeTotals(
        flops_fma2=Flops(flops_total),
        macs=Macs(macs_total),
        macs_unknown_split=tuple(macs_unknown),
        known=counts["known"],
        zero_by_rule=counts["zero_by_rule"],
        not_applicable=counts["not_applicable"],
        unknown=counts["unknown"],
        unknown_ops=tuple(unknown_ops),
    )


@dataclass(frozen=True)
class UnknownOpGroup:
    """One named group of unknown-FLOPs ops (the burn-down work queue).

    Parameters
    ----------
    func_name:
        Captured op name shared by the group.
    count:
        Number of captured ops with this name and unknown FLOPs.
    example_labels:
        Up to three example op labels (truncation disclosed by ``count``).
    example_shapes:
        Output shapes of the example ops.
    remedy:
        The exact remedy invocation for this group.
    """

    func_name: str
    count: int
    example_labels: tuple[str, ...]
    example_shapes: tuple[tuple[int, ...] | None, ...]
    remedy: str


def unknown_op_ledger(trace: Trace) -> tuple[UnknownOpGroup, ...]:
    """Group the trace's unknown-FLOPs ops by op name, with named remedies.

    The ledger is a WORK QUEUE (costreport D3): every unknown op is named,
    reasoned, and remedied -- never a bare count over an anonymous exclusion.
    """

    groups: dict[str, list[Any]] = {}
    for op in trace.layer_list:
        if classify_row(op) != "unknown":
            continue
        groups.setdefault(str(getattr(op, "func_name", "?")), []).append(op)
    ledger = []
    for func_name in sorted(groups):
        ops = groups[func_name]
        ledger.append(
            UnknownOpGroup(
                func_name=func_name,
                count=len(ops),
                example_labels=tuple(str(op.layer_label) for op in ops[:3]),
                example_shapes=tuple(
                    tuple(op.shape) if getattr(op, "shape", None) else None for op in ops[:3]
                ),
                remedy=(f"torchlens.capture.flops.register_op_rule({func_name!r}, <flops_fn>)"),
            )
        )
    return tuple(ledger)


def forward_flops_total(trace: Trace, *, fma: int = 2) -> Flops:
    """Total forward FLOPs under an explicit FMA convention.

    ``fma=2`` is the stored convention (one multiply-accumulate = 2 FLOPs).
    ``fma=1`` recounts every op as ``fma_macs + other_flops`` from its
    two-term compute record; an op whose exact split is unavailable makes the
    conversion impossible and REFUSES typed -- never silently scaled from the
    aggregate, never accepted-and-ignored.
    """

    if fma == 2:
        return aggregate_forward_compute(trace).flops_fma2
    if fma != 1:
        raise InvalidArgumentError(
            f"fma must be 1 or 2; got {fma!r}.",
            code="flop_convention_invalid",
            remedy="Pass fma=2 (stored convention, one MAC = 2 FLOPs) or fma=1.",
        )
    total = 0
    unconvertible: list[str] = []
    for op in trace.layer_list:
        if classify_row(op) in ("not_applicable", "unknown"):
            continue
        record = op.compute_record
        converted = None if record is None else record.flops(fma=1)
        if converted is None:
            unconvertible.append(str(op.layer_label))
            continue
        total += converted
    if unconvertible:
        shown = ", ".join(unconvertible[:5])
        extra = "" if len(unconvertible) <= 5 else f" (+{len(unconvertible) - 5} more)"
        raise InvalidArgumentError(
            f"The fma=1 convention is unavailable: {len(unconvertible)} op(s) have no "
            f"derivable MAC split ({shown}{extra}).",
            code="flop_convention_unavailable",
            remedy=(
                "Report under the stored fma=2 convention, or register a cost rule "
                "declaring the op's MAC split (torchlens.capture.flops.register_op_rule)."
            ),
        )
    return Flops(total)
