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


#: Phase enum for compute rows (costreport item 2). Backward rows land when
#: the C07 backward-fire fields persist (F09); the vocabulary is closed NOW
#: so consumers never invent a third spelling.
COMPUTE_PHASES: tuple[str, ...] = ("forward", "backward")

#: Row-kind vocabulary (identity partition): real compute ops, boundary
#: pseudo-rows (own nothing), buffer state rows (own nothing).
COMPUTE_ROW_KINDS: tuple[str, ...] = ("op", "boundary", "buffer")


@dataclass(frozen=True)
class ComputeRow:
    """One accounting-unit row of the canonical aggregation (D1/D5/D7).

    Boundary and buffer rows RENDER but own nothing: every additive value
    is ``None`` there, and the CI plant pins that a non-``None`` additive
    cell on a non-op row fails. All numeric fields are plain ints or None
    (the raw-numbers pin).
    """

    row_id: str
    label: str
    site_key: str | None
    kind: str
    phase: str
    func_name: str | None
    layer_type: str | None
    dtype: str | None
    flops_fma2: int | None
    fma_macs: int | None
    other_flops: int | None
    coverage_class: str
    evidence: str
    applicability: str
    reason: str | None


@dataclass(frozen=True)
class RatioFact:
    """A ratio with NAMED numerator/denominator row/section IDs (D6).

    A percent without its denominator identity is how the 3.878x naive-sum
    class of error ships; the IDs make the scope auditable.
    """

    value: float | None
    numerator_id: str
    denominator_id: str
    scope: str


@dataclass(frozen=True)
class ComputeAggregation:
    """The canonical aggregation service result (costreport item 2 / D1).

    ``partition_total`` is returned SEPARATELY and is never a column sum;
    ``rows`` carry per-cell evidence and applicability; ``coverage`` is the
    four-way classification; ``convention`` and ``scope`` are explicit.
    This object is the compute FACE of the C02 FactCore -- summary,
    profile, flop_count, and the report family are projections of it, and
    the import lint (tests/test_factcore_import_lint.py) keeps this module
    the only aggregation-layer reader of raw per-op compute fields.
    """

    rows: tuple[ComputeRow, ...]
    partition_total: Flops
    macs_total: Macs
    coverage: ComputeTotals
    convention: str
    scope: str
    by_dtype: tuple[tuple[str, int], ...]
    execution_modes: tuple[tuple[str, int], ...]

    def ratio(self, numerator_row_id: str, denominator: str = "partition_total") -> RatioFact:
        """Return a ratio fact against the partition total (or a named row)."""

        numerator_row = next((row for row in self.rows if row.row_id == numerator_row_id), None)
        numerator = None if numerator_row is None else numerator_row.flops_fma2
        if denominator == "partition_total":
            denominator_value: int | None = int(self.partition_total)
        else:
            denominator_row = next((row for row in self.rows if row.row_id == denominator), None)
            denominator_value = None if denominator_row is None else denominator_row.flops_fma2
        value = (
            None if numerator is None or not denominator_value else numerator / denominator_value
        )
        return RatioFact(
            value=value,
            numerator_id=numerator_row_id,
            denominator_id=denominator,
            scope=self.scope,
        )


def compute_aggregation(trace: Trace) -> ComputeAggregation:
    """Build the canonical per-row compute aggregation over one trace.

    ONE walk over the identity partition; every reporting surface derives
    from these rows or the separately-returned partition total -- never a
    private re-sum. Backward rows are honestly ABSENT until the persisted
    backward-fire fields land (C07/F09); the phase vocabulary is closed
    already.
    """

    rows: list[ComputeRow] = []
    by_dtype: dict[str, int] = {}
    execution_modes: dict[str, int] = {}
    for index, op in enumerate(trace.layer_list):
        row_class = classify_row(op)
        if getattr(op, "is_buffer", False):
            kind = "buffer"
        elif getattr(op, "is_input", False) or getattr(op, "is_output", False):
            kind = "boundary"
        else:
            kind = "op"
        label = str(getattr(op, "layer_label", index))
        flops = getattr(op, "flops_forward", None) if kind == "op" else None
        record = getattr(op, "compute_record", None) if kind == "op" else None
        fma_macs = None if record is None else getattr(record, "fma_macs", None)
        other_flops = None if record is None else getattr(record, "other_flops", None)
        if kind != "op":
            evidence, applicability = "unknown", "not_applicable"
            reason = "boundary/buffer rows own no compute (identity partition)"
        elif row_class == "unknown":
            evidence, applicability = "unknown", "applicable"
            reason = "no cost rule for this op (see trace.unknown_flop_ops)"
        elif row_class == "zero_by_rule":
            evidence, applicability = "formula_exact", "applicable"
            reason = "zero by named rule"
        else:
            evidence, applicability = "formula_exact", "applicable"
            reason = None
        dtype = getattr(op, "dtype", None)
        dtype_token = str(dtype).replace("torch.", "") if dtype is not None else None
        if kind == "op" and flops is not None:
            if dtype_token is not None:
                by_dtype[dtype_token] = by_dtype.get(dtype_token, 0) + int(flops)
            execution_modes["forward"] = execution_modes.get("forward", 0) + int(flops)
        rows.append(
            ComputeRow(
                row_id=f"op:{label}",
                label=label,
                site_key=getattr(op, "site_key", None),
                kind=kind,
                phase="forward",
                func_name=getattr(op, "func_name", None),
                layer_type=getattr(op, "layer_type", None),
                dtype=dtype_token,
                flops_fma2=None if kind != "op" or flops is None else int(flops),
                fma_macs=None if fma_macs is None else int(fma_macs),
                other_flops=None if other_flops is None else int(other_flops),
                coverage_class=row_class,
                evidence=evidence,
                applicability=applicability,
                reason=reason,
            )
        )
    totals = aggregate_forward_compute(trace)
    return ComputeAggregation(
        rows=tuple(rows),
        partition_total=totals.flops_fma2,
        macs_total=totals.macs,
        coverage=totals,
        convention="fma2",
        scope="whole_trace",
        by_dtype=tuple(sorted(by_dtype.items())),
        execution_modes=tuple(sorted(execution_modes.items())),
    )


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
