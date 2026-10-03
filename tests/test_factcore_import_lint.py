"""C02 import lints: one compute reader, no health truthiness (costreport D1, sumfam D5).

Two package-wide AST lints with reason-bearing ledgers (the op-lane
inplace-writer precedent): a NEW module reading raw per-op compute fields
or branching on ``nonfinite_ops`` goes red until consciously ledgered.
"one source" as a convention is unenforceable; as a lint it is checkable.
"""

import ast
import pathlib

import pytest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "torchlens"

#: Raw per-op compute fields the aggregation service owns (costreport D1).
_RAW_COMPUTE_FIELDS = frozenset(
    {"flops_forward", "macs_forward", "compute_record", "fma_macs", "other_flops"}
)

#: LICENSED readers of raw per-op compute fields, with reasons. Everything
#: else derives from torchlens.report._compute_truth's aggregation (or the
#: FactCore compute face). Adding a row is a conscious, reviewed act.
LICENSED_COMPUTE_READERS = {
    # The substrate itself: the ONE aggregation walk (costreport D1).
    "report/_compute_truth.py": "the canonical aggregation service (the one licensed walk)",
    # Producers: build/derive the fields they then re-read for verification.
    "capture/compute_record.py": "two-term record producer (verifies its own split)",
    "capture/projections.py": "journal commit tail (writes the persisted field)",
    "ir/op_record_scatter.py": "generated ingest scatter (field plumbing, not aggregation)",
    # Record classes: field definitions + per-record accessors (a single
    # op/layer's own value is record surface, not aggregation).
    "data_classes/op.py": "field definition + per-record accessors",
    "data_classes/layer.py": "per-layer mirrors/rollups of its OWN ops (record surface)",
    "data_classes/_trace_stats.py": (
        "reads ComputeRow.fma_macs from the aggregation service's own rows "
        "(name-collision with the raw op field; the totals themselves come "
        "from aggregate_forward_compute/compute_aggregation)"
    ),
    "report/_summary_report.py": (
        "reads ComputeRow.fma_macs and its OWN SummaryTotals.macs_forward "
        "(service-row/report-field name-collisions with the raw op fields; "
        "all numbers arrive through compute_aggregation/factcore)"
    ),
    # The F08 summary split moved the same licensed reads into siblings
    # (ladder/render/result carved out of _summary_report.py); identical
    # name-collision class, same factcore-served numbers.
    "report/_summary_ladder.py": (
        "reads ComputeRow.fma_macs from the aggregation service's own rows "
        "(name-collision with the raw op field)"
    ),
    "report/_summary_render.py": (
        "reads its OWN SummaryTotals.macs_forward report field "
        "(name-collision with the raw op field)"
    ),
    "report/_summary_result.py": (
        "reads ComputeRow.fma_macs from the aggregation service's own rows "
        "(name-collision with the raw op field)"
    ),
}

#: Builder/report modules where ``nonfinite_ops`` reads are banned outright
#: (sumfam D5): surfaces branch on ``nonfinite_verdict``/``health_facts``.
_NO_TRUTHINESS_DIRS = (
    "report",
    "visualization/_summary_internal",
)
#: ...except the health substrate itself, which SERVES the record.
_TRUTHINESS_EXEMPT = frozenset({"report/_health.py"})


def _relative(path: pathlib.Path) -> str:
    """Repo-relative module path under torchlens/."""

    return str(path.relative_to(_PACKAGE_ROOT)).replace("\\", "/")


@pytest.mark.heavy
def test_only_licensed_modules_read_raw_compute_fields() -> None:
    """costreport D1: the import lint that makes 'one source' checkable.

    heavy tier: the whole-package AST scan crossed the 7s smoke budget once
    the render substrates landed (T42; 7.4s cpu on a quiet box).
    """

    offenders: dict[str, list[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative = _relative(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = sorted(
            {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr in _RAW_COMPUTE_FIELDS
                and isinstance(node.ctx, ast.Load)
            }
        )
        if hits and relative not in LICENSED_COMPUTE_READERS:
            offenders[relative] = hits
    assert not offenders, (
        "unlicensed raw compute-field reads (derive from "
        "torchlens.report.compute_aggregation / trace.factcore.compute instead, "
        f"or ledger the module with a reason): {offenders}"
    )


def test_licensed_compute_ledger_stays_true() -> None:
    """A licensed module that stops reading the fields leaves the ledger."""

    for relative in LICENSED_COMPUTE_READERS:
        path = _PACKAGE_ROOT / relative
        assert path.exists(), f"ledgered module {relative} no longer exists"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in _RAW_COMPUTE_FIELDS
            and isinstance(node.ctx, ast.Load)
        }
        assert hits, f"ledger row {relative} is stale (no raw compute reads remain)"


def test_no_builder_reads_nonfinite_ops() -> None:
    """sumfam D5: builders branch on nonfinite_verdict, never the tuple.

    The false negative lives inside an ``if trace.nonfinite_ops:`` on a
    payload-stripped artifact; report/summary builders may not read the
    spelling at all.
    """

    offenders: dict[str, int] = {}
    for directory in _NO_TRUTHINESS_DIRS:
        for path in sorted((_PACKAGE_ROOT / directory).rglob("*.py")):
            relative = _relative(path)
            if relative in _TRUTHINESS_EXEMPT:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            count = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr == "nonfinite_ops"
                and isinstance(node.ctx, ast.Load)
            )
            if count:
                offenders[relative] = count
    assert not offenders, (
        "builder-side nonfinite_ops reads (branch on trace.nonfinite_verdict / "
        f"trace.health_facts instead): {offenders}"
    )
