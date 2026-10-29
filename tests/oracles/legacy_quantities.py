"""Legacy quantity census (Table A quantities; launch bar U40.2).

The launch bar asks for zero red on the legacy QUANTITY rows: one row per
named public number, with the doors that serve the same number collapsed
into one quantity. This module is the reproducible count.

It walks every instance of the in-scope public classes on two fixtures and
lists each numeric door (``Class.field`` whose value is an ``int``/``float``
that is not a ``bool``, typed quantities such as ``Bytes``/``Flops``
included). The committed ``data/legacy_quantities.tsv`` maps each door to a
quantity id and a status:

- ``identifier``: an id, index, ordinal, line number, timestamp or version;
  not a quantity, listed so the exclusion is visible.
- ``red-needs-row``: an independent (I1/I2) oracle test exists and passes;
  the quantity is red only because Table A has no row for it yet.
- ``red-needs-oracle``: no independent oracle; one must be written.
- ``green``: a Table A row (``OBL-Q-<QUANTITY>`` in ``obligations.tsv``) with a
  passing I1/I2 oracle, and a Table B binding for every door of the quantity
  naming that oracle as its positive control. The check enforces both
  directions: green needs the rows, and every ``OBL-Q-`` row needs a green
  quantity.

A quantity's status is the worst status over its doors. Usage::

    python tests/oracles/legacy_quantities.py           # check TSV == live doors, print the count
    python tests/oracles/legacy_quantities.py --list    # also print every red quantity
    python tests/oracles/legacy_quantities.py --new     # print TSV skeleton rows for unrowed doors

``--check`` (the default) exits 1 when a live door has no row, a row has no
live door, a quantity is red, or Tables A and B disagree with the census, so
the count cannot silently drift from the surface. :func:`census_findings` is
the same check as a list; ``test_legacy_census_zero_red.py`` runs it in every
pytest backstop.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

TSV_PATH = Path(__file__).resolve().parent / "data" / "legacy_quantities.tsv"
COLUMNS = ("door", "value_type", "kind", "quantity_id", "status", "oracle_ref", "note")
STATUSES = ("green", "red-needs-row", "red-needs-oracle")
KINDS = ("quantity", "identifier")

#: Counted set, disclosed with every published count.
COUNTED_SET = (
    "public non-callable attributes whose value is int/float (bool excluded, typed"
    " quantities included) on every instance of Trace, Layer, Op, Module,"
    " ModuleCall, Param, Buffer, SummaryReport, SummaryTotals and CaptureFacts"
    " reached from two fixtures (eval forward; train mode with save_grads,"
    " backward, input/output distances and peak-memory measurement)"
)

#: Dotted public path of each in-scope class, as Table B names doors.
DOOR_PATHS = {
    "Trace": "torchlens.Trace",
    "Layer": "torchlens.Layer",
    "Op": "torchlens.Op",
    "Module": "torchlens.data_classes.module.Module",
    "ModuleCall": "torchlens.data_classes.module.ModuleCall",
    "Param": "torchlens.data_classes.param.Param",
    "Buffer": "torchlens.data_classes.buffer.Buffer",
    "SummaryReport": "torchlens.report._summary_report.SummaryReport",
    "SummaryTotals": "torchlens.report._summary_report.SummaryTotals",
    "CaptureFacts": "torchlens.report._summary_report.CaptureFacts",
}

SCOPE = (
    "Trace",
    "Layer",
    "Op",
    "Module",
    "ModuleCall",
    "Param",
    "Buffer",
    "SummaryReport",
    "SummaryTotals",
    "CaptureFacts",
)


def _fixture_model() -> Any:
    import torch
    from torch import nn

    class Fixture(nn.Module):
        """Conv + BatchNorm (a buffer), a twice-called Linear (two passes), a head."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3)
            self.bn = nn.BatchNorm2d(4)
            self.fc = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.bn(self.conv(x)))
            x = self.fc(x.mean(dim=(2, 3)))
            return self.head(self.fc(x))

    torch.manual_seed(0)
    return Fixture()


def _traces() -> Iterator[Any]:
    import torch

    import torchlens as tl

    torch.manual_seed(0)
    inputs = torch.randn(2, 3, 8, 8)
    yield tl.trace(_fixture_model().eval(), inputs)
    from torchlens.options import CaptureOptions

    trace = tl.trace(
        _fixture_model().train(),
        inputs,
        capture=CaptureOptions(
            save_grads=True,
            compute_input_output_distances=True,
            measure_python_peak_memory=True,
            measure_host_peak_memory=True,
        ),
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    yield trace


def _instances(trace: Any) -> Iterator[tuple[str, Any]]:
    report = trace.summary()
    yield "Trace", trace
    yield "SummaryReport", report
    yield "SummaryTotals", report.totals
    yield "CaptureFacts", report.capture
    for name, accessor in (
        ("Layer", trace.compute_layers),
        ("Op", trace.compute_ops),
        ("Module", trace.modules),
        ("ModuleCall", trace.module_calls),
        ("Param", trace.param_logs),
        ("Buffer", trace.buffers),
    ):
        for item in accessor:
            yield name, item


def _numeric_fields(obj: Any) -> Iterator[tuple[str, str]]:
    import inspect

    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            static = inspect.getattr_static(obj, name)
        except AttributeError:
            static = None
        if inspect.isfunction(static) or isinstance(static, (staticmethod, classmethod)):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - a raising property is not a numeric door
            value = None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        yield name, type(value).__name__


def live_doors() -> dict[str, str]:
    """Return ``{"Class.field": value_type}`` for every numeric door in scope."""

    doors: dict[str, set[str]] = defaultdict(set)
    for trace in _traces():
        try:
            for class_name, obj in _instances(trace):
                assert type(obj).__name__ == class_name, (class_name, type(obj))
                for field, value_type in _numeric_fields(obj):
                    doors[f"{class_name}.{field}"].add(value_type)
        finally:
            trace.cleanup()
    return {door: "|".join(sorted(types)) for door, types in sorted(doors.items())}


def load_rows(path: Path = TSV_PATH) -> list[dict[str, str]]:
    """Parse the committed census rows (``#`` lines are comments)."""

    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    for row in rows:
        assert tuple(row) == COLUMNS, f"bad columns in {path}: {tuple(row)}"
        assert row["kind"] in KINDS, row
        if row["kind"] == "quantity":
            assert row["status"] in STATUSES, row
            assert row["status"] != "red-needs-row" or row["oracle_ref"] != "-", row
    return rows


def quantity_status(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    """Collapse doors into quantities; a quantity takes its worst door status."""

    rank = {status: index for index, status in enumerate(STATUSES)}
    status: dict[str, str] = {}
    for row in rows:
        if row["kind"] != "quantity":
            continue
        current = status.get(row["quantity_id"])
        if current is None or rank[row["status"]] > rank[current]:
            status[row["quantity_id"]] = row["status"]
    return dict(sorted(status.items()))


def obligation_id(quantity_id: str) -> str:
    """Return the Table A id a quantity's row must carry."""

    return "OBL-Q-" + quantity_id.upper().replace("_", "-")


def registry_findings(rows: list[dict[str, str]]) -> list[str]:
    """Cross-check green quantities against Tables A and B, both directions."""

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tests.oracles._registry import load_registry, validate

    registry = load_registry()
    findings = list(validate(registry))
    obligations = {row.obligation_id: row for row in registry.obligations}
    bindings = {(row.obligation_id, row.door): row for row in registry.bindings}
    status = quantity_status(rows)
    for row in rows:
        if row["kind"] != "quantity" or status[row["quantity_id"]] != "green":
            continue
        oid = obligation_id(row["quantity_id"])
        cls, field = row["door"].split(".", 1)
        binding = bindings.get((oid, f"{DOOR_PATHS[cls]}.{field}"))
        if oid not in obligations:
            findings.append(f"{row['door']}: green without its Table A row {oid}")
        elif binding is None:
            findings.append(f"{row['door']}: green without a Table B binding to {oid}")
        elif binding.positive_control != row["oracle_ref"].split(" (")[0]:
            findings.append(
                f"{row['door']}: binding names {binding.positive_control!r}, not its oracle"
            )
    green = {obligation_id(q) for q, s in status.items() if s == "green"}
    for oid in sorted(o for o in obligations if o.startswith("OBL-Q-") and o not in green):
        findings.append(f"{oid}: Table A quantity row without a green census quantity")
    return findings


def drift_findings(live: dict[str, str], rows: Iterable[dict[str, str]]) -> list[str]:
    """Return one finding per live door without a row and per row without a live door."""

    rowed = {row["door"] for row in rows}
    unrowed = sorted(set(live) - rowed)
    stale = sorted(rowed - set(live))
    return [f"{door}: live numeric door without a census row" for door in unrowed] + [
        f"{door}: census row without a live door" for door in stale
    ]


def red_findings(rows: Iterable[dict[str, str]]) -> list[str]:
    """Return one finding per quantity whose worst door status is not green."""

    status = quantity_status(rows)
    return [f"{quantity}: {state}" for quantity, state in status.items() if state != "green"]


def census_findings(live: dict[str, str], rows: list[dict[str, str]]) -> list[str]:
    """Return every reason the zero-red bar fails: drift, red quantities, registry gaps."""

    return drift_findings(live, rows) + red_findings(rows) + registry_findings(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="print every red quantity")
    parser.add_argument("--new", action="store_true", help="print skeleton rows for unrowed doors")
    args = parser.parse_args(argv)

    live = live_doors()
    rows = load_rows()
    rowed = {row["door"] for row in rows}
    missing = sorted(set(live) - rowed)
    stale = sorted(rowed - set(live))
    if args.new:
        for door in missing:
            print("\t".join((door, live[door], "?", "?", "?", "-", "-")))
    quantities = quantity_status(rows)
    counts = defaultdict(int)
    for status in quantities.values():
        counts[status] += 1
    n_ident = sum(row["kind"] == "identifier" for row in rows)
    print(f"counted set: {COUNTED_SET}")
    print(f"doors: {len(live)} live, {len(rows)} rowed ({n_ident} identifiers excluded)")
    print(f"quantities: {len(quantities)}; " + ", ".join(f"{s}={counts[s]}" for s in STATUSES))
    print(f"red: {counts['red-needs-row'] + counts['red-needs-oracle']}")
    if args.list:
        for quantity, status in quantities.items():
            if status != "green":
                print(f"  {status}\t{quantity}")
    findings = registry_findings(rows)
    for finding in findings:
        print(f"REGISTRY: {finding}")
    if missing or stale:
        print(
            f"DRIFT: {len(missing)} unrowed live doors {missing[:10]}; {len(stale)} stale rows {stale[:10]}"
        )
    red = red_findings(rows)
    return 1 if (missing or stale or findings or red) else 0


if __name__ == "__main__":
    sys.exit(main())
