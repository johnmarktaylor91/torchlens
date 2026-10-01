"""The SG-regression manifest gate (compo memo stratum 3; F36 waves A-D).

"No SG number without a node": all fifty DIGEST-AUDIT section-1 silent gaps
are enumerated in ``data/sg_manifest.tsv``. Regression-node rows must name a
test node that EXISTS on disk (file, or file::test verified against source);
ledgered-open rows are the burn-down queue whose count only shrinks -- the
frozen ceiling below is the ratchet, and lowering it is the ONLY sanctioned
edit direction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "pure-data manifest gate (no products)"

DATA_PATH = Path(__file__).resolve().parent / "data" / "sg_manifest.tsv"
REPO_ROOT = Path(__file__).resolve().parents[2]

#: FROZEN CEILING (lane law: burns DOWN only). DESIGN: 32 of 50 gaps had no
#: dedicated regression node at the F36 freeze (2026-08-30); each fix lane
#: that lands a node flips its row to regression-node and LOWERS this
#: number in the same commit. Raising it requires a new DIGEST-AUDIT row,
#: which means a new SG id, not a bigger ceiling.
LEDGERED_OPEN_CEILING = 32

EXPECTED_SG_IDS = tuple(f"SG#{n}" for n in range(1, 51))
DISPOSITIONS = {"regression-node", "ledgered-open"}


def _load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in DATA_PATH.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        assert len(parts) == len(header), f"malformed manifest row: {line!r}"
        rows.append(dict(zip(header, parts, strict=True)))
    assert header == ["sg_id", "disposition", "node", "owner", "title"]
    return rows


def test_every_sg_number_has_exactly_one_row() -> None:
    """The fifty-row list is closed: SG#1..SG#50, once each, no strays."""

    rows = _load_rows()
    ids = [row["sg_id"] for row in rows]
    assert ids == list(EXPECTED_SG_IDS), (
        "the SG manifest must enumerate SG#1..SG#50 in order, exactly once"
        f" each; got {len(ids)} rows"
    )


def test_rows_carry_disposition_owner_and_title() -> None:
    """Schema teeth: closed dispositions, named owner, human title."""

    for row in _load_rows():
        assert row["disposition"] in DISPOSITIONS, row
        assert row["owner"], f"{row['sg_id']}: no owner"
        assert len(row["title"]) >= 10, f"{row['sg_id']}: title too thin to identify the gap"


def _node_exists(node: str) -> bool:
    if "::" in node:
        path_part, test_name = node.split("::", 1)
        path = REPO_ROOT / path_part
        if not path.is_file():
            return False
        return bool(re.search(rf"def {re.escape(test_name)}\b", path.read_text()))
    return (REPO_ROOT / node).is_file()


def test_regression_nodes_exist_on_disk() -> None:
    """A regression-node row whose node vanished is a broken tripwire."""

    missing = [
        f"{row['sg_id']} -> {row['node']}"
        for row in _load_rows()
        if row["disposition"] == "regression-node" and not _node_exists(row["node"])
    ]
    assert not missing, f"SG regression nodes missing from the tree: {missing}"


def test_open_probe_nodes_exist_on_disk() -> None:
    """Open rows MAY carry an auto-probe; a named probe must exist."""

    missing = [
        f"{row['sg_id']} -> {row['node']}"
        for row in _load_rows()
        if row["disposition"] == "ledgered-open" and row["node"] and not _node_exists(row["node"])
    ]
    assert not missing, f"SG auto-probe nodes missing from the tree: {missing}"


def test_ledgered_open_count_only_burns_down() -> None:
    """The monotone ratchet: open rows <= the frozen ceiling, and the
    ceiling itself must be re-trued DOWNWARD when rows flip (a stale slack
    ceiling hides regressions exactly like a stale floor)."""

    open_rows = [row for row in _load_rows() if row["disposition"] == "ledgered-open"]
    count = len(open_rows)
    assert count <= LEDGERED_OPEN_CEILING, (
        f"{count} ledgered-open SG rows exceed the frozen ceiling"
        f" {LEDGERED_OPEN_CEILING}: an SG gap was REOPENED (or a new row was"
        " misfiled -- new gaps get new DIGEST-AUDIT ids, never a raise here)"
    )
    assert count == LEDGERED_OPEN_CEILING, (
        f"only {count} open rows remain but the ceiling still reads"
        f" {LEDGERED_OPEN_CEILING}: lower the ceiling in the SAME commit that"
        " flips a row (stale slack is how burn-downs silently stall)"
    )
