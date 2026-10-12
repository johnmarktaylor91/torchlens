"""Keyed union-merged baselines refuse a duplicate key instead of passing on row order."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.composition_expectations._keyed_tsv import parse_keyed_tsv

DATA_DIR = Path(__file__).resolve().parent / "data"

KEYED_UNION_FILES = (
    ("diagnostic_ratchets.tsv", "metric"),
    ("universe_cardinalities.tsv", "universe_id"),
    ("harvest_counts.tsv", "metric"),
)


@pytest.mark.parametrize(("filename", "header_key"), KEYED_UNION_FILES)
def test_committed_keyed_baselines_have_unique_keys(filename: str, header_key: str) -> None:
    path = DATA_DIR / filename
    rows = parse_keyed_tsv(
        path.read_text(encoding="utf-8"), filename=filename, header_key=header_key
    )
    assert rows, f"{filename} parsed empty"


def test_keyed_union_files_are_union_merged() -> None:
    """The guard exists because these files merge as a union; keep them listed."""

    attributes = (DATA_DIR.parents[2] / ".gitattributes").read_text(encoding="utf-8")
    for filename, _header in KEYED_UNION_FILES:
        assert f"tests/composition_expectations/data/{filename} merge=union" in attributes


def test_duplicate_key_from_a_union_merge_raises_naming_both_values() -> None:
    planted = "# comment\nmetric\tcount\ns17_total_sites\t3444\nother\t1\ns17_total_sites\t3443\n"
    with pytest.raises(AssertionError) as excinfo:
        parse_keyed_tsv(planted, filename="planted.tsv", header_key="metric")
    message = str(excinfo.value)
    assert "'s17_total_sites'" in message
    assert "'3444'" in message and "'3443'" in message
    assert "train value plus your delta" in message
