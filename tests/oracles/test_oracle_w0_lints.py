"""The five wave-0 static lints, each with its plant (build item 4).

All sub-second except the numeral census (marked heavy: it AST-walks ~1,500
files in one go, well inside the 20 s heavy budget). None can flake: every
lint is a set difference over static parses plus D25 KNOWN-GAP mechanics.
"""

from __future__ import annotations

from pathlib import Path

from ._censuses import census_numerals
from ._known_gaps import licensed_keys, load_known_gaps, partition_monotone
from ._lints import (
    cache_key_uncovered,
    static_all_count,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Lint 1: cache-key set difference (D14).
# ---------------------------------------------------------------------------


def test_plant_new_option_field_blocks() -> None:
    """PLANT: an option name in no bucket registers as a new delta."""

    uncovered = frozenset({*cache_key_uncovered(), "CaptureOptions.planted_option"})
    licensed = licensed_keys(load_known_gaps(), "cache_key_coverage")
    new, _ = partition_monotone(uncovered, licensed)
    assert "CaptureOptions.planted_option" in new, (
        "a planted unkeyed option slipped the cache-key lint; the burden inversion is dead"
    )


# ---------------------------------------------------------------------------
# Lint 2: default-truth 3-source (D20).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lint 3: positive-control / unasserted-before (D7).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lint 4: denominator 3-root (D8).
# ---------------------------------------------------------------------------


def test_plant_denominator_drift_goes_red(tmp_path: Path) -> None:
    """PLANT: a doctored ``__all__`` diverges from the other roots."""

    import torchlens

    planted = tmp_path / "planted_init.py"
    planted.write_text('__all__ = ["only_one_name"]\n')
    assert static_all_count(planted) != len(torchlens.__all__), (
        "the static root failed to see a doctored __all__; root A is dead"
    )


# ---------------------------------------------------------------------------
# Lint 5: numeral census (D19).
# ---------------------------------------------------------------------------


def test_plant_numeral_is_found_and_stale_row_blocks(tmp_path: Path) -> None:
    """PLANT: the census sees a planted numeral; a stale row registers."""

    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("This claims 11600 models and 89 percent.\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "torchlens").mkdir()
    result = census_numerals(tmp_path)
    assert ("docs-md", "README.md", "11600") in result.rows
    assert ("docs-md", "README.md", "89") in result.rows
    live = frozenset(result.rows)
    baseline = frozenset({("docs-md", "README.md", "99999")})
    assert sorted(baseline - live), "a stale planted baseline row was not detected"


def test_numeral_floor_excludes_single_digits(tmp_path: Path) -> None:
    """The counted set excludes sub-floor numerals (disclosed, not silent)."""

    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "torchlens").mkdir()
    (tmp_path / "README.md").write_text("run 3 times on 2 GPUs with 512 tokens\n")
    rows = census_numerals(tmp_path).rows
    assert ("docs-md", "README.md", "512") in rows
    assert not any(key[2] in {"3", "2"} for key in rows)
