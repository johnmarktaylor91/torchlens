"""The five wave-0 static lints, each with its plant (build item 4).

All sub-second except the numeral census (marked heavy: it AST-walks ~1,500
files in one go, well inside the 20 s heavy budget). None can flake: every
lint is a set difference over static parses plus D25 KNOWN-GAP mechanics.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ._censuses import NUMERAL_COUNTED_SET, census_numerals
from ._known_gaps import licensed_keys, load_known_gaps, partition_monotone
from ._lints import (
    cache_config_keys,
    cache_key_uncovered,
    declared_ledger_constant,
    declared_ledger_length,
    load_dont_care,
    option_universe,
    static_all_count,
    unasserted_before_findings,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Lint 1: cache-key set difference (D14).
# ---------------------------------------------------------------------------


def test_cache_key_dont_care_rows_are_not_also_keyed() -> None:
    """A dont-care row for a field the key actually covers is stale."""

    keys = cache_config_keys()
    stale = sorted(field for field in load_dont_care() if field.split(".")[-1] in keys)
    assert not stale, f"dont-care rows shadow real key coverage: {stale}"


def test_plant_new_option_field_blocks() -> None:
    """PLANT: an option name in no bucket registers as a new delta."""

    uncovered = frozenset({*cache_key_uncovered(), "CaptureOptions.planted_option"})
    licensed = licensed_keys(load_known_gaps(), "cache_key_coverage")
    new, _ = partition_monotone(uncovered, licensed)
    assert "CaptureOptions.planted_option" in new, (
        "a planted unkeyed option slipped the cache-key lint; the burden inversion is dead"
    )


def test_option_universe_publishes_its_denominator() -> None:
    """D8: the lint's own universe publishes with counts."""

    universe = option_universe()
    keys = cache_config_keys()
    assert len(universe) > 100, "the option universe collapsed; wrong denominator"
    print(
        f"cache-key lint universe: {len(universe)} option/entry parameters; "
        f"{len(keys)} cache_config keys; "
        f"{len(cache_key_uncovered())} gap-rowed (owned, dated)"
    )


# ---------------------------------------------------------------------------
# Lint 2: default-truth 3-source (D20).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lint 3: positive-control / unasserted-before (D7).
# ---------------------------------------------------------------------------


def test_no_unasserted_before_snapshots_in_the_harness() -> None:
    """The oracle harness's own files carry no dead before-channels."""

    paths = tuple(sorted(Path(__file__).resolve().parent.glob("*.py")))
    findings = unasserted_before_findings(paths)
    assert findings == (), (
        f"dead measurement channels (captured before-values never read): "
        f"{findings} -- the vacuous-postcondition family (D7)"
    )


def test_plant_unasserted_before_is_found(tmp_path: Path) -> None:
    """PLANT: an unread ``*_before`` snapshot registers."""

    planted = tmp_path / "planted_module.py"
    planted.write_text(
        textwrap.dedent(
            """
            def probe(model):
                picklable_before = try_pickle(model)
                run(model)
                assert try_pickle(model)
            """
        )
    )
    findings = unasserted_before_findings((planted,))
    assert findings == ("planted_module.py::probe::picklable_before",), findings


def test_plant_read_before_is_not_flagged(tmp_path: Path) -> None:
    """PLANT (specificity): a READ before-value is not a finding."""

    planted = tmp_path / "planted_module.py"
    planted.write_text(
        textwrap.dedent(
            """
            def probe(model):
                picklable_before = try_pickle(model)
                assert picklable_before, "channel dead before the call"
                run(model)
                assert try_pickle(model)
            """
        )
    )
    assert unasserted_before_findings((planted,)) == ()


# ---------------------------------------------------------------------------
# Lint 4: denominator 3-root (D8).
# ---------------------------------------------------------------------------


def test_surface_denominator_agrees_across_roots() -> None:
    """The declared-surface denominator agrees from independent roots.

    Roots: static AST parse of the ``__all__`` literal, the runtime length,
    the declared-gate constant, and the hand-ledger length. A green gate
    over a wrong denominator (``PUBLIC_SURFACE_SIZE = 119`` guarding a set
    short by 45%) is the purest instance of a defense on the wrong set.
    """

    import torchlens

    roots = {
        "static-ast": static_all_count(),
        "runtime": len(torchlens.__all__),
        "declared-constant": declared_ledger_constant(),
        "hand-ledger": declared_ledger_length(),
    }
    assert len(set(roots.values())) == 1, f"denominator roots disagree: {roots}"


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


@pytest.mark.heavy
def test_numeral_census_baseline_mechanics() -> None:
    """The census runs; stale baseline rows block; new numerals disclose.

    Wave 0 lands the census and the dated legacy baseline. Stale rows fail
    (monotone-down). NEW numerals are DISCLOSED here as the Wave-2 work
    queue but do not block yet: FactRef enforcement is staged with H5
    (build items 13/20, owner F36) -- the wave-0 contract is the census and
    its baseline, not the ratchet.
    """

    import csv

    result = census_numerals()
    live = frozenset(result.rows)
    with (DATA_DIR / "numeral_baseline.tsv").open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        baseline = frozenset(tuple(row) for row in reader)
    stale = sorted(baseline - live)
    new = sorted(live - baseline)
    assert not stale, (
        f"stale numeral baseline rows (first 10): {stale[:10]} -- the numerals "
        "left the corpus; delete their rows (python tests/oracles/_regen.py)"
    )
    print(
        f"numeral census: {len(live)} keys [counted set: {NUMERAL_COUNTED_SET}]; "
        f"{len(new)} NEW since baseline (wave-2 FactRef queue): {new[:10]}"
    )


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
