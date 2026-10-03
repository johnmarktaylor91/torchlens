"""Registry Tables A/B, the nine censuses, waivers, KNOWN-GAP mechanics (item 3).

Everything else binds into this; new deltas block from this merge (a set
difference cannot flake, D25). The H0 corruption plants prove the validation
gates can go red, and the waiver schema's refusals are live from the first
merge so the first real waiver cannot invent its own rules (D24).
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import replace
from pathlib import Path

import pytest

from ._censuses import (
    CENSUS_ROOTS,
    census_memoization_sites,
    census_options_fields,
    census_signature_params,
    load_schema_table,
)
from ._known_gaps import KnownGap, load_known_gaps, partition_monotone
from ._registry import ExposureBinding, Registry, load_registry, validate
from ._waivers import Waiver, load_waivers, validate_waiver

DATA_DIR = Path(__file__).resolve().parent / "data"

_REGEN_HINT = "regenerate consciously: python tests/oracles/_regen.py"


def _single_column_baseline(filename: str) -> frozenset[str]:
    """Load one single-column census baseline.

    Parameters
    ----------
    filename:
        Baseline under ``data/``.

    Returns
    -------
    frozenset[str]
        Row keys.
    """

    with (DATA_DIR / filename).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        return frozenset(row[0] for row in reader)


def test_registry_loads_and_validates_exactly() -> None:
    """The committed seed registry is valid: exact A/B join, closed vocab."""

    registry = load_registry()
    findings = validate(registry)
    assert findings == (), f"registry corruption: {findings}"
    assert registry.obligations, "Table A empty -- the registry lost its seed"
    assert registry.bindings, "Table B empty -- the registry lost its seed"


def test_every_census_root_is_described_and_baselined() -> None:
    """All nine generator roots exist, are typed, and name real baselines."""

    assert len(CENSUS_ROOTS) == 9
    assert {root.kind for root in CENSUS_ROOTS} == {"machine_walk", "schema_table"}
    for root in CENSUS_ROOTS:
        assert (DATA_DIR / root.baseline).exists(), (
            f"{root.root_id} names a missing baseline {root.baseline}"
        )


@pytest.mark.smoke_cells(
    "test_machine_census_matches_baseline[census_signature_params-signature_params.tsv]"
)
@pytest.mark.parametrize(
    ("census", "baseline"),
    [
        (census_options_fields, "options_fields.tsv"),
        (census_signature_params, "signature_params.tsv"),
        (census_memoization_sites, "memoization_sites.tsv"),
    ],
)
def test_machine_census_matches_baseline(census, baseline: str) -> None:
    """Each machine census diffs clean against its committed baseline.

    A NEW options field, entry parameter, or memoization site is a registry
    event (it needs a cache-key/witness decision, D14) and blocks until the
    baseline is regenerated consciously.
    """

    result = census()
    live = frozenset(result.rows)
    committed = _single_column_baseline(baseline)
    new, stale = sorted(live - committed), sorted(committed - live)
    assert not new, f"{result.root_id} NEW census rows {new}; {_REGEN_HINT}"
    assert not stale, f"{result.root_id} stale baseline rows {stale}; {_REGEN_HINT}"
    assert result.counted_set, "a census must publish its counted-set identity (D8)"


@pytest.mark.parametrize(
    ("filename", "required"),
    [
        ("state_axes.tsv", ("axis", "kind", "status")),
        ("qualifier_lattice.tsv", ("label", "trust_rank")),
        ("artifact_leaves.tsv", ("leaf", "door", "parser_status")),
        ("history_cells.tsv", ("cell", "definition", "mandatory_scope")),
    ],
)
def test_schema_table_validates(filename: str, required: tuple[str, ...]) -> None:
    """Wave-0 schema tables parse, are non-empty, and column-complete."""

    rows = load_schema_table(filename, required)
    assert rows, f"{filename} is empty -- schema plumbing must land populated"


def test_history_cells_are_exactly_the_four() -> None:
    """D16: the histories table ships with all four rows from wave 0."""

    rows = load_schema_table("history_cells.tsv", ("cell", "definition", "mandatory_scope"))
    assert [row["cell"] for row in rows] == ["H-CLEAN", "H-WARM", "H-FAILED", "H-BOTH"]


def test_artifact_leaf_doors_resolve() -> None:
    """CR7: every rendered-artifact leaf names a door that RESOLVES."""

    from ._deprecation import resolve_dotted

    for row in load_schema_table("artifact_leaves.tsv", ("leaf", "door", "parser_status")):
        assert resolve_dotted(row["door"]) is not None, f"{row['leaf']}: dead door {row['door']}"


def test_known_gap_manifest_is_schema_valid() -> None:
    """The manifest parses: unique ids, known predicates, dated, owned."""

    gaps = load_known_gaps()
    assert gaps, "the wave-0 manifest must carry the seeded gap rows"


def test_waiver_ledger_is_empty_and_schema_live() -> None:
    """Wave 0 ships ZERO waivers; the ledger parses and validates."""

    today = datetime.date.today()
    waivers = load_waivers()
    for waiver in waivers:
        assert validate_waiver(waiver, today) == (), waiver
    assert waivers == (), "a wave-0 waiver appeared; waivers need D24 review"


# ---------------------------------------------------------------------------
# H0 corruption plants: every checker must be able to go RED.
# ---------------------------------------------------------------------------


def test_plant_dangling_binding_goes_red() -> None:
    """PLANT: a Table B row citing an unknown obligation fails the join."""

    registry = load_registry()
    planted = replace(registry.bindings[0], binding_id="BND-PLANT", obligation_id="OBL-GHOST")
    findings = validate(Registry(registry.obligations, (*registry.bindings, planted)))
    assert any("OBL-GHOST" in finding for finding in findings), (
        "a dangling binding survived validation; the A/B join gate is dead"
    )


def test_plant_doorless_obligation_goes_red() -> None:
    """PLANT: a Table A row with no binding fails closure."""

    registry = load_registry()
    planted = replace(
        registry.obligations[0], obligation_id="OBL-DOORLESS", title="planted doorless"
    )
    findings = validate(Registry((*registry.obligations, planted), registry.bindings))
    assert any("OBL-DOORLESS" in finding for finding in findings)


def test_plant_parity_grade_cannot_satisfy_an_obligation() -> None:
    """PLANT (D4): an I0/parity evidence grade refuses at validation."""

    registry = load_registry()
    planted = replace(registry.obligations[0], obligation_id="OBL-PARITY", evidence_grade="I0")
    bound = replace(registry.bindings[0], binding_id="BND-PARITY", obligation_id="OBL-PARITY")
    findings = validate(Registry((*registry.obligations, planted), (*registry.bindings, bound)))
    assert any("I0" in finding and "OBL-PARITY" in finding for finding in findings), (
        "an I0 parity grade satisfied an obligation; internal agreement is not evidence"
    )


def test_plant_scale_sensitive_without_profile_goes_red() -> None:
    """PLANT (D23): scale-sensitive bindings need a NAMED profile."""

    registry = load_registry()
    planted = replace(
        registry.bindings[0],
        binding_id="BND-SCALE",
        scale_sensitive="yes",
        scale_profile_id="",
    )
    findings = validate(Registry(registry.obligations, (*registry.bindings, planted)))
    assert any("BND-SCALE" in finding for finding in findings)


def test_plant_expired_waiver_goes_red() -> None:
    """PLANT (D24): an expired waiver refuses."""

    waiver = Waiver(
        waiver_id="WVR-PLANT",
        created="2026-01-01",
        expires="2026-02-01",
        owner="planter",
        issue="ISSUE-1",
        scope="BND-W0-001",
        reason="plant",
        second_reviewer="reviewer",
        silent_wrong_risk="none",
    )
    findings = validate_waiver(waiver, today=datetime.date(2026, 8, 26))
    assert any("EXPIRED" in finding for finding in findings)


def test_plant_silent_wrong_waiver_refuses() -> None:
    """PLANT (D24): a waiver can NEVER authorize a silent wrong answer."""

    waiver = Waiver(
        waiver_id="WVR-PLANT2",
        created="2026-08-01",
        expires="2026-09-01",
        owner="planter",
        issue="ISSUE-2",
        scope="BND-W0-001",
        reason="plant",
        second_reviewer="reviewer",
        silent_wrong_risk="plausible",
    )
    findings = validate_waiver(waiver, today=datetime.date(2026, 8, 26))
    assert any("silent" in finding.lower() for finding in findings)


def test_plant_overlong_waiver_goes_red() -> None:
    """PLANT (D24): the 90-day lifetime ceiling holds."""

    waiver = Waiver(
        waiver_id="WVR-PLANT3",
        created="2026-08-01",
        expires="2026-12-31",
        owner="planter",
        issue="ISSUE-3",
        scope="BND-W0-001",
        reason="plant",
        second_reviewer="reviewer",
        silent_wrong_risk="none",
    )
    findings = validate_waiver(waiver, today=datetime.date(2026, 8, 26))
    assert any("90" in finding for finding in findings)


def test_plant_known_gap_monotone_both_directions() -> None:
    """PLANT (D25): both monotone failure directions register."""

    new, stale = partition_monotone(
        live_red=frozenset({"fresh-defect", "licensed-defect"}),
        licensed=frozenset({"licensed-defect", "healed-defect"}),
    )
    assert new == ("fresh-defect",), "a new unlicensed red slipped the manifest gate"
    assert stale == ("healed-defect",), "a healed gap row survived; monotone-down is dead"


def test_plant_unknown_gap_predicate_refuses(tmp_path: Path) -> None:
    """PLANT: a gap row citing an unknown predicate is a schema error."""

    manifest = tmp_path / "known_gaps.tsv"
    manifest.write_text(
        "gap_id\tpredicate\tkey\tdated\towner\tnote\n"
        "GAP-X\tno_such_predicate\tk\t2026-08-26\tme\tplant\n"
    )
    with pytest.raises(ValueError, match="unknown predicate"):
        load_known_gaps(manifest)


def test_known_gap_type_is_importable_dataclass() -> None:
    """The manifest row type is data, not prose."""

    gap = KnownGap("GAP-T", "surface_classification", "k", "2026-08-26", "me", "note")
    assert gap.predicate == "surface_classification"


def test_binding_schema_is_the_memo_column_set() -> None:
    """Item 3's column list is present on the binding schema by name."""

    from dataclasses import fields

    columns = {field.name for field in fields(ExposureBinding)}
    assert {
        "purity_mechanism",
        "scale_sensitive",
        "disclosure_kind",
        "positive_control",
        "classification",
        "invariance_witness",
        "history_cells",
        "scale_profile_id",
    } <= columns
