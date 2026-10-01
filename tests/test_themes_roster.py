"""F12 roster pins: nine rows, tiers, subjects, compositions, memo settings.

The roster's exact settings are the memo's (trilabs themes memo section 3);
these pins hold the registry rows to the panel's ruling so a later edit
cannot silently move a settled cell (the dims collapse="none" 2-1 cell, the
three view pins, the transformer no-filter rule).
"""

from __future__ import annotations

import pytest

from torchlens.visualization import lenses
from torchlens.visualization.theme_registry import describe_lens, get_lens, list_lenses

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)


def test_roster_has_exactly_nine_rows() -> None:
    """The inclusion rule produced exactly nine rows; the tier split is 6+3."""

    assert lenses.ROSTER == (
        "overview",
        "blueprint",
        "debug",
        "speed",
        "dims",
        "transformer",
        "memory",
        "sequence",
        "compute",
    )
    assert set(lenses.CORE_LENSES) | set(lenses.EXTENDED_LENSES) == set(lenses.ROSTER)
    assert len(lenses.CORE_LENSES) == 6
    assert len(lenses.EXTENDED_LENSES) == 3
    registered = {row.name for row in list_lenses()}
    assert set(lenses.ROSTER) <= registered


def test_view_pins_are_exactly_three() -> None:
    """Only overview, blueprint, and sequence own a view member (memo s13.4)."""

    owners = {
        row.name for row in (get_lens(name) for name in lenses.ROSTER) if "vis_mode" in row.members
    }
    assert owners == {"overview", "blueprint", "sequence"}
    assert get_lens("overview").members["vis_mode"] == "rolled"
    assert get_lens("blueprint").members["vis_mode"] == "unrolled"
    assert get_lens("sequence").members["vis_mode"] == "unrolled"


def test_dims_v1_collapse_cell_is_none() -> None:
    """The dims 2-1 dissent cell: collapse='none' in v1, size channel pinned."""

    dims = get_lens("dims")
    assert dims.members["collapse"] == "none"
    assert dims.members["size_by"] == "dims"
    assert dims.members["scale"] == "sqrt"
    assert "color_by" not in dims.members


def test_blueprint_pins_todays_bare_draw_contract() -> None:
    """Blueprint is the fully PINNED escape row."""

    blueprint = get_lens("blueprint")
    assert blueprint.members["collapse"] == "none"
    assert blueprint.members["fold_repeats"] is False
    assert blueprint.members["show_containers"] is False


def test_perf_rows_declare_families_not_members() -> None:
    """Perf rows carry NO color_by member: N16 binds it per view at resolve."""

    assert lenses.PERF_FAMILIES == {"speed": "time", "memory": "bytes", "compute": "flops"}
    for name in lenses.PERF_FAMILIES:
        assert "color_by" not in get_lens(name).members, name


def test_transformer_never_sets_a_filter_and_leans_leftright() -> None:
    """The transformer row: no filter member, leftright prior, fold_repeats."""

    transformer = get_lens("transformer")
    assert "skip_fn" not in transformer.members
    assert transformer.members["direction"] == "leftright"
    assert transformer.members["fold_repeats"] is True


def test_subjects_table_matches_refusal_taxonomy() -> None:
    """Transformer and sequence refuse on SUBJECT; nobody else does."""

    assert lenses.LENS_SUBJECTS == {
        "transformer": "attention_structure",
        "sequence": "multi_pass",
    }


def test_compositions_are_the_three_named_recipes() -> None:
    """Three compositions, each naming a registered lens plus channel kwargs."""

    assert set(lenses.COMPOSITIONS) == {"runtime_storage", "vision", "debug_edge_shapes"}
    assert lenses.COMPOSITIONS["runtime_storage"]["lens"] == "speed"
    assert lenses.COMPOSITIONS["runtime_storage"]["size_by"] == "bytes"
    assert lenses.COMPOSITIONS["vision"]["lens"] == "dims"
    for recipe in lenses.COMPOSITIONS.values():
        get_lens(recipe["lens"])  # every recipe names a registered row


def test_describe_teaches_exact_settings() -> None:
    """describe() prints real draw parameters with exact values (N12)."""

    text = describe_lens("dims")
    assert "size_by='dims'" in text
    assert "what shape is the data?" in text


@pytest.mark.parametrize("name", ["speed", "memory", "compute", "dims"])
def test_evidence_rows_declare_headlines(name: str) -> None:
    """Evidence-refusing rows declare their headline family."""

    assert get_lens(name).headline_evidence is not None
