"""Graduation items G1 to G6 on synthetic scorecards (SPEC 5; criterion A22).

Each item is driven to met and unmet with hand-built scorecards and a
synthetic repository, and the report embeds in the record as evidence; no
record carries a maturity label.
"""

from __future__ import annotations

from pathlib import Path

from tests.conformance import GraduationItem
from tests.conformance._graduation import GRADUATION_IDS, graduation_items, repository_root
from tests.conformance._scorecard import (
    SCORECARD_VERSION,
    PreviewScorecard,
    ScorecardCell,
    _with_digest,
)
from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS

_ROW_CASES = (
    "c1_capture_completes",
    "c1_reference_parity",
    "c1_deterministic_repeat",
    "c1_graph_integrity",
    "c1_shape_metadata_matches_payload",
    "c1_no_silent_gap",
    "c1_validation_passes",
    "c1_validation_names_reason",
)
_C2_CASES = (
    "c2_roundtrip_completes",
    "c2_reload_is_distinct",
    "c2_structure_equal",
    "c2_payload_values_equal",
    "c2_validation_after_load",
)
_C0_CASES = (
    "c0_capability_flags_boolean",
    "c0_provider_identity",
    "c0_surface_complete",
    "c0_protocol_complete",
    "c0_framework_requirement_consistent",
    "c0_refusal_codes_registered",
    "c0_capability_table_complete",
)


def _cells(
    families: dict[str, str], *, state: str = "passed", c0_state: str = "passed"
) -> tuple[ScorecardCell, ...]:
    """Every cell for ``families`` (family -> group), all in one state."""

    rows = [ScorecardCell("C0", case_id, "", c0_state) for case_id in _C0_CASES]
    for family, group in families.items():
        rows += [ScorecardCell("C1", case_id, family, state, group=group) for case_id in _ROW_CASES]
    rows.append(ScorecardCell("C1", "c1_capabilities_consumed", "", state))
    rows.append(ScorecardCell("C1", "c1_refusal_fastlog", "", state))
    for family, group in families.items():
        rows += [ScorecardCell("C2", case_id, family, state, group=group) for case_id in _C2_CASES]
    return tuple(rows)


def _card(
    backend: str = "stub",
    cells: tuple[ScorecardCell, ...] | None = None,
    *,
    capabilities: dict[str, str] | None = None,
    requirement: dict[str, str] | None = None,
) -> PreviewScorecard:
    """A synthetic scorecard; defaults describe a backend that meets G1 and G2."""

    cells = cells if cells is not None else _cells({"mlp": "config_built", "cnn": "config_built"})
    summary = dict.fromkeys(REFUSABLE_CAPABILITY_FLAGS, "supported")
    summary["fastlog"] = "unsupported"
    summary.update(capabilities or {})
    return _with_digest(
        PreviewScorecard(
            SCORECARD_VERSION,
            backend,
            "9.9",
            "suite",
            "2.0.0",
            cells,
            {"denominator": len(cells)},
            "C2",
            framework_requirement=requirement
            if requirement is not None
            else {
                "minimum": "1.0",
                "below": "",
                "tested_through": "9.9",
                "pinned": "9.9",
                "extra": "stub",
            },
            capabilities=summary,
            roster_groups={"config_built": 2},
        )
    )


def _item(items: tuple[GraduationItem, ...], item_id: str) -> GraduationItem:
    """One item by id."""

    return next(item for item in items if item.id == item_id)


def _repo(tmp_path: Path, backend: str = "stub", *, pinned: str = "9.9") -> Path:
    """A synthetic repository that satisfies G4, G5 and G6 for ``backend``."""

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "torchlens"\n', encoding="utf-8")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "tests.yml").write_text(
        f"on:\n  pull_request:\njobs:\n  backend-protocol:\n    steps:\n"
        f"      - run: pip install {backend}=={pinned}\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs" / "backends"
    docs.mkdir(parents=True)
    (docs / f"{backend}.md").write_text(
        "# Stub\n\n## Tutorial\n\n```python\nimport torchlens\n```\n\n## Can and cannot\n\n| a | b |\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "reference").mkdir()
    (tmp_path / "docs" / "reference" / "glossary.md").write_text(
        f"**torchlens.backends.{backend}.Thing**\n: stable.\n", encoding="utf-8"
    )
    package = tmp_path / "torchlens" / "backends" / backend
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Stub backend."""\n', encoding="utf-8")
    return tmp_path


def test_all_six_items_met_on_a_complete_backend(tmp_path: Path) -> None:
    """A backend with every cell passed, two clean ecosystem families and the files meets G1 to G6."""

    families = {
        "mlp": "config_built",
        "cnn": "config_built",
        "bert": "ecosystem",
        "vit": "ecosystem",
    }
    card = _card(cells=_cells(families))
    items = graduation_items(card, reference=card, repo_root=_repo(tmp_path))
    assert [item.id for item in items] == list(GRADUATION_IDS)
    assert all(item.met for item in items), [(i.id, i.detail) for i in items]


def test_g1_unmet_on_failed_not_run_untested_or_missing_reference(tmp_path: Path) -> None:
    """G1 needs zero failed and not_run per-change cells, torch coverage and no untested flag."""

    repo = _repo(tmp_path)
    card = _card()
    assert _item(graduation_items(card, reference=card, repo_root=repo), "G1").met
    missing_reference = _item(graduation_items(card, reference=None, repo_root=repo), "G1")
    assert not missing_reference.met and "no torch reference" in missing_reference.detail
    failed_cells = list(card.cells)
    failed_cells[8] = ScorecardCell("C1", _ROW_CASES[1], "mlp", "failed", "x", "config_built")
    failing = _card(cells=tuple(failed_cells))
    item = _item(graduation_items(failing, reference=card, repo_root=repo), "G1")
    assert not item.met and "1 failed cells" in item.detail
    not_run_cells = list(card.cells)
    not_run_cells[8] = ScorecardCell("C1", _ROW_CASES[1], "mlp", "not_run", "x", "config_built")
    item = _item(
        graduation_items(_card(cells=tuple(not_run_cells)), reference=card, repo_root=repo), "G1"
    )
    assert not item.met and "not_run" in item.detail
    untested = _card(capabilities={"rng_replay": "untested"})
    item = _item(graduation_items(untested, reference=card, repo_root=repo), "G1")
    assert not item.met and "untested" in item.detail
    # An ecosystem row that did not run never blocks G1 (it is G3's evidence).
    bert_rows = tuple(
        cell for cell in _cells({"bert": "ecosystem"}, state="not_run") if cell.family == "bert"
    )
    eco = _card(cells=_cells({"mlp": "config_built"}) + bert_rows)
    assert _item(graduation_items(eco, reference=card, repo_root=repo), "G1").met
    # A torch-passed cell this backend refuses typed still satisfies G1.
    unsupported = [
        ScorecardCell(c.tier, c.case_id, c.family, "unsupported", "unsupported[x]", c.group)
        if c.case_id == "c1_validation_passes"
        else c
        for c in card.cells
    ]
    assert _item(
        graduation_items(_card(cells=tuple(unsupported)), reference=card, repo_root=repo), "G1"
    ).met
    # A torch-passed cell this backend lacks entirely does not.
    reference = _card(
        cells=card.cells
        + (ScorecardCell("C1", "c1_only_torch_has", "mlp", "passed", group="config_built"),)
    )
    item = _item(graduation_items(card, reference=reference, repo_root=repo), "G1")
    assert not item.met and "c1_only_torch_has" in item.detail


def test_g1_counts_a_supported_capability_in_place_of_torchs_refusal_cell(
    tmp_path: Path,
) -> None:
    """Torch passes ``c1_refusal_<flag>`` by refusing a flag it lacks; a backend that supports
    the flag has no refusal cell, and that absence is covered only when the flag is supported.
    """

    repo = _repo(tmp_path)
    card = _card()
    flag = "intermediate_derived_grads"
    refusal = ScorecardCell("C1", f"c1_refusal_{flag}", "", "passed")
    reference = _card(cells=card.cells + (refusal,))
    assert card.capabilities[flag] == "supported"
    assert _item(graduation_items(card, reference=reference, repo_root=repo), "G1").met
    lacking = _card(capabilities={flag: "unsupported"})
    item = _item(graduation_items(lacking, reference=reference, repo_root=repo), "G1")
    assert not item.met and f"c1_refusal_{flag}" in item.detail
    # A trace-option refusal cell is never covered by a capability flag.
    option = ScorecardCell("C1", f"c1_refusal_trace_option_{flag}", "", "passed")
    item = _item(
        graduation_items(card, reference=_card(cells=card.cells + (option,)), repo_root=repo),
        "G1",
    )
    assert not item.met and f"c1_refusal_trace_option_{flag}" in item.detail


def test_g2_needs_the_flag_and_every_validation_cell(tmp_path: Path) -> None:
    """G2 is unmet when replay is unsupported or a validation cell is not passed on a row."""

    repo = _repo(tmp_path)
    card = _card()
    assert _item(graduation_items(card, reference=card, repo_root=repo), "G2").met
    item = _item(
        graduation_items(
            _card(capabilities={"has_validation_replay": "unsupported"}),
            reference=card,
            repo_root=repo,
        ),
        "G2",
    )
    assert not item.met and "has_validation_replay" in item.detail
    cells = [
        ScorecardCell(c.tier, c.case_id, c.family, "unsupported", "u", c.group)
        if c.case_id == "c2_validation_after_load" and c.family == "cnn"
        else c
        for c in card.cells
    ]
    item = _item(graduation_items(_card(cells=tuple(cells)), reference=card, repo_root=repo), "G2")
    assert not item.met and "c2_validation_after_load" in item.detail


def test_g3_counts_ecosystem_families_with_every_cell_passed(tmp_path: Path) -> None:
    """G3 needs two ecosystem families clean through C1 and C2."""

    repo = _repo(tmp_path)
    card = _card()
    item = _item(graduation_items(card, reference=card, repo_root=repo), "G3")
    assert not item.met and "0 ecosystem families" in item.detail
    one = _card(cells=_cells({"mlp": "config_built", "bert": "ecosystem"}))
    assert not _item(graduation_items(one, reference=card, repo_root=repo), "G3").met
    two = _card(cells=_cells({"mlp": "config_built", "bert": "ecosystem", "vit": "ecosystem"}))
    assert _item(graduation_items(two, reference=card, repo_root=repo), "G3").met
    broken = tuple(
        ScorecardCell(c.tier, c.case_id, c.family, "not_run", "weights not cached", c.group)
        if c.family == "vit" and c.case_id == "c2_structure_equal"
        else c
        for c in two.cells
    )
    assert not _item(
        graduation_items(_card(cells=broken), reference=card, repo_root=repo), "G3"
    ).met


def test_g4_g5_g6_read_the_repository(tmp_path: Path) -> None:
    """The file-based items follow the workflow, the docs page and the glossary."""

    card = _card()
    unavailable = graduation_items(card, reference=card, repo_root=tmp_path / "nowhere")
    for item_id in ("G4", "G5", "G6"):
        item = _item(unavailable, item_id)
        assert not item.met and ("unavailable" in item.detail or "missing" in item.detail), item
    repo = _repo(tmp_path)
    items = graduation_items(card, reference=card, repo_root=repo)
    assert all(_item(items, item_id).met for item_id in ("G4", "G5", "G6")), items
    # G4: the pinned version must be what the pull_request workflow installs.
    other_pin = _card(
        requirement={
            "minimum": "1.0",
            "below": "",
            "tested_through": "9.9",
            "pinned": "8.8",
            "extra": "stub",
        }
    )
    item = _item(graduation_items(other_pin, reference=card, repo_root=repo), "G4")
    assert not item.met and "8.8" in item.detail
    no_range = _card(
        requirement={
            "minimum": "1.0",
            "below": "",
            "tested_through": "",
            "pinned": "9.9",
            "extra": "stub",
        }
    )
    item = _item(graduation_items(no_range, reference=card, repo_root=repo), "G4")
    assert not item.met and "tested_through" in item.detail
    inconsistent = _card(
        cells=tuple(
            ScorecardCell(c.tier, c.case_id, c.family, "failed", "x", c.group)
            if c.case_id == "c0_framework_requirement_consistent"
            else c
            for c in card.cells
        )
    )
    assert not _item(graduation_items(inconsistent, reference=card, repo_root=repo), "G4").met
    # G5: both sections are required.
    page = repo / "docs" / "backends" / "stub.md"
    page.write_text("# Stub\n\n## Tutorial\n", encoding="utf-8")
    item = _item(graduation_items(card, reference=card, repo_root=repo), "G5")
    assert not item.met and "Can and cannot" in item.detail
    page.unlink()
    assert "missing" in _item(graduation_items(card, reference=card, repo_root=repo), "G5").detail
    # G6: an unstable marker in the package or a glossary without the backend fails.
    (repo / "torchlens" / "backends" / "stub" / "api.py").write_text(
        '"""Spellings DOCUMENTED-UNSTABLE."""\n', encoding="utf-8"
    )
    item = _item(graduation_items(card, reference=card, repo_root=repo), "G6")
    assert not item.met and "DOCUMENTED-UNSTABLE" in item.detail
    (repo / "torchlens" / "backends" / "stub" / "api.py").unlink()
    (repo / "docs" / "reference" / "glossary.md").write_text("nothing here\n", encoding="utf-8")
    item = _item(graduation_items(card, reference=card, repo_root=repo), "G6")
    assert not item.met and "glossary" in item.detail


def test_repository_root_is_found_from_the_package_and_absent_elsewhere(tmp_path: Path) -> None:
    """The root detector finds this checkout and returns ``None`` outside one."""

    root = repository_root()
    assert root is not None and (root / "pyproject.toml").is_file()
    assert repository_root(tmp_path / "deep" / "er") is None
