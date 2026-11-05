"""Pins of the 3.0 saved-format design page against the tree.

The design page names the modules that leave with the pickle carrier, the
modules that shrink to a door, the validators that become schema nodes or
cross-record invariants, and the ``format3`` modules that replace them. These
tests hold the page and the tree together: a module deleted before its page
row is flipped, or a validator renamed without the page following, fails here.
Builders flip a deletion row to "gone" (the word in the Lines column) as they
delete.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "maintainers" / "saved_format_3_design.md"
FORMAT3 = REPO / "torchlens" / "_io" / "format3"


def _page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(text: str, number: int) -> str:
    match = re.search(rf"^## {number}\. .*?(?=^## {number + 1}\. |\Z)", text, re.S | re.M)
    assert match is not None, f"design page has no section {number}"
    return match.group(0)


def _table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0].startswith("`") and cells[0].endswith(".py`"):
            rows.append(cells)
    return rows


def _deletion_rows() -> list[tuple[str, str]]:
    rows = _table_rows(_section(_page_text(), 10))
    return [(row[0].strip("`"), row[1]) for row in rows if row[0].startswith("`torchlens/")]


@pytest.mark.parametrize("path,lines", _deletion_rows(), ids=lambda value: str(value))
def test_every_module_the_page_names_is_present_or_marked_gone(path: str, lines: str) -> None:
    exists = (REPO / path).is_file()
    if lines == "gone":
        assert not exists, f"{path} is marked gone on the design page but is still in the tree"
    else:
        assert exists, f"{path} is named on the design page but is not in the tree"
        assert lines.replace(",", "").isdigit(), f"{path}: Lines cell {lines!r} is not a count"


def test_section_one_modules_exist_under_the_cap() -> None:
    rows = _table_rows(_section(_page_text(), 1))
    assert rows, "section 1 names no modules"
    for row in rows:
        name = row[0].strip("`")
        module = FORMAT3 / name
        assert module.is_file(), f"design page names format3/{name} which is absent"
        assert len(module.read_text(encoding="utf-8").splitlines()) <= 800, name


def test_section_one_table_names_the_whole_package() -> None:
    named = {row[0].strip("`") for row in _table_rows(_section(_page_text(), 1))}
    present = {path.name for path in FORMAT3.glob("*.py")} - {"__init__.py"}
    assert named == present, f"page {sorted(named)} vs tree {sorted(present)}"


_VALIDATOR_HOME = {
    "forgery_validation": REPO / "torchlens" / "_io" / "forgery_validation.py",
    "_load_validators": REPO / "torchlens" / "_io" / "_load_validators.py",
    "_forgery_identity_facts": REPO / "torchlens" / "_io" / "_forgery_identity_facts.py",
}


def _validator_names_on_page() -> set[str]:
    return set(re.findall(r"`((?:_validate_|check_)[a-z_]+)`", _section(_page_text(), 9)))


def test_every_validator_in_the_tree_is_classified_on_the_page() -> None:
    on_page = _validator_names_on_page()
    in_tree: set[str] = set()
    for home in _VALIDATOR_HOME.values():
        if home.is_file():
            in_tree |= set(
                re.findall(r"^def ((?:_validate_|check_)[a-z_]+)\(", home.read_text(), re.M)
            )
    if not in_tree:
        pytest.skip("the validator modules are gone; the page's section 9 lists are history")
    assert in_tree <= on_page, (
        f"validators missing from the design page: {sorted(in_tree - on_page)}"
    )
    assert on_page <= in_tree, (
        f"validators on the page but not in the tree: {sorted(on_page - in_tree)}"
    )
