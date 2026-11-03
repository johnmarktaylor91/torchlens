"""The glossary names every public parameter (``tools/name_lockstep/glossary_params.py``).

The glossary is the canonical API spec, so a caller reads there which keywords a callable
takes: ``Bundle.rank_changes(baseline, k=10)``, never a guessed ``top_k=``. A public
parameter with no keyword row naming its callable fails here; add the row (the backfill
tool renders it: ``python -m name_lockstep.glossary_backfill DUMP.json`` with
``PYTHONPATH=tools``) in the same change that adds the parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import glossary_backfill, glossary_params  # noqa: E402

GLOSSARY = REPO_ROOT / "docs" / "reference" / "glossary.md"

#: Public parameters allowed to stay unnamed, each with its reason. Keep it empty: a
#: parameter that needs no row is not public (give it a leading underscore instead).
ALLOWED_UNNAMED: dict[tuple[str, str], str] = {}

TABLE = """\
### Keywords
| Spelling | Meaning |
| --- | --- |
| `k=` | keyword of `Bundle.rank_changes`, `tl.top_changed`: rows to return |
| `metric=` | the pairwise metric; see `Bundle.rank_changes` |
| `seed=` | constructor keyword of `tl.errors.Base` and the classes that inherit it |
"""


def _row(callable_: str, param: str, **extra: object) -> dict[str, object]:
    """Build one dumped parameter row."""

    row: dict[str, object] = {
        "callable": callable_,
        "aliases": [callable_],
        "defined_by": callable_,
        "constructor": False,
        "param": param,
        "kind": "keyword_only",
        "default": "10",
        "meaning": "",
    }
    row.update(extra)
    return row


def test_keyword_row_names_only_the_owners_after_keyword_of() -> None:
    """A spelling a row's meaning mentions in passing does not name an owner."""

    rows = [
        _row("Bundle.rank_changes", "k"),
        _row("tl.top_changed", "k"),
        _row("Bundle.rank_changes", "metric"),
        _row("Bundle.rank_changes", "top_k"),
    ]
    gaps = glossary_params.missing(rows, TABLE)
    assert [(g["callable"], g["param"], g["keyword_row"]) for g in gaps] == [
        ("Bundle.rank_changes", "metric", True),
        ("Bundle.rank_changes", "top_k", False),
    ]


def test_an_alias_or_the_constructor_owner_names_the_callable() -> None:
    """A dotted tail, a ``tl.`` path or an inherited constructor's owner counts."""

    rows = [
        _row("tl.bundle.Bundle.rank_changes", "k", aliases=["tl.bundle.Bundle.rank_changes"]),
        _row(
            "tl.errors.Child",
            "seed",
            aliases=["tl.errors.Child", "tl.errors.Base"],
            defined_by="tl.errors.Base",
        ),
    ]
    assert glossary_params.missing(rows, TABLE) == []


def test_backfill_extends_existing_rows_and_adds_sorted_new_ones() -> None:
    """The writer names every gap, keeps the table sorted and renders kind and default."""

    rows = [
        _row("Bundle.rank_changes", "k"),
        _row("tl.top_changed", "k"),
        _row("EffectsView.rank_changes", "k", kind="positional_or_keyword", default="None"),
        _row("Bundle.rank_changes", "baseline", default=None, meaning="Baseline ``name``."),
        _row("tl.errors.Base", "seed", constructor=True),
        _row("tl.errors.Child", "seed", aliases=["tl.errors.Child"], defined_by="tl.errors.Base"),
    ]
    text, extended, added = glossary_backfill.backfill(TABLE, rows)
    assert (extended, added) == (2, 1)
    assert glossary_params.missing(rows, text) == []
    lines = text.splitlines()
    assert lines[3] == (
        "| `baseline=` | keyword of `Bundle.rank_changes` (keyword-only; required): "
        "Baseline `name` |"
    )
    assert lines[4].endswith("; keyword of `EffectsView.rank_changes` (default `None`) |")
    assert "constructor keyword of `tl.errors.Base` and the classes that inherit" in lines[6]


@pytest.mark.heavy
def test_every_public_parameter_has_a_glossary_row() -> None:
    """Every parameter a caller can pass by name is named by a glossary keyword row."""

    rows = glossary_params.public_parameters(meanings=False)
    assert len(rows) > 1000, "the public walk reached too little of the package"
    gaps = [
        f"{gap['callable']}({gap['param']}=)"
        for gap in glossary_params.missing(rows, GLOSSARY.read_text(encoding="utf-8"))
        if (gap["callable"], gap["param"]) not in ALLOWED_UNNAMED
    ]
    assert not gaps, (
        f"{len(gaps)} public parameters have no glossary keyword row naming their callable "
        f"(first 20): {gaps[:20]}. Add a row to docs/reference/glossary.md, ### Keywords."
    )


def test_a_row_naming_a_keyword_the_callable_lacks_is_stale() -> None:
    """``k=`` on ``Bundle.rank_changes`` is stale once the code takes ``top_k=`` instead."""

    rows = [_row("Bundle.rank_changes", "top_k"), _row("tl.top_changed", "k")]
    assert glossary_params.stale(rows, TABLE) == [("Bundle.rank_changes", "k")]


@pytest.mark.heavy
def test_no_glossary_keyword_row_names_a_parameter_the_code_lacks() -> None:
    """A keyword row never says a public callable takes a keyword it does not take."""

    rows = glossary_params.public_parameters(meanings=False)
    wrong = glossary_params.stale(rows, GLOSSARY.read_text(encoding="utf-8"))
    assert not wrong, f"glossary keyword rows name parameters the code lacks: {wrong[:20]}"


def test_allowlist_entries_carry_reasons() -> None:
    """Any allowlisted unnamed parameter states why."""

    assert all(reason.strip() for reason in ALLOWED_UNNAMED.values())
