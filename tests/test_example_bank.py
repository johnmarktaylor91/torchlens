"""The example bank (``torchlens/_docs/example_bank/``) runs, is indexed and covers the surface.

The bank holds small snippets agents copy: one task each, grouped by object, with the
printed output as a ``# Output:`` comment. ``tools/docs/example_bank.py`` parses and runs
them. This file checks:

* every example parses, has a unique id and a known fixture, spells in its code each name
  it claims to cover, has at most 15 code lines and shows output other than ``...``;
* ``INDEX.md`` is current;
* every example runs after its fixture and prints its ``# Output:`` comment (one cell per
  example; the language-model ones are ``heavy``, the toy ones fit the unmarked budget);
* every public name the bank owes an example (root ``tl.*`` names and the members of the
  core objects) has one, except the names in :data:`EXCLUDED`, each with a reason. The
  exclusion list is checked both ways, so an entry whose name gains an example or leaves
  the surface fails too.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "docs"))

import example_bank  # noqa: E402

#: Owed public names the bank deliberately has no example for: ``name -> reason``.
EXCLUDED: dict[str, str] = {}

_EXAMPLES = example_bank.load_examples()
_FIXTURES = example_bank.load_fixtures()
#: Fixtures whose examples cost a heavy-tier budget (a transformer capture).
_HEAVY_FIXTURES = frozenset({"lm"})


def _cells() -> list[Any]:
    """One parameter per example, heavy-marked when its fixture is heavy."""

    return [
        pytest.param(
            example,
            id=example.id,
            marks=[pytest.mark.heavy] if example.fixture in _HEAVY_FIXTURES else [],
        )
        for example in _EXAMPLES
    ]


def test_examples_parse_with_unique_ids_and_known_fixtures() -> None:
    """Ids are unique, fixtures exist, kinds are known and wrong examples pair with a right."""

    ids = [example.id for example in _EXAMPLES]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})
    assert {example.fixture for example in _EXAMPLES} <= set(_FIXTURES)
    assert {example.kind for example in _EXAMPLES} <= {"example", "wrong", "right"}
    wrong = {e.id.removesuffix("-wrong") for e in _EXAMPLES if e.kind == "wrong"}
    right = {e.id.removesuffix("-right") for e in _EXAMPLES if e.kind == "right"}
    assert wrong == right, f"unpaired trap examples: {sorted(wrong ^ right)}"
    assert all(example.shows for example in _EXAMPLES)


def test_examples_spell_the_names_they_cover() -> None:
    """A covered name appears in the code, the code is short and the output shows something."""

    problems = [p for example in _EXAMPLES for p in example_bank.cover_problems(example)]
    assert not problems, "\n".join(problems)


def test_index_is_current() -> None:
    """INDEX.md matches the examples; regenerate with ``example_bank.py index``."""

    rendered = example_bank.render_index(_EXAMPLES)
    assert example_bank.INDEX.read_text(encoding="utf-8") == rendered


@pytest.mark.parametrize("example", _cells())
def test_example_runs_and_prints_its_output(example: Any, tmp_path: Path) -> None:
    """The example runs after its fixture and prints what its output comment shows."""

    result = example_bank.run_example(example, _FIXTURES, tmp_path)
    if result.status == "skipped":
        pytest.skip(f"needs {result.detail}")
    assert result.status == "ok", result.detail


@functools.lru_cache(maxsize=1)
def _inventory() -> tuple[dict[str, Any], ...]:
    """Build the lockstep inventory once per process."""

    return tuple(example_bank._inventory_rows())


@pytest.mark.heavy
def test_every_owed_public_name_has_an_example() -> None:
    """Root names and core-object members each have an example or an excluded reason."""

    rows = list(_inventory())
    missing = [
        f"{name} ({kind})"
        for name, kind, ids in example_bank.coverage_rows(_EXAMPLES, rows)
        if not ids and name not in EXCLUDED
    ]
    assert not missing, f"{len(missing)} owed names without an example:\n" + "\n".join(missing)


@pytest.mark.heavy
def test_exclusions_are_current_and_covers_are_real_names() -> None:
    """Excluded names are owed and uncovered; every covered name is on the surface."""

    rows = list(_inventory())
    table = {name: ids for name, _, ids in example_bank.coverage_rows(_EXAMPLES, rows)}
    stale = [name for name in EXCLUDED if name not in table or table[name]]
    assert not stale, f"excluded names now covered or gone: {stale}"
    known = example_bank.known_names(rows)
    unknown = sorted({n for e in _EXAMPLES for n in e.covers if n not in known})
    assert not unknown, f"covered names not on the public surface: {unknown}"
