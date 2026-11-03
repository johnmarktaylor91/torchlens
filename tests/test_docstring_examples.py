"""Every public function and class carries a runnable NumPy ``Examples`` section.

The docstring standard (docs/agent-reference/conventions.md, "Docstrings") puts one
runnable example on every public function and class. ``tools/docstring_examples.py``
lists those symbols from the lockstep inventory, reads each ``Examples`` section as
doctest examples and runs them in the ambient namespace the docs pages also assume
(``tl``, ``torch``, ``nn``, ``model``, ``x``). This gate fails when a public symbol has
no example and when an example raises or prints something other than what it shows.

The only escape hatch is :data:`PENDING`, an exact ledger of symbols whose example is
not written yet, each with a reason. It is checked both ways: an unledgered symbol
without an example fails, and a ledger entry whose symbol now carries an example (or no
longer exists) ALSO fails, so the ledger cannot rot into a permanent exemption. An
example that needs an optional dependency starts with ``>>> # requires: <module>`` and
is skipped, not failed, when that module is absent.

Each example captures a small model, so the family is ``heavy`` (worker and backstop
tiers, never smoke); one cell per package (the packages the committed census lists, read
without importing torchlens) keeps every cell inside the heavy budget. A stale census
fails here too, so the cell list cannot drift from the surface.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import docstring_examples  # noqa: E402

pytestmark = pytest.mark.heavy

#: Public symbols whose example is not written yet: ``public path -> reason``.
#: EMPTY is the intended steady state. Adding a row needs a reason and an owner; a row
#: whose symbol gains an example must leave in the same change (the staleness test).
PENDING: dict[str, str] = {}


@functools.lru_cache(maxsize=1)
def _symbols() -> tuple[docstring_examples.PublicSymbol, ...]:
    """Build the public symbol list once per process."""

    return tuple(docstring_examples.public_symbols())


def test_every_public_function_and_class_has_an_example() -> None:
    """Fail on every unledgered public function or class without a ``>>>`` example."""

    missing = [
        f"{symbol.path} ({symbol.module})"
        for symbol in _symbols()
        if not docstring_examples.example_of(symbol.obj) and symbol.path not in PENDING
    ]
    assert not missing, (
        f"{len(missing)} public symbols lack a runnable Examples section "
        "(docs/agent-reference/conventions.md, Docstrings):\n  " + "\n  ".join(missing)
    )


def test_pending_ledger_has_no_stale_entries() -> None:
    """Fail on a PENDING row whose symbol now has an example or no longer exists."""

    by_path = {symbol.path: symbol for symbol in _symbols()}
    stale = []
    for path in PENDING:
        symbol = by_path.get(path)
        if symbol is None:
            stale.append(f"{path}: no such public symbol")
        elif docstring_examples.example_of(symbol.obj):
            stale.append(f"{path}: has an example; remove the row")
    assert not stale, "stale PENDING rows:\n  " + "\n  ".join(stale)


def test_census_is_current() -> None:
    """The committed census matches a fresh render, so its package list stays true."""

    rendered = docstring_examples.render_census(
        list(_symbols()), docstring_examples.public_members()
    )
    committed = docstring_examples.CENSUS_PATH.read_text()
    assert committed == rendered, (
        "docs/maintainers/docstring_examples_census.md is stale: run "
        "`python tools/docstring_examples.py census` and commit the result"
    )


@pytest.mark.parametrize("package", docstring_examples.census_packages())
def test_every_example_runs(package: str) -> None:
    """Execute every example filed under ``package`` and report each failure."""

    symbols = [symbol for symbol in _symbols() if symbol.package == package]
    assert symbols, f"the census lists {package} but no public symbol is filed there"
    namespace = docstring_examples.ambient_namespace()
    failures = []
    for symbol in symbols:
        if symbol.path in PENDING:
            continue
        result = docstring_examples.run_example(symbol, namespace)
        if result.status == "failed":
            failures.append(f"{result.path}:\n{result.detail}")
    assert not failures, f"{len(failures)} examples failed:\n" + "\n".join(failures)
