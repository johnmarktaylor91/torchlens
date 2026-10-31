"""AUD-CODE 3.14 regression: the root teaching redirect table is populated.

The five-step facade's compensating design for the 2026-08 shim-removal pass
was a REDIRECT table teaching every removed top-level spelling its canonical
home. It shipped empty, so the root ``wrap_torch`` spelling raised a bare AttributeError
with Python's did-you-mean (``swap_with?``). These pins keep the table
populated, typed, non-shadowing, and pointing at spellings that resolve.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys

import pytest

import torchlens as tl
from torchlens._errors import FacadeTeachingError

#: The removed shims the audit named explicitly (AUD-CODE 3.14).
#: The two paper-era 1.x names the audit also listed (the pre-2.0 capture verb
#: and the pre-2.0 log class) are pinned in tests/test_w051_capt3_paper_era_redirects.py,
#: the one test file the removed-spelling lint's _ALLOWED ledger admits for them.
_AUDIT_NAMED_SHIMS = (
    "replay",
    "rerun",
    "record_span",
    "wrap_torch",
    "list_logs",
    "show_model_graph",
)

_DOTTED = re.compile(r"use (torchlens(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")

#: Rows naming a torchlens subpackage (forks ruling names.4: teaching rows for
#: mechinterp and transformer instead of root module rows). Importing a
#: subpackage binds it on the package, and attribute lookup then never reaches
#: ``__getattr__``, so these rows teach only until the subpackage is imported.
#: In-process state depends on what ran before, so they are pinned in a fresh
#: interpreter (``test_subpackage_rows_teach_until_imported``).
_SUBPACKAGE_ROWS = sorted(
    name for name in tl._REDIRECTS if importlib.util.find_spec(f"torchlens.{name}") is not None
)


def test_redirect_table_is_populated_with_the_audit_named_shims() -> None:
    """Every shim the audit named is a redirect row."""

    missing = [name for name in _AUDIT_NAMED_SHIMS if name not in tl._REDIRECTS]
    assert not missing, f"redirect table lacks the audit-named shims: {missing}"
    assert len(tl._REDIRECTS) >= 50


@pytest.mark.parametrize("name", sorted(set(tl._REDIRECTS) - set(_SUBPACKAGE_ROWS)))
def test_each_redirect_row_is_typed_and_teaches(name: str) -> None:
    """A redirect row raises the typed facade_redirect AttributeError naming the home."""

    with pytest.raises(AttributeError) as excinfo:
        getattr(tl, name)
    assert type(excinfo.value) is FacadeTeachingError
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert tl._REDIRECTS[name] in str(excinfo.value)
    assert "torchlens." in tl._REDIRECTS[name], "every remedy names a canonical spelling"
    # hasattr never explodes and answers False; dir() never advertises the row.
    assert not hasattr(tl, name)
    assert name not in dir(tl)


def test_no_redirect_row_shadows_a_live_name() -> None:
    """Step 2 beats step 4: a row naming a LIVE attribute would hide it.

    ``summary`` is the live example the migration table still lists as
    removed; it must never enter the redirect table.
    """

    # An imported subpackage sits in vars(tl) but is never shadowed: lookup
    # finds it before __getattr__ (pinned in a fresh interpreter below).
    bound = {k for k in vars(tl) if not k.startswith("_")} - set(_SUBPACKAGE_ROWS)
    live = set(tl.__all__) | set(tl._LAZY_ATTRS) | bound
    shadowing = sorted(set(tl._REDIRECTS) & live)
    assert shadowing == [], f"redirect rows shadow live names: {shadowing}"
    assert "summary" in tl.__all__ and "summary" not in tl._REDIRECTS


_FRESH_SUBPACKAGE_PROBE = """
import importlib, sys
import torchlens as tl
from torchlens._errors import FacadeTeachingError
names = sys.argv[1:]
loaded = [name for name in names if f"torchlens.{name}" in sys.modules]
assert not loaded, f"import torchlens loaded {loaded}"
for name in names:
    try:
        getattr(tl, name)
    except FacadeTeachingError as error:
        assert error.fields["code"] == "facade_redirect"
        assert tl._REDIRECTS[name] in str(error)
    else:
        raise AssertionError(f"tl.{name} resolved in a fresh interpreter")
    assert not hasattr(tl, name) and name not in dir(tl)
for name in names:
    module = importlib.import_module(f"torchlens.{name}")
    assert getattr(tl, name) is module
print("ok")
"""


def test_subpackage_rows_are_the_ruled_three() -> None:
    """names.4: exactly mechinterp and transformer teach at the root."""

    assert _SUBPACKAGE_ROWS == ["mechinterp", "transformer"]
    assert not set(_SUBPACKAGE_ROWS) & (set(tl._LAZY_ATTRS) | set(tl.__all__))


# Heavy: one cold interpreter importing torchlens+torch (~3.3s) plus three
# subpackages, the tests/test_import_hygiene.py precedent.
@pytest.mark.heavy
def test_subpackage_rows_teach_until_imported() -> None:
    """In a fresh interpreter each subpackage row raises the typed redirect, then
    resolves to the subpackage once it is imported."""

    result = subprocess.run(
        [sys.executable, "-c", _FRESH_SUBPACKAGE_PROBE, *_SUBPACKAGE_ROWS],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-2000:]


@pytest.mark.parametrize("name", sorted(tl._REDIRECTS))
def test_each_redirect_target_resolves(name: str) -> None:
    """The canonical dotted spelling in every remedy actually exists."""

    match = _DOTTED.search(tl._REDIRECTS[name])
    assert match, f"remedy for {name!r} names no torchlens dotted path"
    assert _resolve_dotted(match.group(1)), f"{match.group(1)} does not resolve"


def _resolve_dotted(dotted: str) -> bool:
    """Import the longest module prefix of ``dotted``, then walk the rest as attributes."""

    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        module_path = ".".join(parts[:split])
        try:
            if importlib.util.find_spec(module_path) is None:
                continue
        except ModuleNotFoundError:  # a parent of module_path is a class, not a package
            continue
        target: object = importlib.import_module(module_path)
        for attr in parts[split:]:
            if not hasattr(target, attr):
                return False
            target = getattr(target, attr)
        return True
    return False


def test_refusal_rows_if_any_are_typed() -> None:
    """Refusal rows (none at the root today) raise the facade_refusal code."""

    for name in tl._REFUSALS:
        with pytest.raises(AttributeError) as excinfo:
            getattr(tl, name)
        assert excinfo.value.fields["code"] == "facade_refusal"


def test_did_you_mean_stays_plain_for_typos() -> None:
    """Step 5 is untouched: a typo still gets the PLAIN did-you-mean error."""

    with pytest.raises(AttributeError) as excinfo:
        tl.trce  # noqa: B018 - the typo IS the test
    assert type(excinfo.value) is AttributeError
    assert "Did you mean" in str(excinfo.value)
