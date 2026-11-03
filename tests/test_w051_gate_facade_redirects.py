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


def test_each_redirect_target_resolves() -> None:
    """The canonical dotted spelling in every remedy actually exists (one row per redirect)."""

    failures = []
    for name in sorted(tl._REDIRECTS):
        match = _DOTTED.search(tl._REDIRECTS[name])
        if not match:
            failures.append(f"remedy for {name!r} names no torchlens dotted path")
        elif not _resolve_dotted(match.group(1)):
            failures.append(f"{name!r}: {match.group(1)} does not resolve")
    assert not failures, "\n".join(failures)


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


def test_did_you_mean_stays_plain_for_typos() -> None:
    """Step 5 is untouched: a typo still gets the PLAIN did-you-mean error."""

    with pytest.raises(AttributeError) as excinfo:
        tl.trce  # noqa: B018 - the typo IS the test
    assert type(excinfo.value) is AttributeError
    assert "  candidates" in str(excinfo.value)


# The root verbs removed in 02d4fea77 (do, push, push_from, run) and the edit
# helper swap_with are Trace methods or torchlens.intervention.helpers names;
# each gets a teaching row so the old root spelling names its home.
@pytest.mark.parametrize("name", ["do", "push", "push_from", "run", "swap_with"])
def test_removed_root_verb_redirects(name: str) -> None:
    """A removed root verb raises the typed redirect naming its canonical home."""

    from torchlens._errors import FacadeTeachingError

    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(tl, name)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert tl._REDIRECTS[name] in str(excinfo.value)
    assert not hasattr(tl, name) and name not in dir(tl)
