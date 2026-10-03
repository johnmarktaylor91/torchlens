"""AUD-CODE 3.14 regression: the root teaching redirect table is populated.

The five-step facade's compensating design for the 2026-08 shim-removal pass
was a REDIRECT table teaching every removed top-level spelling its canonical
home. It shipped empty, so the root ``wrap_torch`` spelling raised a bare AttributeError
with Python's did-you-mean (``swap_with?``). These pins keep the table
populated, typed, non-shadowing, and pointing at spellings that resolve.
"""

from __future__ import annotations

import importlib
import re

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


def test_redirect_table_is_populated_with_the_audit_named_shims() -> None:
    """Every shim the audit named is a redirect row."""

    missing = [name for name in _AUDIT_NAMED_SHIMS if name not in tl._REDIRECTS]
    assert not missing, f"redirect table lacks the audit-named shims: {missing}"
    assert len(tl._REDIRECTS) >= 50


@pytest.mark.parametrize("name", sorted(tl._REDIRECTS))
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

    live = set(tl.__all__) | set(tl._LAZY_ATTRS) | {k for k in vars(tl) if not k.startswith("_")}
    shadowing = sorted(set(tl._REDIRECTS) & live)
    assert shadowing == [], f"redirect rows shadow live names: {shadowing}"
    assert "summary" in tl.__all__ and "summary" not in tl._REDIRECTS


@pytest.mark.parametrize("name", sorted(tl._REDIRECTS))
def test_each_redirect_target_resolves(name: str) -> None:
    """The canonical dotted spelling in every remedy actually exists."""

    match = _DOTTED.search(tl._REDIRECTS[name])
    assert match, f"remedy for {name!r} names no torchlens dotted path"
    module_path, _, attr = match.group(1).rpartition(".")
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), f"{match.group(1)} does not resolve"


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
