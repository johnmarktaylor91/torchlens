"""The mandatory source-link invariant must bind successes without aborting failures.

This regression exists because a single unfetchable source aborted an entire campaign.

A model's author pinned three files at a commit SHA that does not exist. The fetcher
correctly rejected it (HTTP 404), the driver assembled a terminal failure record through
``driver_models._placeholder_facts`` -- which by construction carries ``sources=[]`` and
``mandatory_link_status="failed"`` -- and the reducer then refused that record with
``missing mandatory exact public primary source link`` and killed the run.

The invariant keyed on the single status code ``failed:source`` rather than on the failure
*kind*, so the other eight terminal failure codes were unsatisfiable whenever the failure
occurred before source resolution completed. The engine was therefore building records its
own reducer categorically rejected, and one bad pin among tens of thousands of models was
enough to halt everything.

What must NOT regress, and is pinned below in both directions:

* a **success** record still requires a real exact public primary link and an honest
  ``mandatory_link_status`` -- that guarantee is the whole point of the check;
* a **failure** record may lack a source, but may never *misreport* whether it has one.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.constants import TERMINAL_STATUS_CODES
from menagerie.crawler.reducer import CanonicalReducer, ReductionError

_EXACT_SOURCE = {
    "source_id": "source-1",
    "url": "https://example.com/model",
}

FAILURE_CODES = sorted(code for code in TERMINAL_STATUS_CODES if str(code).startswith("failed:"))
SUCCESS_CODES = sorted(code for code in TERMINAL_STATUS_CODES if not str(code).startswith("failed:"))


def _validate(model: dict[str, Any]) -> None:
    """Invoke the invariant without constructing a full reducer."""

    reducer = object.__new__(CanonicalReducer)
    CanonicalReducer._validate_source(reducer, model)


def _record(code: str, *, with_source: bool, link_status: str) -> dict[str, Any]:
    """Build the minimal shape the invariant inspects."""

    kind = str(code).split(":", 1)[0]
    return {
        "status": {"code": code, "kind": kind},
        "source_resolution": {
            "primary_source_id": "source-1",
            "sources": [deepcopy(_EXACT_SOURCE)] if with_source else [],
            "mandatory_link_status": link_status,
        },
    }


def test_failure_codes_are_actually_present() -> None:
    """Guard the parametrisation itself: the vocabulary must be non-trivial."""

    assert len(FAILURE_CODES) > 1
    assert "failed:source" in FAILURE_CODES


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_every_failure_code_is_satisfiable_without_a_resolved_source(code: str) -> None:
    """The campaign-killer regression, generalised across the whole failure vocabulary.

    A failure that happens before source resolution has no source to report. Every terminal
    failure code must be able to say so, not just ``failed:source``.
    """

    _validate(_record(code, with_source=False, link_status="failed"))


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_failure_may_not_claim_a_link_it_does_not_have(code: str) -> None:
    """A failure record may lack a source; it may not lie about lacking one."""

    with pytest.raises(ReductionError, match="contradicts source evidence"):
        _validate(_record(code, with_source=False, link_status="ok"))


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_failure_may_not_disclaim_a_link_it_does_have(code: str) -> None:
    """The consistency check binds in both directions."""

    with pytest.raises(ReductionError, match="contradicts source evidence"):
        _validate(_record(code, with_source=True, link_status="failed"))


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_failure_that_did_resolve_a_source_still_records_it(code: str) -> None:
    """Late-stage failures normally *do* have a source, and remain valid with one."""

    _validate(_record(code, with_source=True, link_status="ok"))


@pytest.mark.parametrize("code", SUCCESS_CODES)
def test_success_requires_an_exact_public_primary_link(code: str) -> None:
    """The guarantee being protected: no catalogued success without a real public source."""

    with pytest.raises(ReductionError, match="missing mandatory exact public primary source link"):
        _validate(_record(code, with_source=False, link_status="failed"))


@pytest.mark.parametrize("code", SUCCESS_CODES)
def test_success_requires_an_honest_link_status(code: str) -> None:
    """A success may not carry a link while reporting the mandatory link as unresolved."""

    with pytest.raises(ReductionError, match="require mandatory_link_status=ok"):
        _validate(_record(code, with_source=True, link_status="failed"))


@pytest.mark.parametrize("code", SUCCESS_CODES)
def test_success_with_a_real_link_passes(code: str) -> None:
    """The ordinary catalogued case."""

    _validate(_record(code, with_source=True, link_status="ok"))


def test_success_rejects_a_non_http_primary_link() -> None:
    """'Exact public link' means a real URL, not a local or opaque reference."""

    model = _record("runs", with_source=True, link_status="ok")
    model["source_resolution"]["sources"][0]["url"] = "file:///tmp/model.py"

    with pytest.raises(ReductionError, match="missing mandatory exact public primary source link"):
        _validate(model)


def test_success_rejects_a_source_that_is_not_the_declared_primary() -> None:
    """A link to *something* does not satisfy a link to the *primary* source."""

    model = _record("runs", with_source=True, link_status="ok")
    model["source_resolution"]["sources"][0]["source_id"] = "some-other-source"

    with pytest.raises(ReductionError, match="missing mandatory exact public primary source link"):
        _validate(model)
