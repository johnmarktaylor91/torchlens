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

from menagerie.crawler.constants import (
    LINK_EVIDENCED_DISCOVERY_STATUS_CODES,
    TERMINAL_STATUS_CODES,
)
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
    """A success may not carry a link while reporting the mandatory link as unresolved.

    The regex here deliberately matches on ``mandatory-link``/``mandatory_link_status`` rather
    than a full sentence. The two consistency arms were later unified into one message, and the
    original wording ("require mandatory_link_status=ok") stopped matching even though the
    behaviour was unchanged -- a test failure that said nothing about correctness. Assert the
    invariant, not the prose.
    """

    with pytest.raises(ReductionError, match="mandatory[-_]link"):
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


# ---------------------------------------------------------------------------
# The typed-discovery arm: an absence claim must name what it looked at.
# ---------------------------------------------------------------------------

_DISCOVERY_DIGEST = "sha256:" + "0" * 64

TYPED_DISCOVERY_CODES = sorted(
    {
        "skipped:no-description",
        "skipped:insufficient-description",
        "skipped:not-a-real-NN",
        "deferred:needs-opus-tier",
    }
)


def _discovery_record(code: str, *, links: list[str]) -> dict[str, Any]:
    """Build a typed R5 discovery record with an exactly controlled link list."""

    return {
        "status": {"code": code, "kind": str(code).split(":", 1)[0]},
        "source_resolution": {
            "rung": "R5_SKIP",
            "primary_source_id": "discovery-evidence-1",
            "mandatory_link_status": "failed",
            "sources": [
                {
                    "source_id": "discovery-evidence-1",
                    "kind": "discovery-evidence",
                    "url": "urn:menagerie:source-discovery:" + "0" * 64,
                    "revision_kind": "source-discovery-sha256",
                    "revision": _DISCOVERY_DIGEST,
                    "content_sha256": _DISCOVERY_DIGEST,
                    "mirror_digest": _DISCOVERY_DIGEST,
                }
            ],
            "search_report": {
                "queries": ["ExampleNet architecture"],
                "places_checked": ["publisher index"],
                "links_checked": list(links),
                "languages_checked": ["en"],
                "conclusion": "Bounded search reached its end.",
            },
        },
    }


def test_link_evidenced_vocabulary_is_the_absence_claiming_subset() -> None:
    """Guard the parametrisation: the scoping is the point, so pin it explicitly."""

    assert LINK_EVIDENCED_DISCOVERY_STATUS_CODES == {
        "skipped:no-description",
        "skipped:insufficient-description",
    }
    assert LINK_EVIDENCED_DISCOVERY_STATUS_CODES < set(TERMINAL_STATUS_CODES)


@pytest.mark.parametrize("code", sorted(LINK_EVIDENCED_DISCOVERY_STATUS_CODES))
def test_absence_claim_without_candidate_locators_is_refused(code: str) -> None:
    """The cheapest skip -- zero candidate links -- must no longer be the easiest one.

    With no candidate locators there is nothing for ``_probe_discovery_candidates`` to
    dereference, so no machine receipt can exist to contradict the claim. That made
    silence cheaper than honesty on exactly the two codes that assert a world-fact.
    """

    with pytest.raises(ReductionError, match="candidate locators"):
        _validate(_discovery_record(code, links=[]))


@pytest.mark.parametrize("code", sorted(LINK_EVIDENCED_DISCOVERY_STATUS_CODES))
def test_absence_claim_with_candidate_locators_passes(code: str) -> None:
    """The honest author -- the one that names what it rejected -- is still accepted."""

    _validate(_discovery_record(code, links=["https://example.com/abstract"]))


@pytest.mark.parametrize(
    "code",
    [code for code in TYPED_DISCOVERY_CODES if code not in LINK_EVIDENCED_DISCOVERY_STATUS_CODES],
)
def test_non_absence_typed_discovery_may_have_no_candidate_locators(code: str) -> None:
    """Scoping in the other direction: the requirement must not leak onto sibling arms.

    ``not-a-real-NN`` can rest on a search that surfaced nothing worth retaining, and an
    Opus-tier deferral asserts nothing about the world at all.
    """

    _validate(_discovery_record(code, links=[]))


@pytest.mark.parametrize("code", TYPED_DISCOVERY_CODES)
def test_typed_discovery_still_requires_query_place_language_evidence(code: str) -> None:
    """The pre-existing bounded-search floor is unchanged by the new link requirement."""

    record = _discovery_record(code, links=["https://example.com/abstract"])
    record["source_resolution"]["search_report"]["languages_checked"] = []

    with pytest.raises(ReductionError, match="lacks bounded query/place/language evidence"):
        _validate(record)
