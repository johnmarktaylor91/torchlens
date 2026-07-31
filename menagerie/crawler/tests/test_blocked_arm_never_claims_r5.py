"""A ``BLOCKED`` arm may never be recorded as a checked ``R5_SKIP``.

``R5_SKIP`` is not a shrug. In this codebase it is the CHECKED conclusion that no
faithful source path exists, and a genuine one carries a separately certified
epistemic predicate (``authority._validate_skip_predicate``, reached only through
``_derive_skip`` for a ``skipped:*`` status). A ``BLOCKED`` recommendation claims
something entirely different and far weaker: *a prerequisite stopped us before the
ladder was walked to its end*. Its closed terminals are all ``failed:*`` or
``deferred:*`` (``driver._blocked_terminal``), never ``skipped:*`` -- so nothing ever
re-checks the predicate, and a stamped ``R5_SKIP`` on that arm is a false structured
fact that no later pass can contest.

Two sites in ``driver_models`` disagreed about exactly this:

* the rung site excluded the ``BLOCKED`` arm outright ("A BLOCKED arm is deliberately
  excluded"), while
* the typed-discovery site excluded only the ACCESS-blocked *subset*, so every other
  ``BLOCKED`` arm carrying discovery evidence -- notably ``needs-higher-tier``, which is
  simply "send this to a stronger author" -- was stamped ``R5_SKIP``.

Observed live on ``m_645973ec68aff8d687f0``: ``failed:author`` / ``needs-higher-tier``
recorded with ``rung: R5_SKIP`` and a stale ``attempted_rungs`` still blaming
``author-lane-failed / not-reached`` -- blaming the one stage that worked, since the
author lane published the very verdict being recorded.

Pinned below, in both directions:

* no ``BLOCKED`` reason code, adjudicated or not, earns ``R5_SKIP``;
* a gate-obtained ``BLOCKED`` arm stops claiming the author lane failed, and names the
  author's own blocking reason instead;
* a genuine ``SKIP`` over the *same* discovery evidence still earns ``R5_SKIP`` and its
  bounded-negative-discovery attempted rung -- the fix narrows the stamp, it does not
  make R5 harder to reach honestly;
* the sentinel is still refused for every ``skipped:*`` code at the reducer, so nothing
  here relaxes the tripwire.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    BlockedRecommendation,
    SkipRecommendation,
)
from menagerie.crawler.constants import (
    ACCESS_BLOCKED_REASON_CODE,
    ACCESS_BLOCKED_STATUS_CODE,
    NO_RUNG_SELECTED,
    EnvironmentPhase,
    SourceRung,
)
from menagerie.crawler.driver_contracts import AuthorArtifact, DriverConfig, WorkItem

# Importing the driver installs ``driver_models``' dependency table; without it the
# terminal assembler raises. Depend on it explicitly rather than on conftest import
# order, so this module is runnable on its own.
from menagerie.crawler import driver as _driver  # noqa: F401
from menagerie.crawler.driver_models import _assemble_terminal_model
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.tests.conftest import HASH, make_authority_context

_CREATED_AT = "2026-07-31T00:00:00Z"
_STABLE_ID = "m_645973ec68aff8d687f0"

# The exact BLOCKED reasons ``driver._blocked_terminal`` maps onto a closed terminal.
# Every one of them lands on ``failed:*`` or ``deferred:*`` -- never ``skipped:*`` -- so
# none of them can ever have its R5 predicate certified.
BLOCKED_REASON_TO_TERMINAL: tuple[tuple[str, str, str | None], ...] = (
    ("needs-higher-tier", "failed:author", "needs-higher-tier"),
    ("needs-higher-tier", "deferred:needs-opus-tier", None),
    (ACCESS_BLOCKED_REASON_CODE, ACCESS_BLOCKED_STATUS_CODE, None),
    ("missing-material-source", "failed:source", "missing-material-source"),
    ("unmappable-reason", "failed:author", "malformed-result"),
)

# Two distinct, realistic bodies. A one-source manifest cannot tell a projection that
# reads ``sources[0]`` from one that reads every row, and it cannot exercise the broker
# row projection at all -- a BLOCKED verdict reached AFTER stage 1 answered FOUND
# retains both kinds side by side.
_DISCOVERY_BYTES = b'{"arm":"blocked","conclusion":"stronger author tier required"}'
_PAPER_BYTES = b"MuRE embeds entities in the Poincare ball with per-relation diagonal\n"


def _work_item() -> WorkItem:
    """Return the routed work item behind the observed live record."""

    intake = IntakeItem(
        stable_id=_STABLE_ID,
        name="MuRE",
        zoo="pykeen",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=_STABLE_ID,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(
            stable_id=_STABLE_ID, intent="core", phase=EnvironmentPhase.PYTORCH
        ),
    )


def _search_evidence() -> dict[str, Any]:
    """Return one bounded stage-1 search summary with real, distinct locators."""

    return {
        "queries": ["MuRE knowledge graph embedding reference implementation"],
        "places": ["github.com/ibalazevic/multirelational-poincare", "arxiv.org"],
        "candidate_links": [
            {"url": "https://github.com/ibalazevic/multirelational-poincare"},
            {"url": "https://arxiv.org/abs/1905.09791"},
        ],
        "languages": ["en"],
        "conclusion": (
            "Source material was located but reading it needs a stronger authoring tier."
        ),
    }


def _source_manifest(tmp_path: Path) -> dict[str, Any]:
    """Freeze a realistic MULTI-source terminal manifest into a private CAS.

    Two rows with *different* bodies, ids and projections: the typed discovery-evidence
    row a stage-1 verdict always carries, and one broker-fetched row a BLOCKED verdict
    retains once stage 1 answered FOUND. A single-row manifest hides every
    width-dependent defect in the retention loop.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory used as the private staging root.

    Returns
    -------
    dict[str, Any]
        Hash-bound source manifest with one discovery row and one broker row.
    """

    cas_root = tmp_path / "source-cas"
    cas_root.mkdir(parents=True, exist_ok=True)
    discovery_digest = hash_bytes(_DISCOVERY_BYTES)
    paper_digest = hash_bytes(_PAPER_BYTES)
    for digest, content in (
        (discovery_digest, _DISCOVERY_BYTES),
        (paper_digest, _PAPER_BYTES),
    ):
        (cas_root / f"{digest.removeprefix('sha256:')}.source").write_bytes(content)
    discovery_row: dict[str, Any] = {
        "source_id": f"discovery-evidence-{discovery_digest.removeprefix('sha256:')[:16]}",
        "url": f"urn:menagerie:source-discovery:{discovery_digest.removeprefix('sha256:')}",
        "revision": discovery_digest,
        "content_sha256": discovery_digest,
        "fetched_bytes_len": len(_DISCOVERY_BYTES),
        "retrieval_status": "machine-derived",
        "media_type": "application/json",
        "cas_path": str(cas_root / f"{discovery_digest.removeprefix('sha256:')}.source"),
        "source_kind": "discovery-evidence-v1",
        "discovery_arm": "blocked",
        "search_evidence": _search_evidence(),
        "retained_vague_text": None,
        "candidate_probe_receipts": None,
        "candidate_probes": [
            {
                "url": "https://arxiv.org/abs/1905.09791",
                "result": "reachable",
                "attempted_at": _CREATED_AT,
            }
        ],
    }
    broker_row: dict[str, Any] = {
        "source_id": "broker-paper-1905-09791",
        "broker_role": "documentation",
        "url": "https://arxiv.org/abs/1905.09791",
        "final_url": "https://arxiv.org/abs/1905.09791v2",
        "revision": paper_digest,
        "content_sha256": paper_digest,
        "fetched_bytes_len": len(_PAPER_BYTES),
        "retrieval_status": "fetched",
        "media_type": "text/html",
        "requested_repo": False,
        "cas_path": str(cas_root / f"{paper_digest.removeprefix('sha256:')}.source"),
    }
    rows = [discovery_row, broker_row]
    return {"manifest_sha256": stable_hash(rows), "sources": rows}


def _binding(kind: str) -> AuthorResultBinding:
    """Return one hash-bound author-result binding for a terminal arm."""

    raw_fields = {
        "result_id": f"result-{kind.lower()}",
        "result_sha256": HASH,
        "stable_id": _STABLE_ID,
        "work_id": f"work-{_STABLE_ID}",
        "campaign_id": f"work-{_STABLE_ID}",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": _CREATED_AT,
    }
    return AuthorResultBinding(
        raw_result={**raw_fields, "kind": kind, "payload": {}}, **raw_fields
    )


def _blocked_artifact(tmp_path: Path, *, reason_code: str) -> AuthorArtifact:
    """Stage one BLOCKED terminal artifact over the multi-source manifest."""

    manifest = _source_manifest(tmp_path)
    result = BlockedRecommendation(
        binding=_binding("BLOCKED"),
        stage="source",
        reason_code=reason_code,
        prerequisite_ids=("prereq-author-tier",),
        evidence_ids=("ev-blocked-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
        research_summary=_search_evidence(),
    )
    return AuthorArtifact(
        author_result=result, source_manifest=manifest, model_dir=tmp_path / "model"
    )


def _skip_artifact(tmp_path: Path) -> AuthorArtifact:
    """Stage one genuine SKIP terminal artifact over the SAME manifest.

    The positive control. It differs from the blocked artifact in exactly one thing --
    the arm -- so a fix that merely stopped stamping ``R5_SKIP`` everywhere would be
    caught here rather than read as success.
    """

    manifest = _source_manifest(tmp_path)
    result = SkipRecommendation(
        binding=_binding("SKIP"),
        status_code="skipped:no-description",
        source_ids=tuple(str(row["source_id"]) for row in manifest["sources"]),
        evidence_ids=("ev-skip-1",),
        evidence_identity=HASH,
        search_report_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
    )
    return AuthorArtifact(
        author_result=result, source_manifest=manifest, model_dir=tmp_path / "model"
    )


def _assemble(
    artifact: AuthorArtifact,
    *,
    status_code: str,
    reason_code: str | None,
    terminal_gate_obtained: bool = True,
) -> dict[str, Any]:
    """Run the real terminal assembler; never a local re-derivation of it."""

    return _assemble_terminal_model(
        _work_item(),
        artifact,
        status_code,
        reason_code,
        None,
        [],
        [],
        DriverConfig(),
        _CREATED_AT,
        human_review=status_code == "failed:accuracy-gate",
        root_cause_fingerprint="fingerprint-blocked",
        terminal_gate_obtained=terminal_gate_obtained,
    )


# ---------------------------------------------------------------------------
# 1. the defect: a BLOCKED arm stamped with a checked no-source conclusion
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("reason_code", "status_code", "terminal_reason"), BLOCKED_REASON_TO_TERMINAL
)
def test_no_blocked_reason_earns_a_checked_r5(
    tmp_path: Path, reason_code: str, status_code: str, terminal_reason: str | None
) -> None:
    """Every closed BLOCKED terminal keeps the sentinel, not ``R5_SKIP``.

    This is the failing direction. Before the fix only the ACCESS subset was excluded,
    so the four other rows here -- all carrying typed discovery evidence, because a
    BLOCKED verdict always publishes its bounded stage-1 summary -- were stamped with a
    conclusion nothing in the pipeline would ever re-check.
    """

    model = _assemble(
        _blocked_artifact(tmp_path, reason_code=reason_code),
        status_code=status_code,
        reason_code=terminal_reason,
    )

    resolution = model["source_resolution"]
    assert resolution["rung"] == NO_RUNG_SELECTED
    assert resolution["rung"] != SourceRung.SKIP.value


@pytest.mark.smoke
def test_blocked_arm_stops_blaming_the_author_lane_that_worked(tmp_path: Path) -> None:
    """The stale placeholder blamed the one stage that succeeded.

    ``author-lane-failed / not-reached`` is what ``_placeholder_facts`` writes when the
    model-local lane died before source resolution. An adjudicated BLOCKED verdict is
    the opposite: the author lane ran to completion and published this very result. The
    correction was gated on ``isinstance(..., SkipRecommendation)``, so the BLOCKED arm
    kept the placeholder.
    """

    model = _assemble(
        _blocked_artifact(tmp_path, reason_code="needs-higher-tier"),
        status_code="failed:author",
        reason_code="needs-higher-tier",
    )

    attempted = model["source_resolution"]["attempted_rungs"]
    assert [entry["rung"] for entry in attempted] == [NO_RUNG_SELECTED]
    reasons = [entry["reason_code"] for entry in attempted]
    assert "author-lane-failed" not in reasons
    # The author's OWN blocking reason, not an invented one and not a stage name.
    assert reasons == ["needs-higher-tier"]


@pytest.mark.smoke
def test_blocked_attempted_rung_is_not_vacuously_equal_to_the_placeholder(
    tmp_path: Path,
) -> None:
    """Guard the comparison above against passing on identical slots.

    An all-placeholder negative once passed because both compared values happened to be
    the same string. Pin the two arms against each other with reasons that are distinct
    by construction.
    """

    blocked = _assemble(
        _blocked_artifact(tmp_path / "blocked", reason_code="needs-higher-tier"),
        status_code="failed:author",
        reason_code="needs-higher-tier",
    )
    access = _assemble(
        _blocked_artifact(tmp_path / "access", reason_code=ACCESS_BLOCKED_REASON_CODE),
        status_code=ACCESS_BLOCKED_STATUS_CODE,
        reason_code=None,
    )

    blocked_reasons = [
        entry["reason_code"] for entry in blocked["source_resolution"]["attempted_rungs"]
    ]
    access_reasons = [
        entry["reason_code"] for entry in access["source_resolution"]["attempted_rungs"]
    ]
    assert blocked_reasons == ["needs-higher-tier"]
    assert access_reasons == [ACCESS_BLOCKED_REASON_CODE]
    assert blocked_reasons != access_reasons


@pytest.mark.smoke
def test_the_record_no_longer_contradicts_its_own_status_block(tmp_path: Path) -> None:
    """``status.attempted_rungs`` always read the honest sentinel; the rung did not.

    The live record disagreed with ITSELF: ``status.attempted_rungs`` was derived from
    the untouched placeholder facts and said ``NO_RUNG_SELECTED`` while
    ``source_resolution.rung`` said ``R5_SKIP``. One record, two answers.
    """

    model = _assemble(
        _blocked_artifact(tmp_path, reason_code="needs-higher-tier"),
        status_code="failed:author",
        reason_code="needs-higher-tier",
    )

    assert model["status"]["attempted_rungs"] == [NO_RUNG_SELECTED]
    assert model["source_resolution"]["rung"] == model["status"]["attempted_rungs"][0]


@pytest.mark.smoke
def test_an_unadjudicated_blocked_arm_also_keeps_the_sentinel(tmp_path: Path) -> None:
    """No disposition gate means nothing was checked at all, let alone an R5."""

    model = _assemble(
        _blocked_artifact(tmp_path, reason_code="needs-higher-tier"),
        status_code="failed:author",
        reason_code="needs-higher-tier",
        terminal_gate_obtained=False,
    )

    resolution = model["source_resolution"]
    assert resolution["rung"] == NO_RUNG_SELECTED
    assert [entry["reason_code"] for entry in resolution["attempted_rungs"]] == [
        "terminal-disposition-gate-unavailable"
    ]


# ---------------------------------------------------------------------------
# 2. the positive control: a genuine SKIP is untouched
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_genuine_skip_over_the_same_evidence_still_earns_r5(tmp_path: Path) -> None:
    """Narrowing the BLOCKED stamp must not make an honest R5 unreachable.

    Same manifest, same discovery evidence, same gate: only the arm differs. A SKIP is
    the one arm whose closed terminal IS ``skipped:*``, so its ``R5_SKIP`` is the claim
    ``authority._derive_skip`` will actually re-check.
    """

    model = _assemble(
        _skip_artifact(tmp_path),
        status_code="skipped:no-description",
        reason_code=None,
    )

    resolution = model["source_resolution"]
    assert resolution["rung"] == SourceRung.SKIP.value
    assert [entry["rung"] for entry in resolution["attempted_rungs"]] == [
        SourceRung.SKIP.value
    ]
    assert [entry["result"] for entry in resolution["attempted_rungs"]] == [
        "bounded-negative-discovery"
    ]


# ---------------------------------------------------------------------------
# 3. the reducer still refuses the sentinel wherever R5 is the claim
# ---------------------------------------------------------------------------


def _reducer_record(*, status_code: str, rung: str) -> dict[str, Any]:
    """Build the minimal shape ``CanonicalReducer._validate_source`` inspects.

    The discovery row is spelled exactly as ``_assemble_terminal_model`` projects it:
    a ``urn:`` locator that is deliberately NOT an exact public link, which is the whole
    reason the typed-discovery relaxation exists.
    """

    digest = hash_bytes(_DISCOVERY_BYTES)
    source_id = f"discovery-evidence-{digest.removeprefix('sha256:')[:16]}"
    return {
        "status": {"code": status_code, "kind": status_code.split(":", 1)[0]},
        "source_resolution": {
            "rung": rung,
            "primary_source_id": source_id,
            "mandatory_link_status": "failed",
            "search_report": {
                "queries": ["MuRE reference implementation"],
                "places_checked": ["github.com", "arxiv.org"],
                "links_checked": ["https://arxiv.org/abs/1905.09791"],
                "languages_checked": ["en"],
                "archives_checked": [],
                "started_at": _CREATED_AT,
                "finished_at": _CREATED_AT,
                "conclusion": "bounded search concluded",
            },
            "sources": [
                {
                    "source_id": source_id,
                    "kind": "discovery-evidence",
                    "url": f"urn:menagerie:source-discovery:{digest.removeprefix('sha256:')}",
                    "revision_kind": "source-discovery-sha256",
                    "revision": digest,
                    "content_sha256": digest,
                    "mirror_digest": digest,
                }
            ],
        },
    }


@pytest.fixture(name="reducer")
def _reducer(tmp_path: Path) -> Iterator[CanonicalReducer]:
    """Yield a REAL canonical reducer over isolated ledgers.

    The production class over real ledger paths and a real authority context, so these
    cases exercise the shipped ``_validate_source`` rather than a stand-in for it.
    """

    ledgers = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(ledgers, make_authority_context([_STABLE_ID])) as instance:
        yield instance


@pytest.mark.smoke
def test_reducer_admits_the_sentinel_for_a_blocked_tier_deferral(
    reducer: CanonicalReducer,
) -> None:
    """The tier deferral is a BLOCKED arm, so the sentinel is its honest rung.

    Before the fix the reducer's typed-discovery shape required ``R5_SKIP`` for every
    code except the access deferral -- which meant a ``deferred:needs-opus-tier`` record
    was only *acceptable* if it lied. The carve-out follows the arm, not one code.
    """

    reducer._validate_source(
        _reducer_record(status_code="deferred:needs-opus-tier", rung=NO_RUNG_SELECTED)
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "status_code",
    ["skipped:no-description", "skipped:insufficient-description", "skipped:not-a-real-NN"],
)
def test_reducer_still_refuses_the_sentinel_for_every_skipped_code(
    reducer: CanonicalReducer, status_code: str
) -> None:
    """The tripwire is untouched: a ``skipped:*`` claim still has to carry its R5.

    This is the direction that proves the reducer change relaxed nothing. Every
    ``skipped:*`` code is an epistemic claim, so the sentinel there is a record with no
    mandatory public link AND no checked conclusion -- exactly what the mandatory-link
    invariant exists to refuse.
    """

    with pytest.raises(ReductionError) as excinfo:
        reducer._validate_source(
            _reducer_record(status_code=status_code, rung=NO_RUNG_SELECTED)
        )
    # Match the specific refusal, not any error the whole record dump happens to contain.
    assert str(excinfo.value) == "missing mandatory exact public primary source link"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "status_code",
    ["skipped:no-description", "deferred:needs-opus-tier", ACCESS_BLOCKED_STATUS_CODE],
)
def test_reducer_still_accepts_a_checked_r5_typed_discovery(
    reducer: CanonicalReducer, status_code: str
) -> None:
    """A record that really did conclude ``R5_SKIP`` keeps validating everywhere."""

    reducer._validate_source(
        _reducer_record(status_code=status_code, rung=SourceRung.SKIP.value)
    )
