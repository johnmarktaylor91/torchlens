"""A platform ``DEFER`` records the rung its executable handoff actually selected.

``R5_SKIP`` is the CHECKED conclusion that *no faithful source path exists*. A platform
deferral asserts the exact opposite. Its ``handoff_execution`` is REQUIRED
(``author-result-v4`` lists it in ``required`` for the defer payload alone) and carries a
complete executable proposal whose ``source_resolution.rung`` was already validated
through ``SourceRung(...)`` -- a genuine ladder rung, sentinel excluded. The model has
working code; what is missing is a host with the named capability, and the Linux deferred
sweep will run exactly that proposal.

Recording ``R5_SKIP`` there was therefore a false structured fact about a model we hold a
faithful source path FOR, and it was worse than the sibling ``BLOCKED`` case just fixed:

* the R5 certification (``authority._validate_skip_predicate``) is reachable only through
  ``_derive_skip``, which runs only for a ``skipped:*`` status -- so on a ``deferred:*``
  record nothing ever contests the claim, while ``status.funnel_counts`` counts ``rung:``
  unconditionally;
* nothing rewrites a record in place. The deferred sweep emits a NEW revision, so for any
  model the sweep never reaches, the false ``R5_SKIP`` is permanent -- on a run-once
  campaign with a whole deferred-platform lane, that is a large population of records
  asserting the strongest possible negative about models we have working code for.

Measured on the shipped assembler before this change, with an ``R1_LIBRARY`` handoff::

    handoff proposal rung  : R1_LIBRARY
    recorded source rung   : R5_SKIP
    source attempted_rungs : [('NO_RUNG_SELECTED', 'not-reached', 'author-lane-failed')]

Pinned below, in both directions:

* every platform deferral records its handoff's own rung, on both platforms and at every
  ladder rung, and never ``R5_SKIP``;
* the deferral stops blaming an author lane that ran to completion and published the very
  handoff being deferred;
* an unreadable handoff rung DEGRADES to the sentinel rather than aborting -- one
  malformed handoff may not kill a campaign, and the sentinel is strictly weaker than the
  ``R5_SKIP`` it replaces;
* the positive control: a genuine ``skipped:*`` over the SAME multi-source manifest still
  earns its ``R5_SKIP``, and ``authority`` still refuses to certify any ``skipped:*``
  predicate that does not carry one. The fix narrows one arm; it does not disable R5.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    DeferRecommendation,
    HandoffExecution,
    SkipRecommendation,
)
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    _validate_skip_predicate,
)
from menagerie.crawler.constants import (
    NO_RUNG_SELECTED,
    SKIPPED_STATUS_CODES,
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
from menagerie.crawler.status import funnel_counts
from menagerie.crawler.tests.conftest import (
    HASH,
    make_author_proposal,
    make_authority_context,
)

_CREATED_AT = "2026-07-31T00:00:00Z"
_STABLE_ID = "m_7f21c0a94be3d8115a0e"

# Every ladder rung a handoff can legitimately carry. The sentinel is deliberately absent:
# ``SourceRung`` is a pure ladder and a deferral's proposal could not have been parsed
# with anything else in it.
LADDER_RUNGS: tuple[str, ...] = tuple(member.value for member in SourceRung)

# Two distinct, realistic bodies. A one-source manifest cannot tell a projection that
# reads ``sources[0]`` from one that reads every row, and it cannot exercise the broker
# row projection at all -- a deferral reached AFTER stage 1 answered FOUND retains both
# kinds side by side. A repo fixture that froze exactly one source once hid a
# width-dependent bug for a whole sprint.
_DISCOVERY_BYTES = b'{"arm":"defer","conclusion":"located; fused CUDA kernels required"}'
_REPO_BYTES = b"class MoELayer(nn.Module):  # top-2 gating, capacity_factor=1.25\n"


def _work_item() -> WorkItem:
    """Return the routed work item behind a deferred-platform terminal."""

    intake = IntakeItem(
        stable_id=_STABLE_ID,
        name="DeepSpeed-MoE",
        zoo="deepspeed",
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
        "queries": ["DeepSpeed MoE reference implementation"],
        "places": ["github.com/microsoft/DeepSpeed", "arxiv.org"],
        "candidate_links": [
            {"url": "https://github.com/microsoft/DeepSpeed"},
            {"url": "https://arxiv.org/abs/2201.05596"},
        ],
        "languages": ["en"],
        "conclusion": "Real source located; running it needs the named platform.",
    }


def _source_manifest(tmp_path: Path) -> dict[str, Any]:
    """Freeze a realistic MULTI-source terminal manifest into a private CAS.

    Two rows with *different* bodies, ids and projections: the typed discovery-evidence
    row a stage-1 verdict always carries, and one broker-fetched repository row a
    deferral retains once stage 1 answered FOUND -- the row whose public link is what
    makes the deferral reducible at all.

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
    repo_digest = hash_bytes(_REPO_BYTES)
    for digest, content in (
        (discovery_digest, _DISCOVERY_BYTES),
        (repo_digest, _REPO_BYTES),
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
        "discovery_arm": "defer",
        "search_evidence": _search_evidence(),
        "retained_vague_text": None,
        "candidate_probe_receipts": None,
        "candidate_probes": [
            {
                "url": "https://github.com/microsoft/DeepSpeed",
                "result": "reachable",
                "attempted_at": _CREATED_AT,
            }
        ],
    }
    broker_row: dict[str, Any] = {
        "source_id": "broker-repo-deepspeed-moe",
        "broker_role": "primary-implementation",
        "url": "https://github.com/microsoft/DeepSpeed",
        "final_url": "https://github.com/microsoft/DeepSpeed",
        "revision": repo_digest,
        "content_sha256": repo_digest,
        "fetched_bytes_len": len(_REPO_BYTES),
        "retrieval_status": "fetched",
        "media_type": "text/html",
        "requested_repo": True,
        "cas_path": str(cas_root / f"{repo_digest.removeprefix('sha256:')}.source"),
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


def _handoff(rung: Any) -> HandoffExecution:
    """Return one separately authenticated handoff carrying an exact ladder rung.

    Only the selected rung is varied. Everything else is the shipped author-proposal
    fixture, so the value under test is the one the assembler is supposed to read.

    Parameters
    ----------
    rung:
        Value to place in the handoff proposal's ``source_resolution.rung``. A
        non-``SourceRung`` value stands for a malformed or migrated handoff.
    """

    proposal = deepcopy(make_author_proposal(_STABLE_ID))
    proposal["proposal_id"] = "proposal-defer-1"
    if rung is None:
        proposal["proposed_facts"]["source_resolution"].pop("rung", None)
    else:
        proposal["proposed_facts"]["source_resolution"]["rung"] = rung
    return HandoffExecution(
        proposal=proposal,
        proposal_sha256=HASH,
        code_manifest_identity=HASH,
        source_manifest_identity=HASH,
        handoff_sha256=HASH,
    )


def _defer_artifact(
    tmp_path: Path,
    *,
    platform: str = "cuda",
    rung: Any = SourceRung.LIBRARY.value,
    public_source_only: bool = False,
) -> AuthorArtifact:
    """Stage one platform-deferral artifact over the multi-source manifest.

    Parameters
    ----------
    tmp_path:
        Private staging root.
    platform:
        Deferred platform capability (``cuda`` or ``x86``).
    rung:
        Ladder rung carried by the executable handoff.
    public_source_only:
        Retain only the broker row, the shape whose exact public primary link is what
        ``reducer._validate_source`` requires of a non-failure terminal.
    """

    manifest = _source_manifest(tmp_path)
    rows = [
        row
        for row in manifest["sources"]
        if not public_source_only or row.get("broker_role") is not None
    ]
    result = DeferRecommendation(
        binding=_binding("DEFER"),
        platform=platform,
        source_ids=tuple(str(row["source_id"]) for row in rows),
        evidence_ids=("ev-defer-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
        handoff_execution=_handoff(rung),
    )
    return AuthorArtifact(
        author_result=result, source_manifest=manifest, model_dir=tmp_path / "model"
    )


def _skip_artifact(tmp_path: Path) -> AuthorArtifact:
    """Stage one genuine SKIP terminal artifact over the SAME manifest.

    The positive control. It differs from the deferral in exactly one thing -- the arm --
    so a fix that merely stopped stamping ``R5_SKIP`` everywhere would be caught here
    rather than read as success.
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
    terminal_gate_obtained: bool = True,
) -> dict[str, Any]:
    """Run the real terminal assembler; never a local re-derivation of it."""

    return _assemble_terminal_model(
        _work_item(),
        artifact,
        status_code,
        None,
        None,
        [],
        [],
        DriverConfig(),
        _CREATED_AT,
        human_review=False,
        root_cause_fingerprint="fingerprint-defer",
        terminal_gate_obtained=terminal_gate_obtained,
    )


# ---------------------------------------------------------------------------
# 1. the defect: a deferral stamped with a checked no-source conclusion
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("platform", ["cuda", "x86"])
@pytest.mark.parametrize("rung", LADDER_RUNGS)
def test_platform_defer_records_the_rung_its_handoff_selected(
    tmp_path: Path, platform: str, rung: str
) -> None:
    """The recorded rung is the handoff's own, on either platform, at every rung.

    This is the failing direction. Before the fix every row here recorded ``R5_SKIP``
    regardless of what the handoff selected -- including the ``R1_LIBRARY`` case, where
    the campaign holds a complete declarative library recipe the Linux sweep will run.
    """

    model = _assemble(
        _defer_artifact(tmp_path, platform=platform, rung=rung),
        status_code=f"deferred:needs-{platform}",
    )

    assert model["source_resolution"]["rung"] == rung


@pytest.mark.smoke
@pytest.mark.parametrize("rung", [value for value in LADDER_RUNGS if value != "R5_SKIP"])
def test_a_deferral_with_working_code_is_never_recorded_as_unresolvable(
    tmp_path: Path, rung: str
) -> None:
    """No deferral holding an executable non-R5 handoff may assert ``R5_SKIP``.

    Stated as its own claim rather than as a corollary of the equality above: this is the
    assertion whose violation is permanent, because nothing on a ``deferred:*`` record
    ever re-checks an R5 and no later revision rewrites this one in place.
    """

    model = _assemble(
        _defer_artifact(tmp_path, rung=rung), status_code="deferred:needs-cuda"
    )

    assert model["source_resolution"]["rung"] != SourceRung.SKIP.value


@pytest.mark.smoke
def test_platform_defer_stops_blaming_the_author_lane_that_worked(
    tmp_path: Path,
) -> None:
    """The placeholder's ``author-lane-failed`` cannot survive beside a selected rung.

    ``author-lane-failed / not-reached`` is what ``_placeholder_facts`` writes when the
    model-local lane died before source resolution. An adjudicated deferral is the
    opposite: the author lane ran to completion and published the very handoff being
    deferred. Leaving it would make the corrected block contradict itself.
    """

    model = _assemble(
        _defer_artifact(tmp_path, rung=SourceRung.LIBRARY.value),
        status_code="deferred:needs-cuda",
    )

    attempted = model["source_resolution"]["attempted_rungs"]
    assert [entry["rung"] for entry in attempted] == [SourceRung.LIBRARY.value]
    assert [entry["result"] for entry in attempted] == ["selected-deferred-to-platform"]
    assert [entry["reason_code"] for entry in attempted] == ["needs-cuda"]
    assert "author-lane-failed" not in {entry["reason_code"] for entry in attempted}
    # The cited evidence is this record's own terminal evidence, never the handoff
    # proposal's ids, which name a different evidence block entirely.
    assert [entry["evidence_ids"] for entry in attempted] == [["ev-defer-1"]]


@pytest.mark.smoke
def test_the_defer_and_skip_arms_are_not_vacuously_equal(tmp_path: Path) -> None:
    """Guard the comparisons above against passing on identical slots.

    An all-placeholder negative once passed because both compared values happened to be
    the same string. Pin the two arms against each other over the SAME manifest, with
    values that are distinct by construction.
    """

    deferred = _assemble(
        _defer_artifact(tmp_path / "defer", rung=SourceRung.VENDOR.value),
        status_code="deferred:needs-cuda",
    )
    skipped = _assemble(
        _skip_artifact(tmp_path / "skip"), status_code="skipped:no-description"
    )

    assert deferred["source_resolution"]["rung"] == SourceRung.VENDOR.value
    assert skipped["source_resolution"]["rung"] == SourceRung.SKIP.value
    assert deferred["source_resolution"]["rung"] != skipped["source_resolution"]["rung"]
    deferred_results = [
        entry["result"] for entry in deferred["source_resolution"]["attempted_rungs"]
    ]
    skipped_results = [
        entry["result"] for entry in skipped["source_resolution"]["attempted_rungs"]
    ]
    assert deferred_results == ["selected-deferred-to-platform"]
    assert skipped_results == ["bounded-negative-discovery"]
    assert deferred_results != skipped_results


@pytest.mark.smoke
def test_an_unadjudicated_platform_defer_keeps_the_sentinel(tmp_path: Path) -> None:
    """No disposition gate means nothing was adjudicated, so no rung is awarded."""

    model = _assemble(
        _defer_artifact(tmp_path, rung=SourceRung.LIBRARY.value),
        status_code="deferred:needs-cuda",
        terminal_gate_obtained=False,
    )

    resolution = model["source_resolution"]
    assert resolution["rung"] == NO_RUNG_SELECTED
    assert [entry["reason_code"] for entry in resolution["attempted_rungs"]] == [
        "terminal-disposition-gate-unavailable"
    ]


@pytest.mark.smoke
@pytest.mark.parametrize("rung", [None, "R9_INVENTED", NO_RUNG_SELECTED, "", 7])
def test_an_unreadable_handoff_rung_degrades_and_never_aborts(
    tmp_path: Path, rung: Any
) -> None:
    """A malformed handoff yields the sentinel, not ``R5_SKIP`` and not an exception.

    Totality matters here for the same reason it does for the BLOCKED terminal mapping:
    one odd model may not kill a 28k-model campaign. The sentinel is also strictly weaker
    than the ``R5_SKIP`` it replaces, so degrading never manufactures a stronger claim.
    """

    model = _assemble(
        _defer_artifact(tmp_path, rung=rung), status_code="deferred:needs-cuda"
    )

    resolution = model["source_resolution"]
    assert resolution["rung"] == NO_RUNG_SELECTED
    assert [entry["result"] for entry in resolution["attempted_rungs"]] == ["not-reached"]


@pytest.mark.smoke
def test_the_funnel_stops_counting_a_deferred_model_as_an_r5_skip(
    tmp_path: Path,
) -> None:
    """``funnel_counts`` reads the structured field unconditionally -- so it must be true.

    This is the consumer the whole defect routes through: no narrative repairs it, and on
    a run-once campaign the count is what the roster is read from.
    """

    model = _assemble(
        _defer_artifact(tmp_path, rung=SourceRung.LIBRARY.value),
        status_code="deferred:needs-cuda",
    )

    counts = funnel_counts([model])
    assert counts["rung:R1_LIBRARY"] == 1
    assert "rung:R5_SKIP" not in counts
    assert counts["status:deferred:needs-cuda"] == 1


# ---------------------------------------------------------------------------
# 2. the positive control: R5 stays exactly as reachable, and exactly as checked
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_genuine_skip_over_the_same_manifest_still_earns_r5(tmp_path: Path) -> None:
    """Narrowing the deferral stamp must not make an honest R5 unreachable.

    Same manifest, same discovery evidence, same gate: only the arm differs. A SKIP is
    the one arm whose closed terminal IS ``skipped:*``, so its ``R5_SKIP`` is the claim
    ``authority._derive_skip`` will actually re-check.
    """

    model = _assemble(_skip_artifact(tmp_path), status_code="skipped:no-description")

    resolution = model["source_resolution"]
    assert resolution["rung"] == SourceRung.SKIP.value
    assert [entry["rung"] for entry in resolution["attempted_rungs"]] == [
        SourceRung.SKIP.value
    ]


def _skip_source_resolution(rung: str, predicate: str) -> dict[str, Any]:
    """Return the accepted R5 facts ``_validate_skip_predicate`` re-checks.

    Each predicate carries the material its own clause demands -- the sufficiency gap for
    ``insufficient-description``, its absence for ``no-description`` -- so a refusal in
    these cases can only ever be the rung clause under test, never a missing field.
    """

    insufficient = predicate == "insufficient-description"
    return {
        "rung": rung,
        "sufficiency_gap": (
            "retained description lacks the layer widths needed to specify a forward pass"
            if insufficient
            else None
        ),
        "search_report": {"conclusion": "bounded search concluded over named locators"},
    }


def _skip_evidence(predicate: str) -> tuple[dict[str, Any], ...]:
    """Return the literal excerpt each typed skip predicate is checked against."""

    return (
        {
            "evidence_id": "ev-skip-1",
            "disposition": (
                "insufficient-for-faithful-reimpl"
                if predicate == "insufficient-description"
                else "supporting"
            ),
            "supports": [predicate, f"skipped:{predicate}"],
        },
    )


@pytest.mark.smoke
@pytest.mark.parametrize("status_code", sorted(SKIPPED_STATUS_CODES))
@pytest.mark.parametrize("rung", [value for value in LADDER_RUNGS if value != "R5_SKIP"])
def test_every_skipped_code_still_requires_its_checked_r5(
    status_code: str, rung: str
) -> None:
    """A ``skipped:*`` predicate is refused for every non-R5 rung, sentinel included.

    This is the direction that proves nothing here relaxes the tripwire. If a deferral's
    propagated rung ever leaked onto a skip, the R5 certification still refuses it under
    its own name.
    """

    predicate = status_code.removeprefix("skipped:")
    with pytest.raises(AuthorityDerivationError) as excinfo:
        _validate_skip_predicate(
            predicate,
            _skip_source_resolution(rung, predicate),
            _skip_evidence(predicate),
            ("ev-skip-1",),
        )
    # Exact equality, never a substring: an error that dumps the whole instance would
    # make a substring assertion pass regardless of which refusal fired.
    assert str(excinfo.value) == "skip proof does not resolve to R5 source facts"


@pytest.mark.smoke
@pytest.mark.parametrize("status_code", sorted(SKIPPED_STATUS_CODES))
def test_every_skipped_code_still_refuses_the_sentinel(status_code: str) -> None:
    """The sentinel is refused under its own name, ahead of the ladder comparison."""

    predicate = status_code.removeprefix("skipped:")
    with pytest.raises(AuthorityDerivationError) as excinfo:
        _validate_skip_predicate(
            predicate,
            _skip_source_resolution(NO_RUNG_SELECTED, predicate),
            _skip_evidence(predicate),
            ("ev-skip-1",),
        )
    assert str(excinfo.value) == "no rung was selected, so no epistemic R5 skip can be proven"


@pytest.mark.smoke
@pytest.mark.parametrize("status_code", sorted(SKIPPED_STATUS_CODES))
def test_a_certified_r5_skip_still_passes_for_every_skipped_code(
    status_code: str,
) -> None:
    """The positive half of the same check: a real R5 is still accepted.

    Without this, the two refusal cases above would also pass if R5 had been disabled
    outright, which is precisely the failure mode this file must not be able to hide.
    """

    predicate = status_code.removeprefix("skipped:")
    _validate_skip_predicate(
        predicate,
        _skip_source_resolution(SourceRung.SKIP.value, predicate),
        _skip_evidence(predicate),
        ("ev-skip-1",),
    )


# ---------------------------------------------------------------------------
# 3. the reducer accepts the corrected record and never demanded the lie
# ---------------------------------------------------------------------------


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
@pytest.mark.parametrize("rung", LADDER_RUNGS + (NO_RUNG_SELECTED,))
def test_every_reducer_rule_a_platform_defer_meets_is_rung_agnostic(
    reducer: CanonicalReducer, tmp_path: Path, rung: str
) -> None:
    """No rule on this arm was ever satisfiable only by the false ``R5_SKIP``.

    The sibling ``BLOCKED`` fix had to change a rule that *required* ``R5_SKIP`` for the
    shape it governed -- a rule satisfiable only by an untrue record, which is what forced
    the driver to write the lie. This arm has no such rule, and that is worth pinning
    rather than assuming: ``deferred:needs-cuda`` sits outside the typed-discovery
    relaxation entirely, and the deferral rules are rung-blind. So the driver wrote its
    falsehood UNFORCED, purely by sharing one stamp with the SKIP arm, and correcting it
    needed no reducer change at all.

    ``_validate_gates`` is included deliberately: it is the one rule that keys on the
    rung (``R3_PORT``/``R4_REIMPLEMENT`` require a current fidelity gate), so a corrected
    ``R3``/``R4`` deferral is exactly where a propagated rung could have made the record
    unreducible. It does not, because a deferral is a pre-fidelity terminal.
    """

    model = _assemble(
        _defer_artifact(tmp_path, rung=rung, public_source_only=True),
        status_code="deferred:needs-cuda",
    )
    assert model["source_resolution"]["mandatory_link_status"] == "ok"

    reducer._validate_source(model)
    reducer._validate_status(model)
    reducer._validate_gates(model)
    reducer._validate_completeness(model)
    reducer._validate_execution(model)


@pytest.mark.smoke
def test_a_deferral_without_its_public_link_is_still_refused(
    reducer: CanonicalReducer, tmp_path: Path,
) -> None:
    """The corrected rung buys no relaxation of the mandatory-source-link invariant.

    A deferral retaining only its ``urn:`` discovery row has no exact public primary link
    and no typed-discovery carve-out, so it is refused -- at ``R1_LIBRARY`` exactly as it
    was at ``R5_SKIP``.
    """

    model = _assemble(
        _defer_artifact(tmp_path, rung=SourceRung.LIBRARY.value),
        status_code="deferred:needs-cuda",
    )

    with pytest.raises(ReductionError) as excinfo:
        reducer._validate_source(model)
    assert str(excinfo.value) == "missing mandatory exact public primary source link"
