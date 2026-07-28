"""``NO_RUNG_SELECTED`` must be representable, countable, and powerless.

This regression exists because a model whose source was *found, fetched and fully
understood at R1* was recorded as ``source_resolution.rung = "R5_SKIP"`` (observed live on
``m2323``, PyKEEN-MuRE). It hit its effort cap before a rung was selected, and the enum had
no honest member for that, so the code wrote the one value that asserts the opposite of
what happened.

Why the surrounding narrative did not repair it, which is the argument that settled the
design debate:

* ``status.funnel_counts`` counts **every** record by ``source_resolution.rung``, failures
  included -- the structured field, not the prose, is what every consumer reads;
* a genuine R5 carries a separately certified epistemic predicate in ``authority``;
* the closed schemas admitted only R1-R5, so the builder wrote ``R5_SKIP`` while *knowing*
  nothing had been selected.

At 28,482 models in a run-once campaign that permanently conflates budget casualties with
genuine "no faithful source path exists", poisoning retry targeting, R5-rate analysis, and
any later triage of what still needs work.

What must NOT regress, pinned below in both directions:

* work that ended before the ladder was walked carries the sentinel, never ``R5_SKIP``;
* the funnel counts the sentinel as its own category and does not inflate the R5 count;
* the sentinel can never satisfy the certified R5 epistemic predicate, at either the source
  facts or the gate -- the tripwire is *stricter* than before, not merely reshaped;
* a genuine checked R5 still validates and still counts as R5;
* the sentinel is structurally inexpressible everywhere a real ladder rung is required
  (author proposals, checker gate rung checks), and a real rung still validates there.
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.authority import AuthorityDerivationError, _validate_skip_predicate
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    GATE_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION_V3,
    NO_RUNG_SELECTED,
    SOURCE_RESOLUTION_RUNG_VALUES,
    EnvironmentPhase,
    SourceRung,
    StatusKind,
)
from menagerie.crawler.driver_contracts import WorkItem

# Importing the driver is what installs ``driver_models``' dependency table; without it
# ``_placeholder_facts`` raises. Depend on it explicitly rather than on conftest's import
# order, so this module is runnable on its own.
from menagerie.crawler import driver as _driver  # noqa: F401
from menagerie.crawler.driver_models import _placeholder_facts
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.schema import _resolve_schema_reference, load_schema
from menagerie.crawler.status import _status_completeness_failures, funnel_counts

_CREATED_AT = "2026-07-28T00:00:00Z"

# The reason codes the driver maps onto "the author never got to pick a rung". The first
# two are cap exhaustion; the third is the generic author-stage failure.
CAP_EXHAUSTED_REASONS = (
    "effort-cap-exhausted",
    "effort-exhausted:tool-calls",
    "effort-exhausted:wall-seconds",
)
AUTHOR_FAILURE_REASONS = (None, "session-crashed", "research-tools-unavailable")


def _work_item() -> WorkItem:
    """Return a minimal routed work item; only its intake identity is read here."""

    intake = IntakeItem(
        stable_id="m2323",
        name="MuRE",
        zoo="pykeen",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id="m2323",
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id="m2323", intent="core", phase=EnvironmentPhase.PYTORCH),
    )


def _deref(root: Any, node: dict[str, Any], label: str) -> tuple[Any, dict[str, Any]]:
    """Chase ``$ref`` through the real crawler resolver to a concrete node.

    The v3 schemas are thin: most ``$defs`` entries are bare ``$ref`` passthroughs into the
    ``*-common`` resources, so a naive pointer walk never reaches the enum. Reusing the
    production resolver means these tests read the same graph the validator does.
    """

    seen: set[str] = set()
    while "$ref" in node and "enum" not in node:
        reference = node["$ref"]
        assert isinstance(reference, str), f"{label} has a malformed $ref"
        assert reference not in seen, f"cyclic $ref at {label}"
        seen.add(reference)
        root, resolved = _resolve_schema_reference(root, reference)
        node = dict(resolved)
    return root, node


def _walk(schema: Any, pointer: str) -> tuple[Any, dict[str, Any]]:
    """Follow one ``#/`` JSON pointer, dereferencing at every step."""

    root: Any = schema
    node: dict[str, Any] = dict(schema)
    for token in pointer.lstrip("#/").split("/"):
        root, node = _deref(root, node, pointer)
        child = node[token]
        assert isinstance(child, dict), f"{pointer} does not address a schema node"
        node = dict(child)
    return root, node


def _resolve(schema: Any, pointer: str) -> dict[str, Any]:
    """Return the node one pointer addresses, without forcing it to an enum."""

    return _walk(schema, pointer)[1]


def _follow_to_enum(root: Any, node: dict[str, Any], label: str) -> set[str]:
    """Return the closed enum reached from one node."""

    _, resolved = _deref(root, dict(node), label)
    assert "enum" in resolved, f"{label} is not a closed enum"
    return set(resolved["enum"])


def _enum_at(schema_version: str, pointer: str) -> set[str]:
    """Return the closed enum a schema admits at one pointer, following ``$ref``."""

    root, node = _walk(load_schema(schema_version), pointer)
    return _follow_to_enum(root, node, pointer)


def _model_record(*, rung: str, code: str) -> dict[str, Any]:
    """Build the minimal record shape the funnel and completeness checks inspect."""

    kind = code.split(":", 1)[0]
    return {
        "stable_id": "m2323",
        "status": {
            "code": code,
            "kind": kind,
            "stage": code.split(":", 1)[1] if kind == "failed" else None,
            "reason_code": "effort-cap-exhausted" if kind == "failed" else None,
            "traceback": None,
            "no_traceback_reason": "author never started" if kind == "failed" else None,
            "root_cause_fingerprint": "fingerprint" if kind == "failed" else None,
            "human_review": {"required": True},
        },
        "source_resolution": {"rung": rung},
        "authored_metadata_state": "failed",
        "implementation": {"run_framework": "pytorch"},
        "modes": {"meaningful_modes": []},
        "completeness": {"source_read_fields_complete": False},
        "fidelity": {"required": False, "current": False},
        "execution": {"accepted_attempt_ids": []},
    }


# ---------------------------------------------------------------------------
# 1. the record itself
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("reason_code", CAP_EXHAUSTED_REASONS)
def test_cap_exhausted_record_carries_the_sentinel_not_r5(reason_code: str) -> None:
    """The original defect: cap exhaustion claimed a checked no-source conclusion.

    ``m2323``'s source was found, fetched and understood at R1. The session then ran out of
    its effort grant. That is a budget outcome, and the record must say so structurally.
    """

    facts = _placeholder_facts(_work_item(), _CREATED_AT, reason_code=reason_code)

    resolution = facts["source_resolution"]
    assert resolution["rung"] == NO_RUNG_SELECTED
    assert resolution["rung"] != SourceRung.SKIP.value
    assert [entry["rung"] for entry in resolution["attempted_rungs"]] == [NO_RUNG_SELECTED]


@pytest.mark.smoke
@pytest.mark.parametrize("reason_code", AUTHOR_FAILURE_REASONS)
def test_author_failed_record_carries_the_sentinel_not_r5(reason_code: str | None) -> None:
    """Any author-stage failure ends without a selected rung, cap-related or not."""

    facts = _placeholder_facts(_work_item(), _CREATED_AT, reason_code=reason_code)

    assert facts["source_resolution"]["rung"] == NO_RUNG_SELECTED


@pytest.mark.smoke
def test_placeholder_narrative_still_refuses_to_imply_a_bounded_search() -> None:
    """The honest narrative is kept as well as the honest structure, not instead of it."""

    facts = _placeholder_facts(
        _work_item(), _CREATED_AT, reason_code="effort-cap-exhausted"
    )

    resolution = facts["source_resolution"]
    assert "exhausted its effort grant before a rung was selected" in resolution["decision"]
    assert "unfinished, not unresolvable" in resolution["search_report"]["conclusion"]


# ---------------------------------------------------------------------------
# 2. the funnel counts it as itself
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_funnel_counts_the_sentinel_as_its_own_category() -> None:
    """The evidence that defeated 'an honest narrative repairs it'.

    ``funnel_counts`` reads the structured field unconditionally, so a cap casualty
    recorded as ``R5_SKIP`` was indistinguishable from a genuine skip in every report.
    """

    counts = funnel_counts(
        [
            _model_record(rung=NO_RUNG_SELECTED, code="failed:author"),
            _model_record(rung=NO_RUNG_SELECTED, code="failed:source"),
            _model_record(rung=SourceRung.SKIP.value, code="skipped:no-description"),
        ]
    )

    assert counts[f"rung:{NO_RUNG_SELECTED}"] == 2
    assert counts[f"rung:{SourceRung.SKIP.value}"] == 1


@pytest.mark.smoke
def test_sentinel_does_not_inflate_the_r5_count() -> None:
    """Before the migration these three records reported ``rung:R5_SKIP: 3``."""

    counts = funnel_counts(
        [
            _model_record(rung=NO_RUNG_SELECTED, code="failed:author"),
            _model_record(rung=NO_RUNG_SELECTED, code="failed:source"),
            _model_record(rung=NO_RUNG_SELECTED, code="failed:fetch"),
        ]
    )

    assert counts.get(f"rung:{SourceRung.SKIP.value}", 0) == 0
    assert counts[f"rung:{NO_RUNG_SELECTED}"] == 3


# ---------------------------------------------------------------------------
# 3. the authority guard -- the sentinel can never buy R5 authority
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sentinel_cannot_satisfy_the_r5_epistemic_predicate() -> None:
    """The guard that makes this a strengthening rather than a relabel."""

    with pytest.raises(AuthorityDerivationError, match="no rung was selected"):
        _validate_skip_predicate(
            "no-description",
            {
                "rung": NO_RUNG_SELECTED,
                "search_report": {"conclusion": "the author ran out of effort grant"},
                "sufficiency_gap": "anything",
            },
            [],
            (),
        )


@pytest.mark.smoke
def test_sentinel_cannot_satisfy_the_r5_predicate_even_with_complete_evidence() -> None:
    """A sentinel record carrying an otherwise-well-formed skip pack is still refused.

    This is the case that matters: the refusal must key on "nothing was selected", not on
    some incidental missing field that a future record might happen to fill in.
    """

    excerpt = {
        "evidence_id": "e1",
        "source_id": "s1",
        "disposition": "insufficient-for-faithful-reimpl",
    }
    with pytest.raises(AuthorityDerivationError, match="no rung was selected"):
        _validate_skip_predicate(
            "insufficient-description",
            {
                "rung": NO_RUNG_SELECTED,
                "search_report": {"conclusion": "bounded search"},
                "sufficiency_gap": "no implementation detail",
            },
            [excerpt],
            ("e1",),
        )


@pytest.mark.smoke
@pytest.mark.parametrize("kind", sorted({StatusKind.RUNS.value, StatusKind.SKIPPED.value}))
def test_sentinel_is_incoherent_under_runs_and_skipped_statuses(kind: str) -> None:
    """A record that runs necessarily selected a rung; every skip must carry a checked R5."""

    code = "runs" if kind == StatusKind.RUNS.value else "skipped:no-description"
    failures = _status_completeness_failures(
        [_model_record(rung=NO_RUNG_SELECTED, code=code)]
    )

    assert f"m2323:no-rung-selected-under-{kind}" in failures


@pytest.mark.smoke
def test_sentinel_is_coherent_under_failed_statuses() -> None:
    """The sentinel's whole purpose: work that genuinely stopped short may record it."""

    failures = _status_completeness_failures(
        [_model_record(rung=NO_RUNG_SELECTED, code="failed:author")]
    )

    assert not [failure for failure in failures if "no-rung-selected" in failure]


# ---------------------------------------------------------------------------
# 4. a genuine R5 is untouched
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_genuine_r5_still_satisfies_the_epistemic_predicate() -> None:
    """A checked conclusion that no faithful source path exists still validates."""

    excerpt = {
        "evidence_id": "e1",
        "source_id": "s1",
        "disposition": "insufficient-for-faithful-reimpl",
    }
    _validate_skip_predicate(
        "insufficient-description",
        {
            "rung": SourceRung.SKIP.value,
            "search_report": {"conclusion": "bounded search found no usable code"},
            "sufficiency_gap": "retained description lacks implementation detail",
        },
        [excerpt],
        ("e1",),
    )


@pytest.mark.smoke
def test_genuine_r5_still_counts_as_r5_and_stays_coherent_when_skipped() -> None:
    """The R5 population must survive the migration intact, not shrink into the sentinel."""

    record = _model_record(rung=SourceRung.SKIP.value, code="skipped:no-description")

    assert funnel_counts([record])[f"rung:{SourceRung.SKIP.value}"] == 1
    assert not [
        failure
        for failure in _status_completeness_failures([record])
        if "no-rung-selected" in failure
    ]


# ---------------------------------------------------------------------------
# 5. the closed vocabularies, in both directions
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_ladder_enum_stays_a_pure_ladder() -> None:
    """``SourceRung`` must not gain the sentinel: rung ordering and author parsing key on it."""

    assert NO_RUNG_SELECTED not in {member.value for member in SourceRung}
    assert SOURCE_RESOLUTION_RUNG_VALUES == {member.value for member in SourceRung} | {
        NO_RUNG_SELECTED
    }


@pytest.mark.smoke
@pytest.mark.parametrize(
    "pointer",
    [
        "#/$defs/source_resolution/properties/rung",
        "#/$defs/attempted_rung/properties/rung",
        "#/$defs/status/properties/attempted_rungs/items",
    ],
)
def test_model_record_admits_the_sentinel_where_no_rung_may_have_been_selected(
    pointer: str,
) -> None:
    """The sentinel must be *expressible* on a driver-observed record, or it is unusable."""

    assert _enum_at(MODEL_SCHEMA_VERSION_V3, pointer) == SOURCE_RESOLUTION_RUNG_VALUES


@pytest.mark.smoke
@pytest.mark.parametrize(
    "pointer",
    [
        "#/$defs/rung_check/properties/selected_rung",
        "#/$defs/rung_check/properties/highest_applicable",
    ],
)
def test_checker_gate_rejects_the_sentinel_where_a_real_rung_is_required(
    pointer: str,
) -> None:
    """A gate rung check is a positive checker judgement about a ladder rung.

    It has no honest reading of "nothing was selected", so the sentinel must be
    structurally inexpressible rather than merely discouraged.
    """

    admitted = _enum_at(GATE_SCHEMA_VERSION_V3, pointer)
    assert NO_RUNG_SELECTED not in admitted
    assert admitted == {member.value for member in SourceRung}


@pytest.mark.smoke
def test_author_proposal_rejects_the_sentinel_where_a_real_rung_is_required() -> None:
    """The sentinel is a driver-observed outcome; an author may only name a real rung."""

    schema = load_schema(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    node = _resolve(schema, "#/$defs/proposed_facts/properties/source_resolution")
    narrowing = [
        branch for branch in node["allOf"] if "properties" in branch and "rung" in branch["properties"]
    ]

    assert narrowing, "author proposals must narrow rung back to the strict ladder"
    admitted = _follow_to_enum(
        schema, narrowing[0]["properties"]["rung"], "author proposal source_resolution.rung"
    )
    assert NO_RUNG_SELECTED not in admitted
    assert admitted == {member.value for member in SourceRung}
