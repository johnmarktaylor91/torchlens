"""A record with no authored facts must not be judged by another proposal's gate.

An authored-metadata gate adjudicates the AUTHORED facts of the exact proposal it
bound. Two terminal record shapes carry no such facts at all:

* the ARTIFACT-FREE minimal fallback -- the last rung of ``_terminalize``'s
  ladder, which exists precisely to guarantee that a model whose intended
  terminal bookkeeping refused still gets SOME honest record; and
* a typed terminal recommendation (DEFER/SKIP/BLOCKED), whose facts start from
  the same machine placeholders.

Both declare authored metadata absent (``external_metadata: None`` in
``_placeholder_facts``). Passing ``proposal=None`` into ``_find_gate`` made the
metadata-gate lookup a stable-id WILDCARD: it matched the latest accepted gate
of a DROPPED proposal generation and validated the placeholder facts against
it, which refuses deterministically.

Observed live in rung 2 of the pilot (archived at
``crawler-launch-sprint/rung-archive/rung2-80fcbe9c-census``): ``m5915``'s
primary terminal append refused on a stale artifact/gate pair
(``extraneous authored field check: identity`` -- the tripwire working), and the
artifact-free fallback then refused too (``external_metadata must be an
object``), because it inherited the same foreign accepted gate over placeholder
facts. Result: NO ``records/models`` entry at all, and the campaign died
``terminal partition invalid: missing=['m5915']``. On a permanent run-once
catalog, a model with no record is worse than a model with a failure record.

Both errors were reproduced byte-for-class against the frozen census with the
real code before the fix; with it, the same artifact-free assembly over the
same archived gates builds a ``authored_metadata_state="failed"`` record that a
real reducer hydrated with the archived artifact/attempt ledgers appends.

No check is weakened: a PROPOSED artifact's facts are still validated against
its exactly-bound gate (the third test pins that direction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import NO_RUNG_SELECTED, EnvironmentPhase

# Importing the driver installs ``driver_models``' dependency table; without it
# the terminal assembler raises. Depend on it explicitly rather than on conftest
# import order, so this module is runnable on its own.
from menagerie.crawler import driver as _driver  # noqa: F401
from menagerie.crawler.driver_contracts import DriverConfig, WorkItem
from menagerie.crawler.driver_models import _assemble_terminal_model
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.metadata import MetadataValidationError
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.tests.conftest import make_gate

# The BLOCKED-artifact factory over a realistic two-row frozen manifest; reused
# rather than re-derived so the terminal-recommendation arm here is byte-shaped
# exactly like the arm the R5 tests already pin.
from menagerie.crawler.tests.test_blocked_arm_never_claims_r5 import (
    _blocked_artifact,
)

_CREATED_AT = "2026-08-04T08:17:24Z"
_STABLE_ID = "m5915"


def _work_item() -> WorkItem:
    """Return the routed work item shaped like the model that vanished."""

    intake = IntakeItem(
        stable_id=_STABLE_ID,
        name="mixtral",
        zoo="huggingface_transformers",
        variant="",
        discovery_source="master_catalog",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=("legacy-source-unresolved",),
        variant_scope="family",
        family_representative_id=_STABLE_ID,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(
            stable_id=_STABLE_ID, intent="core", phase=EnvironmentPhase.PYTORCH
        ),
    )


def _foreign_accepted_gate() -> dict[str, Any]:
    """Return an accepted metadata gate bound to a DROPPED proposal generation.

    ``make_gate`` builds a complete, currently-prompt-hashed, fully accurate
    metadata_batch gate. Nothing in the record under test carries the proposal
    it bound, which is exactly the m5915 situation: the gate is real and
    accepted, and it has nothing to say about a record that holds only machine
    placeholder facts.
    """

    return make_gate([_STABLE_ID], gate_id="gate-foreign-accepted")


def _assemble(
    artifact: Any,
    gates: list[dict[str, Any]],
    *,
    status_code: str = "failed:runner",
    reason_code: str | None = "protocol-violation",
    terminal_gate_obtained: bool = False,
) -> dict[str, Any]:
    """Run the real terminal assembler; never a local re-derivation of it."""

    return _assemble_terminal_model(
        _work_item(),
        artifact,
        status_code,
        reason_code,
        "terminal bookkeeping failed: synthetic primary refusal",
        (),
        gates,
        DriverConfig(),
        _CREATED_AT,
        human_review=True,
        root_cause_fingerprint=None,
        terminal_gate_obtained=terminal_gate_obtained,
    )


@pytest.mark.smoke
def test_artifact_free_fallback_survives_a_foreign_accepted_metadata_gate() -> None:
    """The last-resort rung must be able to record a model the gate never saw.

    Before the fix this raised ``MetadataValidationError: external_metadata must
    be an object`` -- verbatim the ``fallback_error`` in m5915's two
    ``terminal-unrecordable`` operational events -- because the wildcard gate
    lookup validated ``_placeholder_facts`` (``external_metadata: None``)
    against the accepted gate of a proposal this record deliberately dropped.
    """

    model = _assemble(None, [_foreign_accepted_gate()])

    assert model["authored_metadata_state"] == "failed"
    assert model["status"]["code"] == "failed:runner"
    assert model["source_resolution"]["rung"] == NO_RUNG_SELECTED


@pytest.mark.smoke
def test_fallback_does_not_stamp_the_foreign_gate_as_its_own_adjudication() -> None:
    """Surviving is not enough: the record must not CLAIM the foreign gate.

    ``metadata_accepted`` over placeholder facts would assert that a checker
    accepted authored metadata this record does not carry. The honest projection
    is a gateless ``accuracy_gate``: no gate id, no verdict, not current.
    """

    model = _assemble(None, [_foreign_accepted_gate()])

    accuracy = model["accuracy_gate"]
    assert accuracy["gate_id"] is None
    assert accuracy["verdict"] is None
    assert accuracy["current"] is False
    assert accuracy["vet_identity"] is None


@pytest.mark.smoke
def test_blocked_terminal_recommendation_survives_the_same_foreign_gate(
    tmp_path: Path,
) -> None:
    """The terminal-recommendation arms hit the identical wildcard.

    A BLOCKED verdict reached one generation after an ACCEPTED proposal (the
    m5888 shape) assembles its record from placeholder-based facts too, so the
    same foreign-gate validation refused it identically. Same fix, same
    honesty: the record assembles and does not claim the foreign gate.
    """

    model = _assemble(
        _blocked_artifact(tmp_path, reason_code="needs-higher-tier"),
        [_foreign_accepted_gate()],
        status_code="failed:author",
        reason_code="needs-higher-tier",
        terminal_gate_obtained=True,
    )

    assert model["authored_metadata_state"] == "failed"
    assert model["accuracy_gate"]["gate_id"] is None


@pytest.mark.smoke
def test_a_proposed_artifact_is_still_gated_by_its_exactly_bound_gate() -> None:
    """The tripwire direction: proposed facts still refuse a mismatched gate.

    The fix scopes the gate lookup, it does not disarm the authored-facts
    validation. A PROPOSED artifact whose exactly-bound gate carries a check
    outside the machine-derived claim vocabulary (the frozen rung-2 gates named
    a grouped ``identity`` check) must still refuse -- that refusal is the
    tripwire that correctly stopped m5915's PRIMARY append.
    """

    from menagerie.crawler.metadata import validate_authored_facts_for_write
    from menagerie.crawler.tests.conftest import make_author_proposal

    proposal = make_author_proposal(_STABLE_ID)
    gate = _foreign_accepted_gate()
    gate_item = dict(gate["items"][0])
    gate_item["field_checks"] = [
        {
            "field": "identity",
            "verdict": "accurate",
            "evidence_ids": ["evidence-1"],
            "checked_source_ids": ["source-1"],
            "reason": "supported",
            "required_repair": None,
        }
    ]

    with pytest.raises(MetadataValidationError, match="identity"):
        validate_authored_facts_for_write(proposal["proposed_facts"], gate_item)
