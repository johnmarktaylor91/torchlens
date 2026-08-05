"""A failed terminal's placeholder blames the stage that failed, not the author lane.

Rung 7 (2026-08-05, archive ``rung7-a843ca11-abort``): m9666 and m9819 published
valid R1 proposals -- the author lane SUCCEEDED -- and then died at
``failed:evidence / coverage-incomplete``. Their durable records' attempt entries
read ``author-lane-failed / not-reached``, because ``_placeholder_facts`` hardcoded
that reason for every non-exhaustion failure. That violates the standard the same
file already enforces for the BLOCKED/DEFER arms (which rewrite the placeholder
entry precisely because "blaming the one stage that worked" is a false durable
fact), and it misleads exactly the triage the records exist to serve: a census of
rung-7 read 15 ``author-lane-failed`` entries where the author lane had failed in
only a fraction of them.

The fix threads the terminal's own ``failed:<stage>`` stage into the placeholder;
these tests pin the honest projection in both directions.
"""

from __future__ import annotations

import pytest

from menagerie.crawler.constants import NO_RUNG_SELECTED, EnvironmentPhase

# Importing the driver installs ``driver_models``' dependency table; without it
# ``_placeholder_facts`` raises. Explicit so this module runs on its own.
from menagerie.crawler import driver as _driver  # noqa: F401
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.driver_models import _placeholder_facts
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute

_CREATED_AT = "2026-08-05T00:00:00Z"


def _work_item(stable_id: str = "m9666") -> WorkItem:
    """Return a minimal routed work item; only its intake identity is read here."""

    intake = IntakeItem(
        stable_id=stable_id,
        name="smp_PAN_se_resnet50",
        zoo="smp",
        variant="se_resnet50",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=stable_id,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id=stable_id, intent="core", phase=EnvironmentPhase.PYTORCH),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("stage", ["evidence", "accuracy-gate", "environment", "runner"])
def test_post_author_failure_blames_its_own_stage(stage: str) -> None:
    """A stage reached only after publication never records ``author-lane-failed``."""

    facts = _placeholder_facts(
        _work_item(),
        _CREATED_AT,
        reason_code="coverage-incomplete",
        failing_stage=stage,
    )

    resolution = facts["source_resolution"]
    (attempted,) = resolution["attempted_rungs"]
    assert attempted["reason_code"] == f"{stage}-lane-failed"
    assert attempted["rung"] == NO_RUNG_SELECTED
    # The narrative agrees with the structured field: the author published, the
    # named stage failed.
    assert f"the {stage} stage failed after the author published" in resolution["decision"]
    assert "author lane published a complete result" in (
        resolution["search_report"]["conclusion"]
    )


@pytest.mark.smoke
def test_pre_author_failure_keeps_the_historical_projection() -> None:
    """Stages at or before authoring keep the exact historical wording."""

    for stage in (None, "author", "source", "fetch"):
        facts = _placeholder_facts(
            _work_item(),
            _CREATED_AT,
            reason_code="session-crashed" if stage == "author" else None,
            failing_stage=stage,
        )
        resolution = facts["source_resolution"]
        (attempted,) = resolution["attempted_rungs"]
        expected = f"{stage}-lane-failed" if stage else "author-lane-failed"
        assert attempted["reason_code"] == expected
        assert resolution["decision"] == "source resolution did not complete"


@pytest.mark.smoke
def test_cap_exhaustion_still_wins_over_the_stage_projection() -> None:
    """An effort-exhaustion reason keeps its own honest projection, stage or not."""

    facts = _placeholder_facts(
        _work_item(),
        _CREATED_AT,
        reason_code="effort-cap-exhausted",
        failing_stage="evidence",
    )

    resolution = facts["source_resolution"]
    (attempted,) = resolution["attempted_rungs"]
    assert attempted["reason_code"] == "effort-cap-exhausted"
    assert "exhausted its effort grant" in resolution["decision"]
