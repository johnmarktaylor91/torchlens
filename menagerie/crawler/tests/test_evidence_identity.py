"""Trusted-intake identity coverage regressions."""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.driver_admission import _validate_trusted_intake_identity
from menagerie.crawler.driver_contracts import DriverIntegrationError, WorkItem
from menagerie.crawler.evidence import trusted_intake_identity_mismatches
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import EnvironmentPhase, IntentRoute


FROZEN_SMP_REPLAYS = (
    pytest.param(
        {
            "intake_name": "smp_PAN_se_resnet50",
            "intake_zoo": "segmentation_models_pytorch-0.5.0",
            "proposed_variant": "se_resnet50",
            "stable_id": "m9666",
        },
        id="m9666-pan-se-resnet50",
    ),
    pytest.param(
        {
            "intake_name": "smp_Segformer_se_resnext101_32x4d",
            "intake_zoo": "segmentation_models_pytorch-0.5.0",
            "proposed_variant": "se_resnext101_32x4d",
            "stable_id": "m9819",
        },
        id="m9819-segformer-se-resnext101",
    ),
)


def _trusted_identity(case: dict[str, str]) -> dict[str, str]:
    """Return the trusted identity projection from the frozen replay row.

    Parameters
    ----------
    case:
        Minimal frozen replay fixture.

    Returns
    -------
    dict[str, str]
        Trusted identity leaves projected by the intake layer at rung 7.
    """

    return {
        "variant": case["intake_name"],
        "variant_scope": "family",
        "family_representative_id": case["stable_id"],
    }


def _proposed_identity(case: dict[str, str], **overrides: Any) -> dict[str, Any]:
    """Return the author-proposed identity from the frozen replay row.

    Parameters
    ----------
    case:
        Minimal frozen replay fixture.
    overrides:
        Identity fields to replace for negative coverage.

    Returns
    -------
    dict[str, Any]
        Proposed identity leaves from the author result.
    """

    identity: dict[str, Any] = {
        "variant": case["proposed_variant"],
        "variant_scope": "family",
        "family_representative_id": case["stable_id"],
    }
    identity.update(overrides)
    return identity


def _intake_item(case: dict[str, str]) -> IntakeItem:
    """Return a real trusted intake row for a frozen SMP replay case.

    Parameters
    ----------
    case:
        Minimal frozen replay fixture.

    Returns
    -------
    IntakeItem
        Trusted intake row whose identity projection matches the rung-7 row.
    """

    return IntakeItem(
        stable_id=case["stable_id"],
        name=case["intake_name"],
        zoo=case["intake_zoo"],
        variant="",
        discovery_source="frozen-rung7-replay",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="family",
        family_representative_id=case["stable_id"],
    )


def _work_item(case: dict[str, str]) -> WorkItem:
    """Return the minimal work item shape consumed by the trusted-intake validator.

    Parameters
    ----------
    case:
        Minimal frozen replay fixture.

    Returns
    -------
    WorkItem
        Real driver work item carrying the trusted intake row.
    """

    return WorkItem(
        intake=_intake_item(case),
        route=IntentRoute(
            stable_id=case["stable_id"],
            intent="core",
            phase=EnvironmentPhase.PYTORCH,
        ),
    )


def _facts(case: dict[str, str], **identity_overrides: Any) -> dict[str, Any]:
    """Return canonical proposal facts with identity overrides applied.

    Parameters
    ----------
    case:
        Minimal frozen replay fixture.
    identity_overrides:
        Identity fields to replace for negative coverage.

    Returns
    -------
    dict[str, Any]
        Proposal fact block accepted by the real trusted-intake validator.
    """

    return {"identity": _proposed_identity(case, **identity_overrides)}


@pytest.mark.parametrize("case", FROZEN_SMP_REPLAYS)
def test_smp_legacy_full_token_accepts_source_grounded_encoder_variant(
    case: dict[str, str],
) -> None:
    """Replay m9666/m9819: SMP roster names prove their encoder variant suffix.

    Parameters
    ----------
    case:
        Minimal frozen m9666 or m9819 replay fixture.
    """

    assert (
        trusted_intake_identity_mismatches(
            _proposed_identity(case),
            _trusted_identity(case),
            intake_name=case["intake_name"],
            intake_zoo=case["intake_zoo"],
        )
        == {}
    )


@pytest.mark.parametrize("case", FROZEN_SMP_REPLAYS)
def test_driver_trusted_intake_identity_accepts_frozen_smp_replay(case: dict[str, str]) -> None:
    """Replay m9666/m9819 through the real driver trusted-intake tripwire.

    Parameters
    ----------
    case:
        Minimal frozen m9666 or m9819 replay fixture.
    """

    _validate_trusted_intake_identity(_facts(case), _work_item(case))


def test_driver_trusted_intake_identity_rejects_genuine_variant_mismatch() -> None:
    """The real driver tripwire still refuses an unrelated SMP variant claim."""

    case = {
        "intake_name": "smp_PAN_se_resnet50",
        "intake_zoo": "segmentation_models_pytorch-0.5.0",
        "proposed_variant": "se_resnet50",
        "stable_id": "m9666",
    }

    with pytest.raises(DriverIntegrationError, match="identity contradicts trusted intake"):
        _validate_trusted_intake_identity(
            _facts(case, variant="se_resnet101"),
            _work_item(case),
        )


@pytest.mark.parametrize("case", FROZEN_SMP_REPLAYS)
def test_smp_equivalence_keeps_non_variant_identity_fields_strict(case: dict[str, str]) -> None:
    """SMP variant normalization does not launder representative drift.

    Parameters
    ----------
    case:
        Minimal frozen m9666 or m9819 replay fixture.
    """

    mismatches = trusted_intake_identity_mismatches(
        _proposed_identity(case, family_representative_id="m_other"),
        _trusted_identity(case),
        intake_name=case["intake_name"],
        intake_zoo=case["intake_zoo"],
    )

    assert mismatches == {
        "family_representative_id": {"proposed": "m_other", "trusted": case["stable_id"]}
    }


def test_smp_equivalence_rejects_unrelated_encoder_suffix() -> None:
    """Only the exact encoder suffix of the trusted SMP token is equivalent."""

    case = {
        "intake_name": "smp_PAN_se_resnet50",
        "intake_zoo": "segmentation_models_pytorch-0.5.0",
        "proposed_variant": "se_resnet101",
        "stable_id": "m9666",
    }

    mismatches = trusted_intake_identity_mismatches(
        _proposed_identity(case),
        _trusted_identity({**case, "proposed_variant": "se_resnet50"}),
        intake_name=case["intake_name"],
        intake_zoo=case["intake_zoo"],
    )

    assert mismatches == {
        "variant": {"proposed": "se_resnet101", "trusted": "smp_PAN_se_resnet50"}
    }


def test_smp_equivalence_rejects_non_smp_zoo() -> None:
    """Legacy SMP token normalization is unavailable outside the SMP zoo."""

    case = {
        "intake_name": "smp_PAN_se_resnet50",
        "intake_zoo": "discovered-pytorch",
        "proposed_variant": "se_resnet50",
        "stable_id": "m9666",
    }

    mismatches = trusted_intake_identity_mismatches(
        _proposed_identity(case),
        _trusted_identity(case),
        intake_name=case["intake_name"],
        intake_zoo=case["intake_zoo"],
    )

    assert mismatches == {
        "variant": {"proposed": "se_resnet50", "trusted": "smp_PAN_se_resnet50"}
    }
