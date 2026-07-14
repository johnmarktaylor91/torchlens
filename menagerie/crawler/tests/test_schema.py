"""Executable crawler schema contract tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.schema import PayloadValidationError, validate_payload
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
    make_operational_event,
)


@pytest.mark.parametrize(
    "payload",
    [
        make_model(accepted=True),
        make_model(accepted=False),
        make_attempt(),
        make_gate(),
        make_gate(["m_example"], gate_kind="fidelity", fidelity_identity="sha256:" + "a" * 64),
        make_author_proposal(),
        make_operational_event(),
    ],
)
def test_representative_records_validate(payload: dict[str, Any]) -> None:
    """Every representative full schema record validates.

    Parameters
    ----------
    payload:
        Full schema payload under test.
    """

    validate_payload(payload)


def test_unknown_fields_are_rejected(valid_model: dict[str, Any]) -> None:
    """Unknown fields fail at both root and nested typed objects.

    Parameters
    ----------
    valid_model:
        Valid accepted model fixture.
    """

    root_unknown = deepcopy(valid_model)
    root_unknown["surprise"] = True
    nested_unknown = deepcopy(valid_model)
    nested_unknown["external_metadata"]["surprise"] = True
    with pytest.raises(PayloadValidationError):
        validate_payload(root_unknown)
    with pytest.raises(PayloadValidationError):
        validate_payload(nested_unknown)


def test_missing_mandatory_fields_are_rejected(valid_model: dict[str, Any]) -> None:
    """Missing mandatory fields fail strict validation.

    Parameters
    ----------
    valid_model:
        Valid accepted model fixture.
    """

    malformed = deepcopy(valid_model)
    del malformed["external_metadata"]["architecture_class"]
    with pytest.raises(PayloadValidationError):
        validate_payload(malformed)


def test_authored_blocks_are_atomic(valid_model: dict[str, Any]) -> None:
    """Accepted metadata cannot be null and pending metadata cannot be populated.

    Parameters
    ----------
    valid_model:
        Valid accepted model fixture.
    """

    accepted_with_null = deepcopy(valid_model)
    accepted_with_null["website"] = None
    pending_with_text = deepcopy(valid_model)
    pending_with_text["authored_metadata_state"] = "pending"
    with pytest.raises(PayloadValidationError):
        validate_payload(accepted_with_null)
    with pytest.raises(PayloadValidationError):
        validate_payload(pending_with_text)


def test_metadata_gate_batch_size_is_enforced() -> None:
    """Metadata gates require 10--20 independently judged items."""

    undersized = make_gate([f"m_{index}" for index in range(9)])
    with pytest.raises(PayloadValidationError):
        validate_payload(undersized)


def test_forward_attempt_requires_mode() -> None:
    """A meaningful forward attempt cannot omit its runtime mode."""

    malformed = make_attempt(mode=None)
    with pytest.raises(PayloadValidationError):
        validate_payload(malformed)
