"""Executable crawler schema contract tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import TERMINAL_STATUS_CODES
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


def test_metadata_gate_final_tail_size_is_enforced() -> None:
    """Metadata gates permit a final one-item tail but never an empty result."""

    validate_payload(make_gate(["m_tail"]))
    undersized = make_gate([])
    with pytest.raises(PayloadValidationError):
        validate_payload(undersized)


def test_forward_attempt_requires_mode() -> None:
    """A meaningful forward attempt cannot omit its runtime mode."""

    malformed = make_attempt(mode=None)
    with pytest.raises(PayloadValidationError):
        validate_payload(malformed)


def test_schema_properties_have_nonempty_descriptions() -> None:
    """Every declared schema property carries a non-empty self-documenting description.

    Returns
    -------
    None
        The assertion validates all crawler schemas.
    """

    def assert_descriptions(node: object, path: str = "$") -> None:
        """Recursively assert descriptions for every JSON Schema property.

        Parameters
        ----------
        node:
            JSON-compatible schema node.
        path:
            Human-readable node location for assertion failures.

        Returns
        -------
        None
            Raises when a property is undocumented.
        """

        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                for name, property_schema in properties.items():
                    assert isinstance(property_schema, dict)
                    assert property_schema.get("description", "").strip(), f"{path}.{name}"
                    assert_descriptions(property_schema, f"{path}.{name}")
            for key, value in node.items():
                if key not in {"properties", "description"}:
                    assert_descriptions(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                assert_descriptions(value, f"{path}[{index}]")

    schema_root = Path(__file__).parents[1] / "schemas"
    for schema_path in schema_root.glob("*.json"):
        assert_descriptions(json.loads(schema_path.read_text()), schema_path.name)


def test_split_skip_reasons_require_vague_text_and_sufficiency_gap(
    valid_model: dict[str, Any],
) -> None:
    """The three skip terminals retain the ruled evidence distinctions.

    Parameters
    ----------
    valid_model:
        Complete accepted model fixture.

    Returns
    -------
    None
        The assertion validates all three terminal skip paths.
    """

    skip_codes = {
        "skipped:insufficient-description",
        "skipped:no-description",
        "skipped:not-a-real-NN",
    }
    assert skip_codes.issubset(TERMINAL_STATUS_CODES)

    for code in skip_codes - {"skipped:insufficient-description"}:
        record = deepcopy(valid_model)
        record["status"]["kind"] = "skipped"
        record["status"]["code"] = code
        validate_payload(record)

    insufficient = deepcopy(valid_model)
    insufficient["status"]["kind"] = "skipped"
    insufficient["status"]["code"] = "skipped:insufficient-description"
    insufficient["source_resolution"]["rung"] = "R5_SKIP"
    insufficient["source_resolution"]["sufficiency_gap"] = (
        "concept described but no layer configs/dims/connectivity"
    )
    insufficient["evidence"]["excerpts"][0]["text"] = "A novel neural architecture for vision."
    insufficient["evidence"]["excerpts"][0]["disposition"] = "insufficient-for-faithful-reimpl"
    validate_payload(insufficient)

    missing_gap = deepcopy(insufficient)
    missing_gap["source_resolution"]["sufficiency_gap"] = None
    with pytest.raises(PayloadValidationError):
        validate_payload(missing_gap)

    missing_vague_excerpt = deepcopy(insufficient)
    missing_vague_excerpt["evidence"]["excerpts"][0]["disposition"] = "supporting"
    with pytest.raises(PayloadValidationError):
        validate_payload(missing_vague_excerpt)
