"""JSON Schema loading and strict payload validation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Union

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)

SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
SCHEMA_FILES = {
    MODEL_SCHEMA_VERSION: "model-v2.schema.json",
    ATTEMPT_SCHEMA_VERSION: "attempt-v2.schema.json",
    GATE_SCHEMA_VERSION: "gate-v2.schema.json",
    AUTHOR_PROPOSAL_SCHEMA_VERSION: "author-proposal-v2.schema.json",
    OPERATIONAL_EVENT_SCHEMA_VERSION: "operational-event-v1.schema.json",
}


class PayloadValidationError(ValueError):
    """Raised when a payload violates its executable schema."""


@lru_cache(maxsize=None)
def load_schema(schema_version: str) -> dict[str, Any]:
    """Load and cache one supported crawler schema.

    Parameters
    ----------
    schema_version:
        Exact schema-version constant embedded in a payload.

    Returns
    -------
    dict[str, Any]
        Parsed JSON Schema.

    Raises
    ------
    KeyError
        If the version is not supported.
    SchemaError
        If the bundled schema is invalid.
    """

    try:
        path = SCHEMA_DIRECTORY / SCHEMA_FILES[schema_version]
    except KeyError as exc:
        raise KeyError(f"unsupported crawler schema version: {schema_version!r}") from exc
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SchemaError(f"schema root must be an object: {path}")
    Draft202012Validator.check_schema(loaded)
    return loaded


@lru_cache(maxsize=None)
def get_validator(schema_version: str) -> Draft202012Validator:
    """Return a cached strict validator for a crawler schema.

    Parameters
    ----------
    schema_version:
        Exact supported schema version.

    Returns
    -------
    Draft202012Validator
        Validator configured with RFC date/time format checking.
    """

    resources: list[tuple[str, Resource[Any]]] = []
    for supported_version in SCHEMA_FILES:
        schema = load_schema(supported_version)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry: Registry[Any] = Registry[Any]().with_resources(resources)
    return Draft202012Validator(
        load_schema(schema_version), format_checker=FormatChecker(), registry=registry
    )


def validate_payload(payload: Mapping[str, Any], schema_version: Union[str, None] = None) -> None:
    """Validate a payload and reject every schema or format violation.

    Parameters
    ----------
    payload:
        JSON-like mapping to validate.
    schema_version:
        Expected schema version. When omitted, the payload value is used.

    Raises
    ------
    PayloadValidationError
        If the schema version is absent/mismatched or validation fails.
    """

    actual = payload.get("schema_version")
    if not isinstance(actual, str):
        raise PayloadValidationError("schema_version must be present and be a string")
    expected = schema_version or actual
    if actual != expected:
        raise PayloadValidationError(
            f"schema_version mismatch: expected {expected!r}, received {actual!r}"
        )
    try:
        get_validator(expected).validate(dict(payload))
    except (KeyError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            message = f"{expected} validation failed at {location}: {exc.message}"
        else:
            message = str(exc)
        raise PayloadValidationError(message) from exc


def is_valid_payload(payload: Mapping[str, Any], schema_version: Union[str, None] = None) -> bool:
    """Return whether a payload passes strict schema validation.

    Parameters
    ----------
    payload:
        JSON-like mapping to validate.
    schema_version:
        Optional expected schema version.

    Returns
    -------
    bool
        True only for a fully valid payload.
    """

    try:
        validate_payload(payload, schema_version)
    except PayloadValidationError:
        return False
    return True
