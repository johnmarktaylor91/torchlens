"""External-metadata completeness and block-at-write demarcation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    AccuracyVerdict,
)
from menagerie.crawler.schema import load_schema


def _required_external_fields(schema_version: str) -> tuple[str, ...]:
    """Return the schema-required authored external-metadata leaves.

    Parameters
    ----------
    schema_version:
        Crawler schema whose ``external_metadata`` fact tree is inspected.

    Returns
    -------
    tuple[str, ...]
        Required top-level external-metadata leaf names in schema order.

    Raises
    ------
    RuntimeError
        If the bundled schema does not expose a well-formed required field list.
    """

    schema = load_schema(schema_version)
    definitions = schema.get("$defs")
    external_metadata = (
        definitions.get("external_metadata") if isinstance(definitions, Mapping) else None
    )
    required = external_metadata.get("required") if isinstance(external_metadata, Mapping) else None
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(field, str) and field for field in required)
    ):
        raise RuntimeError(f"{schema_version} does not define required external_metadata fields")
    if len(required) != len(set(required)):
        raise RuntimeError(f"{schema_version} duplicates required external_metadata fields")
    return tuple(required)


_MODEL_EXTERNAL_FIELDS = _required_external_fields(MODEL_SCHEMA_VERSION)
_AUTHOR_EXTERNAL_FIELDS = _required_external_fields(AUTHOR_PROPOSAL_SCHEMA_VERSION)
if _MODEL_EXTERNAL_FIELDS != _AUTHOR_EXTERNAL_FIELDS:
    raise RuntimeError("model-v2 and author-proposal-v2 external metadata requirements diverge")

MANDATORY_EXTERNAL_FIELDS = _MODEL_EXTERNAL_FIELDS

TORCHLENS_DERIVABLE_FIELDS = frozenset(
    {
        "parameter_count_total",
        "parameter_count_trainable",
        "input_shapes",
        "output_shapes",
        "op_types",
        "layer_types",
        "flops",
        "graph",
        "dtype",
        "device",
    }
)

_REQUIRED_NONEMPTY_ARRAYS = frozenset(
    {"modality", "architecture_class", "domain", "task", "paradigm", "authors", "institution"}
)
_ARRAY_FIELDS = _REQUIRED_NONEMPTY_ARRAYS | {"lineage", "predecessors", "tags", "keywords"}
_REQUIRED_NONEMPTY_STRINGS = frozenset(
    {"family", "era", "key_contribution", "description", "original_framework", "run_framework"}
)


class MetadataValidationError(ValueError):
    """Raised when mandatory external metadata is missing, empty, or ungated."""


@dataclass(frozen=True)
class MetadataValidationReport:
    """External metadata validation summary.

    Parameters
    ----------
    present_fields:
        Mandatory externally sourced fields present in the object.
    gated_fields:
        Mandatory fields independently gated accurate.
    derivable_fields_present:
        Optional structural observations present incidentally.
    """

    present_fields: frozenset[str]
    gated_fields: frozenset[str]
    derivable_fields_present: frozenset[str]


def validate_external_metadata(
    metadata: Mapping[str, Any],
    *,
    field_checks: Optional[Sequence[Mapping[str, Any]]] = None,
) -> MetadataValidationReport:
    """Validate external metadata under the source-read/derivable demarcation.

    Parameters
    ----------
    metadata:
        Proposed ``external_metadata`` object.
    field_checks:
        Optional exhaustive checker field results. Supplying them enforces that
        every mandatory external field received an ``accurate`` verdict.

    Returns
    -------
    MetadataValidationReport
        Completeness, gate coverage, and optional structural observations.

    Raises
    ------
    MetadataValidationError
        If a mandatory external field is absent, malformed, or not accurate.
    """

    missing = [field for field in MANDATORY_EXTERNAL_FIELDS if field not in metadata]
    if missing:
        raise MetadataValidationError(f"missing mandatory external metadata: {missing}")
    for field in _ARRAY_FIELDS:
        value = metadata[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise MetadataValidationError(f"external_metadata.{field} must be a string array")
        if field in _REQUIRED_NONEMPTY_ARRAYS and not value:
            raise MetadataValidationError(f"external_metadata.{field} must be non-empty")
    for field in _REQUIRED_NONEMPTY_STRINGS:
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise MetadataValidationError(f"external_metadata.{field} must be non-empty")
    if not isinstance(metadata["citation"], Mapping):
        raise MetadataValidationError("external_metadata.citation must be an object")
    modes = metadata["modes"]
    if not isinstance(modes, Mapping):
        raise MetadataValidationError("external_metadata.modes must be an object")
    meaningful_modes = modes.get("meaningful_modes")
    if (
        not isinstance(meaningful_modes, list)
        or not meaningful_modes
        or not all(mode in {"train", "eval"} for mode in meaningful_modes)
        or len(meaningful_modes) != len(set(meaningful_modes))
    ):
        raise MetadataValidationError(
            "external_metadata.modes.meaningful_modes must contain unique train/eval modes"
        )
    if modes.get("train_eval_divergence") not in {"none", "statistical", "structural"}:
        raise MetadataValidationError(
            "external_metadata.modes.train_eval_divergence is not canonical"
        )
    gated = _validate_field_checks(field_checks) if field_checks is not None else frozenset()
    derivable = frozenset(field for field in TORCHLENS_DERIVABLE_FIELDS if field in metadata)
    return MetadataValidationReport(
        present_fields=frozenset(MANDATORY_EXTERNAL_FIELDS),
        gated_fields=gated,
        derivable_fields_present=derivable,
    )


def validate_external_metadata_for_write(
    metadata: Mapping[str, Any], gate_item: Mapping[str, Any]
) -> MetadataValidationReport:
    """Enforce mandatory metadata and its accurate block-at-write gate.

    Parameters
    ----------
    metadata:
        Proposed external metadata object.
    gate_item:
        Independently bound metadata gate item.

    Returns
    -------
    MetadataValidationReport
        Fully write-eligible metadata report.

    Raises
    ------
    MetadataValidationError
        If the item or any mandatory field is not accurate.
    """

    if gate_item.get("verdict") != AccuracyVerdict.ACCURATE.value:
        raise MetadataValidationError("metadata gate is not accurate; canonical write is blocked")
    integrity = gate_item.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("verdict") != AccuracyVerdict.ACCURATE.value
    ):
        raise MetadataValidationError(
            "metadata integrity is not accurate; canonical write is blocked"
        )
    checks = gate_item.get("field_checks")
    if not isinstance(checks, list):
        raise MetadataValidationError("metadata gate has no per-field checks")
    return validate_external_metadata(metadata, field_checks=checks)


def _validate_field_checks(
    field_checks: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Validate exhaustive accurate checks for mandatory external fields.

    Parameters
    ----------
    field_checks:
        Checker field results.

    Returns
    -------
    frozenset[str]
        Mandatory fields checked accurate.

    Raises
    ------
    MetadataValidationError
        If a leaf is missing, duplicated, inaccurate, or cannot be verified.
    """

    verdicts: dict[str, str] = {}
    for check in field_checks:
        raw_field = check.get("field")
        if not isinstance(raw_field, str):
            raise MetadataValidationError("metadata field check has no field name")
        field = raw_field.removeprefix("external_metadata.")
        if field not in MANDATORY_EXTERNAL_FIELDS:
            continue
        if field in verdicts:
            raise MetadataValidationError(f"duplicate metadata field check: {field}")
        verdicts[field] = str(check.get("verdict"))
    missing = set(MANDATORY_EXTERNAL_FIELDS) - set(verdicts)
    if missing:
        raise MetadataValidationError(f"ungated mandatory external metadata: {sorted(missing)}")
    failed = {
        field: verdict
        for field, verdict in verdicts.items()
        if verdict != AccuracyVerdict.ACCURATE.value
    }
    if failed:
        raise MetadataValidationError(f"non-accurate external metadata fields: {failed}")
    return frozenset(verdicts)
