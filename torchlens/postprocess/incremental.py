"""Opt-in incremental enrichments for fastlog recordings."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Literal, cast

from ..fastlog.exceptions import RecordingConfigError
from ..fastlog.types import ActivationRecord, Recording

EnrichmentStep = Literal["module_path_strings", "param_addresses"]
_ALL_FEASIBLE: tuple[EnrichmentStep, ...] = ("module_path_strings",)
_ALL_STEPS: set[str] = {"module_path_strings", "param_addresses"}


def _copy_record_with_metadata(
    record: ActivationRecord, metadata: dict[str, object]
) -> ActivationRecord:
    """Return a copy of an activation record with merged metadata."""

    merged_metadata = dict(record.metadata)
    merged_metadata.update(metadata)
    return replace(record, metadata=merged_metadata)


def _replace_recording_records(recording: Recording, records: list[ActivationRecord]) -> Recording:
    """Return a recording copy with a new records list and count."""

    return replace(recording, records=records, n_records=len(records))


def add_module_path_strings(recording: Recording) -> Recording:
    """Annotate records with module address strings from their captured stack.

    Parameters
    ----------
    recording:
        Source fastlog recording.

    Returns
    -------
    Recording
        New recording whose records include ``metadata["module_path_strings"]`` and
        ``metadata["module_path"]``.
    """

    records = []
    for record in recording.records:
        module_path_strings = tuple(frame.module_address for frame in record.ctx.module_stack)
        module_path = ".".join(address for address in module_path_strings if address)
        records.append(
            _copy_record_with_metadata(
                record,
                {
                    "module_path_strings": module_path_strings,
                    "module_path": module_path,
                },
            )
        )
    return _replace_recording_records(recording, records)


def _activation_record_has_param_addresses() -> bool:
    """Return whether capture-time param-address data exists on ActivationRecord."""

    return any(field.name == "parent_param_addresses" for field in fields(ActivationRecord))


def add_param_addresses(recording: Recording) -> Recording:
    """Attach captured parent parameter addresses to record metadata.

    Parameters
    ----------
    recording:
        Source fastlog recording.

    Returns
    -------
    Recording
        New recording with ``metadata["parent_param_addresses"]`` populated.

    Raises
    ------
    RecordingConfigError
        If this build did not capture ``ActivationRecord.parent_param_addresses``.
    """

    if not _activation_record_has_param_addresses():
        raise RecordingConfigError(
            "param_addresses enrichment requires capture-time "
            "ActivationRecord.parent_param_addresses data, which is not available"
        )
    records = []
    for record in recording.records:
        addresses = getattr(record, "parent_param_addresses")
        records.append(
            _copy_record_with_metadata(record, {"parent_param_addresses": tuple(addresses or ())})
        )
    return _replace_recording_records(recording, records)


def _normalize_steps(steps: list[str] | str) -> tuple[EnrichmentStep, ...]:
    """Normalize public enrichment step input."""

    if steps == "all-feasible":
        return _ALL_FEASIBLE
    if isinstance(steps, str):
        requested: tuple[str, ...] = (steps,)
    else:
        requested = tuple(steps)
    unknown = sorted(set(requested) - _ALL_STEPS)
    if unknown:
        raise RecordingConfigError(
            "Recording.enrich steps must be 'module_path_strings', "
            "'param_addresses', or 'all-feasible'"
        )
    return tuple(cast("EnrichmentStep", step) for step in requested)


def enrich_recording(recording: Recording, steps: list[str] | str) -> Recording:
    """Apply requested incremental enrichments to a recording."""

    enriched = recording
    for step in _normalize_steps(steps):
        if step == "module_path_strings":
            enriched = add_module_path_strings(enriched)
        elif step == "param_addresses":
            enriched = add_param_addresses(enriched)
    return enriched
