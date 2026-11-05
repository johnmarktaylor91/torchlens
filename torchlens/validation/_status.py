"""Compatibility re-export; the status objects live in ``torchlens._validation_status``."""

# layering: re-export kept for record importers; remove after the record-tier fold
from .._validation_status import (  # noqa: F401
    _VALIDATION_REASON_DEFINITIONS as _VALIDATION_REASON_DEFINITIONS,
    REGION_REPLAY_CLASS as REGION_REPLAY_CLASS,
    REGION_REPLAY_CLASS_KEY as REGION_REPLAY_CLASS_KEY,
    REGION_REPLAY_IMPORTER_PROVENANCE as REGION_REPLAY_IMPORTER_PROVENANCE,
    REGION_REPLAY_PROVENANCE_KEY as REGION_REPLAY_PROVENANCE_KEY,
    VALIDATION_REASONS as VALIDATION_REASONS,
    ValidationReplaySource as ValidationReplaySource,
    ValidationReplayState as ValidationReplayState,
    ValidationStatus as ValidationStatus,
    _failed_replay_message as _failed_replay_message,
    _partial_replay_message as _partial_replay_message,
    count_importer_region_annotations as count_importer_region_annotations,
    has_importer_region_provenance as has_importer_region_provenance,
    is_region_replay_annotation as is_region_replay_annotation,
)
