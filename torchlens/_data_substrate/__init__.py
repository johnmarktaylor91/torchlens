"""Extraction data substrates: digest primitives + artifact v2 + model identity.

The C04 substrate half of dataset extraction (extract memo items 1, 2, 6):
:mod:`digests` builds the two pinned digest primitives once (threaded Merkle
crypto fold, order-sensitive value reduction) for every consumer;
:mod:`artifact` owns the v2 artifact layout (bounded manifest, append-only
fsynced ledger, commit protocol, completed-v1 migration, field-by-field
signature compare); :mod:`model_identity` computes the D6 identity record
whose resume comparison kills the random-init T-MODELSWAP hazard.

Private substrate: the public doors are :func:`torchlens.extract_dataset` /
:func:`torchlens.load_extraction`. Every spelling is DOCUMENTED-UNSTABLE
pending the naming sprint.
"""

from __future__ import annotations

from .artifact import (
    LEDGER_FILENAME,
    MANIFEST_SCHEMA_V1,
    MANIFEST_SCHEMA_V2,
    SIGNATURE_SEMANTIC_FIELDS,
    STIMULUS_IDS_FILENAME,
    UNRECORDED_V1,
    ArtifactWriter,
    ExtractionArtifactError,
    compare_model_identity,
    compare_signatures,
    migrate_v1_artifact,
    read_trusted_rows,
    repair_ledger_tail,
    shard_filename,
    stimulus_ids_digest,
)
from .digests import (
    MERKLE_ALGORITHM_ID,
    MERKLE_ALGORITHM_VERSION,
    VALUE_REDUCTION_ALGORITHM_ID,
    VALUE_REDUCTION_ALGORITHM_VERSION,
    MerkleDigest,
    merkle_digest,
    value_reduction,
)
from .model_identity import (
    MODEL_IDENTITY_LEVELS,
    MODEL_STATE_DIGEST_ID,
    MODEL_STATE_DIGEST_VERSION,
    compute_model_identity,
)

__tl_layer__ = "L3"

__all__ = [
    "LEDGER_FILENAME",
    "MANIFEST_SCHEMA_V1",
    "MANIFEST_SCHEMA_V2",
    "MERKLE_ALGORITHM_ID",
    "MERKLE_ALGORITHM_VERSION",
    "MODEL_IDENTITY_LEVELS",
    "MODEL_STATE_DIGEST_ID",
    "MODEL_STATE_DIGEST_VERSION",
    "SIGNATURE_SEMANTIC_FIELDS",
    "STIMULUS_IDS_FILENAME",
    "UNRECORDED_V1",
    "VALUE_REDUCTION_ALGORITHM_ID",
    "VALUE_REDUCTION_ALGORITHM_VERSION",
    "ArtifactWriter",
    "ExtractionArtifactError",
    "MerkleDigest",
    "compare_model_identity",
    "compare_signatures",
    "compute_model_identity",
    "merkle_digest",
    "migrate_v1_artifact",
    "read_trusted_rows",
    "repair_ledger_tail",
    "shard_filename",
    "stimulus_ids_digest",
    "value_reduction",
]
