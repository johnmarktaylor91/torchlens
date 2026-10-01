"""Extraction v2 runtime (lane F18; extract memo items 4-19).

The private implementation package behind :mod:`torchlens.dataset_extraction`:
the typed batch envelope + collate door (:mod:`envelope`), the batch context
with row and key validation (:mod:`context`), mask-aware pool presets
(:mod:`pool`), save-dtype routing (:mod:`dtype_policy`), the D-RAGGED gate
and trimmed carrier (:mod:`ragged`), the shard codecs (:mod:`shards`), the
lazy reader + views (:mod:`reader`, :mod:`views`), selector
resolve-and-freeze (:mod:`selector_plan`), the streaming exporters
(:mod:`export`), and the callable-identity classifier
(:mod:`callable_identity`).

Public doors live on :mod:`torchlens.dataset_extraction`; every spelling is
DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from .callable_identity import classify_callable, register_pure_module, registered_pure_modules
from .context import BatchContext, sanitize_key, validate_output_keys, validate_row_counts
from .dtype_policy import cast_for_store, resolve_dtype_policy
from .envelope import (
    BatchEnvelope,
    coerce_envelope,
    default_collate,
    envelope_input_digest,
    hf_collate,
    resolve_tokenizer_once,
)
from .export import EXPORT_CONTRACT_FILENAME, EXPORT_FORMATS, export_extraction
from .pool import POOL_PRESETS, apply_pool, resolve_pool_policy
from .ragged import (
    RAGGED_MODES,
    RaggedBatch,
    mask_row_geometry,
    raise_ragged_refusal,
    to_padded,
    trim_batch,
    validate_ragged_mode,
)
from .reader import ExtractionReader, open_extraction
from .selector_plan import (
    SelectorPlan,
    attest_selector_batch,
    freeze_selector_plan,
    is_selector_request,
    selector_request_record,
)
from .shards import (
    SHARD_FORMATS,
    read_shard,
    shard_extension,
    validate_shard_format,
    write_shard,
)
from .views import as_torch_dataset, feature_matrix, shuffled_batches

__tl_layer__ = "L5"

__all__ = [
    "EXPORT_CONTRACT_FILENAME",
    "EXPORT_FORMATS",
    "POOL_PRESETS",
    "RAGGED_MODES",
    "SHARD_FORMATS",
    "BatchContext",
    "BatchEnvelope",
    "ExtractionReader",
    "RaggedBatch",
    "SelectorPlan",
    "apply_pool",
    "as_torch_dataset",
    "attest_selector_batch",
    "cast_for_store",
    "classify_callable",
    "coerce_envelope",
    "default_collate",
    "envelope_input_digest",
    "export_extraction",
    "feature_matrix",
    "freeze_selector_plan",
    "hf_collate",
    "is_selector_request",
    "mask_row_geometry",
    "open_extraction",
    "raise_ragged_refusal",
    "read_shard",
    "register_pure_module",
    "registered_pure_modules",
    "resolve_dtype_policy",
    "resolve_pool_policy",
    "resolve_tokenizer_once",
    "sanitize_key",
    "selector_request_record",
    "shard_extension",
    "shuffled_batches",
    "to_padded",
    "trim_batch",
    "validate_output_keys",
    "validate_ragged_mode",
    "validate_row_counts",
    "validate_shard_format",
    "write_shard",
]
