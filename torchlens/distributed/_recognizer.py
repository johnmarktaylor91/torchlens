"""Compatibility re-export; the module lives in ``torchlens._distributed._recognizer``."""

# layering: re-export kept for record importers; remove after the S2 fold
from .._distributed._recognizer import (  # noqa: F401
    _LAYER2_TYPE_MARKERS as _LAYER2_TYPE_MARKERS,
    _REGISTRATION_MODULES as _REGISTRATION_MODULES,
    COLLECTIVE_NAMESPACES as COLLECTIVE_NAMESPACES,
    UNCAPTURED_COLLECTIVE_OP as UNCAPTURED_COLLECTIVE_OP,
    VETTED_NAMESPACE_SNAPSHOTS as VETTED_NAMESPACE_SNAPSHOTS,
    CollectiveRecognizer as CollectiveRecognizer,
    UncapturedCollectiveOpError as UncapturedCollectiveOpError,
    _all_dispatcher_schemas as _all_dispatcher_schemas,
    _force_registration_imports as _force_registration_imports,
    _layer2_offenders as _layer2_offenders,
    _match_vetted_snapshot as _match_vetted_snapshot,
    _runtime_namespace_sets as _runtime_namespace_sets,
    derive_collective_recognizer as derive_collective_recognizer,
    is_vetted_build as is_vetted_build,
)
