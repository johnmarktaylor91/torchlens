"""Validation subpackage for verifying saved activations and metadata invariants."""

from .core import validate_saved_activations
from .invariants import check_metadata_invariants, MetadataInvariantError

__all__ = [
    "validate_saved_activations",
    "check_metadata_invariants",
    "MetadataInvariantError",
]
