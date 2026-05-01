"""Validation subpackage for saved activations, backward capture, and invariants."""

from ..intervention.save import check_spec_compat
from ..intervention.resolver import resolve_sites
from ..user_funcs import (
    validate_backward_pass,
    validate_batch_of_models_and_inputs,
    validate_forward_pass,
    validate_saved_activations,
)
from .core import validate_saved_activations as validate_model_log_saved_activations
from .consolidated import InterventionValidationReport, validate
from .invariants import MetadataInvariantError, check_metadata_invariants

__all__ = [
    "InterventionValidationReport",
    "validate_backward_pass",
    "validate_batch_of_models_and_inputs",
    "validate",
    "validate_forward_pass",
    "validate_saved_activations",
    "validate_model_log_saved_activations",
    "check_metadata_invariants",
    "check_spec_compat",
    "MetadataInvariantError",
    "resolve_sites",
]
