"""Lazy public namespace for TorchLens receptive-field analysis."""

from __future__ import annotations

from ._errors import (
    AmbiguousCallError,
    AmbiguousInputError,
    AmbiguousPassError,
    BackendUnsupportedError,
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
    ReceptiveFieldValidationError,
)
from ._types import (
    GradientReceptiveField,
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAlignment,
    ReceptiveFieldAxis,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldProfile,
    ReceptiveFieldStatus,
    ReceptiveFieldValidation,
    ReceptiveFieldValidationStatus,
    ReceptiveFieldViolation,
)
from ._rules import ReceptiveFieldRule, ReceptiveFieldRuleContext, register_rf_rule, rules
from ._view import ReceptiveFieldView
from ._viz import node_spec


__all__ = [
    "AmbiguousCallError",
    "AmbiguousInputError",
    "AmbiguousPassError",
    "BackendUnsupportedError",
    "GradientReceptiveField",
    "GridLayout",
    "ReceptiveField",
    "ReceptiveFieldAlignment",
    "ReceptiveFieldAxis",
    "ReceptiveFieldBox",
    "ReceptiveFieldBoxAxis",
    "ReceptiveFieldError",
    "ReceptiveFieldProfile",
    "ReceptiveFieldRule",
    "ReceptiveFieldRuleContext",
    "ReceptiveFieldStatus",
    "ReceptiveFieldUnavailableError",
    "ReceptiveFieldValidation",
    "ReceptiveFieldValidationError",
    "ReceptiveFieldValidationStatus",
    "ReceptiveFieldView",
    "ReceptiveFieldViolation",
    "register_rf_rule",
    "node_spec",
    "rules",
]
