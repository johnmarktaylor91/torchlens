"""Lazy public namespace for TorchLens receptive-field analysis."""

from __future__ import annotations

from ._errors import (
    AmbiguousCallError,
    AmbiguousInputError,
    AmbiguousTargetError,
    AmbiguousPassError,
    BackendUnsupportedError,
    ReceptiveFieldError,
    NoInfluencePathError,
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
    ReceptiveFieldDirection,
    ReceptiveFieldProfile,
    ReceptiveFieldStatus,
    ReceptiveFieldValidation,
    ReceptiveFieldValidationStatus,
    ReceptiveFieldViolation,
)
from ._rules import ReceptiveFieldRule, ReceptiveFieldRuleContext, register_rf_rule, rules
from ._validation import cross_validate
from ._view import ReceptiveFieldView
from ._viz import node_spec


__all__ = [
    "AmbiguousCallError",
    "AmbiguousInputError",
    "AmbiguousTargetError",
    "AmbiguousPassError",
    "BackendUnsupportedError",
    "GradientReceptiveField",
    "GridLayout",
    "ReceptiveField",
    "ReceptiveFieldAlignment",
    "ReceptiveFieldAxis",
    "ReceptiveFieldBox",
    "ReceptiveFieldBoxAxis",
    "ReceptiveFieldDirection",
    "ReceptiveFieldError",
    "NoInfluencePathError",
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
    "cross_validate",
    "node_spec",
    "rules",
]
