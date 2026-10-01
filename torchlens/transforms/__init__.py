"""Built-in activation transforms: one contract, declared not sniffed.

The data-substrate half of the transforms library (transforms memo B1-B4):
frozen :class:`TransformSpec` steps with a validated ``plan``/pure ``apply``
protocol, ordered chains with ONE canonical numeric record (the resume
identity), a versioned registry behind :func:`register_transform`, the single
:func:`coerce_transform` door, role-free kernels, and evidence-based axis
roles with the frozen :class:`TransformContext`.

Semantic pooling (``pool_tokens``), SRP, and the fitted-projection family land
in the feature wave (memo B5+). Every spelling here is DOCUMENTED-UNSTABLE
pending the naming sprint; ``tl.transforms`` is the placeholder home (the
collision with the input-side ``transform=`` slot is a named naming-sprint
item).
"""

from __future__ import annotations

from ._coerce import DEFAULT, chain, coerce_transform, coerce_transform_mapping
from ._context import (
    BTD,
    NCHW,
    ROLE_EVIDENCE,
    ContextTransform,
    RoleDeclaration,
    TransformContext,
    axis_for_role,
    wants_context,
    with_context,
)
from ._errors import TransformContractError
from ._kernels import cast, flatten, magnitude, reduce, take_index, unit_norm
from ._pipeline import (
    PIPELINE_SCHEMA,
    OpaqueStep,
    TransformPipeline,
    apply_dispatch,
    pipeline_from_record,
    pipeline_record,
)
from ._registry import lookup_transform, register_transform, registered_transform_names
from ._spec import PlannedStep, TensorSpec, TransformDefinition, TransformSpec, canonical_json

__tl_layer__ = "L4"

__all__ = [
    "BTD",
    "DEFAULT",
    "NCHW",
    "PIPELINE_SCHEMA",
    "ROLE_EVIDENCE",
    "ContextTransform",
    "OpaqueStep",
    "PlannedStep",
    "RoleDeclaration",
    "TensorSpec",
    "TransformContext",
    "TransformContractError",
    "TransformDefinition",
    "TransformPipeline",
    "TransformSpec",
    "apply_dispatch",
    "axis_for_role",
    "canonical_json",
    "cast",
    "chain",
    "coerce_transform",
    "coerce_transform_mapping",
    "flatten",
    "lookup_transform",
    "magnitude",
    "pipeline_from_record",
    "pipeline_record",
    "reduce",
    "register_transform",
    "registered_transform_names",
    "take_index",
    "unit_norm",
    "wants_context",
    "with_context",
]
