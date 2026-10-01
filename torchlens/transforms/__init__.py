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
    SpecialTokenFacts,
    TransformContext,
    axis_for_role,
    wants_context,
    with_context,
)
from ._errors import TransformContractError
from ._helpers import ProbeReport, SrpSizing, srp_dims_for, srp_fidelity_probe
from ._kernels import cast, flatten, magnitude, reduce, take_index, unit_norm
from ._pipeline import (
    PIPELINE_SCHEMA,
    OpaqueStep,
    TransformPipeline,
    apply_dispatch,
    pipeline_from_record,
    pipeline_record,
)
from ._pooling import channel_mean, cls_token, pool_spatial, pool_tokens
from ._projection import pca_apply, project
from ._registry import (
    _seal_builtins,
    lookup_transform,
    register_transform,
    registered_transform_names,
)
from ._spec import PlannedStep, TensorSpec, TransformDefinition, TransformSpec, canonical_json
from ._srp import srp, srp_realized_facts, srp_verify_matrix

__tl_layer__ = "L4"

# Every builtin module above has registered through the one door; the builtin
# name set is a CLOSED SET from here on (memo s7).
_seal_builtins()

__all__ = [
    "BTD",
    "DEFAULT",
    "NCHW",
    "PIPELINE_SCHEMA",
    "ROLE_EVIDENCE",
    "ContextTransform",
    "OpaqueStep",
    "PlannedStep",
    "ProbeReport",
    "RoleDeclaration",
    "SpecialTokenFacts",
    "SrpSizing",
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
    "channel_mean",
    "cls_token",
    "coerce_transform",
    "coerce_transform_mapping",
    "flatten",
    "lookup_transform",
    "magnitude",
    "pca_apply",
    "pipeline_from_record",
    "pipeline_record",
    "pool_spatial",
    "pool_tokens",
    "project",
    "reduce",
    "register_transform",
    "registered_transform_names",
    "srp",
    "srp_dims_for",
    "srp_fidelity_probe",
    "srp_realized_facts",
    "srp_verify_matrix",
    "take_index",
    "unit_norm",
    "wants_context",
    "with_context",
]
