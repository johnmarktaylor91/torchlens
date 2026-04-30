"""Import surface for the planned TorchLens intervention API.

The subpackage exists in Phase 0 only to reserve ownership boundaries and
public import paths. Runtime behavior remains unimplemented and fails closed.
"""

from .bundle import Bundle
from .helpers import (
    bwd_hook,
    clamp,
    gradient_scale,
    gradient_zero,
    mean_ablate,
    noise,
    project_off,
    project_onto,
    resample_ablate,
    scale,
    splice_module,
    steer,
    swap_with,
    zero_ablate,
)
from .replay import replay, replay_from
from .resolver import SiteTable, resolve_sites
from .rerun import rerun
from .runtime import do
from .save import load_intervention_spec, save_intervention
from .selectors import contains, func, in_module, label, module, where
from .types import (
    CapturedArgTemplate,
    EdgeUseRecord,
    FireRecord,
    ForkFieldPolicy,
    FrozenInterventionSpec,
    FrozenTargetSpec,
    FunctionRegistryKey,
    HelperSpec,
    InterventionSpec,
    Relationship,
    TargetSpec,
    TensorSliceSpec,
)

__all__ = [
    "Bundle",
    "CapturedArgTemplate",
    "EdgeUseRecord",
    "FireRecord",
    "ForkFieldPolicy",
    "FrozenInterventionSpec",
    "FrozenTargetSpec",
    "FunctionRegistryKey",
    "HelperSpec",
    "InterventionSpec",
    "Relationship",
    "SiteTable",
    "TargetSpec",
    "TensorSliceSpec",
    "bwd_hook",
    "clamp",
    "contains",
    "do",
    "func",
    "gradient_scale",
    "gradient_zero",
    "in_module",
    "label",
    "load_intervention_spec",
    "mean_ablate",
    "module",
    "noise",
    "project_off",
    "project_onto",
    "replay",
    "replay_from",
    "rerun",
    "resample_ablate",
    "resolve_sites",
    "save_intervention",
    "scale",
    "splice_module",
    "steer",
    "swap_with",
    "where",
    "zero_ablate",
]
