"""Import surface for TorchLens intervention selectors, hooks, reruns, and bundles.

Two of this package's public names are resolved LAZILY through the module
``__getattr__`` below rather than imported eagerly, because their modules import
BACK into modules whose own import pulls in ``torchlens.intervention``:

* ``Bundle`` lives in :mod:`torchlens.bundle`, which imports
  ``torchlens.intervention._metrics`` / ``._super`` / ``._topology``; importing any
  of those first executes THIS ``__init__``, so an eager ``from .bundle import
  Bundle`` here made ``import torchlens.bundle`` -- a PUBLIC module -- fail with a
  partially-initialized-module ``ImportError``.
* ``rerun`` / ``run`` live in :mod:`torchlens.intervention.rerun`, which imports
  ``torchlens._chunking``; ``_chunking`` imports ``.intervention.errors``, so an
  eager ``from .rerun import ...`` here made ``import torchlens._chunking`` fail
  the same way.

Deferring exactly these three names keeps every module in the package standalone
importable (gated by ``tests/test_module_import_isolation.py``) while leaving the
public surface, ``__all__``, and static types unchanged.
"""

from typing import TYPE_CHECKING, Any

from ._metrics import (
    METRIC_REGISTRY as METRIC_REGISTRY,
    cosine_distance,
    pearson_correlation_distance as pearson_correlation_distance,
    relative_l1_scalar,
    relative_l2,
    resolve_metric as resolve_metric,
)
from ._super.super_op import (
    SuperAtenOp,
    SuperLayer as SuperLayer,
    SuperLayerAccessor as SuperLayerAccessor,
    SuperOp,
    SuperOpAccessor as SuperOpAccessor,
    TraceAccessor as TraceAccessor,
)
from ._topology.topology import (
    Supergraph as Supergraph,
    SupergraphNode as SupergraphNode,
    TopologyDiff as TopologyDiff,
    build_supergraph as build_supergraph,
    compare_topology,
)

# F01 surgery live lanes: the capture-free bound executor, the bind-side
# steer wrapper (module named ``steering`` -- the ``steer`` attribute is the
# historical helper from .helpers), and the region noun (derived
# admissibility on the existing verbs).
from .binding import BindReport, BoundInterventionExecutor
from .compose import compose
from .errors import (
    AppendBatchDependenceError,
    AppendMismatchError,
    AppendStateValidationWarning,
    AppendStreamingNotSupportedError,
    AxisAmbiguityError,
    BaselineUndeterminedError,
    BatchNormTrainModeWarning,
    BundleMemberError,
    BundleRelationshipError,
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
    DeadParentError,
    DirectActivationWriteWarning,
    DirectWriteInExecutableSaveError,
    EngineDispatchError,
    GraphShapeMismatchError,
    HelperMountError,
    HookSignatureError,
    HookSiteCoverageError,
    HookValueError,
    InterventionReadyConflictError,
    LiveModeLabelError,
    ModelMismatchError,
    MultiMatchWarning,
    MultiOutputModuleError,
    MutateInPlaceWarning,
    NoParentError,
    OpaqueCallableInExecutableSaveError,
    RecursiveTracingError,
    RegionError,
    ReplayPreconditionError,
    SelectionError,
    SelectorCompositionError,
    SiteAmbiguityError,
    SiteResolutionError,
    SpecMutationError,
    SpecPortabilityError,
    SpliceModuleDeviceError,
    SpliceModuleDtypeError,
    UnclassifiedSelectorError,
    UntrustedCallableError,
)
from .handles import HookHandle
from .helpers import (
    _RESAMPLE_ABLATE_REDIRECT,
    bwd_hook,
    clamp,
    grad_clamp,
    grad_clip,
    grad_noise,
    grad_scale,
    grad_zero,
    mean_ablate,
    noise,
    patch_from,
    project_off,
    project_onto,
    scale,
    scramble_elements,
    splice_module,
    steer,
    swap_with,
    zero_ablate,
)
from .hooks import (
    HookContext,
    NormalizedHookEntry,
    make_hook_context,
    normalize_hook as normalize_hook,
    normalize_hook_plan,
)
from .population import OneDatum, PerRowDatums, Reference, reference
from .predicates import add, replace_with
from .regions import RegionBoundary, RegionEdge, RegionInstance, RegionTarget, region
from .replay import push, push_from
from .resolver import SiteTable, resolve_sites
from .runtime import do
from .save import (
    SaveLevel,
    SpecCompat,
    TargetManifestDiff,
    check_spec_compat,
    load_intervention_spec,
    save_intervention,
)
from .selectors import (
    at_step,
    contains,
    facet,
    followed_by,
    func,
    func_transform,
    grad_fn,
    grad_fn_label,
    grad_input,
    grad_output,
    head,
    in_backward_pass,
    in_module,
    input_at,
    label,
    module,
    output,
    output_at,
    preceded_by,
    regex,
    site,
    where,
    without_op,
)
from .sites import SiteCollection, SiteSpec as SiteSpec, sites

# The PUBLIC immutable multi-clause spec (C03 substrate; surgery memo 3.1)
# claims the package-level ``InterventionSpec`` name. The historical mutable
# sticky-hook recipe keeps its identity at ``.types.InterventionSpec`` --
# internal code and persisted pickles are untouched.
from .spec import (
    InterventionRule,
    InterventionSpec,
    classify_where,
    refuse_unreplayable_rules,
    when,
)
from .steering import SteerResult, steer_generate
from .stochastic import (
    SamplingPlan,
    mean_fill,
    mean_from,
    permute_batch,
    resample_rows_from,
    sample_from,
    sampling_records,
    set_direction_mean,
)
from .types import (
    ArgComponent as ArgComponent,
    CapturedArgTemplate,
    ContainerSpec,
    DataclassField,
    DictKey,
    EdgeUseRecord,
    Edit,
    FireRecord,
    ForkFieldPolicy,
    FrozenInterventionSpec as FrozenInterventionSpec,
    FrozenTargetSpec,
    FunctionRegistryKey,
    HelperSpec,
    HFKey,
    InterventionDecision,
    LiteralTensor,
    LiteralValue,
    NamedField,
    ParentRef,
    Relationship,
    TargetSpec,
    TensorSliceSpec,
    TupleIndex,
    Unsupported,
    rebuild_container_from_spec,
)

if TYPE_CHECKING:  # typing-only mirror of the lazy names below
    from .bundle import Bundle
    from .rerun import run

__all__ = [
    "AppendBatchDependenceError",
    "AppendMismatchError",
    "AppendStateValidationWarning",
    "AppendStreamingNotSupportedError",
    "add",
    "at_step",
    "AxisAmbiguityError",
    "BaselineUndeterminedError",
    "BatchNormTrainModeWarning",
    "BindReport",
    "BoundInterventionExecutor",
    "Bundle",
    "BundleMemberError",
    "BundleRelationshipError",
    "CapturedArgTemplate",
    "ContainerSpec",
    "ControlFlowDivergenceError",
    "ControlFlowDivergenceWarning",
    "DataclassField",
    "DeadParentError",
    "DictKey",
    "DirectActivationWriteWarning",
    "DirectWriteInExecutableSaveError",
    "EdgeUseRecord",
    "EngineDispatchError",
    "FireRecord",
    "ForkFieldPolicy",
    "FrozenTargetSpec",
    "FunctionRegistryKey",
    "GraphShapeMismatchError",
    "HFKey",
    "Edit",
    "HelperSpec",
    "InterventionDecision",
    "HookSignatureError",
    "HelperMountError",
    "HookHandle",
    "HookContext",
    "HookSiteCoverageError",
    "HookValueError",
    "InterventionRule",
    "InterventionSpec",
    "InterventionReadyConflictError",
    "LiveModeLabelError",
    "LiteralTensor",
    "LiteralValue",
    "ModelMismatchError",
    "MultiMatchWarning",
    "MultiOutputModuleError",
    "MutateInPlaceWarning",
    "NamedField",
    "NoParentError",
    "OpaqueCallableInExecutableSaveError",
    "RegionBoundary",
    "RegionEdge",
    "RegionError",
    "RegionInstance",
    "RegionTarget",
    "Relationship",
    "RecursiveTracingError",
    "ReplayPreconditionError",
    "SaveLevel",
    "ParentRef",
    "SiteAmbiguityError",
    "SiteCollection",
    "SelectionError",
    "SiteResolutionError",
    "SelectorCompositionError",
    "SpecCompat",
    "SpecMutationError",
    "SpecPortabilityError",
    "SuperAtenOp",
    "SiteTable",
    "SpliceModuleDeviceError",
    "SpliceModuleDtypeError",
    "SuperOp",
    "TargetManifestDiff",
    "TargetSpec",
    "TensorSliceSpec",
    "TupleIndex",
    "Unsupported",
    "bwd_hook",
    "clamp",
    "check_spec_compat",
    "classify_where",
    "compare_topology",
    "compose",
    "contains",
    "cosine_distance",
    "do",
    "facet",
    "func",
    "func_transform",
    "followed_by",
    "grad_clamp",
    "grad_clip",
    "grad_fn",
    "grad_fn_label",
    "grad_input",
    "head",
    "label",
    "grad_noise",
    "grad_output",
    "grad_scale",
    "grad_zero",
    "input_at",
    "in_backward_pass",
    "in_module",
    "regex",
    "load_intervention_spec",
    "make_hook_context",
    "mean_ablate",
    "mean_fill",
    "mean_from",
    "OneDatum",
    "PerRowDatums",
    "permute_batch",
    "Reference",
    "reference",
    "SamplingPlan",
    "module",
    "noise",
    "normalize_hook_plan",
    "NormalizedHookEntry",
    "output",
    "output_at",
    "preceded_by",
    "project_off",
    "project_onto",
    "refuse_unreplayable_rules",
    "region",
    "replace_with",
    "push",
    "push_from",
    "run",
    "resample_rows_from",
    "sample_from",
    "sampling_records",
    "set_direction_mean",
    "relative_l1_scalar",
    "relative_l2",
    "resolve_sites",
    "save_intervention",
    "scramble_elements",
    "scale",
    "site",
    "splice_module",
    "sites",
    "steer",
    "steer_generate",
    "SteerResult",
    "swap_with",
    "where",
    "when",
    "without_op",
    "UnclassifiedSelectorError",
    "UntrustedCallableError",
    "patch_from",
    "zero_ablate",
    "rebuild_container_from_spec",
]

# Names whose defining module imports back into a module that imports this
# package; see the module docstring. Resolved on first attribute access, after
# this ``__init__`` has finished executing, so no partially-initialized module is
# ever observed.
_LAZY_NAMES: dict[str, str] = {
    "Bundle": ".bundle",
    "run": ".rerun",
}

#: Removed spellings: each raises the typed ``facade_redirect`` error naming
#: its replacement (clean break, no alias).
_REDIRECTS: dict[str, str] = {
    "resample_ablate": _RESAMPLE_ABLATE_REDIRECT,
    "intervening": "use torchlens.intervention.without_op -- intervening was renamed",
    "replay_from": "use torchlens.intervention.push_from -- replay_from was renamed",
}


def __getattr__(name: str) -> Any:
    """Resolve the cycle-deferred public names on first access.

    Parameters
    ----------
    name:
        Attribute name being looked up on ``torchlens.intervention``.

    Returns
    -------
    Any
        The resolved attribute.

    Raises
    ------
    AttributeError
        For any name this package does not define.
    """

    module_name = _LAZY_NAMES.get(name)
    if module_name is None:
        from ..utils.facade import resolve_facade_attr

        return resolve_facade_attr(
            owner=__name__, name=name, module_globals=globals(), redirects=_REDIRECTS
        )
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # memoize; subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    """Return the public surface including the lazily-resolved names."""

    return sorted(set(__all__) | set(globals()))
