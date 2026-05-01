"""Shared dataclass ownership for TorchLens intervention schemas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias

GraphShapeHash: TypeAlias = str


@dataclass(frozen=True)
class TupleIndex:
    """Index component for tuple/list output paths."""

    index: int


@dataclass(frozen=True)
class DictKey:
    """Key component for dict output paths."""

    key: Any


@dataclass(frozen=True)
class NamedField:
    """Field-name component for namedtuple output paths."""

    name: str


@dataclass(frozen=True)
class DataclassField:
    """Field-name component for dataclass output paths."""

    name: str


@dataclass(frozen=True)
class HFKey:
    """Key component for HuggingFace ``ModelOutput`` output paths."""

    key: Any


OutputPathComponent: TypeAlias = (
    TupleIndex | DictKey | NamedField | DataclassField | HFKey | str | int
)


@dataclass(frozen=True)
class TensorSliceSpec:
    """Portable tensor slicing metadata for an intervention target."""

    positions: Any | None = None
    heads: Any | None = None
    batch: Any | None = None
    output_index: int | None = None
    position_axis: int | None = None
    head_axis: int | None = None
    query_axis: int | None = None
    key_axis: int | None = None
    feature_axis: int | None = None


@dataclass
class TargetSpec:
    """Mutable internal selector target specification."""

    selector_kind: str
    selector_value: Any | None = None
    strict: bool = False
    slice_spec: TensorSliceSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> "FrozenTargetSpec":
        """Return an immutable view of this target spec.

        Returns
        -------
        FrozenTargetSpec
            Frozen target spec with shallow-frozen metadata.
        """

        return FrozenTargetSpec(
            selector_kind=self.selector_kind,
            selector_value=self.selector_value,
            strict=self.strict,
            slice_spec=self.slice_spec,
            metadata=tuple(sorted(self.metadata.items())),
        )


@dataclass(frozen=True)
class FrozenTargetSpec:
    """Immutable public selector target specification."""

    selector_kind: str
    selector_value: Any | None = None
    strict: bool = False
    slice_spec: TensorSliceSpec | None = None
    metadata: tuple[tuple[str, Any], ...] = ()


HelperKind: TypeAlias = Literal["forward", "backward"]
HelperPortability: TypeAlias = Literal["builtin", "import_ref", "opaque_audit"]


@dataclass(frozen=True)
class HelperSpec:
    """Portable identity and hook factory for helper-built interventions."""

    helper_name: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    kind: HelperKind = "forward"
    portability: HelperPortability = "builtin"
    factory: Callable[[], Callable[..., Any]] | None = field(
        default=None, compare=False, repr=False
    )
    metadata: tuple[tuple[str, Any], ...] = ()
    batch_independent: bool = False
    compatible_with_append: bool = False

    @property
    def name(self) -> str:
        """Return this helper's public name.

        Returns
        -------
        str
            Stable helper name.
        """

        return self.helper_name

    def __call__(self) -> Callable[..., Any]:
        """Build this helper's normalized hook callable.

        Returns
        -------
        Callable[..., Any]
            Hook callable with signature ``hook(activation, *, hook)``.

        Raises
        ------
        TypeError
            If the spec does not carry a runtime factory.
        """

        if self.factory is None:
            raise TypeError(f"HelperSpec {self.helper_name!r} has no hook factory")
        return self.factory()


@dataclass(frozen=True)
class FunctionRegistryKey:
    """Portable identity for a captured Python function."""

    namespace: Literal[
        "torch",
        "torch.Tensor",
        "torch.nn.functional",
        "operator",
        "custom",
    ]
    qualname: str
    dispatch_kind: Literal["function", "method", "dunder", "namespace_alias"]
    version: int = 1
    import_path: str | None = None


@dataclass(frozen=True)
class ContainerSpec:
    """Portable description of an output container seen during capture."""

    kind: Literal["tuple", "list", "dict", "namedtuple", "dataclass", "hf_model_output"]
    length: int | None = None
    keys: tuple[Any, ...] = ()
    fields: tuple[str, ...] = ()
    type_module: str | None = None
    type_qualname: str | None = None
    child_specs: tuple[tuple[OutputPathComponent, "ContainerSpec"], ...] = ()


@dataclass(frozen=True)
class ParentRef:
    """Template component that resolves to a previously captured tensor."""

    parent_label: str


@dataclass(frozen=True)
class LiteralTensor:
    """Template component that stores a tensor literal for replay."""

    value: Any


@dataclass(frozen=True)
class LiteralValue:
    """Template component that stores a non-tensor literal for replay."""

    value: Any


@dataclass(frozen=True)
class Unsupported:
    """Template component for values replay cannot reconstruct yet."""

    reason: str
    value_type: str


ArgComponent: TypeAlias = ParentRef | LiteralTensor | LiteralValue | Unsupported | tuple[Any, ...]


@dataclass(frozen=True)
class CapturedArgTemplate:
    """Replay template for one captured function call."""

    args: tuple[ArgComponent, ...] = ()
    kwargs: tuple[tuple[str, ArgComponent], ...] = ()
    func_id: FunctionRegistryKey | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FireRecord:
    """Runtime record for one intervention firing."""

    target_label: str = ""
    pass_label: str | None = None
    func_call_id: int | None = None
    output_path: tuple[OutputPathComponent, ...] = ()
    engine: str | None = None
    helper: HelperSpec | None = None
    site_label: str | None = None
    timing: Literal["pre", "post"] | None = None
    direction: Literal["forward", "backward"] | None = None
    helper_name: str | None = None
    seed: int | None = None
    determinism_note: str | None = None
    timestamp: float | None = None


@dataclass(frozen=True)
class EdgeUseRecord:
    """Provenance for one parent tensor use by a child operation."""

    parent_label: str
    child_label: str
    arg_kind: Literal["positional", "keyword"]
    arg_path: tuple[OutputPathComponent, ...]
    view_or_copy: Literal["view", "copy", "unknown"] | None
    parent_func_call_id: int | None
    child_func_call_id: int


@dataclass
class TargetValueSpec:
    """Mutable set-replacement entry in an intervention recipe."""

    site_target: TargetSpec
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> "FrozenTargetValueSpec":
        """Return an immutable view of this value replacement.

        Returns
        -------
        FrozenTargetValueSpec
            Frozen value spec with immutable target and metadata containers.
        """

        return FrozenTargetValueSpec(
            site_target=self.site_target.freeze(),
            value=self.value,
            metadata=tuple(sorted(self.metadata.items())),
        )


@dataclass(frozen=True)
class FrozenTargetValueSpec:
    """Immutable set-replacement entry in a frozen intervention recipe."""

    site_target: FrozenTargetSpec
    value: Any
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass
class HookSpec:
    """Mutable sticky hook entry in an intervention recipe."""

    site_target: TargetSpec
    hook: Any
    helper: HelperSpec | None = None
    handle: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> "FrozenHookSpec":
        """Return an immutable view of this sticky hook spec.

        Returns
        -------
        FrozenHookSpec
            Frozen hook spec with immutable target and metadata containers.
        """

        return FrozenHookSpec(
            site_target=self.site_target.freeze(),
            hook=self.hook,
            helper=self.helper,
            handle=self.handle,
            metadata=tuple(sorted(self.metadata.items())),
        )


@dataclass(frozen=True)
class FrozenHookSpec:
    """Immutable sticky hook entry in a frozen intervention recipe."""

    site_target: FrozenTargetSpec
    hook: Any
    helper: HelperSpec | None = None
    handle: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass
class InterventionSpec:
    """Mutable internal intervention recipe."""

    targets: list[TargetSpec] = field(default_factory=list)
    helper: HelperSpec | None = None
    value: Any | None = None
    hook: Any | None = None
    target_value_specs: list[TargetValueSpec] = field(default_factory=list)
    hook_specs: list[HookSpec] = field(default_factory=list)
    records: list[FireRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_set(
        self,
        site_target: TargetSpec,
        value: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TargetValueSpec:
        """Append a set-replacement entry to this mutable recipe.

        Parameters
        ----------
        site_target:
            Portable target spec for the replacement site.
        value:
            Static replacement value or one-shot callable.
        metadata:
            Optional per-entry metadata.

        Returns
        -------
        TargetValueSpec
            The appended value-replacement spec.
        """

        value_spec = TargetValueSpec(site_target=site_target, value=value, metadata=metadata or {})
        self.target_value_specs.append(value_spec)
        return value_spec

    def add_hook(
        self,
        site_target: TargetSpec,
        hook: Any,
        *,
        helper: HelperSpec | None = None,
        handle: str | None = None,
        metadata: dict[str, Any] | None = None,
        prepend: bool = False,
    ) -> HookSpec:
        """Add a sticky hook entry to this mutable recipe.

        Parameters
        ----------
        site_target:
            Portable target spec for the hook site.
        hook:
            Hook callable or helper spec.
        helper:
            Optional helper spec when ``hook`` came from a helper.
        handle:
            Optional removable handle identifier.
        metadata:
            Optional per-entry metadata.
        prepend:
            Whether to insert the hook before existing sticky hooks.

        Returns
        -------
        HookSpec
            The added sticky hook spec.
        """

        hook_spec = HookSpec(
            site_target=site_target,
            hook=hook,
            helper=helper,
            handle=handle,
            metadata=metadata or {},
        )
        if prepend:
            self.hook_specs.insert(0, hook_spec)
        else:
            self.hook_specs.append(hook_spec)
        return hook_spec

    def remove_hook(
        self,
        *,
        site_target: TargetSpec | None = None,
        handle: str | None = None,
    ) -> int:
        """Remove sticky hook specs matching a site target or handle.

        Parameters
        ----------
        site_target:
            Optional target spec. When provided without a handle, all sticky
            hooks for that target are removed.
        handle:
            Optional hook handle. Phase 8a does not issue handles, but this
            path removes a matching stored handle if future code populated one.

        Returns
        -------
        int
            Number of hook specs removed.
        """

        original_len = len(self.hook_specs)
        self.hook_specs = [
            hook_spec
            for hook_spec in self.hook_specs
            if not _hook_spec_matches(hook_spec, site_target=site_target, handle=handle)
        ]
        return original_len - len(self.hook_specs)

    def clear(self) -> None:
        """Clear all Phase 8a sticky hook entries.

        Returns
        -------
        None
            This spec is mutated in place.
        """

        self.hook_specs.clear()

    def freeze(self) -> "FrozenInterventionSpec":
        """Return an immutable public view of this intervention spec.

        Returns
        -------
        FrozenInterventionSpec
            Frozen recipe containing immutable target and record containers.
        """

        return FrozenInterventionSpec(
            targets=tuple(target.freeze() for target in self.targets),
            helper=self.helper,
            value=self.value,
            hook=self.hook,
            target_value_specs=tuple(value_spec.freeze() for value_spec in self.target_value_specs),
            hook_specs=tuple(hook_spec.freeze() for hook_spec in self.hook_specs),
            records=tuple(self.records),
            metadata=tuple(sorted(self.metadata.items())),
        )


@dataclass(frozen=True)
class FrozenInterventionSpec:
    """Immutable public intervention recipe view."""

    targets: tuple[FrozenTargetSpec, ...] = ()
    helper: HelperSpec | None = None
    value: Any | None = None
    hook: Any | None = None
    target_value_specs: tuple[FrozenTargetValueSpec, ...] = ()
    hook_specs: tuple[FrozenHookSpec, ...] = ()
    records: tuple[FireRecord, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()


def _hook_spec_matches(
    hook_spec: HookSpec,
    *,
    site_target: TargetSpec | None,
    handle: str | None,
) -> bool:
    """Return whether a hook spec matches a removal request.

    Parameters
    ----------
    hook_spec:
        Sticky hook spec to inspect.
    site_target:
        Optional target spec to match.
    handle:
        Optional handle to match.

    Returns
    -------
    bool
        ``True`` when the hook spec should be removed.
    """

    if handle is not None and hook_spec.handle != handle:
        return False
    if site_target is not None and hook_spec.site_target != site_target:
        return False
    return handle is not None or site_target is not None


class Relationship(str, Enum):
    """Evidence level for relationships between bundle members."""

    SAME_OBJECT = "same_object"
    SAME_MODEL_OBJECT_AT_CAPTURE = "same_model_at_capture"
    SHARED_GRAPH_SAME_INPUT = "shared_graph_same_input"
    SHARED_GRAPH_DIFFERENT_INPUT = "shared_graph_diff_input"
    SHARED_ARCHITECTURE = "shared_architecture"
    SAME_PARAM_SHAPES = "same_param_shapes"
    DIFF_MODEL = "diff_model"
    UNKNOWN = "unknown"


class ForkFieldPolicy(str, Enum):
    """Copy policy for fields when a ModelLog is forked."""

    FORK_SHARE = "fork_share"
    FORK_COPY = "fork_copy"
    FORK_RECONSTRUCT = "fork_reconstruct"


def _fork_policy_table(
    field_order: list[str],
    *,
    share: set[str] | None = None,
    reconstruct: set[str] | None = None,
) -> dict[str, ForkFieldPolicy]:
    """Build a fork policy table for a canonical field-order list.

    Parameters
    ----------
    field_order:
        Ordered field names for one TorchLens data class.
    share:
        Fields that should be shared by reference across a fork.
    reconstruct:
        Fields that should be rebuilt for the forked owner.

    Returns
    -------
    dict[str, ForkFieldPolicy]
        Per-field fork policy table.
    """

    share = share or set()
    reconstruct = reconstruct or set()
    table: dict[str, ForkFieldPolicy] = {}
    for field_name in field_order:
        if field_name in reconstruct:
            table[field_name] = ForkFieldPolicy.FORK_RECONSTRUCT
        elif field_name in share:
            table[field_name] = ForkFieldPolicy.FORK_SHARE
        else:
            table[field_name] = ForkFieldPolicy.FORK_COPY
    return table


def _build_model_log_fork_policy() -> dict[str, ForkFieldPolicy]:
    """Build the ModelLog fork policy table.

    Returns
    -------
    dict[str, ForkFieldPolicy]
        Fork policies for ModelLog fields.
    """

    from ..constants import MODEL_LOG_FIELD_ORDER

    return _fork_policy_table(
        MODEL_LOG_FIELD_ORDER,
        share={
            "activation_postfunc",
            "gradient_postfunc",
            "_source_code_blob",
            "_source_model_ref",
            "_optimizer",
        },
        reconstruct={"parent_run"},
    )


def _build_layer_pass_log_fork_policy() -> dict[str, ForkFieldPolicy]:
    """Build the LayerPassLog fork policy table.

    Returns
    -------
    dict[str, ForkFieldPolicy]
        Fork policies for LayerPassLog fields.
    """

    from ..constants import LAYER_PASS_LOG_FIELD_ORDER

    return _fork_policy_table(
        LAYER_PASS_LOG_FIELD_ORDER,
        share={
            "activation",
            "transformed_activation",
            "gradient",
            "transformed_gradient",
            "func_applied",
            "grad_fn_object",
            "corresponding_grad_fn",
            "source_model_log",
        },
        reconstruct={"source_model_log", "_construction_done"},
    )


MODEL_LOG_FORK_POLICY = _build_model_log_fork_policy()
LAYER_PASS_LOG_FORK_POLICY = _build_layer_pass_log_fork_policy()

__all__ = [
    "CapturedArgTemplate",
    "ArgComponent",
    "ContainerSpec",
    "DataclassField",
    "DictKey",
    "EdgeUseRecord",
    "FireRecord",
    "ForkFieldPolicy",
    "FrozenHookSpec",
    "FrozenInterventionSpec",
    "FrozenTargetSpec",
    "FrozenTargetValueSpec",
    "FunctionRegistryKey",
    "GraphShapeHash",
    "HFKey",
    "HelperSpec",
    "HookSpec",
    "InterventionSpec",
    "LAYER_PASS_LOG_FORK_POLICY",
    "LiteralTensor",
    "LiteralValue",
    "MODEL_LOG_FORK_POLICY",
    "NamedField",
    "OutputPathComponent",
    "ParentRef",
    "Relationship",
    "TargetSpec",
    "TargetValueSpec",
    "TensorSliceSpec",
    "TupleIndex",
    "Unsupported",
]
