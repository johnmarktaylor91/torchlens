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

    module: str | None
    qualname: str | None
    name: str | None = None
    registry_id: str | None = None


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
class InterventionSpec:
    """Mutable internal intervention recipe."""

    targets: list[TargetSpec] = field(default_factory=list)
    helper: HelperSpec | None = None
    value: Any | None = None
    hook: Any | None = None
    records: list[FireRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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
    records: tuple[FireRecord, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()


class Relationship(str, Enum):
    """Evidence level for relationships between logs, models, inputs, and graphs."""

    UNKNOWN = "unknown"
    SAME_OBJECT = "same_object"
    SAME_FINGERPRINT = "same_fingerprint"
    DIFFERENT = "different"


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
    "FrozenInterventionSpec",
    "FrozenTargetSpec",
    "FunctionRegistryKey",
    "GraphShapeHash",
    "HFKey",
    "HelperSpec",
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
    "TensorSliceSpec",
    "TupleIndex",
    "Unsupported",
]
