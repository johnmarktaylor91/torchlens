"""Shared dataclass ownership for TorchLens intervention schemas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, TypeAlias

from .._io import FieldPolicy
from ..ir.container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
    rebuild_container_from_spec,
)
from ..selection import _SelectionOperand

GraphShapeHash: TypeAlias = str
InterventionAction: TypeAlias = Literal["replace", "add_hook", "scale", "transform"]


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


def _freeze_value(value: Any) -> Any:
    """Recursively freeze built-in mutable containers.

    Parameters
    ----------
    value:
        Value to freeze.

    Returns
    -------
    Any
        Immutable equivalent for built-in list, dict, and set containers, or
        the original value for opaque objects.
    """

    if isinstance(value, dict):
        return tuple(
            (key, _freeze_value(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_stable(value: Any) -> bool:
    """Return whether ``_freeze_value(value)`` is invariant while ``value`` is held.

    Parameters
    ----------
    value:
        Candidate selector value.

    Returns
    -------
    bool
        False when the value reaches a built-in dict/list/set whose contents
        ``_freeze_value`` snapshots (in-place mutation would stale a cached
        freeze), True for pass-through and tuple-of-stable values.
    """

    if isinstance(value, dict | list | set | frozenset):
        return False
    if isinstance(value, tuple):
        return all(_freeze_stable(item) for item in value)
    return True


@dataclass
class TargetSpec:
    """Mutable internal selector target specification."""

    selector_kind: str
    selector_value: Any | None = None
    strict: bool = False
    slice_spec: TensorSliceSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> FrozenTargetSpec:
        """Return an immutable view of this target spec.

        Returns
        -------
        FrozenTargetSpec
            Frozen target spec with shallow-frozen metadata.
        """

        # Dedup scans over spec.targets refreeze both sides per pair, which is
        # quadratic in unique targets. The cache is only stored for specs whose
        # freeze output cannot drift under in-place mutation (empty metadata,
        # snapshot-free selector_value) and is only returned while every field
        # still holds the exact cached value, so a hit is byte-identical to a
        # fresh freeze.
        cached = self.__dict__.get("_tl_frozen_cache")
        if (
            cached is not None
            and not self.metadata
            and cached[0] is self.selector_value
            and cached[1] == self.selector_kind
            and cached[2] is self.strict
            and cached[3] is self.slice_spec
        ):
            return cached[4]
        frozen = FrozenTargetSpec(
            selector_kind=self.selector_kind,
            selector_value=_freeze_value(self.selector_value),
            strict=self.strict,
            slice_spec=self.slice_spec,
            metadata=tuple(
                (key, _freeze_value(value))
                for key, value in sorted(self.metadata.items(), key=lambda pair: repr(pair[0]))
            ),
        )
        if not self.metadata and _freeze_stable(self.selector_value):
            self.__dict__["_tl_frozen_cache"] = (
                self.selector_value,
                self.selector_kind,
                self.strict,
                self.slice_spec,
                frozen,
            )
        return frozen

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state without the transient freeze cache."""

        state = dict(self.__dict__)
        state.pop("_tl_frozen_cache", None)
        return state


@dataclass(frozen=True)
class FrozenTargetSpec:
    """Immutable public selector target specification."""

    selector_kind: str
    selector_value: Any | None = None
    strict: bool = False
    slice_spec: TensorSliceSpec | None = None
    metadata: tuple[tuple[str, Any], ...] = ()


HelperKind: TypeAlias = Literal["forward", "backward"]
HelperDirection: TypeAlias = Literal["forward", "backward", "both"]
HelperPortability: TypeAlias = Literal["builtin", "import_ref", "opaque_audit"]


@dataclass(frozen=True)
class HelperSpec:
    """Portable identity and hook factory for helper-built interventions."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "helper_name": FieldPolicy.KEEP,
        "args": FieldPolicy.KEEP,
        "kwargs": FieldPolicy.KEEP,
        "kind": FieldPolicy.KEEP,
        "portability": FieldPolicy.KEEP,
        "factory": FieldPolicy.DROP,
        "metadata": FieldPolicy.KEEP,
        "direction": FieldPolicy.KEEP,
        "batch_independent": FieldPolicy.KEEP,
        "compatible_with_append": FieldPolicy.KEEP,
        # L6 Query-Selection recipe family: persists as of tlspec v8 as
        # BLOB_RECURSIVE (recipe ASTs may embed unit-term masks), with the
        # audit digest relation validated at load. NEVER smuggled through
        # the KEEP args/kwargs fields.
        "selection_recipe": FieldPolicy.BLOB_RECURSIVE,
    }

    helper_name: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    kind: HelperKind = "forward"
    portability: HelperPortability = "builtin"
    factory: Callable[[], Callable[..., Any]] | None = field(
        default=None, compare=False, repr=False
    )
    metadata: tuple[tuple[str, Any], ...] = ()
    direction: HelperDirection | None = None
    batch_independent: bool = False
    compatible_with_append: bool = False
    selection_recipe: Any = field(default=None, compare=False)

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
            Hook callable with signature ``hook(out, *, hook)``.

        Raises
        ------
        TypeError
            If the spec does not carry a runtime factory.
        """

        if self.factory is None:
            raise TypeError(f"HelperSpec {self.helper_name!r} has no hook factory")
        return self.factory()

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state, dropping factories pickle cannot carry.

        Builtin factories are local closures derived entirely from the
        stable ``(helper_name, args, kwargs)`` identity; ``__setstate__``
        rebuilds them through the same builtin registry ``tl.load`` uses,
        so plain ``pickle`` and ``tl.save`` agree on helper-carrying specs.
        ``opaque_audit`` factories (load-time raising placeholders) drop to
        the canonical factory-less audit-only form.
        """

        state = dict(self.__dict__)
        if self.portability in ("builtin", "opaque_audit"):
            state["factory"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state, rebuilding a dropped builtin factory.

        Raises
        ------
        InvalidArgumentError
            If a builtin helper name is unknown to this torchlens (same
            typed refusal as an intervention-spec load).
        """

        if state.get("factory") is None and state.get("portability") == "builtin":
            from .helpers import rebuild_builtin_helper

            rebuilt = rebuild_builtin_helper(
                state["helper_name"],
                tuple(state.get("args", ())),
                dict(state.get("kwargs", ())),
            )
            state = {**state, "factory": rebuilt.factory}
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __deepcopy__(self, memo: dict[int, Any]) -> HelperSpec:
        """Field-wise deepcopy preserving factory identity.

        Explicit so ``copy.deepcopy`` keeps its pre-pickle-hook semantics
        (functions are deepcopy-atomic, so the factory closure is shared by
        identity) instead of routing through ``__getstate__``'s
        factory-dropping pickle path.
        """

        import copy as _copy

        cls = type(self)
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            object.__setattr__(clone, key, _copy.deepcopy(value, memo))
        return clone


@dataclass(frozen=True, slots=True)
class InterventionDecision:
    """Active predicate-time intervention decision.

    Parameters
    ----------
    action:
        Active intervention action kind.
    hook:
        Helper spec or callable consumed by the existing hook normalizer.
    template_ref:
        Optional template reference associated with the intervention.
    keep_grad:
        Whether the replacement should preserve gradient connectivity when possible.
    isolate:
        Whether backend isolation is required before applying the action.
    """

    action: InterventionAction
    hook: Any | None = None
    template_ref: Any | None = None
    keep_grad: bool = False
    isolate: bool = False
    direction: HelperDirection = "forward"
    #: Provenance of the SPEC rule that produced this decision (leverage B7):
    #: the user's WHERE expression must survive into the persisted spec, so
    #: the spec threads it through the decision to the staging site. It is
    #: a provenance rider only — predicate-door persistence still stamps
    #: spec_derived=True (the addressing is lowered per-site labels either
    #: way). Bare-predicate decisions carry None.
    rule_id: str | None = None
    where_repr: str | None = None


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
class ParentRef:
    """Template component that resolves to a previously captured tensor."""

    parent_label: str


@dataclass(frozen=True)
class LiteralTensor:
    """Template component that stores a tensor literal for replay.

    ``param_barcode`` (r75 F2) snapshots, AT CLASSIFICATION TIME (mid-capture, model
    provably alive), the per-capture random barcode model prep stamped on an
    ``nn.Parameter`` argument -- the same barcode mirrored onto the cooked ``Param``
    record. The runnable producer's parameter matcher uses it as its gc-immune identity
    rung: session cleanup strips the parameter's weak registry meta and postprocess
    releases ``Param._param_ref``, so without this snapshot a caller that never held the
    model (plus one ``gc.collect()`` before the save) left two same-shape+dtype BN params
    unmatchable -- a nondeterministic honest-save over-refusal. ``None`` for non-Parameter
    literals and foreign/unstamped parameters (matching falls through to the other rungs);
    optional-with-default keeps unpickling of pre-r75 payloads intact (readers use
    ``getattr`` with a ``None`` fallback).
    """

    value: Any
    param_barcode: str | None = None


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

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "target_label": FieldPolicy.KEEP,
        "call_label": FieldPolicy.KEEP,
        "func_call_id": FieldPolicy.KEEP,
        "container_path": FieldPolicy.KEEP,
        "engine": FieldPolicy.KEEP,
        "helper": FieldPolicy.KEEP,
        "site_label": FieldPolicy.KEEP,
        "timing": FieldPolicy.KEEP,
        "direction": FieldPolicy.KEEP,
        "helper_name": FieldPolicy.KEEP,
        "seed": FieldPolicy.KEEP,
        "determinism_note": FieldPolicy.KEEP,
        "timestamp": FieldPolicy.KEEP,
        "backward_pass_index": FieldPolicy.KEEP,
        "call_index": FieldPolicy.KEEP,
        "grad_kind": FieldPolicy.KEEP,
        "tuple_index": FieldPolicy.KEEP,
        "replaced": FieldPolicy.KEEP,
        # L6 stage 3: (child_func_call_id, arg_kind, arg_path) occurrence
        # address on edge-substitution FireRecords; persists as of tlspec v8.
        "edge_address": FieldPolicy.KEEP,
    }

    target_label: str = ""
    call_label: str | None = None
    func_call_id: int | None = None
    container_path: tuple[OutputPathComponent, ...] = ()
    engine: str | None = None
    helper: HelperSpec | None = None
    site_label: str | None = None
    timing: Literal["pre", "post"] | None = None
    direction: Literal["forward", "backward"] | None = None
    helper_name: str | None = None
    seed: int | None = None
    determinism_note: str | None = None
    timestamp: float | None = None
    backward_pass_index: int | None = None
    call_index: int | None = None
    grad_kind: Literal["grad_input", "grad_output"] | None = None
    tuple_index: int | None = None
    edge_address: tuple | None = None
    replaced: bool | None = None


@dataclass(frozen=True)
class EdgeUseRecord(_SelectionOperand):
    """Provenance for one parent tensor use by a child operation.

    The canonical occurrence address is ``(child_func_call_id, arg_kind,
    arg_path)`` — stable within a trace and across its save/load. Records are
    region-shaped producers: ``__selection__`` lifts one edge occurrence as
    an EDGE-kind selection (whole-edge granularity), so edge sets compose
    with the ``| & - ~`` algebra.
    """

    parent_label: str
    child_label: str
    arg_kind: Literal["positional", "keyword"]
    arg_path: tuple[OutputPathComponent, ...]
    view_or_copy: Literal["view", "copy", "unknown"] | None
    parent_func_call_id: int | None
    child_func_call_id: int
    edge_use: str = "arg"

    def __selection__(self) -> Any:
        """Lift this edge occurrence as an EDGE selection term."""

        from ..selection import _selection_from_edge

        return _selection_from_edge(self)

    def __repr__(self) -> str:
        """Designed per-edge line (F10): source -> target (arg k) + relations.

        The storage relation prints only when POPULATED (``view``/``copy``);
        the honest sentinel ``unknown`` stays silent rather than reading as
        a fact (lovely bug 24's sentinel discipline).
        """

        path = ".".join(str(component) for component in self.arg_path) or "?"
        line = f"edge {self.parent_label} -> {self.child_label} ({self.arg_kind} {path})"
        if self.view_or_copy in ("view", "copy"):
            line += f" [{self.view_or_copy}]"
        return line


@dataclass
class TargetValueSpec:
    """Mutable set-replacement entry in an intervention recipe."""

    site_target: TargetSpec
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> FrozenTargetValueSpec:
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

    def freeze(self) -> FrozenHookSpec:
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

        from ._helper_fingerprint import stamp_metadata

        hook_spec = HookSpec(
            site_target=site_target,
            hook=hook,
            helper=helper,
            handle=handle,
            # Record the helper's tensors as staged: helpers alias them (F9).
            metadata=stamp_metadata(metadata, hook, helper),
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
            Optional hook handle. Matching stored handles are removed when
            present.

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
        """Clear all sticky hook entries.

        Returns
        -------
        None
            This spec is mutated in place.
        """

        self.hook_specs.clear()

    def freeze(self) -> FrozenInterventionSpec:
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
    """Copy policy for fields when a Trace is forked."""

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
    # Seed from the canonical field order plus any share/reconstruct names that
    # live outside it (e.g. `_optimizer`, which is a real Trace attribute carried
    # by the fork but is intentionally NOT a tabular display field in
    # MODEL_LOG_FIELD_ORDER). Iterating field_order alone would silently drop
    # those entries, letting the fork fall through to the deepcopy default and
    # sever the shared reference the `share` set explicitly asked for.
    seen: set[str] = set()
    for field_name in [*field_order, *sorted(share | reconstruct)]:
        if field_name in seen:
            continue
        seen.add(field_name)
        if field_name in reconstruct:
            table[field_name] = ForkFieldPolicy.FORK_RECONSTRUCT
        elif field_name in share:
            table[field_name] = ForkFieldPolicy.FORK_SHARE
        else:
            table[field_name] = ForkFieldPolicy.FORK_COPY
    return table


def _build_trace_fork_policy() -> dict[str, ForkFieldPolicy]:
    """Build the Trace fork policy table.

    Returns
    -------
    dict[str, ForkFieldPolicy]
        Fork policies for Trace fields.
    """

    from ..constants import MODEL_LOG_FIELD_ORDER

    table = _fork_policy_table(
        MODEL_LOG_FIELD_ORDER,
        share={
            "activation_transform",
            "grad_transform",
            "_output_transform",
            "_source_code_blob",
            "_source_model_ref",
            "_optimizer",
        },
        # A fork never inherits the parent's settled capture outcome: the
        # fork is the sanctioned MUTATION surface, so carrying the parent's
        # blessed attestation by identity would let a hand-edited fork save
        # as a bit-identical attested COMPLETE. ``build_fork`` settles a
        # DERIVED outcome via ``capture.outcome.stamp_forked`` instead.
        reconstruct={"parent_run", "_capture_outcome"},
    )
    # `_trace_core` (the columnar op row store) is not in MODEL_LOG_FIELD_ORDER
    # (private runtime storage, FieldPolicy.DROP). The generic field pass
    # reconstructs (drops) it; the M11 COW fork builder installs the forked
    # core (per-fork store views over the shared sealed base) explicitly
    # after the field pass.
    table["_trace_core"] = ForkFieldPolicy.FORK_RECONSTRUCT
    return table


def _build_op_log_fork_policy() -> dict[str, ForkFieldPolicy]:
    """Build the Op fork policy table.

    Returns
    -------
    dict[str, ForkFieldPolicy]
        Fork policies for Op fields.
    """

    from ..constants import LAYER_PASS_LOG_FIELD_ORDER

    table = _fork_policy_table(
        LAYER_PASS_LOG_FIELD_ORDER,
        share={
            "out",
            "transformed_out",
            "grad",
            "transformed_grad",
            "func",
            "grad_fn_handle",
        },
        reconstruct={"source_trace", "_construction_done"},
    )
    # `_facets_cache` is not part of LAYER_PASS_LOG_FIELD_ORDER (it is a lazily
    # populated runtime cache, not a portable/user-facing field), so the loop
    # above never assigns it a policy. Left unset, a forked Op would fall
    # through to the generic default policy and attempt copy.deepcopy() on the
    # cached FacetView -- which self-references its owning Op and crashes with
    # RecursionError. Reconstruct it instead (discarded immediately afterward
    # by _rebind_fork_owner_refs's `del layer_pass.facets` in any case),
    # mirroring how Op.__getstate__ already excludes this field for pickling.
    table["_facets_cache"] = ForkFieldPolicy.FORK_RECONSTRUCT
    return table


MODEL_LOG_FIELD_FORK_POLICY = _build_trace_fork_policy()
LAYER_PASS_LOG_FIELD_FORK_POLICY = _build_op_log_fork_policy()

#: Public edit-object type (slate 5.5, ratified subject to D7 default-keep):
#: ``tl.Edit`` is the public spelling; ``HelperSpec`` is its deprecated alias
#: (stable surface, no removal scheduled).
Edit = HelperSpec

# The tlspec v8 coordinated bump retired this module's S3 pre-release
# registrations (HelperSpec.selection_recipe, FireRecord.edge_address).

__all__ = [
    "Edit",
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
    "InterventionAction",
    "InterventionDecision",
    "HookSpec",
    "InterventionSpec",
    "LAYER_PASS_LOG_FIELD_FORK_POLICY",
    "LiteralTensor",
    "LiteralValue",
    "MODEL_LOG_FIELD_FORK_POLICY",
    "NamedField",
    "OutputPathComponent",
    "ParentRef",
    "Relationship",
    "TargetSpec",
    "TargetValueSpec",
    "TensorSliceSpec",
    "TupleIndex",
    "Unsupported",
    "rebuild_container_from_spec",
]
