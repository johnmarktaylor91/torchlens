"""Hook parsing and hook-spec ownership for TorchLens interventions."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import torch

from .errors import (
    HelperMountError,
    HookSignatureError,
    HookSiteCoverageError,
    LiveModeLabelError,
    SiteResolutionError,
)
from .selectors import (
    BaseSelector,
    CompositeSelector,
    FacetSelector,
    NotSelector,
    SelectorLike,
    _classify_selector_direction,
    in_module,
)
from .types import HelperSpec, HookSpec, InterventionSpec, TargetSpec, TargetValueSpec

HookTiming: TypeAlias = Literal["pre", "post"]
HookDirection: TypeAlias = Literal["forward", "backward"]
HookCallable: TypeAlias = Callable[..., torch.Tensor]
HookInput: TypeAlias = Callable[..., Any] | HelperSpec
_FINAL_LABEL_PATTERN = re.compile(r"(?:_\d+_\d+(?::\d+)?$|:\d+$)")
_DEFAULT_HEAD_FACET_NAMES = ("q", "k", "v")

_LAYER_LOG_CONTEXT_FIELDS = (
    "layer_label",
    "layer_type",
    "shape",
    "dtype",
    "tensor_device",
    "address",
    "call_index",
)


@dataclass(frozen=True, slots=True)
class HookContext:
    """Frozen metadata snapshot passed to hook callables.

    Parameters
    ----------
    name:
        Display name of the hook.
    timing:
        Hook timing. Phase 3 stores both pre and post; MVP execution uses post.
    direction:
        ``"forward"`` for outs or ``"backward"`` for grads.
    layer_log:
        Mapping proxy over selected layer metadata, never a live Op.
    ctx:
        Per resolved ``(site, hook instance)`` scratch dictionary.
    run_ctx:
        Mutable dictionary shared across all hooks in a run.
    step_index:
        Trace index for bundle operations.
    trace_name:
        Trace name for bundle operations.
    args:
        Frozen positional tensor arguments at the hook site.
    kwargs:
        Mapping proxy of tensor keyword arguments at the hook site.
    """

    name: str
    timing: HookTiming
    direction: HookDirection
    layer_log: Mapping[str, Any]
    ctx: dict[str, Any]
    run_ctx: dict[str, Any]
    step_index: int
    trace_name: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NormalizedHookEntry:
    """Uniform representation of one requested hook attachment."""

    site_target: Any
    normalized_callable: HookCallable
    helper_spec: HelperSpec | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def make_hook_context(
    *,
    name: str,
    timing: HookTiming = "post",
    direction: HookDirection = "forward",
    layer_log: Any | None = None,
    ctx: dict[str, Any] | None = None,
    run_ctx: dict[str, Any] | None = None,
    step_index: int = 0,
    trace_name: str = "",
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> HookContext:
    """Build a frozen hook context with mapping-proxy snapshots.

    Parameters
    ----------
    name:
        Display name of the hook.
    timing:
        Hook timing.
    direction:
        Hook direction.
    layer_log:
        Optional Op-like object or mapping to snapshot.
    ctx:
        Per-hook scratch dictionary. A new dictionary is created when omitted.
    run_ctx:
        Shared run dictionary. A new dictionary is created when omitted.
    step_index:
        Trace index.
    trace_name:
        Trace name.
    args:
        Positional tensor args at this site.
    kwargs:
        Keyword tensor args at this site.

    Returns
    -------
    HookContext
        Frozen hook context object.
    """

    return HookContext(
        name=name,
        timing=timing,
        direction=direction,
        layer_log=MappingProxyType(_snapshot_layer_log(layer_log)),
        ctx={} if ctx is None else ctx,
        run_ctx={} if run_ctx is None else run_ctx,
        step_index=step_index,
        trace_name=trace_name,
        args=tuple(args),
        kwargs=MappingProxyType(dict(kwargs or {})),
    )


def normalize_hook(fn: HookInput, *, direction: HookDirection = "forward") -> HookCallable:
    """Normalize one callable or helper spec into a hook callable.

    Parameters
    ----------
    fn:
        User callable or helper spec.
    direction:
        Expected hook direction.

    Returns
    -------
    HookCallable
        Signature-validated hook callable.
    """

    helper_spec: HelperSpec | None = fn if isinstance(fn, HelperSpec) else None
    hook_callable = helper_spec() if helper_spec is not None else fn
    if not callable(hook_callable):
        raise HookSignatureError(f"hook object {hook_callable!r} is not callable")
    _validate_hook_signature(hook_callable, direction=direction)
    return cast(HookCallable, hook_callable)


def normalize_hook_plan(
    hooks_or_site: Any,
    hook: HookInput | None = None,
    *,
    default_site_target: Any | None = None,
    force_shape_change: bool = False,
) -> list[NormalizedHookEntry]:
    """Normalize all supported attach-hook shapes into hook-plan entries.

    Parameters
    ----------
    hooks_or_site:
        Callable/helper, mapping, list-of-tuples, or site target.
    hook:
        Optional hook for the ``(site, hook)`` shape.
    default_site_target:
        Site target for bare callable/helper input. Without this, bare hooks
        fail closed because Phase 3 does not resolve model logs.
    force_shape_change:
        Escape hatch metadata consumed by the later execution layer.

    Returns
    -------
    list[NormalizedHookEntry]
        Uniform hook plan in left-to-right composition order.
    """

    pairs = _dispatch_hook_pairs(hooks_or_site, hook, default_site_target=default_site_target)
    entries: list[NormalizedHookEntry] = []
    for order, (site_target, hook_like) in enumerate(pairs):
        helper_spec = hook_like if isinstance(hook_like, HelperSpec) else None
        for direction in _hook_directions(hook_like, helper_spec):
            _validate_helper_mount(site_target, helper_spec)
            normalized_callable = _normalize_directional_hook(hook_like, direction=direction)
            entries.append(
                NormalizedHookEntry(
                    site_target=site_target,
                    normalized_callable=normalized_callable,
                    helper_spec=helper_spec,
                    metadata=MappingProxyType(
                        {
                            "attach_order": order,
                            "composition": "left_to_right",
                            "force_shape_change": force_shape_change,
                            "direction": direction,
                            "timing": "post",
                        }
                    ),
                )
            )
    return entries


def _hook_directions(
    hook_like: HookInput, helper_spec: HelperSpec | None
) -> tuple[HookDirection, ...]:
    """Return hook directions requested by a hook-like object.

    Parameters
    ----------
    hook_like:
        User callable, observer, or helper spec.
    helper_spec:
        Helper spec when ``hook_like`` is a built-in helper.

    Returns
    -------
    tuple[HookDirection, ...]
        One or two concrete hook directions.
    """

    if helper_spec is not None:
        return (helper_spec.kind,)
    direction = getattr(hook_like, "direction", "forward")
    if direction == "both":
        return ("forward", "backward")
    if direction in {"forward", "backward"}:
        return (cast(HookDirection, direction),)
    raise HookSignatureError("hook direction must be 'forward', 'backward', or 'both'")


def _normalize_directional_hook(hook_like: HookInput, *, direction: HookDirection) -> HookCallable:
    """Normalize a hook for a concrete direction.

    Parameters
    ----------
    hook_like:
        User callable, observer, or helper spec.
    direction:
        Concrete hook direction.

    Returns
    -------
    HookCallable
        Direction-specific hook callable.
    """

    if direction == "backward" and hasattr(hook_like, "record_backward"):
        return normalize_hook(getattr(hook_like, "record_backward"), direction=direction)
    return normalize_hook(hook_like, direction=direction)


def normalize_hooks_from_spec(spec: InterventionSpec | None) -> list[NormalizedHookEntry]:
    """Normalize an intervention spec into live-capture hook-plan entries.

    Parameters
    ----------
    spec:
        Mutable intervention spec attached to a ``Trace``. ``None`` or an
        empty spec produces no hook entries.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook entries suitable for ``_state._active_hook_plan`` during rerun.
    """

    if spec is None:
        return []

    entries: list[NormalizedHookEntry] = []
    entries.extend(_normalize_value_specs(spec.target_value_specs))
    entries.extend(_normalize_sticky_hook_specs(spec.hook_specs))

    if spec.targets:
        hook_like = _hook_like_from_spec(spec)
        if hook_like is not None:
            for target in spec.targets:
                entries.extend(normalize_hook_plan(target, hook_like))
    return entries


def is_facet_target(site_target: Any) -> bool:
    """Return whether a site target is a semantic facet selector.

    Parameters
    ----------
    site_target:
        Selector-like site target.

    Returns
    -------
    bool
        Whether the target should be expanded before storage.
    """

    return _facet_selector_from_target(site_target) is not None


def expand_facet_hook_entries(
    log: Any, entries: Sequence[NormalizedHookEntry]
) -> list[NormalizedHookEntry]:
    """Expand facet hook entries into ordinary home-op hook entries.

    Parameters
    ----------
    log:
        Trace that owns the facet registry snapshot and captured home ops.
    entries:
        Normalized hook entries, possibly including ``tl.facet`` or
        ``tl.head`` targets.

    Returns
    -------
    list[NormalizedHookEntry]
        Entries whose facet targets have been replaced by home-op label
        targets and slice-aware wrapper hooks.
    """

    expanded: list[NormalizedHookEntry] = []
    claimed = _existing_facet_write_claims(log)
    for entry in entries:
        selector = _facet_selector_from_target(entry.site_target)
        if selector is None:
            expanded.append(entry)
            continue
        specs = _resolve_facet_specs(log, selector)
        if not specs:
            if selector.name in {"pattern", "scores", "z", "result"}:
                raise SiteResolutionError(
                    f"facet selector {selector!r} matched no write-capable real tensor facets. "
                    "Fused attention reconstruction facets are read-only; re-run with a scoped "
                    "eager attention implementation so the requested internal facet is captured "
                    "as an op before attaching this intervention."
                )
            raise SiteResolutionError(
                f"facet selector {selector!r} matched 0 write-capable facets."
            )
        for facet_name, spec in specs:
            try:
                mask = spec.write_mask().detach().clone()
            except RuntimeError as exc:
                if selector.name in {"pattern", "scores", "z", "result"}:
                    raise SiteResolutionError(
                        f"facet selector {selector!r} resolved only read-only reconstructed "
                        "attention facets. Re-run with a scoped eager attention implementation "
                        "so the requested internal facet is captured as a real tensor op."
                    ) from exc
                raise
            home_label = _facet_home_label(spec)
            _check_facet_write_conflicts(claimed, home_label=home_label, mask=mask)
            claimed.append((home_label, mask, facet_name))
            metadata = {
                **entry.metadata,
                "facet_write": True,
                "facet_name": facet_name,
                "facet_home_label": home_label,
                "facet_write_mask": mask,
            }
            expanded.append(
                NormalizedHookEntry(
                    site_target=TargetSpec("label", home_label),
                    normalized_callable=_facet_scatter_hook(
                        entry.normalized_callable,
                        spec,
                        facet_name=facet_name,
                    ),
                    helper_spec=entry.helper_spec,
                    metadata=MappingProxyType(metadata),
                )
            )
    return expanded


def _normalize_value_specs(value_specs: Sequence[TargetValueSpec]) -> list[NormalizedHookEntry]:
    """Normalize set-replacement specs into hook-plan entries.

    Parameters
    ----------
    value_specs:
        Mutable set-replacement specs from an intervention recipe.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook-plan entries that replace matching outs.
    """

    entries: list[NormalizedHookEntry] = []
    for order, value_spec in enumerate(value_specs):
        value = value_spec.value

        def _set_value_hook(
            out: torch.Tensor,
            *,
            hook: HookContext,
            replacement: Any = value,
        ) -> torch.Tensor:
            """Return a static or one-shot callable replacement.

            Parameters
            ----------
            out:
                Activation at the matched site.
            hook:
                Hook context supplied by TorchLens.
            replacement:
                Replacement tensor or callable captured by default argument.

            Returns
            -------
            torch.Tensor
                Replacement out.
            """

            del hook
            if callable(replacement):
                return cast(torch.Tensor, replacement(out))
            return cast(torch.Tensor, replacement)

        metadata = {
            "attach_order": order,
            "composition": "left_to_right",
            "force_shape_change": False,
            "direction": "forward",
            "timing": "post",
            **value_spec.metadata,
        }
        entries.append(
            NormalizedHookEntry(
                site_target=value_spec.site_target,
                normalized_callable=_set_value_hook,
                helper_spec=None,
                metadata=MappingProxyType(metadata),
            )
        )
    return entries


def _normalize_sticky_hook_specs(hook_specs: Sequence[HookSpec]) -> list[NormalizedHookEntry]:
    """Normalize sticky hook specs into hook-plan entries.

    Parameters
    ----------
    hook_specs:
        Mutable sticky hook specs from an intervention recipe.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook-plan entries in stored composition order.
    """

    entries: list[NormalizedHookEntry] = []
    for hook_spec in hook_specs:
        hook_like = hook_spec.helper if hook_spec.helper is not None else hook_spec.hook
        normalized_entries = normalize_hook_plan(hook_spec.site_target, hook_like)
        for entry in normalized_entries:
            metadata = {**entry.metadata, **hook_spec.metadata}
            entries.append(
                NormalizedHookEntry(
                    site_target=entry.site_target,
                    normalized_callable=entry.normalized_callable,
                    helper_spec=entry.helper_spec,
                    metadata=MappingProxyType(metadata),
                )
            )
    return entries


def _hook_like_from_spec(spec: InterventionSpec) -> HookInput | None:
    """Return the callable/helper represented by an intervention spec.

    Parameters
    ----------
    spec:
        Intervention spec to inspect.

    Returns
    -------
    HookInput | None
        Normalizable hook input, or ``None`` when the spec is empty.
    """

    if spec.hook is not None:
        return cast(HookInput, spec.hook)
    if spec.helper is not None:
        return spec.helper
    if spec.value is not None:
        value = spec.value

        def _value_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return a fixed replacement value from an intervention spec.

            Parameters
            ----------
            out:
                Activation being replaced.
            hook:
                Hook context supplied by TorchLens.

            Returns
            -------
            torch.Tensor
                Replacement tensor.
            """

            del out, hook
            if callable(value):
                return cast(torch.Tensor, value())
            return cast(torch.Tensor, value)

        return _value_hook
    return None


def live_selector_matches_site(selector_like: Any, site: Any) -> bool:
    """Return whether a selector can match one capture-time site.

    Parameters
    ----------
    selector_like:
        Selector or target-spec-like object from a normalized hook entry.
    site:
        Capture-time pass proxy with raw labels and module context populated.

    Returns
    -------
    bool
        Whether the selector matches the site during live capture.

    Raises
    ------
    LiveModeLabelError
        If a finalized-looking ``tl.label(...)`` selector cannot exist yet.
    SiteResolutionError
        If the selector kind is unsupported or a predicate fails.
    """

    selector = _normalize_live_selector(selector_like)
    matched = _live_selector_matches_unchecked(selector, site)
    if not matched and selector.selector_kind == "label":
        label_value = str(selector.selector_value)
        if _looks_like_finalized_label(label_value):
            raise LiveModeLabelError(_live_label_error_message(label_value))
    return matched


def _normalize_live_selector(selector_like: Any) -> BaseSelector:
    """Normalize a live-capture selector input.

    Parameters
    ----------
    selector_like:
        Selector-like object from the hook plan.

    Returns
    -------
    BaseSelector
        Normalized selector.
    """

    if isinstance(selector_like, BaseSelector):
        return selector_like
    if isinstance(selector_like, TargetSpec):
        return _selector_from_target_spec(selector_like)
    if isinstance(selector_like, str):
        from .selectors import label

        return label(selector_like)
    if hasattr(selector_like, "layer_label"):
        from .selectors import label

        return label(str(selector_like.layer_label))
    raise SiteResolutionError(f"Unsupported live hook selector {selector_like!r}.")


def _selector_from_target_spec(target: TargetSpec) -> BaseSelector:
    """Build a selector from a live target spec.

    Parameters
    ----------
    target:
        Target spec to convert.

    Returns
    -------
    BaseSelector
        Selector equivalent to the target spec.
    """

    from .selectors import contains, func, grad_fn, intervening, label, module, where

    if target.selector_kind == "label":
        return label(str(target.selector_value))
    if target.selector_kind == "func":
        if isinstance(target.selector_value, dict):
            return func(
                str(target.selector_value.get("name")),
                output=target.selector_value.get("output"),
            )
        return func(str(target.selector_value))
    if target.selector_kind == "module":
        return module(str(target.selector_value))
    if target.selector_kind == "output":
        from .selectors import output

        return output(cast("int | str", target.selector_value))
    if target.selector_kind == "contains":
        return contains(str(target.selector_value))
    if target.selector_kind == "in_module":
        from .selectors import in_module as make_in_module

        selector = make_in_module(str(target.selector_value))
        if isinstance(selector, BaseSelector):
            return selector
    if target.selector_kind == "predicate" and callable(target.selector_value):
        return where(target.selector_value, name_hint=target.metadata.get("name_hint"))
    if target.selector_kind == "grad_fn":
        payload = dict(target.selector_value or {})
        return grad_fn(
            payload.get("type"),
            label=payload.get("grad_fn_label_pattern"),
            is_custom=payload.get("is_custom"),
        )
    if target.selector_kind == "intervening":
        return intervening()
    if target.selector_kind == "label":
        return label(str(target.selector_value))
    if target.selector_kind == "not":
        return ~_normalize_live_selector(target.selector_value)
    raise SiteResolutionError(f"Unsupported live hook selector kind {target.selector_kind!r}.")


def live_backward_selector_matches(
    selector_like: Any,
    grad_fn_handle: Any,
    call_index: int,
) -> bool:
    """Return whether a selector can match one grad_fn_handle callback site.

    Parameters
    ----------
    selector_like:
        Selector or target spec from a normalized hook entry.
    grad_fn_handle:
        GradFn receiving a backward hook callback.
    call_index:
        One-based callback index.

    Returns
    -------
    bool
        Whether the selector matches this backward site.
    """

    del call_index
    from .resolver import _resolve_unchecked

    selector = _normalize_live_selector(selector_like)
    return bool(_resolve_unchecked((grad_fn_handle,), selector, strict=False))


def _validate_helper_mount(site_target: Any, helper_spec: HelperSpec | None) -> None:
    """Validate helper/selector mount compatibility at attach time.

    Parameters
    ----------
    site_target:
        Selector-like site target.
    helper_spec:
        Helper spec, if the hook came from a built-in helper.

    Returns
    -------
    None
        Raises for incompatible helper mount shapes.
    """

    if helper_spec is None:
        return
    helper_metadata = dict(helper_spec.metadata)
    mount_shape = helper_metadata.get("mount_shape", "tensor")
    from .resolver import _selector_resolution_direction

    selector_direction = _selector_resolution_direction(site_target)
    if mount_shape == "tuple" and selector_direction != "backward":
        raise HelperMountError(
            f"{helper_spec.name} is a grad_fn_handle helper and must be mounted on a backward selector."
        )
    if mount_shape == "tensor" and selector_direction == "backward":
        raise HelperMountError(
            f"{helper_spec.name} is a tensor-gradient helper and cannot mount on grad_fn_handle sites."
        )
    if helper_metadata.get("requires_grad_output") and selector_direction == "backward":
        raise HelperMountError(
            f"{helper_spec.name} requires grad_output and cannot mount on AccumulateGrad prehooks."
        )


def _facet_selector_from_target(site_target: Any) -> FacetSelector | None:
    """Return a facet selector from a target object when possible.

    Parameters
    ----------
    site_target:
        Selector-like target.

    Returns
    -------
    FacetSelector | None
        Normalized facet selector, or ``None`` for ordinary targets.
    """

    if isinstance(site_target, FacetSelector):
        return site_target
    if isinstance(site_target, TargetSpec) and site_target.selector_kind == "facet":
        return _facet_selector_from_payload(site_target.selector_value)
    return None


def _facet_selector_from_payload(payload: Any) -> FacetSelector:
    """Build a ``FacetSelector`` from stored target payload data.

    Parameters
    ----------
    payload:
        TargetSpec selector payload.

    Returns
    -------
    FacetSelector
        Normalized semantic facet selector.
    """

    if isinstance(payload, Mapping):
        name = payload.get("name")
        head_index = payload.get("head_index")
        return FacetSelector(
            None if name is None else str(name),
            head_index=None if head_index is None else int(head_index),
        )
    if isinstance(payload, str):
        return FacetSelector(payload)
    raise SiteResolutionError(f"Unsupported facet selector payload {payload!r}.")


def _resolve_facet_specs(log: Any, selector: FacetSelector) -> list[tuple[str, Any]]:
    """Resolve a semantic facet selector against captured module facet views.

    Parameters
    ----------
    log:
        Trace whose modules should be searched.
    selector:
        Facet selector to resolve.

    Returns
    -------
    list[tuple[str, Any]]
        Pairs of public facet names and ``FacetSpec`` objects.
    """

    from ..semantic.facets import Facet, FacetSpec

    resolved: list[tuple[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    facet_names = _DEFAULT_HEAD_FACET_NAMES if selector.name is None else (selector.name,)
    for record in _iter_facet_records(log):
        view = getattr(record, "facets", None)
        if view is None:
            continue
        for facet_name in facet_names:
            if not view.has(facet_name):
                continue
            try:
                if selector.head_index is None:
                    value = view[facet_name]
                else:
                    value = view.head(selector.head_index)[facet_name]
            except (AttributeError, KeyError, RuntimeError, ValueError):
                continue
            spec = (
                value.spec
                if isinstance(value, Facet)
                else value
                if isinstance(value, FacetSpec)
                else None
            )
            if spec is None:
                continue
            dedupe_key = (id(spec.home), repr(spec.transforms), facet_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            resolved.append((facet_name, spec))
    return resolved


def _iter_facet_records(log: Any) -> Iterator[Any]:
    """Yield records that can own semantic facet views.

    Parameters
    ----------
    log:
        Trace to inspect.

    Yields
    ------
    Any
        Module records followed by op records.
    """

    modules = getattr(log, "modules", ())
    try:
        yield from modules
    except TypeError:
        pass
    for op in getattr(log, "layer_list", ()):
        yield op


def _facet_home_label(spec: Any) -> str:
    """Return the live hook label for a facet home op.

    Parameters
    ----------
    spec:
        FacetSpec-like object.

    Returns
    -------
    str
        Home op layer label.
    """

    home = getattr(spec, "home", None)
    label_value = (
        getattr(home, "_layer_label_raw", None)
        or getattr(home, "_label_raw", None)
        or getattr(home, "layer_label", None)
        or getattr(home, "label", None)
    )
    if label_value is None:
        label_value = getattr(spec, "home_label", None)
    if label_value is None:
        raise SiteResolutionError("Facet selector resolved to a home without a layer label.")
    return str(label_value)


def _existing_facet_write_claims(log: Any) -> list[tuple[str, torch.Tensor, str]]:
    """Return existing facet write claims from a trace intervention spec.

    Parameters
    ----------
    log:
        Trace whose sticky hook specs should be inspected.

    Returns
    -------
    list[tuple[str, torch.Tensor, str]]
        Claimed ``(home_label, mask, facet_name)`` tuples.
    """

    spec = getattr(log, "_intervention_spec", None)
    if spec is None:
        return []
    claims: list[tuple[str, torch.Tensor, str]] = []
    for hook_spec in getattr(spec, "hook_specs", ()):
        metadata = getattr(hook_spec, "metadata", {})
        if not metadata.get("facet_write"):
            continue
        mask = metadata.get("facet_write_mask")
        home_label = metadata.get("facet_home_label")
        facet_name = metadata.get("facet_name", "<unknown>")
        if isinstance(mask, torch.Tensor) and home_label is not None:
            claims.append((str(home_label), mask, str(facet_name)))
    return claims


def _check_facet_write_conflicts(
    claims: Sequence[tuple[str, torch.Tensor, str]],
    *,
    home_label: str,
    mask: torch.Tensor,
) -> None:
    """Raise when a facet write overlaps an existing same-home claim.

    Parameters
    ----------
    claims:
        Existing write claims.
    home_label:
        Home label for the candidate write.
    mask:
        Candidate boolean write mask.

    Returns
    -------
    None
        Raises on overlapping writes.
    """

    for claimed_home, claimed_mask, claimed_name in claims:
        if claimed_home != home_label:
            continue
        if tuple(claimed_mask.shape) != tuple(mask.shape):
            continue
        if bool(torch.logical_and(claimed_mask.to(mask.device), mask).any()):
            raise SiteResolutionError(
                f"Facet intervention conflict on home {home_label!r}: write overlaps "
                f"existing facet {claimed_name!r}."
            )


def _facet_scatter_hook(
    user_hook: HookCallable,
    spec: Any,
    *,
    facet_name: str,
) -> HookCallable:
    """Return a home-op hook that edits one facet slice.

    Parameters
    ----------
    user_hook:
        Normalized user hook that receives the facet slice.
    spec:
        FacetSpec used to read and scatter the slice.
    facet_name:
        Public facet name for diagnostics.

    Returns
    -------
    HookCallable
        Hook with the standard whole-output TorchLens contract.
    """

    def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Apply a user hook to a facet slice and scatter it into ``out``.

        Parameters
        ----------
        out:
            Full home-op output tensor.
        hook:
            Hook context supplied by TorchLens.

        Returns
        -------
        torch.Tensor
            Full edited home-op output tensor.
        """

        from .runtime import validate_hook_output

        facet_slice = spec.apply(out)
        edited_slice = user_hook(facet_slice, hook=hook)
        checked_slice = validate_hook_output(edited_slice, facet_slice, hook_context=hook)
        return spec.scatter_update(out, checked_slice, mode="replace")

    _hook.__name__ = f"facet_{facet_name}_scatter_hook"
    return _hook


def _selector_direction_recursive(selector: BaseSelector) -> HookDirection | None:
    """Return explicit selector direction for a selector tree, if any.

    Parameters
    ----------
    selector:
        Selector tree to inspect.

    Returns
    -------
    HookDirection | None
        Explicit forward/backward direction, or None.
    """

    if isinstance(selector, CompositeSelector):
        found: HookDirection | None = None
        for child in selector.selectors:
            child_dir = _selector_direction_recursive(_normalize_live_selector(child))
            if child_dir is not None:
                found = child_dir
        return found
    if isinstance(selector, NotSelector):
        return _selector_direction_recursive(_normalize_live_selector(selector.selector))
    return _classify_selector_direction(selector)


def _live_selector_matches_unchecked(selector: BaseSelector, site: Any) -> bool:
    """Match one normalized selector against one live site.

    Parameters
    ----------
    selector:
        Normalized selector.
    site:
        Capture-time site proxy.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    kind = selector.selector_kind
    value = selector.selector_value
    if kind == "and" and isinstance(selector, CompositeSelector):
        return all(
            _live_selector_matches_unchecked(_normalize_live_selector(child), site)
            for child in selector.selectors
        )
    if kind == "or" and isinstance(selector, CompositeSelector):
        return any(
            _live_selector_matches_unchecked(_normalize_live_selector(child), site)
            for child in selector.selectors
        )
    if kind == "not" and isinstance(selector, NotSelector):
        return not _live_selector_matches_unchecked(
            _normalize_live_selector(selector.selector), site
        )
    if kind == "label":
        return _live_label_matches(site, str(value))
    if kind == "func":
        if isinstance(value, dict):
            return bool(getattr(site, "func_name", None) == value.get("name")) and (
                _live_output_matches(site, value.get("output"))
            )
        return bool(getattr(site, "func_name", None) == value)
    if kind == "module":
        return _live_module_matches(site, str(value))
    if kind == "output":
        return _live_output_matches(site, value)
    if kind == "contains":
        return str(value).lower() in str(getattr(site, "_layer_label_raw", "")).lower()
    if kind == "in_module":
        return _live_module_matches(site, str(value))
    if kind == "predicate":
        predicate = value[0] if isinstance(value, tuple) and callable(value[0]) else value
        if not callable(predicate):
            raise SiteResolutionError("tl.where(...) requires a callable predicate.")
        try:
            return bool(predicate(site))
        except Exception as exc:
            raise SiteResolutionError(
                f"live predicate selector failed at {getattr(site, '_layer_label_raw', '<unknown>')}"
            ) from exc
    raise SiteResolutionError(f"Unsupported live hook selector kind {kind!r}.")


def _live_label_matches(site: Any, label_value: str) -> bool:
    """Return whether a label selector matches capture-time labels.

    Parameters
    ----------
    site:
        Capture-time site proxy.
    label_value:
        Requested label literal.

    Returns
    -------
    bool
        Whether the literal matches a raw label available during capture.
    """

    candidates = (
        getattr(site, "_layer_label_raw", None),
        getattr(site, "_label_raw", None),
        getattr(site, "layer_label", None),
    )
    return label_value in candidates


def _live_module_matches(site: Any, address: str) -> bool:
    """Return whether a live site belongs to a module address.

    Parameters
    ----------
    site:
        Capture-time site proxy.
    address:
        Module address or pass label.

    Returns
    -------
    bool
        Whether the site is currently inside or exiting the module.
    """

    candidates = tuple(getattr(site, "output_of_module_calls", ()) or ()) + tuple(
        getattr(site, "modules", ()) or ()
    )
    return any(_module_label_matches(str(candidate), address) for candidate in candidates)


def _live_output_matches(site: Any, value: Any) -> bool:
    """Return whether a live site matches an output index or role.

    Parameters
    ----------
    site:
        Capture-time site proxy.
    value:
        Output index or semantic role.

    Returns
    -------
    bool
        Whether the site matches the requested output.
    """

    if isinstance(value, int):
        return getattr(site, "multi_output_index", None) == value
    return getattr(site, "multi_output_name", None) == str(value)


def _module_label_matches(module_pass: str, address: str) -> bool:
    """Return whether a module-pass label matches an address.

    Parameters
    ----------
    module_pass:
        Module pass label or ``(address, call_index)`` tuple string.
    address:
        Requested module address.

    Returns
    -------
    bool
        Whether the labels refer to the same module.
    """

    if module_pass.startswith("("):
        return address in module_pass
    address = module_pass.rsplit(":", 1)[0]
    return module_pass == address or address == address


def _looks_like_finalized_label(label_value: str) -> bool:
    """Return whether a label literal looks postprocessed.

    Parameters
    ----------
    label_value:
        Label literal from ``tl.label``.

    Returns
    -------
    bool
        True for pass suffixes like ``:2`` or ``relu_4_27``-style names.
    """

    return bool(_FINAL_LABEL_PATTERN.search(label_value)) and not label_value.endswith("_raw")


def _live_label_error_message(label_value: str) -> str:
    """Build a copy-pasteable finalized-label diagnostic.

    Parameters
    ----------
    label_value:
        Finalized-looking label.

    Returns
    -------
    str
        User-facing error message.
    """

    return (
        f"tl.label({label_value!r}) looks like a finalized postprocess label, but live hooks "
        "run during capture before those labels exist. For post-capture selection use "
        f'tl.where(lambda p: p.layer_label == "{label_value}"). For live hooks, prefer '
        'a capture-time selector such as tl.func("relu") or tl.module("encoder.layer.4").'
    )


def make_live_site_proxy(
    *,
    _layer_label_raw: str,
    func_name: str,
    layer_type: str,
    tensor: torch.Tensor,
    func_call_id: int,
    container_path: tuple[Any, ...],
    fields: Mapping[str, Any],
) -> Any:
    """Build a minimal Op-like object for live hook matching.

    Parameters
    ----------
    _layer_label_raw:
        Predicted raw label for the output tensor.
    func_name:
        Decorated function name.
    layer_type:
        Normalized layer type.
    tensor:
        Output tensor at the hook site.
    func_call_id:
        Function-call id allocated by the wrapper.
    container_path:
        Stable output path for multi-output containers.
    fields:
        Shared capture metadata from output logging.

    Returns
    -------
    Any
        Simple namespace with capture-time fields used by selectors and hooks.
    """

    return SimpleNamespace(
        layer_label=_layer_label_raw,
        _layer_label_raw=_layer_label_raw,
        _label_raw=_layer_label_raw,
        raw_index=fields.get("raw_index"),
        layer_type=layer_type,
        func_name=func_name,
        func_call_id=func_call_id,
        container_path=container_path,
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        tensor_device=tensor.device,
        memory=tensor.nelement() * tensor.element_size(),
        module=fields.get("module"),
        modules=fields.get("modules", []),
        output_of_module_calls=fields.get("output_of_module_calls", []),
        output_of_modules=fields.get("output_of_modules", []),
        call_index=1,
        lookup_keys=[],
    )


def _dispatch_hook_pairs(
    hooks_or_site: Any,
    hook: HookInput | None,
    *,
    default_site_target: Any | None,
) -> list[tuple[Any, HookInput]]:
    """Parse supported hook input shapes into ``(site, hook)`` pairs.

    Parameters
    ----------
    hooks_or_site:
        Attach-hooks input object.
    hook:
        Optional hook paired with a site.
    default_site_target:
        Site target used for bare callable/helper input.

    Returns
    -------
    list[tuple[Any, HookInput]]
        Parsed site/hook pairs.
    """

    if hook is not None:
        if hasattr(hooks_or_site, "to_hook_pairs"):
            return list(hooks_or_site.to_hook_pairs(hook))
        return [(hooks_or_site, hook)]

    observer_site = getattr(hooks_or_site, "site", None)
    if observer_site is not None and _is_hook_like(hooks_or_site):
        return [(observer_site, hooks_or_site)]

    if _is_hook_like(hooks_or_site):
        if default_site_target is None:
            raise HookSiteCoverageError(
                "bare hook input has no site target; pass a site/hook pair or default_site_target"
            )
        return [(default_site_target, hooks_or_site)]

    if isinstance(hooks_or_site, Mapping):
        return [(site, hook_like) for site, hook_like in hooks_or_site.items()]

    if _is_single_pair(hooks_or_site):
        site_target, hook_like = hooks_or_site
        return [(site_target, hook_like)]

    if isinstance(hooks_or_site, Sequence) and not isinstance(hooks_or_site, (str, bytes)):
        pairs = []
        for item in hooks_or_site:
            if not _is_single_pair(item):
                raise HookSignatureError(
                    "hook sequence entries must be explicit (site, callable_or_helper) pairs"
                )
            site_target, hook_like = item
            pairs.append((site_target, hook_like))
        return pairs

    raise HookSignatureError(f"unsupported hook input shape: {type(hooks_or_site).__name__}")


def _is_hook_like(value: Any) -> bool:
    """Return whether an object can be normalized as a hook.

    Parameters
    ----------
    value:
        Object to test.

    Returns
    -------
    bool
        Whether the object is callable or a helper spec.
    """

    return isinstance(value, HelperSpec) or callable(value)


def _is_single_pair(value: Any) -> bool:
    """Return whether ``value`` is an unambiguous ``(site, hook)`` pair.

    Parameters
    ----------
    value:
        Object to test.

    Returns
    -------
    bool
        Whether the object is a two-item tuple ending in a hook-like object.
    """

    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_site_like(value[0])
        and _is_hook_like(value[1])
    )


def _is_site_like(value: Any) -> bool:
    """Return whether ``value`` can name a future hook site.

    Parameters
    ----------
    value:
        Object to test.

    Returns
    -------
    bool
        Whether the value is accepted as a site target by the contract layer.
    """

    return isinstance(value, (BaseSelector, TargetSpec, str)) or hasattr(value, "layer_label")


def _validate_hook_signature(fn: Callable[..., Any], *, direction: HookDirection) -> None:
    """Validate the MVP hook callable signature.

    Parameters
    ----------
    fn:
        Hook callable to inspect.
    direction:
        Hook direction for diagnostics.

    Raises
    ------
    HookSignatureError
        If the callable cannot be called as ``fn(out, *, hook=ctx)``.
        Backward hooks receive a grad as the first positional argument; the
        parameter name is conventional only, so ``g`` or any other name is
        accepted.
    """

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise HookSignatureError(f"could not inspect hook signature for {fn!r}") from exc

    parameters = tuple(signature.parameters.values())
    has_positional = any(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    has_var_positional = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    hook_param = signature.parameters.get("hook")
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if not has_positional and not has_var_positional:
        noun = "grad" if direction == "backward" else "out"
        raise HookSignatureError(f"hook must accept a first positional {noun} argument")
    if direction == "backward" and (
        {"grad_output", "grad_fn_handle", "call_index", "run_ctx"} & set(signature.parameters)
    ):
        return
    if hook_param is None and not has_var_keyword:
        raise HookSignatureError("hook must accept required keyword-only argument 'hook'")
    if hook_param is not None and hook_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise HookSignatureError("'hook' must be a keyword-only parameter")


def _snapshot_layer_log(layer_log: Any | None) -> dict[str, Any]:
    """Snapshot selected Op metadata.

    Parameters
    ----------
    layer_log:
        Op-like object or mapping.

    Returns
    -------
    dict[str, Any]
        Plain dictionary suitable for ``MappingProxyType``.
    """

    snapshot: dict[str, Any] = {}
    if layer_log is None:
        return {field_name: None for field_name in _LAYER_LOG_CONTEXT_FIELDS}
    for field_name in _LAYER_LOG_CONTEXT_FIELDS:
        if isinstance(layer_log, Mapping):
            value = layer_log.get(field_name)
        else:
            value = getattr(layer_log, field_name, None)
        snapshot[field_name] = value
    return snapshot


__all__ = [
    "HookCallable",
    "HookContext",
    "HookDirection",
    "HookTiming",
    "NormalizedHookEntry",
    "expand_facet_hook_entries",
    "is_facet_target",
    "make_hook_context",
    "make_live_site_proxy",
    "live_selector_matches_site",
    "live_backward_selector_matches",
    "normalize_hook",
    "normalize_hook_plan",
]
