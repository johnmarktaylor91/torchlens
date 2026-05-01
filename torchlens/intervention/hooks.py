"""Hook parsing and hook-spec ownership for TorchLens interventions."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import torch

from .errors import (
    HookSignatureError,
    HookSiteCoverageError,
    LiveModeLabelError,
    SiteResolutionError,
)
from .selectors import BaseSelector, CompositeSelector, NotSelector, SelectorLike, in_module
from .types import HelperSpec, HookSpec, InterventionSpec, TargetSpec, TargetValueSpec

HookTiming: TypeAlias = Literal["pre", "post"]
HookDirection: TypeAlias = Literal["forward", "backward"]
HookCallable: TypeAlias = Callable[..., torch.Tensor]
HookInput: TypeAlias = Callable[..., Any] | HelperSpec
_FINAL_LABEL_PATTERN = re.compile(r"(?:_\d+_\d+(?::\d+)?$|:\d+$)")

_LAYER_LOG_CONTEXT_FIELDS = (
    "layer_label",
    "layer_type",
    "tensor_shape",
    "tensor_dtype",
    "tensor_device",
    "module_address",
    "pass_num",
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
        ``"forward"`` for activations or ``"backward"`` for gradients.
    layer_log:
        Mapping proxy over selected layer metadata, never a live LayerPassLog.
    ctx:
        Per resolved ``(site, hook instance)`` scratch dictionary.
    run_ctx:
        Mutable dictionary shared across all hooks in a run.
    trace_index:
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
    trace_index: int
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
    trace_index: int = 0,
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
        Optional LayerPassLog-like object or mapping to snapshot.
    ctx:
        Per-hook scratch dictionary. A new dictionary is created when omitted.
    run_ctx:
        Shared run dictionary. A new dictionary is created when omitted.
    trace_index:
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
        trace_index=trace_index,
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
        direction: HookDirection = helper_spec.kind if helper_spec is not None else "forward"
        normalized_callable = normalize_hook(hook_like, direction=direction)
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


def normalize_hooks_from_spec(spec: InterventionSpec | None) -> list[NormalizedHookEntry]:
    """Normalize an intervention spec into live-capture hook-plan entries.

    Parameters
    ----------
    spec:
        Mutable intervention spec attached to a ``ModelLog``. ``None`` or an
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


def _normalize_value_specs(value_specs: Sequence[TargetValueSpec]) -> list[NormalizedHookEntry]:
    """Normalize set-replacement specs into hook-plan entries.

    Parameters
    ----------
    value_specs:
        Mutable set-replacement specs from an intervention recipe.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook-plan entries that replace matching activations.
    """

    entries: list[NormalizedHookEntry] = []
    for order, value_spec in enumerate(value_specs):
        value = value_spec.value

        def _set_value_hook(
            activation: torch.Tensor,
            *,
            hook: HookContext,
            replacement: Any = value,
        ) -> torch.Tensor:
            """Return a static or one-shot callable replacement.

            Parameters
            ----------
            activation:
                Activation at the matched site.
            hook:
                Hook context supplied by TorchLens.
            replacement:
                Replacement tensor or callable captured by default argument.

            Returns
            -------
            torch.Tensor
                Replacement activation.
            """

            del hook
            if callable(replacement):
                return cast(torch.Tensor, replacement(activation))
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

        def _value_hook(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Return a fixed replacement value from an intervention spec.

            Parameters
            ----------
            activation:
                Activation being replaced.
            hook:
                Hook context supplied by TorchLens.

            Returns
            -------
            torch.Tensor
                Replacement tensor.
            """

            del activation, hook
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

    from .selectors import contains, func, label, module, where

    if target.selector_kind == "label":
        return label(str(target.selector_value))
    if target.selector_kind == "func":
        return func(str(target.selector_value))
    if target.selector_kind == "module":
        return module(str(target.selector_value))
    if target.selector_kind == "contains":
        return contains(str(target.selector_value))
    if target.selector_kind == "in_module":
        from .selectors import in_module as make_in_module

        selector = make_in_module(str(target.selector_value))
        if isinstance(selector, BaseSelector):
            return selector
    if target.selector_kind == "predicate" and callable(target.selector_value):
        return where(target.selector_value, name_hint=target.metadata.get("name_hint"))
    if target.selector_kind == "not":
        return ~_normalize_live_selector(target.selector_value)
    raise SiteResolutionError(f"Unsupported live hook selector kind {target.selector_kind!r}.")


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
        return getattr(site, "func_name", None) == value
    if kind == "module":
        return _live_module_matches(site, str(value))
    if kind == "contains":
        return str(value).lower() in str(getattr(site, "layer_label_raw", "")).lower()
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
                f"live predicate selector failed at {getattr(site, 'layer_label_raw', '<unknown>')}"
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
        getattr(site, "layer_label_raw", None),
        getattr(site, "tensor_label_raw", None),
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

    candidates = tuple(getattr(site, "module_passes_exited", ()) or ()) + tuple(
        getattr(site, "containing_modules", ()) or ()
    )
    return any(_module_label_matches(str(candidate), address) for candidate in candidates)


def _module_label_matches(module_pass: str, address: str) -> bool:
    """Return whether a module-pass label matches an address.

    Parameters
    ----------
    module_pass:
        Module pass label or ``(address, pass_num)`` tuple string.
    address:
        Requested module address.

    Returns
    -------
    bool
        Whether the labels refer to the same module.
    """

    if module_pass.startswith("("):
        return address in module_pass
    module_address = module_pass.rsplit(":", 1)[0]
    return module_pass == address or module_address == address


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
    layer_label_raw: str,
    func_name: str,
    layer_type: str,
    tensor: torch.Tensor,
    func_call_id: int,
    output_path: tuple[Any, ...],
    fields: Mapping[str, Any],
) -> Any:
    """Build a minimal LayerPassLog-like object for live hook matching.

    Parameters
    ----------
    layer_label_raw:
        Predicted raw label for the output tensor.
    func_name:
        Decorated function name.
    layer_type:
        Normalized layer type.
    tensor:
        Output tensor at the hook site.
    func_call_id:
        Function-call id allocated by the wrapper.
    output_path:
        Stable output path for multi-output containers.
    fields:
        Shared capture metadata from output logging.

    Returns
    -------
    Any
        Simple namespace with capture-time fields used by selectors and hooks.
    """

    return SimpleNamespace(
        layer_label=layer_label_raw,
        layer_label_raw=layer_label_raw,
        tensor_label_raw=layer_label_raw,
        layer_type=layer_type,
        func_name=func_name,
        func_call_id=func_call_id,
        output_path=output_path,
        tensor_shape=tuple(tensor.shape),
        tensor_dtype=tensor.dtype,
        tensor_device=tensor.device,
        tensor_memory=tensor.nelement() * tensor.element_size(),
        containing_module=fields.get("containing_module"),
        containing_modules=fields.get("containing_modules", []),
        module_passes_exited=fields.get("module_passes_exited", []),
        modules_exited=fields.get("modules_exited", []),
        pass_num=1,
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
        If the callable cannot be called as ``fn(activation, *, hook=ctx)``.
        Backward hooks receive a gradient as the first positional argument; the
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
        noun = "gradient" if direction == "backward" else "activation"
        raise HookSignatureError(f"hook must accept a first positional {noun} argument")
    if hook_param is None and not has_var_keyword:
        raise HookSignatureError("hook must accept required keyword-only argument 'hook'")
    if hook_param is not None and hook_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise HookSignatureError("'hook' must be a keyword-only parameter")


def _snapshot_layer_log(layer_log: Any | None) -> dict[str, Any]:
    """Snapshot selected LayerPassLog metadata.

    Parameters
    ----------
    layer_log:
        LayerPassLog-like object or mapping.

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
    "make_hook_context",
    "make_live_site_proxy",
    "live_selector_matches_site",
    "normalize_hook",
    "normalize_hook_plan",
]
