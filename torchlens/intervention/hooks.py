"""Hook parsing and hook-spec ownership for TorchLens interventions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import torch

from .errors import HookSignatureError, HookSiteCoverageError
from .selectors import BaseSelector, SelectorLike
from .types import HelperSpec, TargetSpec

HookTiming: TypeAlias = Literal["pre", "post"]
HookDirection: TypeAlias = Literal["forward", "backward"]
HookCallable: TypeAlias = Callable[..., torch.Tensor]
HookInput: TypeAlias = Callable[..., Any] | HelperSpec

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
        return [(hooks_or_site, hook)]

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
    "normalize_hook",
    "normalize_hook_plan",
]
