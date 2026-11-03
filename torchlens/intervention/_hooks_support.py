"""Leaf helpers for hook normalization (including the hook signature check), live site proxies and facet bookkeeping."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from .._state import pause_logging
from .errors import HookSignatureError, ReplayPreconditionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .hooks import HookCallable, HookDirection
    from .types import Edit, HookSpec

#: Selector kind of the ``tl.grad(...)`` wrapper (SPEC section 0, THE_WRAPPER).
_GRAD_SELECTOR_KIND = "grad"


_LAYER_LOG_CONTEXT_FIELDS = (
    "layer_label",
    # Pass-qualified spelling + pass ordinal: on a multi-pass
    # (recurrence-grouped) layer the bare layer_label alone cannot tell a
    # hook WHICH pass it is firing at.
    "label",
    "pass_number",
    # L1 structural site key (where minted): the derived-seed law's logical
    # firing coordinate prefers it over the display label (F02, edits memo D6).
    "site_key",
    "layer_type",
    "shape",
    "dtype",
    "tensor_device",
    "address",
    "call_index",
)


def gradient_inner(site_target: Any) -> Any | None:
    """Return the forward site inside a ``tl.grad(...)`` address, else ``None``.

    Duck-typed on the selector kind so a saved ``TargetSpec`` of kind
    ``"grad"`` (whose value is the inner frozen target) reads the same way as
    a live ``GradSelector``.

    Parameters
    ----------
    site_target:
        Selector-like target.

    Returns
    -------
    Any | None
        The inner forward selector, or ``None`` when ``site_target`` is not a
        gradient address.
    """

    if getattr(site_target, "selector_kind", None) != _GRAD_SELECTOR_KIND:
        return None
    return getattr(site_target, "selector_value", None)


def _needs_tensor_backward_adapter(
    hook_callable: HookCallable,
    helper_spec: Edit | None,
) -> bool:
    """Return whether a backward hook expects one tensor instead of grad tuples.

    Parameters
    ----------
    hook_callable:
        Normalized callable being mounted on the backward signal.
    helper_spec:
        Helper spec when the hook came from a built-in helper.

    Returns
    -------
    bool
        Whether the callable should be adapted to a ``GradFn`` tuple callback.
    """

    if helper_spec is not None and dict(helper_spec.metadata).get("mount_shape") == "tuple":
        return False
    try:
        signature = inspect.signature(hook_callable)
    except (TypeError, ValueError):
        return True
    tuple_callback_params = {"grad_output", "grad_fn_handle", "call_index", "run_ctx"}
    return not bool(tuple_callback_params & set(signature.parameters))


def _display_label(log: Any, home_label: str) -> str:
    """Return the final layer label behind a raw home-op hook label, for messages."""

    try:
        layer = log._layer_logs.get(home_label)
        return str((layer if layer is not None else log[home_label]).layer_label)
    except (AttributeError, KeyError, TypeError, ValueError):
        return home_label


def _selector_call_index(selector_like: Any) -> int | None:
    """Return an explicit one-based call-index selector, if supplied.

    Parameters
    ----------
    selector_like:
        Selector or target spec.

    Returns
    -------
    int | None
        Expected callback index, or ``None`` when unconstrained.
    """

    metadata = getattr(selector_like, "metadata", None)
    if isinstance(metadata, Mapping) and isinstance(metadata.get("call_index"), int):
        return int(metadata["call_index"])
    value = getattr(selector_like, "selector_value", None)
    if isinstance(value, Mapping) and isinstance(value.get("call_index"), int):
        return int(value["call_index"])
    return None


def _targets_accumulate_grad(site_target: Any) -> bool:
    """Return whether a node selector target names ``AccumulateGrad``.

    Parameters
    ----------
    site_target:
        Selector-like target.

    Returns
    -------
    bool
        Whether the target is an AccumulateGrad selector.
    """

    value = getattr(site_target, "selector_value", None)
    if isinstance(value, Mapping):
        type_value = value.get("type")
        return type_value is not None and "accumulategrad" in str(type_value).lower()
    return "accumulategrad" in repr(site_target).lower()


def _multi_call_addresses(record: Any) -> tuple[str, ...] | None:
    """Return the call-qualified addresses of a module record that ran more than once.

    ``None`` for any other record. The test mirrors the module record's own
    single-call guard, so a record whose ``facets`` would raise
    ``module_call_ambiguous`` is never asked for its view.
    """

    num_calls = getattr(record, "num_calls", None)
    if num_calls is None or hasattr(record, "call_label") or not hasattr(record, "calls"):
        return None
    if num_calls == 1 and len(getattr(record, "ops", ())) == 1:
        return None
    address = getattr(record, "address", "")
    return tuple(f"{address}:{index}" for index in range(1, max(int(num_calls), 1) + 1))


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

    # TorchLens-internal size read: paused so it is never mistaken for a user call.
    with pause_logging():
        memory = tensor.nelement() * tensor.element_size()
    return SimpleNamespace(
        label=_layer_label_raw,
        raw_label=_layer_label_raw,
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
        memory=memory,
        module=fields.get("module"),
        modules=fields.get("modules", []),
        output_of_module_calls=fields.get("output_of_module_calls", []),
        output_of_modules=fields.get("output_of_modules", []),
        _tl_module_boundary=bool(fields.get("_tl_module_boundary", False)),
        _tl_input_snapshot=bool(fields.get("_tl_input_snapshot", False)),
        is_inplace=bool(fields.get("is_inplace", False)),
        is_transform=bool(fields.get("is_transform", False)),
        transform_kind=fields.get("transform_kind"),
        transform_chain=tuple(fields.get("transform_chain", ())),
        transform_config=dict(fields.get("transform_config", {})),
        call_index=1,
        lookup_keys=[],
    )


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
        return dict.fromkeys(_LAYER_LOG_CONTEXT_FIELDS)
    for field_name in _LAYER_LOG_CONTEXT_FIELDS:
        if isinstance(layer_log, Mapping):
            value = layer_log.get(field_name)
        else:
            value = getattr(layer_log, field_name, None)
        snapshot[field_name] = value
    return snapshot


def _validate_hook_signature(fn: Callable[..., Any], *, signal: HookDirection) -> None:
    """Validate the MVP hook callable signature.

    Parameters
    ----------
    fn:
        Hook callable to inspect.
    signal:
        Pass of the site branch, for diagnostics and the backward tuple protocol.

    Raises
    ------
    HookSignatureError
        If the callable cannot be called as ``fn(out, *, ctx=context)``.
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
    ctx_param = signature.parameters.get("ctx")
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if not has_positional and not has_var_positional:
        noun = "grad" if signal == "backward" else "out"
        raise HookSignatureError(f"hook must accept a first positional {noun} argument")
    if signal == "backward" and (
        {"grad_output", "grad_fn_handle", "call_index", "run_ctx"} & set(signature.parameters)
    ):
        return
    if ctx_param is None and not has_var_keyword:
        raise HookSignatureError(
            "hook must accept required keyword-only argument 'ctx'; write it as "
            "def edit(t, *, ctx): return t * 0 (ctx may go unused)"
        )
    if ctx_param is not None and ctx_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise HookSignatureError(
            "'ctx' must be a keyword-only parameter; write it as def edit(t, *, ctx): ..."
        )


def _stored_facet_scatter_hook(hook_spec: HookSpec) -> Any:
    """Return a sticky facet-slice hook's stored scatter wrapper, or refuse.

    Facet-slice hooks fire through the scatter wrapper built at attach time
    (``_facet_scatter_hook``), which closes over the resolved FacetSpec and
    CANNOT be rebuilt from the raw helper: the generic normalization path
    would silently apply the helper to the WHOLE home tensor
    (``tl.head(0, "q")`` plus ``tl.zero_ablate()`` zeroing every head). Fail
    closed when the wrapper is absent (for example a spec loaded from disk).
    """

    stored_hook = hook_spec.hook
    if not getattr(stored_hook, "_tl_facet_scatter", False):
        facet_name = hook_spec.metadata.get("facet_name", "<unknown>")
        home_label = hook_spec.metadata.get("facet_home_label", "<unknown>")
        raise ReplayPreconditionError(
            "Sticky facet-slice hook lost its scatter wrapper: the stored hook "
            f"for facet {facet_name!r} on home {home_label!r} is not the live "
            "facet scatter wrapper, so firing it would write the whole home "
            "tensor instead of the selected facet slice. Facet hooks cannot be "
            "rebuilt from a serialized helper; re-attach the facet intervention "
            "on a live trace (e.g. trace.do(tl.head(...), helper, model=model, inputs=x))."
        )
    return stored_hook
