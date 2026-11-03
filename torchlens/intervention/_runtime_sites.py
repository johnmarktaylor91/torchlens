"""Site-scope and provisional-site helpers for live intervention hooks."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hooks import HookContext, NormalizedHookEntry


def _hook_call_inputs_for_site(
    entry: NormalizedHookEntry,
    *,
    site: Any,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any] | None,
    call_input_snapshots: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the positional and keyword inputs for one hook fire.

    Parameters
    ----------
    entry:
        Normalized hook entry being executed.
    site:
        Live site proxy.
    call_args:
        Captured positional inputs.
    call_kwargs:
        Captured keyword inputs.
    call_input_snapshots:
        Optional pre-execution input snapshots captured for in-place ops.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Hook positional and keyword inputs.
    """

    if not _helper_routes_inputs(entry.helper_spec):
        return call_args, dict(call_kwargs or {})
    if not getattr(site, "_tl_module_boundary", False):
        if bool(getattr(site, "_tl_input_snapshot", False)) and call_input_snapshots is not None:
            return call_input_snapshots
        if bool(getattr(site, "is_inplace", False)):
            warnings.warn(
                "An input-routed intervention fired on an in-place operation without a "
                "pre-mutation input snapshot; the intervention may be seeing post-mutation "
                "inputs. This can occur with out= aliasing or a conservative pre-gate miss.",
                UserWarning,
                stacklevel=3,
            )
        return call_args, dict(call_kwargs or {})
    return call_args, dict(call_kwargs or {})


def _selector_uses_only_provisional_fields(selector: Any) -> bool:
    """Return whether a selector can be faithfully evaluated before execution.

    Parameters
    ----------
    selector:
        Selector-like target to inspect.

    Returns
    -------
    bool
        Whether a false pre-execution match result is definitive.
    """

    try:
        normalized = getattr(selector, "selector_kind", None)
        # A provisional op explicitly carries ``_tl_module_boundary=False`` and
        # ``output_of_module_calls=()``. Therefore ``tl.module`` is definitively
        # false here; its eventual true match is evaluated by the module-exit hook.
        if normalized in {"func", "module", "in_module"}:
            return True
        if normalized in {"and", "or"}:
            return all(
                _selector_uses_only_provisional_fields(child) for child in selector.selectors
            )
        if normalized == "not":
            return _selector_uses_only_provisional_fields(selector.selector)
    except Exception:
        return False
    return False


def _helper_routes_inputs(helper: Any | None) -> bool:
    """Return whether a helper consumes call inputs from ``ctx.args``.

    Parameters
    ----------
    helper:
        Helper spec-like object.

    Returns
    -------
    bool
        Whether the helper is ``splice_module(input_route='in')``.
    """

    if helper is None or getattr(helper, "name", None) != "splice_module":
        return False
    metadata = dict(getattr(helper, "metadata", ()))
    return metadata.get("input_route") == "in"


def _input_splice_module_scope(site_target: Any, entry: NormalizedHookEntry) -> str | None:
    """Return the module address for input-spliced module-scoped selectors.

    Parameters
    ----------
    site_target:
        Selector-like hook target.
    entry:
        Normalized hook entry.

    Returns
    -------
    str | None
        Module address when the entry is an input-spliced module scope.
    """

    helper = entry.helper_spec
    if helper is None or helper.name != "splice_module":
        return None
    helper_metadata = dict(helper.metadata)
    if helper_metadata.get("input_route") != "in":
        return None
    return _module_scope_address(site_target)


def _module_scope_address(site_target: Any) -> str | None:
    """Return a module address when a selector tree contains module scope.

    Parameters
    ----------
    site_target:
        Selector-like hook target.

    Returns
    -------
    str | None
        First module address found in the selector tree.
    """

    kind = getattr(site_target, "selector_kind", None)
    if kind in {"module", "in_module"}:
        value = getattr(site_target, "selector_value", None)
        return None if value is None else str(value)
    selectors = getattr(site_target, "selectors", None)
    if selectors is not None:
        for selector in selectors:
            address = _module_scope_address(selector)
            if address is not None:
                return address
    selector = getattr(site_target, "selector", None)
    if selector is not None:
        return _module_scope_address(selector)
    return None


def _whole_call_site_view(site: Any) -> Any:
    """Return ``site`` as a whole-module-call site for input-splice matching.

    ``splice_module(input_route="in")`` replaces a module call as a whole, so its
    ``tl.in_module`` target fires at the boundary even when an op inside the
    call produced the output; drop the boundary's inner-alias mark (which
    otherwise keeps ``tl.in_module`` from re-firing there) for that match.

    Parameters
    ----------
    site:
        Live site proxy.

    Returns
    -------
    Any
        ``site`` itself, or a shallow copy without the inner-alias mark.
    """

    if not getattr(site, "_tl_boundary_inner_alias", False):
        return site
    return SimpleNamespace(**{**vars(site), "_tl_boundary_inner_alias": False})


def _is_plain_module_selector(site_target: Any) -> bool:
    """Return whether a selector is exactly a module boundary or containment selector.

    Parameters
    ----------
    site_target:
        Selector-like hook target.

    Returns
    -------
    bool
        Whether the selector is exactly ``tl.module`` or ``tl.in_module``.
    """

    return getattr(site_target, "selector_kind", None) in {"module", "in_module"}


def _forward_site_label(op: Any) -> str:
    """Return the forward label a gradient fire records (``relu_3``, ``relu_3:2``).

    The bare layer label names a single-pass layer; a multi-pass layer keeps
    the pass-qualified label so each pass's fire names its own pass.

    Parameters
    ----------
    op:
        Final forward op record.

    Returns
    -------
    str
        Layer label, pass-qualified only on a multi-pass layer.
    """

    layer_label = str(getattr(op, "layer_label", "") or "")
    if layer_label and int(getattr(op, "num_passes", 1) or 1) <= 1:
        return layer_label
    return str(getattr(op, "label", "") or layer_label)


def _op_output_slot(op: Any) -> int | None:
    """Return the output slot of a multi-output op's record, else ``None``.

    Parameters
    ----------
    op:
        Final forward op record.

    Returns
    -------
    int | None
        ``multi_output_index`` of the record when it is one output of several.
    """

    slot = getattr(op, "multi_output_index", None)
    return slot if isinstance(slot, int) else None


def _site_name(hook_context: HookContext | None) -> str:
    """Return a readable site name for hook diagnostics.

    Parameters
    ----------
    hook_context:
        Optional hook context.

    Returns
    -------
    str
        Layer label or ``"<unknown site>"``.
    """

    if hook_context is None:
        return "<unknown site>"
    layer_label = hook_context.op.get("layer_label")
    if layer_label is None:
        return "<unknown site>"
    return str(layer_label)
