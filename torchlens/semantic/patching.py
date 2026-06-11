"""Prebuilt activation and attribution patching helpers for facets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from ..intervention.selectors import facet
from ..user_funcs import trace
from .facets import Facet, MissingGradient

Metric = Callable[[Any], torch.Tensor]


def activation_patch_residual_stream(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    metric: Metric,
    *,
    facet_name: str = "resid_pre",
    position_axis: int = 1,
    positions: Sequence[int] | None = None,
    patch_positions: bool = True,
    trace_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Patch clean residual-stream activations into a corrupted run.

    Parameters
    ----------
    model:
        Model to trace and rerun.
    clean_input:
        Input for the clean baseline run.
    corrupted_input:
        Input for the corrupted baseline and patched runs.
    metric:
        Callable receiving a ``Trace`` and returning a scalar tensor metric.
    facet_name:
        Residual facet to patch, usually ``"resid_pre"``, ``"resid_mid"``, or
        ``"resid_post"``.
    position_axis:
        Axis containing sequence positions in the residual tensor.
    positions:
        Explicit positions to patch. When omitted, all positions along
        ``position_axis`` are patched.
    patch_positions:
        If true, return one metric per ``[layer, pos]`` patch. If false, patch
        each full residual tensor and return ``[layer]``.
    trace_kwargs:
        Extra keyword arguments forwarded to ``tl.trace``.

    Returns
    -------
    torch.Tensor
        Metric values shaped ``[layer, pos]`` or ``[layer]``.
    """

    clean_log, corrupted_log = _baseline_traces(
        model, clean_input, corrupted_input, trace_kwargs=trace_kwargs
    )
    metric_template = _baseline_metric_template(clean_log, corrupted_log, metric)
    modules = _modules_with_facet(clean_log, facet_name)
    _ensure_matching_modules(corrupted_log, modules, facet_name=facet_name)
    if not patch_positions:
        return _activation_patch_by_module(
            model,
            corrupted_input,
            corrupted_log,
            clean_log,
            modules,
            facet_name=facet_name,
            metric=metric,
            metric_template=metric_template,
        )

    if not modules:
        raise ValueError(f"No modules expose facet {facet_name!r}.")
    first_value = _facet_tensor(clean_log.modules[modules[0]].facets[facet_name])
    normalized_axis = position_axis % first_value.ndim
    patch_positions_list = (
        list(range(first_value.shape[normalized_axis])) if positions is None else list(positions)
    )
    result = torch.empty(
        (len(modules), len(patch_positions_list)),
        dtype=metric_template.dtype,
        device=metric_template.device,
    )
    for layer_index, address in enumerate(modules):
        clean_value = _facet_tensor(clean_log.modules[address].facets[facet_name]).detach().clone()
        selector = facet(facet_name).in_module(address)
        for pos_index, position in enumerate(patch_positions_list):

            def _patch_position(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return ``out`` with one position replaced by the clean activation."""

                del hook
                patched = out.clone(memory_format=torch.preserve_format)
                target = patched.select(normalized_axis, position)
                source = clean_value.select(normalized_axis, position)
                target.copy_(source)
                return patched

            patched_log = _run_patch(
                model,
                corrupted_input,
                corrupted_log,
                selector,
                _patch_position,
                name=f"patch_{facet_name}_{layer_index}_{position}",
            )
            result[layer_index, pos_index] = _metric_scalar(metric(patched_log), like=result)
    return result


def activation_patch_attention_output(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    metric: Metric,
    *,
    facet_name: str = "attn_out",
    trace_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Patch each clean attention output into the corrupted run.

    Parameters
    ----------
    model:
        Model to trace and rerun.
    clean_input:
        Input for the clean baseline run.
    corrupted_input:
        Input for the corrupted baseline and patched runs.
    metric:
        Callable receiving a ``Trace`` and returning a scalar tensor metric.
    facet_name:
        Attention output facet to patch.
    trace_kwargs:
        Extra keyword arguments forwarded to ``tl.trace``.

    Returns
    -------
    torch.Tensor
        Metric values shaped ``[layer]``.
    """

    clean_log, corrupted_log = _baseline_traces(
        model, clean_input, corrupted_input, trace_kwargs=trace_kwargs
    )
    metric_template = _baseline_metric_template(clean_log, corrupted_log, metric)
    modules = _modules_with_facet(clean_log, facet_name)
    _ensure_matching_modules(corrupted_log, modules, facet_name=facet_name)
    return _activation_patch_by_module(
        model,
        corrupted_input,
        corrupted_log,
        clean_log,
        modules,
        facet_name=facet_name,
        metric=metric,
        metric_template=metric_template,
    )


def activation_patch_attention_heads(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    metric: Metric,
    *,
    facet_name: str = "result",
    trace_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Patch each clean attention head output into the corrupted run.

    Parameters
    ----------
    model:
        Model to trace and rerun.
    clean_input:
        Input for the clean baseline run.
    corrupted_input:
        Input for the corrupted baseline and patched runs.
    metric:
        Callable receiving a ``Trace`` and returning a scalar tensor metric.
    facet_name:
        Per-head attention output facet. The default ``"result"`` follows the
        P3 facet convention ``[batch, pos, head, d_model]``.
    trace_kwargs:
        Extra keyword arguments forwarded to ``tl.trace``.

    Returns
    -------
    torch.Tensor
        Metric values shaped ``[layer, head]``.
    """

    clean_log, corrupted_log = _baseline_traces(
        model, clean_input, corrupted_input, trace_kwargs=trace_kwargs
    )
    metric_template = _baseline_metric_template(clean_log, corrupted_log, metric)
    return _activation_patch_heads(
        model,
        corrupted_input,
        corrupted_log,
        clean_log,
        facet_name=facet_name,
        metric=metric,
        metric_template=metric_template,
    )


def activation_patch_mlp_output(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    metric: Metric,
    *,
    facet_name: str = "output",
    trace_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Patch each clean MLP output into the corrupted run.

    Parameters
    ----------
    model:
        Model to trace and rerun.
    clean_input:
        Input for the clean baseline run.
    corrupted_input:
        Input for the corrupted baseline and patched runs.
    metric:
        Callable receiving a ``Trace`` and returning a scalar tensor metric.
    facet_name:
        MLP output facet to patch.
    trace_kwargs:
        Extra keyword arguments forwarded to ``tl.trace``.

    Returns
    -------
    torch.Tensor
        Metric values shaped ``[layer]``.
    """

    clean_log, corrupted_log = _baseline_traces(
        model, clean_input, corrupted_input, trace_kwargs=trace_kwargs
    )
    metric_template = _baseline_metric_template(clean_log, corrupted_log, metric)
    modules = [
        address
        for address in _modules_with_facet(clean_log, facet_name)
        if _looks_like_mlp_module(clean_log.modules[address])
    ]
    _ensure_matching_modules(corrupted_log, modules, facet_name=facet_name)
    return _activation_patch_by_module(
        model,
        corrupted_input,
        corrupted_log,
        clean_log,
        modules,
        facet_name=facet_name,
        metric=metric,
        metric_template=metric_template,
    )


def attribution_patch_attention_heads(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    metric: Metric,
    *,
    facet_name: str = "result",
    trace_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Approximate per-head activation patching with ``grad * delta``.

    Parameters
    ----------
    model:
        Model to trace.
    clean_input:
        Input for the clean baseline run.
    corrupted_input:
        Input for the corrupted baseline run.
    metric:
        Callable receiving a ``Trace`` and returning a scalar tensor metric.
    facet_name:
        Per-head attention output facet. The default ``"result"`` follows the
        P3 facet convention ``[batch, pos, head, d_model]``.
    trace_kwargs:
        Extra keyword arguments forwarded to ``tl.trace``. By default this
        helper captures all gradients; passing ``save_grads=None`` is a
        useful way to verify the missing-gradient error path.

    Returns
    -------
    torch.Tensor
        Approximate metric effects shaped ``[layer, head]``.
    """

    grad_trace_kwargs = dict(trace_kwargs or {})
    grad_trace_kwargs.setdefault("save_grads", True)
    clean_log, corrupted_log = _baseline_traces(
        model, clean_input, corrupted_input, trace_kwargs=grad_trace_kwargs
    )
    clean_metric = _require_scalar_metric(metric(clean_log))
    corrupted_metric = _require_scalar_metric(metric(corrupted_log))
    clean_log.log_backward(clean_metric)
    corrupted_log.log_backward(corrupted_metric)
    modules = _modules_with_facet(clean_log, facet_name)
    _ensure_matching_modules(corrupted_log, modules, facet_name=facet_name)
    if not modules:
        raise ValueError(f"No modules expose facet {facet_name!r}.")
    n_heads = _num_heads(clean_log.modules[modules[0]], facet_name=facet_name)
    result = torch.empty(
        (len(modules), n_heads), dtype=corrupted_metric.dtype, device=corrupted_metric.device
    )
    for layer_index, address in enumerate(modules):
        for head_index in range(n_heads):
            clean_value = _facet_tensor(
                clean_log.modules[address].facets.head(head_index)[facet_name]
            )
            corrupted_facet = corrupted_log.modules[address].facets.head(head_index)[facet_name]
            corrupted_value = _facet_tensor(corrupted_facet)
            grad = _facet_grad_tensor(corrupted_facet, address=address, facet_name=facet_name)
            result[layer_index, head_index] = (grad * (clean_value - corrupted_value)).sum()
    return result


def _baseline_traces(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    *,
    trace_kwargs: Mapping[str, Any] | None,
) -> tuple[Any, Any]:
    """Return clean and corrupted traces with all layer activations saved."""

    kwargs = _trace_kwargs(trace_kwargs)
    clean_log = trace(model, clean_input, **kwargs)
    corrupted_log = trace(model, corrupted_input, **kwargs)
    return clean_log, corrupted_log


def _trace_kwargs(trace_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return trace keyword arguments with P4-safe defaults."""

    kwargs = dict(trace_kwargs or {})
    kwargs.setdefault("layers_to_save", "all")
    kwargs.setdefault("save_arg_values", True)
    return kwargs


def _activation_patch_by_module(
    model: nn.Module,
    corrupted_input: Any,
    corrupted_log: Any,
    clean_log: Any,
    modules: Sequence[str],
    *,
    facet_name: str,
    metric: Metric,
    metric_template: torch.Tensor,
) -> torch.Tensor:
    """Patch one whole facet per module and return metric values."""

    if not modules:
        raise ValueError(f"No modules expose facet {facet_name!r}.")
    first_value = _facet_tensor(clean_log.modules[modules[0]].facets[facet_name])
    del first_value
    result = torch.empty(
        (len(modules),), dtype=metric_template.dtype, device=metric_template.device
    )
    for layer_index, address in enumerate(modules):
        clean_value = _facet_tensor(clean_log.modules[address].facets[facet_name]).detach().clone()

        def _patch_whole(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            """Return the clean activation for this facet slice."""

            del out, hook
            return clean_value

        patched_log = _run_patch(
            model,
            corrupted_input,
            corrupted_log,
            facet(facet_name).in_module(address),
            _patch_whole,
            name=f"patch_{facet_name}_{layer_index}",
        )
        result[layer_index] = _metric_scalar(metric(patched_log), like=result)
    return result


def _activation_patch_heads(
    model: nn.Module,
    corrupted_input: Any,
    corrupted_log: Any,
    clean_log: Any,
    *,
    facet_name: str,
    metric: Metric,
    metric_template: torch.Tensor,
) -> torch.Tensor:
    """Patch one clean head per attention module and return metric values."""

    modules = _modules_with_facet(clean_log, facet_name)
    _ensure_matching_modules(corrupted_log, modules, facet_name=facet_name)
    if not modules:
        raise ValueError(f"No modules expose facet {facet_name!r}.")
    n_heads = _num_heads(clean_log.modules[modules[0]], facet_name=facet_name)
    result = torch.empty(
        (len(modules), n_heads), dtype=metric_template.dtype, device=metric_template.device
    )
    for layer_index, address in enumerate(modules):
        for head_index in range(n_heads):
            clean_value = (
                _facet_tensor(clean_log.modules[address].facets.head(head_index)[facet_name])
                .detach()
                .clone()
            )

            def _patch_head(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return the clean activation for this head facet slice."""

                del out, hook
                return clean_value

            patched_log = _run_patch(
                model,
                corrupted_input,
                corrupted_log,
                facet(facet_name).head(head_index).in_module(address),
                _patch_head,
                name=f"patch_{facet_name}_{layer_index}_{head_index}",
            )
            result[layer_index, head_index] = _metric_scalar(metric(patched_log), like=result)
    return result


def _run_patch(
    model: nn.Module,
    corrupted_input: Any,
    corrupted_log: Any,
    selector: Any,
    hook: Callable[..., torch.Tensor],
    *,
    name: str,
) -> Any:
    """Fork the corrupted trace, attach one facet hook, and rerun."""

    patched_log = corrupted_log.fork(name)
    patched_log.attach_hooks(selector, hook)
    patched_log.rerun(model, corrupted_input)
    return patched_log


def _modules_with_facet(log: Any, facet_name: str) -> list[str]:
    """Return module addresses exposing a facet."""

    return [
        str(module.address)
        for module in log.modules
        if getattr(module, "address", None) != "self" and module.facets.has(facet_name)
    ]


def _ensure_matching_modules(log: Any, modules: Sequence[str], *, facet_name: str) -> None:
    """Raise if a corrupted trace lacks a clean-trace facet owner."""

    missing = [
        address
        for address in modules
        if address not in log.modules or not log.modules[address].facets.has(facet_name)
    ]
    if missing:
        raise ValueError(
            f"Corrupted trace is missing facet {facet_name!r} on modules {tuple(missing)!r}."
        )


def _baseline_metric_template(clean_log: Any, corrupted_log: Any, metric: Metric) -> torch.Tensor:
    """Run clean and corrupted baseline metrics and return the corrupted scalar."""

    _require_scalar_metric(metric(clean_log))
    return _require_scalar_metric(metric(corrupted_log)).detach()


def _num_heads(module: Any, *, facet_name: str) -> int:
    """Return the number of heads for a per-head facet module."""

    view = module.facets
    n_heads = view.get("n_q_heads", view.get("n_heads", None))
    if isinstance(n_heads, int):
        return n_heads
    value = _facet_tensor(view[facet_name])
    if value.ndim < 3:
        raise ValueError(f"Facet {facet_name!r} on module {module.address!r} has no head axis.")
    return int(value.shape[-2])


def _looks_like_mlp_module(module: Any) -> bool:
    """Return whether a module exposes the built-in MLP facet family."""

    view = module.facets
    return any(view.has(name) for name in ("up_out", "down_out", "gated_out", "intermediate"))


def _facet_tensor(value: Any) -> torch.Tensor:
    """Return a tensor value from a facet-like object."""

    if isinstance(value, Facet):
        value = value.value
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor facet value, got {type(value).__name__}.")
    return value


def _facet_grad_tensor(value: Any, *, address: str, facet_name: str) -> torch.Tensor:
    """Return a tensor gradient from a facet-like object or raise clearly."""

    grad = value.grad if isinstance(value, Facet) else getattr(value, "grad", None)
    if isinstance(grad, MissingGradient):
        raise RuntimeError(
            f"Attribution patching requires grad capture for facet {facet_name!r} "
            f"on module {address!r}. {grad.reason}"
        )
    if not isinstance(grad, torch.Tensor):
        raise RuntimeError(
            f"Attribution patching requires grad capture for facet {facet_name!r} "
            f"on module {address!r}."
        )
    return grad


def _metric_scalar(value: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    """Return a detached scalar metric converted for assignment."""

    scalar = _require_scalar_metric(value).detach()
    return scalar.to(device=like.device, dtype=like.dtype)


def _require_scalar_metric(value: torch.Tensor) -> torch.Tensor:
    """Validate and return a scalar tensor metric."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Patch metric must return a scalar tensor, got {type(value).__name__}.")
    if value.numel() != 1:
        raise ValueError(
            f"Patch metric must return one scalar value, got shape {tuple(value.shape)}."
        )
    return value.reshape(())
