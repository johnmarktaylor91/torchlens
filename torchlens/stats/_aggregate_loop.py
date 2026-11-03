"""The dataset loops behind :func:`torchlens.stats.aggregate`.

``aggregate`` validates its arguments and keys the metrics, then runs one of
three loops here: a per-batch capture plus backward when a metric key is a
gradient site, a per-batch full capture for selector-keyed activation
metrics, and the activation-only loop over string keys with the compiled
sparse fast path (the plan machinery stays in ``_aggregate``).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn

from .._user_funcs import trace
from ..intervention.spec import is_gradient_site
from ..options import CaptureOptions
from ._aggregate import (
    _FAST_LOOP_GEN0_THRESHOLD,
    _compile_aggregate_plan,
    _CompiledAggregatePlan,
    _is_gradient_key,
    _metric_grad_from_log,
    _metric_value_from_log,
    _model_state_allows_compiled_plan,
    _resolve_metric_out,
    _rng_restore,
    _rng_snapshot,
    _run_compiled_aggregate_batch,
    _split_batch_for_loss,
    _warn_on_ambiguous_metric_match,
)
from ._streaming import StreamingStat

__tl_layer__ = "L5"


def _trace_entry() -> Callable[..., Any]:
    """Return ``tl.trace`` from its real home.

    Real-home import (not the lazy root facade) for the same SCC/implicit-Any
    reason as the fastlog.record import in ``_aggregate``.
    """

    return trace


def _aggregate_loop(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    sites: Mapping[str, Any],
    loss_fn: Callable[..., torch.Tensor] | None,
) -> dict[str, Any]:
    """Run the dataset loop of :func:`aggregate` (arguments already validated)."""

    if loss_fn is not None and any(_is_gradient_key(site) for site in sites.values()):
        return _aggregate_backward_loop(model, dataloader, metrics, sites=sites, loss_fn=loss_fn)
    if any(not isinstance(site, str) for site in sites.values()):
        return _aggregate_traced_loop(model, dataloader, metrics, sites=sites)
    return _aggregate_out_loop(model, dataloader, metrics)


def _site_label(site: Any) -> str | None:
    """Return the exact label a metric site names, or ``None`` for a pattern.

    A ``tl.grad`` key names its inner site; a bare string inner or an exact
    ``tl.label`` selector is a label.
    """

    inner = getattr(site, "inner", None) if is_gradient_site(site) else site
    if isinstance(inner, str):
        return inner
    label = getattr(inner, "selector_value", None)
    if getattr(inner, "selector_kind", None) == "label" and isinstance(label, str):
        return label
    return None


def _metric_capture_layers(sites: Mapping[str, Any]) -> tuple[str | list[str], str | list[str]]:
    """Return ``(layers_to_save, save_grads)`` for the metric keys' sites."""

    activation: list[str] = []
    gradient: list[str] = []
    for site in sites.values():
        if isinstance(site, str):
            if site != "output":
                activation.append(site)
            continue
        label = _site_label(site)
        if label is None:
            return "all", "all"
        (gradient if _is_gradient_key(site) else activation).append(label)
    return (activation + gradient or "all"), (gradient or "all")


def _site_value(log: Any, name: str, site: Any) -> Any:
    """Read one metric's tensor (activation or output gradient) from a trace."""

    if isinstance(site, str):
        return _metric_value_from_log(log, name)
    gradient = is_gradient_site(site)
    target = getattr(site, "inner", site) if gradient else site
    matches = log._resolve_sites(target, max_fanout=max(1, len(log._layer_list)))
    if not matches:
        raise KeyError(f"No site matched metric {name!r}.")
    _warn_on_ambiguous_metric_match(name, list(matches))
    layer = matches[0]
    value = layer.grad if gradient else layer.out
    if value is None:
        kind = "grad" if gradient else "out"
        raise KeyError(f"No saved {kind} matched metric {name!r}.")
    return value


def _aggregate_backward_loop(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    sites: Mapping[str, Any],
    loss_fn: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    """Per-batch capture plus backward for metric keys that include gradients."""

    trace = _trace_entry()
    layers_to_save, save_grads = _metric_capture_layers(sites)
    for batch in dataloader:
        model_input, loss_args = _split_batch_for_loss(batch)
        log = trace(
            model,
            model_input,
            capture=CaptureOptions(layers_to_save=layers_to_save, save_grads=save_grads),
        )
        try:
            loss = loss_fn(_metric_value_from_log(log, "output"), *loss_args)
            log.backward(loss)
            for name, stat in metrics.items():
                site = sites[name]
                if isinstance(site, str) or not _is_gradient_key(site):
                    stat.update(_site_value(log, name, site))
                    continue
                label = _site_label(site)
                if label is not None:
                    stat.update(_metric_grad_from_log(log, label))
                else:
                    stat.update(_site_value(log, name, site))
        finally:
            log.cleanup()
    return {name: stat.compute() for name, stat in metrics.items()}


def _aggregate_traced_loop(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    sites: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-batch full capture for activation metrics keyed by selectors."""

    trace = _trace_entry()
    layers_to_save, _save_grads = _metric_capture_layers(sites)
    for batch in dataloader:
        model_input, _loss_args = _split_batch_for_loss(batch)
        log = trace(model, model_input, capture=CaptureOptions(layers_to_save=layers_to_save))
        try:
            for name, stat in metrics.items():
                stat.update(_site_value(log, name, sites[name]))
        finally:
            log.cleanup()
    return {name: stat.compute() for name, stat in metrics.items()}


def _aggregate_out_loop(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
) -> dict[str, Any]:
    """Activation-only loop over string keys, with the compiled sparse fast path."""

    trace = _trace_entry()
    layers = [name for name in metrics if name != "output"]
    capture_layers: str | list[str] = layers if layers else "all"

    # Trace the structure once, then execute a compiled sparse
    # measurement per batch. Any surprise -- unstable structure, non-eval
    # modules, unmappable sites -- keeps or restores the exact per-batch
    # full-trace path. Trace and sparse record share the per-capture seed draw,
    # so the global RNG streams the model and the stats see stay identical; a
    # fingerprint mismatch rewinds the RNG engines before re-tracing so the
    # fallback consumes the exact stream the full path would have.
    plan: _CompiledAggregatePlan | None = None
    compile_allowed = _model_state_allows_compiled_plan(model)
    saved_gc_thresholds: tuple[int, int, int] | None = None
    try:
        for batch in dataloader:
            model_input, _loss_args = _split_batch_for_loss(batch)
            if plan is not None:
                snapshot = _rng_snapshot()
                payloads = _compiled_payloads(model, model_input, plan, snapshot)
                if payloads is not None:
                    _update_from_payloads(metrics, plan, payloads)
                    continue
                # Structure drifted from the discovery batch (or the RNG
                # engines could not be protected): re-trace this batch exactly
                # and stay on the full path for the rest of the loop.
                plan = None
                compile_allowed = False
                if saved_gc_thresholds is not None:
                    gc.set_threshold(*saved_gc_thresholds)
                    saved_gc_thresholds = None
                if snapshot is not None:
                    _rng_restore(snapshot)
            log = trace(
                model,
                model_input,
                capture=CaptureOptions(layers_to_save=capture_layers),
            )
            try:
                resolved_sites = _update_from_trace(log, metrics)
                if compile_allowed:
                    plan = _compile_aggregate_plan(log, resolved_sites)
                    compile_allowed = False
                    if plan is not None:
                        saved_gc_thresholds = _raise_gen0_threshold()
            finally:
                log.cleanup()
    finally:
        if saved_gc_thresholds is not None:
            gc.collect(1)
            gc.set_threshold(*saved_gc_thresholds)
    return {name: stat.compute() for name, stat in metrics.items()}


def _compiled_payloads(
    model: nn.Module,
    model_input: Any,
    plan: _CompiledAggregatePlan,
    snapshot: tuple[Any, Any, Any, Any] | None,
) -> Any:
    """Run the compiled sparse measurement for one batch (``None``: fall back).

    Parameters
    ----------
    model:
        Model under measurement.
    model_input:
        This batch's model input.
    plan:
        The discovery-batch plan.
    snapshot:
        The RNG snapshot taken before the batch; ``None`` (the engines could
        not be protected) skips the sparse run.

    Returns
    -------
    Any
        The per-position payloads, or ``None`` when the batch must re-trace.
    """

    return _run_compiled_aggregate_batch(model, model_input, plan) if snapshot is not None else None


def _update_from_payloads(
    metrics: Mapping[str, StreamingStat], plan: _CompiledAggregatePlan, payloads: Any
) -> None:
    """Feed one compiled batch's payloads to the metrics, then collect generation 1."""

    for metric_name, stat in metrics.items():
        stat.update(payloads[plan.sites[metric_name]])
    gc.collect(1)


def _update_from_trace(log: Any, metrics: Mapping[str, StreamingStat]) -> dict[str, Any]:
    """Feed one traced batch to the metrics; return each metric's resolved site."""

    resolved_sites: dict[str, Any] = {}
    for metric_name, stat in metrics.items():
        value, site = _resolve_metric_out(log, metric_name)
        resolved_sites[metric_name] = site
        stat.update(value)
    return resolved_sites


def _raise_gen0_threshold() -> tuple[int, int, int]:
    """Raise the gen-0 GC threshold for the fast loop; return the saved thresholds."""

    saved_gc_thresholds = gc.get_threshold()
    gc.set_threshold(
        max(saved_gc_thresholds[0], _FAST_LOOP_GEN0_THRESHOLD),
        *saved_gc_thresholds[1:],
    )
    return saved_gc_thresholds
