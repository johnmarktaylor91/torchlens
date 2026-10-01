"""Dataloader aggregation over streaming statistics (C01 item 18)."""

from __future__ import annotations

import gc
import random
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..intervention.errors import MultiMatchWarning

__tl_layer__ = "L5"

from ._streaming import (
    StreamingStat,
)


class Aggregator:
    """Combine multiple streaming accumulators in one update pass."""

    def __init__(self, *stats: StreamingStat, name: str | None = None) -> None:
        """Initialize the combined aggregator.

        Parameters
        ----------
        *stats:
            Streaming statistic instances.
        name:
            Optional metric name.
        """

        self.name = name
        self.stats = tuple(stats)

    def update(self, value: Any) -> None:
        """Update each child statistic.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        for stat in self.stats:
            stat.update(value)

    def result(self) -> dict[str, Any]:
        """Return each child statistic result.

        Returns
        -------
        dict[str, Any]
            Mapping from child names/classes to finalized results.
        """

        results: dict[str, Any] = {}
        for index, stat in enumerate(self.stats):
            key = stat.name or type(stat).__name__
            if key in results:
                key = f"{key}_{index}"
            results[key] = stat.result()
        return results


def _resolve_metric_out(log: Any, metric_name: str) -> tuple[Any, Any]:
    """Resolve one metric input value and its matched site from a Trace.

    Parameters
    ----------
    log:
        Captured model log.
    metric_name:
        Layer selector or ``"output"``.

    Returns
    -------
    tuple[Any, Any]
        ``(value, site)`` where ``site`` is the matched layer object whose
        ``out`` supplied the value.
    """

    if metric_name == "output" and log.output_layers:
        site = log[log.output_layers[-1]]
        return site.out, site
    try:
        site = log[metric_name]
        return site.out, site
    except Exception:
        matches = _matching_layers(
            log,
            metric_name,
            require_grad=False,
        )
        if not matches:
            raise KeyError(f"No saved out matched metric {metric_name!r}.")
        _warn_on_ambiguous_metric_match(metric_name, matches)
        return matches[0].out, matches[0]


def _metric_value_from_log(log: Any, metric_name: str) -> Any:
    """Resolve one metric input value from a Trace.

    Parameters
    ----------
    log:
        Captured model log.
    metric_name:
        Layer selector or ``"output"``.

    Returns
    -------
    Any
        Tensor-like value for the metric.
    """

    value, _site = _resolve_metric_out(log, metric_name)
    return value


def _metric_grad_from_log(log: Any, metric_name: str) -> Any:
    """Resolve one gradient metric input value from a Trace."""

    if metric_name == "output":
        raise KeyError(f"No saved grad matched metric {metric_name!r}.")
    try:
        value = log[metric_name].grad
    except Exception:
        matches = _matching_layers(
            log,
            metric_name,
            require_grad=True,
        )
        if not matches:
            raise KeyError(f"No saved grad matched metric {metric_name!r}.")
        _warn_on_ambiguous_metric_match(metric_name, matches)
        value = matches[0].grad
    if value is None:
        raise KeyError(f"No saved grad matched metric {metric_name!r}.")
    return value


def _matching_layers(log: Any, metric_name: str, *, require_grad: bool) -> list[Any]:
    """Return saved layers whose labels contain ``metric_name``.

    Parameters
    ----------
    log:
        Captured model trace.
    metric_name:
        User-provided selector substring.
    require_grad:
        Whether to require gradient availability instead of saved activations.

    Returns
    -------
    list[Any]
        Matching saved layers in trace order.
    """

    attribute = "has_grad" if require_grad else "has_saved_activation"
    return [
        layer
        for layer in log.layer_list
        if metric_name in str(layer.layer_label) and bool(getattr(layer, attribute, False))
    ]


def _warn_on_ambiguous_metric_match(metric_name: str, matches: list[Any]) -> None:
    """Warn when a metric substring selector matches multiple saved sites.

    Parameters
    ----------
    metric_name:
        User-provided selector substring.
    matches:
        Matching saved sites in trace order.

    Returns
    -------
    None
        Emits a warning when multiple sites match.
    """

    if len(matches) < 2:
        return
    first_label = str(getattr(matches[0], "layer_label", metric_name))
    warnings.warn(
        (
            f"metric selector {metric_name!r} matched {len(matches)} sites; "
            f"using the first saved site {first_label!r}."
        ),
        MultiMatchWarning,
        stacklevel=3,
    )


def _split_batch_for_loss(batch: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return model input and extra loss arguments from a dataloader batch."""

    if isinstance(batch, tuple) and len(batch) >= 2:
        return batch[0], tuple(batch[1:])
    if isinstance(batch, list) and len(batch) >= 2:
        return batch[0], tuple(batch[1:])
    return batch, ()


# Each sparse record batch leaves a cyclic capture-event graph behind, which
# only a cyclic collection can free. With the default gen-0 threshold that
# garbage gets promoted to gen 2 mid-batch and either accumulates for many
# batches or forces expensive full-heap collections. Raising the gen-0
# threshold for the duration of the fast loop keeps each batch's garbage in
# the young generations, where a per-batch ``gc.collect(1)`` frees it without
# scanning the full heap.
_FAST_LOOP_GEN0_THRESHOLD = 100_000

# One fingerprint entry per operation in capture order: the normalized op type
# plus each parent encoded structurally -- ("op", stream position),
# ("input", label), ("buffer", first-reference ordinal), or ("other", label).
# Raw capture labels are deliberately absent: their numbering diverges between
# exhaustive trace and sparse record when buffer-write events consume indexes.
_StreamEntry = tuple[str, tuple[tuple[str, Any], ...]]


@dataclass(frozen=True)
class _CompiledAggregatePlan:
    """Discovery-batch measurement plan for ``aggregate(target='out')``."""

    fingerprint: tuple[_StreamEntry, ...]
    sites: dict[str, int]


def _trace_reference_stream(log: Any) -> tuple[tuple[_StreamEntry, ...], dict[str, int]] | None:
    """Rebuild the raw op-event stream fingerprint from a finalized Trace.

    Returns ``None`` whenever any structural detail cannot be mapped; callers
    treat that as "do not compile a plan" and keep the exact per-batch path.
    """

    try:
        ops = []
        input_labels: set[str] = set()
        buffer_labels: set[str] = set()
        for layer in log.layer_list:
            if layer.is_input:
                input_labels.add(layer.layer_label)
            elif layer.is_buffer:
                buffer_labels.add(layer.layer_label)
            elif not layer.is_output:
                ops.append(layer)
        ops.sort(key=lambda op: op.raw_index)
        final_to_pos = {op.layer_label: pos for pos, op in enumerate(ops)}
        if len(final_to_pos) != len(ops):
            return None
        buffer_ordinals: dict[str, int] = {}
        entries: list[_StreamEntry] = []
        for op in ops:
            parents: list[tuple[str, Any]] = []
            for parent in op.parents:
                if parent in final_to_pos:
                    parents.append(("op", final_to_pos[parent]))
                elif parent in input_labels:
                    parents.append(("input", parent))
                elif parent in buffer_labels:
                    parents.append(
                        ("buffer", buffer_ordinals.setdefault(parent, len(buffer_ordinals)))
                    )
                else:
                    return None
            layer_type = op.layer_type
            if not isinstance(layer_type, str) or not layer_type:
                return None
            entries.append((layer_type, tuple(parents)))
        return tuple(entries), final_to_pos
    except Exception:
        return None


def _site_stream_position(site: Any, final_to_pos: dict[str, int]) -> int | None:
    """Return the op-stream position measured for a resolved metric site."""

    try:
        if getattr(site, "is_output", False):
            parents = list(getattr(site, "parents", ()))
            if len(parents) != 1:
                return None
            return final_to_pos.get(parents[0])
        label = getattr(site, "layer_label", None)
        if label is None:
            return None
        return final_to_pos.get(label)
    except Exception:
        return None


def _compile_aggregate_plan(
    log: Any, resolved_sites: Mapping[str, Any]
) -> _CompiledAggregatePlan | None:
    """Compile the discovery trace into a sparse measurement plan, or refuse."""

    reference = _trace_reference_stream(log)
    if reference is None:
        return None
    fingerprint, final_to_pos = reference
    sites: dict[str, int] = {}
    for metric_name, site in resolved_sites.items():
        position = _site_stream_position(site, final_to_pos)
        if position is None:
            return None
        sites[metric_name] = position
    return _CompiledAggregatePlan(fingerprint=fingerprint, sites=sites)


@dataclass
class _RecordStreamCursor:
    """Per-batch op-event stream state built inside the record predicate."""

    entries: list[_StreamEntry] = field(default_factory=list)
    raw_to_pos: dict[str, int] = field(default_factory=dict)
    event_to_pos: dict[Any, int] = field(default_factory=dict)
    buffer_ordinals: dict[str, int] = field(default_factory=dict)
    last_event: Any = None
    last_decision: bool = False
    broken: bool = False


def _strip_raw_suffix(label: str) -> str:
    """Return ``label`` without the in-flight ``_raw`` capture suffix."""

    return label[:-4] if label.endswith("_raw") else label


def _make_plan_predicate(
    cursor: _RecordStreamCursor, wanted: frozenset[int]
) -> Callable[[Any], bool]:
    """Build the record predicate that fingerprints the stream and saves sites."""

    def _predicate(ctx: Any) -> bool:
        """Fingerprint the record stream and decide whether this site is wanted.

        Stateful in ``cursor``: a repeat call for the same event index under an
        alias label replays the previous decision instead of advancing the stream.
        """

        try:
            if ctx.kind != "op":
                return False
            event_index = ctx.event_index
            if event_index is not None and event_index == cursor.last_event:
                # Compatibility retry for the same event under an alias label.
                return cursor.last_decision
            cursor.last_event = event_index
            position = len(cursor.entries)
            raw_label = ctx.raw_label
            if isinstance(raw_label, str) and raw_label:
                cursor.raw_to_pos[_strip_raw_suffix(raw_label)] = position
            parents: list[tuple[str, Any]] = []
            for parent in ctx.parent_labels:
                name = _strip_raw_suffix(parent)
                parent_pos = cursor.raw_to_pos.get(name)
                if parent_pos is not None:
                    parents.append(("op", parent_pos))
                elif name.startswith("input_"):
                    parents.append(("input", name))
                elif name.startswith("buffer_"):
                    parents.append(
                        (
                            "buffer",
                            cursor.buffer_ordinals.setdefault(name, len(cursor.buffer_ordinals)),
                        )
                    )
                else:
                    parents.append(("other", name))
            layer_type = ctx.layer_type
            cursor.entries.append(
                (layer_type if isinstance(layer_type, str) else "", tuple(parents))
            )
            decision = position in wanted
            if decision:
                cursor.event_to_pos[event_index] = position
            cursor.last_decision = decision
            return decision
        except Exception:
            cursor.broken = True
            cursor.last_decision = False
            return False

    return _predicate


def _run_compiled_aggregate_batch(
    model: nn.Module,
    model_input: Any,
    plan: _CompiledAggregatePlan,
) -> dict[int, torch.Tensor] | None:
    """Measure one batch with the compiled sparse plan.

    Returns the payload per compiled stream position, or ``None`` when the
    batch's op stream does not match the discovery fingerprint (the caller
    then falls back to the exact full-trace path).
    """

    # Import from the real home, not the lazy root facade: the F10 deferred
    # data_classes -> stats import edge puts this module inside an import
    # SCC where mypy degrades facade names to implicit-Any and refuses the
    # call ("_Any? not callable").
    from ..fastlog import record

    wanted = frozenset(plan.sites.values())
    cursor = _RecordStreamCursor()
    recording = None
    try:
        recording = record(model, model_input, save=_make_plan_predicate(cursor, wanted))
        if cursor.broken or tuple(cursor.entries) != plan.fingerprint:
            return None
        payloads: dict[int, torch.Tensor] = {}
        for entry in recording:
            position = cursor.event_to_pos.get(entry.ctx.event_index)
            if position is None:
                return None
            payload = entry.ram_payload
            if payload is None:
                return None
            payloads[position] = payload
        if any(position not in payloads for position in wanted):
            return None
        return payloads
    finally:
        del recording, cursor


def _rng_snapshot() -> tuple[Any, Any, Any, Any] | None:
    """Snapshot the global RNG engines a capture can consume.

    Covers Python's ``random`` module, the torch CPU generator, torch CUDA
    generators, and NumPy's legacy global generator. Returns ``None`` when any
    engine cannot be snapshotted; callers then skip the sparse fast path.
    """

    try:
        cuda_states = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        try:
            import numpy as np

            numpy_state = np.random.get_state()
        except Exception:
            numpy_state = None
        return (
            random.getstate(),
            torch.random.get_rng_state().clone(),
            cuda_states,
            numpy_state,
        )
    except Exception:
        return None


def _rng_restore(snapshot: tuple[Any, Any, Any, Any]) -> None:
    """Rewind the global RNG engines to a ``_rng_snapshot()`` state."""

    python_state, torch_state, cuda_states, numpy_state = snapshot
    random.setstate(python_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        for device_index, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, device_index)
    if numpy_state is not None:
        import numpy as np

        np.random.set_state(numpy_state)


def _model_state_allows_compiled_plan(model: nn.Module) -> bool:
    """Gate the sparse plan on fully-eval models.

    A fingerprint mismatch re-traces the same batch, so the fast path is only
    compiled when a second forward cannot mutate module state (train-mode
    BatchNorm running stats being the canonical hazard).
    """

    try:
        return not any(module.training for module in model.modules())
    except Exception:
        return False


def aggregate(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    target: str = "out",
    loss_fn: Callable[..., torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Stream outs through metric accumulators.

    Parameters
    ----------
    model:
        Model to capture.
    dataloader:
        Iterable of model inputs.
    metrics:
        Mapping from layer selector to streaming statistic.
    target:
        ``"out"`` for activation statistics or ``"grad"`` for gradient
        statistics.
    loss_fn:
        Callable used to build a loss from ``(output, *batch_tail)`` when
        ``target="grad"``.

    Returns
    -------
    dict[str, Any]
        Finalized metric results.
    """

    # Real-home import (not the lazy root facade) for the same SCC/implicit-
    # Any reason as the fastlog.record import above.
    from ..options import CaptureOptions
    from ..user_funcs import trace

    if target not in {"out", "grad"}:
        raise ValueError("target must be 'out' or 'grad'")
    if target == "grad" and loss_fn is None:
        raise TypeError("aggregate(target='grad') requires loss_fn=")
    grad_loss_fn = loss_fn

    layers = [name for name in metrics if name != "output"]
    capture_layers: str | list[str] = layers if layers else "all"

    if target == "grad":
        for batch in dataloader:
            model_input, loss_args = _split_batch_for_loss(batch)
            log = trace(
                model,
                model_input,
                capture=CaptureOptions(
                    layers_to_save=capture_layers,
                    save_grads=capture_layers,
                ),
            )
            try:
                if grad_loss_fn is None:
                    raise TypeError("aggregate(target='grad') requires loss_fn=")
                loss = grad_loss_fn(_metric_value_from_log(log, "output"), *loss_args)
                log.log_backward(loss)
                for metric_name, stat in metrics.items():
                    stat.update(_metric_grad_from_log(log, metric_name))
            finally:
                log.cleanup()
        return {name: stat.result() for name, stat in metrics.items()}

    # target == "out": trace the structure once, then execute a compiled sparse
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
                payloads = (
                    _run_compiled_aggregate_batch(model, model_input, plan)
                    if snapshot is not None
                    else None
                )
                if payloads is not None:
                    for metric_name, stat in metrics.items():
                        stat.update(payloads[plan.sites[metric_name]])
                    gc.collect(1)
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
                resolved_sites: dict[str, Any] = {}
                for metric_name, stat in metrics.items():
                    value, site = _resolve_metric_out(log, metric_name)
                    resolved_sites[metric_name] = site
                    stat.update(value)
                if compile_allowed:
                    plan = _compile_aggregate_plan(log, resolved_sites)
                    compile_allowed = False
                    if plan is not None:
                        saved_gc_thresholds = gc.get_threshold()
                        gc.set_threshold(
                            max(saved_gc_thresholds[0], _FAST_LOOP_GEN0_THRESHOLD),
                            *saved_gc_thresholds[1:],
                        )
            finally:
                log.cleanup()
    finally:
        if saved_gc_thresholds is not None:
            gc.collect(1)
            gc.set_threshold(*saved_gc_thresholds)
    return {name: stat.result() for name, stat in metrics.items()}
