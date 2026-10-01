"""Prebuilt activation and attribution patching helpers for facets."""

from __future__ import annotations

import copy
import itertools
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from typing import Any

import torch
from torch import nn

from ..errors._base import TorchLensError, TorchLensWarning
from ..intervention.selectors import facet
from ..options import CaptureOptions
from ..user_funcs import trace
from .facets import Facet, MissingGradient

Metric = Callable[[Any], torch.Tensor]


class PatchApplicationError(TorchLensError, RuntimeError):
    """A patched rerun produced no effective activation replacement.

    Raised by the activation-patching helpers when the positive fire ledger
    for a patched run is empty: either the facet hook never fired, or every
    fire was refused (``replaced=False``). Publishing the metric row in that
    state would silently equal the corrupted baseline -- a publishable-looking
    null result -- so the helpers refuse instead (DOCUMENTED-UNSTABLE
    spelling pending naming-session ratification). ``fields["code"]`` is
    ``patch_ineffective``; the ``RuntimeError`` lineage is preserved for
    existing catch sites.
    """


class _CounterfactualStateGuard:
    """Snapshot and restore model state and global RNG around patching runs.

    Activation and attribution patching run several forward passes (a clean
    baseline, a corrupted baseline, and one per patched cell) on the SAME live
    model. Without resetting state between runs, mutable buffers (e.g.
    BatchNorm running statistics or any buffer written during forward) and the
    global RNG drift: every counterfactual after the first starts from a
    different model/RNG state than the clean baseline, so the reported effect is
    a silently wrong comparison, and the caller's model + global RNG are left
    mutated when the helper returns.

    Call :meth:`open` before the first run and :meth:`close` in a ``finally``.
    ``open`` snapshots the model's parameters/buffers and enters
    ``torch.random.fork_rng`` so the caller's global RNG is restored robustly on
    ``close`` no matter what the traced forwards do. Call :meth:`reset` before
    each counterfactual run to return the model and RNG to the captured baseline
    so every counterfactual is a true comparison. ``close`` restores the model
    state and releases the forked RNG.
    """

    def __init__(self, model: nn.Module) -> None:
        """Snapshot the model's parameters/buffers/grads and the global RNG state."""

        self._tensors: list[tuple[torch.Tensor, torch.Tensor]] = [
            (tensor, tensor.detach().clone()) for tensor in _stateful_tensors(model)
        ]
        self._grads: list[tuple[torch.Tensor, torch.Tensor | None]] = [
            (
                tensor,
                None if tensor.grad is None else tensor.grad.detach().clone(),
            )
            for tensor, _saved in self._tensors
        ]
        self._rng: dict[str, Any] = _snapshot_rng()
        self._fork: Any = None

    def open(self) -> None:
        """Enter a forked-RNG scope so the caller's global RNG is preserved."""

        self._fork = torch.random.fork_rng(devices=_fork_rng_devices())
        self._fork.__enter__()

    def close(self) -> None:
        """Restore the model state and release the forked RNG scope."""

        try:
            self._restore_model()
        finally:
            fork, self._fork = self._fork, None
            if fork is not None:
                fork.__exit__(None, None, None)

    def reset(self) -> None:
        """Reset model state and global RNG to the captured baseline."""

        self._restore_model()
        _restore_rng(self._rng)

    def _restore_model(self) -> None:
        """Restore every DRIFTED parameter/buffer value and grad in place.

        The copy is equality-gated on purpose: an unconditional ``copy_``
        bumps the autograd version counter of every parameter consumed by an
        earlier captured forward, so the later ``log_backward`` on the clean
        and corrupted baselines raised "modified by an inplace operation" on
        ANY model with parameters (attribution patching was unusable outside
        zero-parameter toys). A value-equal tensor needs no write and keeps
        its autograd graphs valid; a genuinely drifted tensor (a mutated
        buffer, a forward that writes a parameter) is still written back --
        comparison honesty beats preserving a graph that no longer matches
        the model state, and autograd's in-place tripwire then fires on a
        REAL divergence instead of on bookkeeping.

        Grad slots are restored to their snapshot (usually ``None``) so
        counterfactual runs start from the baseline grad state and the
        caller's model does not accumulate patching-run gradients. Grad
        assignment never touches version counters.
        """

        with torch.no_grad():
            for tensor, saved in self._tensors:
                if not torch.equal(tensor, saved):
                    tensor.copy_(saved)
        for tensor, saved_grad in self._grads:
            current = tensor.grad
            if saved_grad is None:
                if current is not None:
                    tensor.grad = None
            elif current is None or not torch.equal(current, saved_grad):
                tensor.grad = saved_grad.clone()


def _teardown(guard: _CounterfactualStateGuard, *logs: Any) -> None:
    """Run every teardown step even when an earlier one raises.

    The entry points used to tear down as bare sequential statements, so a
    raising ``Trace.cleanup()`` skipped ``guard.close()`` and left the USER's
    model parameters unrestored and the global RNG permanently advanced.
    ``ExitStack`` callbacks run last-registered-first, so the guard is
    registered first (closes LAST, after every log cleanup) and each
    ``logs``-order cleanup still runs even if an earlier one raises.
    """

    with ExitStack() as stack:
        stack.callback(guard.close)
        for log in reversed(logs):
            if log is not None:
                stack.callback(log.cleanup)


def _stateful_tensors(model: nn.Module) -> list[torch.Tensor]:
    """Return every distinct parameter and buffer tensor of a model.

    Uses object identity to include non-persistent buffers and to stage tied /
    double-registered tensors exactly once.
    """

    seen: set[int] = set()
    tensors: list[torch.Tensor] = []
    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        tensors.append(tensor)
    return tensors


def _fork_rng_devices() -> list[int]:
    """Return the CUDA device ordinals to fork RNG for (empty when CPU-only).

    Gated on ``is_initialized()``, not ``is_available()`` (R36-4): if this
    process never initialized CUDA, no CUDA generator fed any captured op,
    and forking RNG for every visible device would allocate the very CUDA
    contexts (~300-600 MB each) this path does not need -- the exact stale
    pre-fix gate ``utils/rng.py`` documents.
    """

    from ..utils.tensor_utils import _is_cuda_initialized

    if _is_cuda_initialized() and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def _snapshot_rng() -> dict[str, Any]:
    """Capture the global CPU (and CUDA, when live) RNG state."""

    from ..utils.rng import _snapshot_cuda_rng_states

    snapshot: dict[str, Any] = {"cpu": torch.get_rng_state()}
    # Initialized-CUDA-only, latch-guarded (R36-4): returns [] on a
    # CUDA-less or never-initialized process without touching the driver.
    cuda_states = _snapshot_cuda_rng_states()
    if cuda_states:
        snapshot["cuda"] = cuda_states
    return snapshot


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    """Restore the global CPU (and CUDA, when available) RNG state."""

    torch.set_rng_state(snapshot["cpu"])
    cuda_states = snapshot.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


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

    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = _baseline_traces(
            model, clean_input, corrupted_input, trace_kwargs=trace_kwargs, guard=guard
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
                guard=guard,
            )

        if not modules:
            raise ValueError(f"No modules expose facet {facet_name!r}.")
        first_value = _facet_tensor(clean_log.modules[modules[0]].facets[facet_name])
        normalized_axis = position_axis % first_value.ndim
        patch_positions_list = (
            list(range(first_value.shape[normalized_axis]))
            if positions is None
            else list(positions)
        )
        result = torch.empty(
            (len(modules), len(patch_positions_list)),
            dtype=metric_template.dtype,
            device=metric_template.device,
        )
        campaign_ledger: dict[str, int] = {}
        for layer_index, address in enumerate(modules):
            clean_value = (
                _facet_tensor(clean_log.modules[address].facets[facet_name]).detach().clone()
            )
            selector = facet(facet_name).in_module(address)
            for pos_index, position in enumerate(patch_positions_list):

                def _patch_position(
                    out: torch.Tensor,
                    *,
                    hook: Any,
                    position: Any = position,
                    clean_value: torch.Tensor = clean_value,
                ) -> torch.Tensor:
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
                    guard=guard,
                    facet_name=facet_name,
                    address=address,
                    campaign_ledger=campaign_ledger,
                )
                try:
                    result[layer_index, pos_index] = _metric_scalar(
                        metric(patched_log), like=result
                    )
                finally:
                    patched_log.cleanup()
        _warn_if_campaign_all_identical(
            campaign_ledger, f"facet {facet_name!r} across modules {tuple(modules)!r}"
        )
        return result
    finally:
        _teardown(guard, clean_log, corrupted_log)


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

    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = _baseline_traces(
            model, clean_input, corrupted_input, trace_kwargs=trace_kwargs, guard=guard
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
            guard=guard,
        )
    finally:
        _teardown(guard, clean_log, corrupted_log)


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

    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = _baseline_traces(
            model, clean_input, corrupted_input, trace_kwargs=trace_kwargs, guard=guard
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
            guard=guard,
        )
    finally:
        _teardown(guard, clean_log, corrupted_log)


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

    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = _baseline_traces(
            model, clean_input, corrupted_input, trace_kwargs=trace_kwargs, guard=guard
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
            guard=guard,
        )
    finally:
        _teardown(guard, clean_log, corrupted_log)


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
        helper captures all gradients; an explicit
        ``capture=CaptureOptions(save_grads=False)`` is honored and is a
        useful way to verify the missing-gradient error path.

    Returns
    -------
    torch.Tensor
        Approximate metric effects shaped ``[layer, head]``.
    """

    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        clean_log, corrupted_log = _baseline_traces(
            model,
            clean_input,
            corrupted_input,
            trace_kwargs=trace_kwargs,
            save_grads=True,
            guard=guard,
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
    finally:
        _teardown(guard, clean_log, corrupted_log)


def _baseline_traces(
    model: nn.Module,
    clean_input: Any,
    corrupted_input: Any,
    *,
    trace_kwargs: Mapping[str, Any] | None,
    save_grads: bool | None = None,
    guard: _CounterfactualStateGuard,
) -> tuple[Any, Any]:
    """Return clean and corrupted traces with all layer activations saved.

    Both baselines start from the guard's pristine model/RNG snapshot so the
    corrupted run is a true counterfactual of the clean run rather than a run on
    a model already mutated by the clean forward.
    """

    kwargs = _trace_kwargs(trace_kwargs, save_grads=save_grads)
    guard.reset()
    clean_log = trace(model, clean_input, **kwargs)
    guard.reset()
    corrupted_log = trace(model, corrupted_input, **kwargs)
    return clean_log, corrupted_log


def _trace_kwargs(
    trace_kwargs: Mapping[str, Any] | None,
    *,
    save_grads: bool | None = None,
) -> dict[str, Any]:
    """Return trace keyword arguments with the helper's REQUIRED fields applied.

    The patching helpers cannot work without ``layers_to_save="all"`` (facet
    values must be readable on every candidate module) and
    ``save_arg_values=True`` (reconstructed facets read saved op args), and
    attribution additionally needs ``save_grads``. These used to be supplied
    only when the caller passed NO ``capture=`` at all -- any user
    ``capture=`` silently dropped every one of them, so facets came up
    partially absent and grids were computed over a silently narrowed module
    set. The required fields now COMPOSE with the user's options: fields the
    user left unspecified are filled in, an explicitly matching value passes,
    and an explicitly CONFLICTING ``layers_to_save``/``save_arg_values``
    refuses with the requirement named (an explicit user ``save_grads``
    predicate is honored as-is -- a too-narrow one fails loudly at the
    gradient read). Flat capture spellings (``layers_to_save=`` etc.) were
    REMOVED from ``tl.trace``; this helper used to forward them verbatim into
    a guaranteed ``TypeError``, so it now refuses them with the grouped
    spelling named.
    """

    from ..options import _CAPTURE_FIELDS

    kwargs = dict(trace_kwargs or {})
    stale_flat = sorted(name for name in kwargs if name != "capture" and name in _CAPTURE_FIELDS)
    if stale_flat:
        raise ValueError(
            f"trace_kwargs contains removed flat capture kwargs {stale_flat!r}; tl.trace "
            "accepts capture knobs only through the grouped spelling "
            "capture=tl.options.CaptureOptions(...). Move these fields into capture=."
        )
    if "capture" in kwargs:
        kwargs["capture"] = _compose_required_capture(kwargs["capture"], save_grads=save_grads)
        return kwargs
    capture_fields: dict[str, Any] = {"layers_to_save": "all", "save_arg_values": True}
    if save_grads is not None:
        capture_fields["save_grads"] = save_grads
    kwargs["capture"] = CaptureOptions(**capture_fields)
    return kwargs


def _compose_required_capture(capture: Any, *, save_grads: bool | None) -> Any:
    """Merge the patching-required capture fields into user capture options.

    Parameters
    ----------
    capture:
        User-supplied ``CaptureOptions``.
    save_grads:
        Required grad-capture setting, or ``None`` when the helper does not
        need gradients.

    Returns
    -------
    Any
        ``CaptureOptions`` with unspecified required fields filled in.

    Raises
    ------
    ValueError
        If the user EXPLICITLY set a required field to a conflicting value;
        honoring it would silently narrow or empty the facet table.
    """

    if not isinstance(capture, CaptureOptions):
        return capture
    required: dict[str, Any] = {"layers_to_save": "all", "save_arg_values": True}
    conflicts = [
        field_name
        for field_name, required_value in required.items()
        if capture.is_field_explicit(field_name) and getattr(capture, field_name) != required_value
    ]
    if conflicts:
        raise ValueError(
            f"Patching helpers require capture options {required!r}, but the supplied "
            f"capture= explicitly sets {conflicts!r} to conflicting values. Facet values "
            "must be readable on every candidate module (layers_to_save='all') and "
            "reconstructed facets read saved op args (save_arg_values=True); a narrower "
            "capture silently shrinks or empties the patch table. Drop these fields from "
            "capture= (they are filled in automatically) or set them to the required "
            "values."
        )
    fills = {
        field_name: required_value
        for field_name, required_value in required.items()
        if not capture.is_field_explicit(field_name)
    }
    if save_grads is not None and not capture.is_field_explicit("save_grads"):
        fills["save_grads"] = save_grads
    if not fills:
        return capture
    values = capture.as_dict()
    values.update(fills)
    return CaptureOptions.from_values(
        values, frozenset(capture._specified_fields) | frozenset(fills)
    )


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
    guard: _CounterfactualStateGuard,
) -> torch.Tensor:
    """Patch one whole facet per module and return metric values."""

    if not modules:
        raise ValueError(f"No modules expose facet {facet_name!r}.")
    first_value = _facet_tensor(clean_log.modules[modules[0]].facets[facet_name])
    del first_value
    result = torch.empty(
        (len(modules),), dtype=metric_template.dtype, device=metric_template.device
    )
    campaign_ledger: dict[str, int] = {}
    for layer_index, address in enumerate(modules):
        clean_value = _facet_tensor(clean_log.modules[address].facets[facet_name]).detach().clone()

        def _patch_whole(
            out: torch.Tensor, *, hook: Any, clean_value: torch.Tensor = clean_value
        ) -> torch.Tensor:
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
            guard=guard,
            facet_name=facet_name,
            address=address,
            campaign_ledger=campaign_ledger,
        )
        try:
            result[layer_index] = _metric_scalar(metric(patched_log), like=result)
        finally:
            patched_log.cleanup()
    _warn_if_campaign_all_identical(
        campaign_ledger, f"facet {facet_name!r} across modules {tuple(modules)!r}"
    )
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
    guard: _CounterfactualStateGuard,
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
    campaign_ledger: dict[str, int] = {}
    for layer_index, address in enumerate(modules):
        for head_index in range(n_heads):
            clean_value = (
                _facet_tensor(clean_log.modules[address].facets.head(head_index)[facet_name])
                .detach()
                .clone()
            )

            def _patch_head(
                out: torch.Tensor, *, hook: Any, clean_value: torch.Tensor = clean_value
            ) -> torch.Tensor:
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
                guard=guard,
                where=f"facet {facet_name!r} head {head_index} on module {address!r}",
                campaign_ledger=campaign_ledger,
            )
            try:
                result[layer_index, head_index] = _metric_scalar(metric(patched_log), like=result)
            finally:
                patched_log.cleanup()
    _warn_if_campaign_all_identical(
        campaign_ledger, f"facet {facet_name!r} heads across modules {tuple(modules)!r}"
    )
    return result


def _run_patch(
    model: nn.Module,
    corrupted_input: Any,
    corrupted_log: Any,
    selector: Any,
    hook: Callable[..., torch.Tensor],
    *,
    name: str,
    guard: _CounterfactualStateGuard,
    facet_name: str | None = None,
    address: str | None = None,
    where: str | None = None,
    campaign_ledger: dict[str, int] | None = None,
) -> Any:
    """Fork the corrupted trace, attach one facet hook, and rerun.

    The model/RNG state is reset to the pristine snapshot before the rerun so
    the only difference between the corrupted baseline and this patched run is
    the injected clean activation. Every hook-path rerun must leave positive
    fire evidence (at least one ``replaced=True`` fire record); a run without
    it raises :class:`PatchApplicationError` instead of returning a trace
    whose metric would silently equal the corrupted baseline. ``where``
    overrides the refusal's site description (used by the per-head caller,
    which must not enable the whole-facet input-patch route).

    Live hooks fire at wrapped-function and module-boundary sites only, so a
    facet homed on a MODEL INPUT op (e.g. ``resid_pre`` of a first block) has
    no site where a hook could ever fire. When the caller identifies the facet
    (``facet_name`` + ``address``) and its home is a model input, the patch is
    applied to the input tensor itself and the rerun proceeds without a hook —
    semantically identical to a hook fire at the home site.
    """

    if where is None:
        where = (
            f"facet {facet_name!r} on module {address!r}"
            if facet_name is not None and address is not None
            else repr(name)
        )
    fire_ledger = {"fires": 0, "identical": 0}

    input_role = None
    if facet_name is not None and address is not None:
        input_role = _model_input_home_role(corrupted_log, facet_name, address)
    if input_role is not None:
        patched_log = _run_input_patch(
            model,
            corrupted_input,
            corrupted_log,
            hook,
            name=name,
            guard=guard,
            input_role=input_role,
            fire_ledger=fire_ledger,
        )
        _fold_fire_ledger(campaign_ledger, fire_ledger)
        return patched_log

    def _tracked_hook(
        out: torch.Tensor, *, hook: Any, _inner: Callable[..., torch.Tensor] = hook
    ) -> torch.Tensor:
        """Run the patch hook while counting fires and value-identical fires."""

        result = _inner(out, hook=hook)
        fire_ledger["fires"] += 1
        if _replaced_identical(result, out):
            fire_ledger["identical"] += 1
        return result

    patched_log = corrupted_log.fork(name)
    patched_log.attach_hooks(selector, _tracked_hook)
    guard.reset()
    patched_log.run(model, corrupted_input)
    try:
        _require_effective_patch(patched_log, where=where, fire_ledger=fire_ledger)
    except PatchApplicationError:
        patched_log.cleanup()
        raise
    _fold_fire_ledger(campaign_ledger, fire_ledger)
    return patched_log


def _replaced_identical(result: Any, original: Any) -> bool:
    """Return True when a patch fire replaced the site value with an identical tensor."""

    return (
        isinstance(result, torch.Tensor)
        and isinstance(original, torch.Tensor)
        and tuple(result.shape) == tuple(original.shape)
        and torch.equal(result.detach(), original.detach())
    )


def _fold_fire_ledger(
    campaign_ledger: dict[str, int] | None, fire_ledger: Mapping[str, int]
) -> None:
    """Accumulate one patched run's fire counts into the campaign ledger, if any."""

    if campaign_ledger is None:
        return
    campaign_ledger["fires"] = campaign_ledger.get("fires", 0) + fire_ledger["fires"]
    campaign_ledger["identical"] = campaign_ledger.get("identical", 0) + fire_ledger["identical"]


def _run_input_patch(
    model: nn.Module,
    corrupted_input: Any,
    corrupted_log: Any,
    hook: Callable[..., torch.Tensor],
    *,
    name: str,
    guard: _CounterfactualStateGuard,
    input_role: Any,
    fire_ledger: dict[str, int],
) -> Any:
    """Apply the patch to the model-input leaf and rerun without a hook."""

    def _tracked_input_patch(leaf: torch.Tensor) -> torch.Tensor:
        """Apply the patch to an input leaf while feeding the fire ledger."""

        result = hook(leaf.detach().clone(), hook=None)
        fire_ledger["fires"] += 1
        if _replaced_identical(result, leaf):
            fire_ledger["identical"] += 1
        return result

    patched_input = _patch_input_leaf(corrupted_input, model, input_role, _tracked_input_patch)
    patched_log = corrupted_log.fork(name)
    guard.reset()
    patched_log.run(model, patched_input)
    return patched_log


def _warn_if_campaign_all_identical(campaign_ledger: Mapping[str, int], where: str) -> None:
    """Disclose a patch campaign whose EVERY fire replaced an identical value.

    A single value-identical cell is ordinary science (an inert head, a
    shared prompt prefix), so per-cell noise would be wrong. But when every
    fire across the WHOLE campaign replaced the site value with an identical
    tensor, the entire table is guaranteed to equal the corrupted baseline --
    and on differing inputs that pattern usually means the facet is anchored
    on an input-derived op (e.g. an attention-mask view) rather than the
    computation it names.
    """

    fires = int(campaign_ledger.get("fires", 0))
    identical = int(campaign_ledger.get("identical", 0))
    if fires and identical == fires:
        warnings.warn(
            TorchLensWarning(
                f"Activation patching campaign for {where}: every hook fire across the "
                "whole table replaced the site value with an IDENTICAL tensor (clean == "
                "corrupted at every patched site), so the table is guaranteed to equal "
                "the corrupted baseline everywhere. A genuine all-zero effect is "
                "possible, but identical values at EVERY site usually mean the facet is "
                "anchored on an input-derived op (e.g. an attention-mask view) rather "
                "than the computation it names. Remedy: inspect "
                "tl.facets.facet_coverage(trace) and re-anchor the facet before "
                "publishing this as a null result",
                code="patch_campaign_all_identical",
            ),
            stacklevel=3,
        )


def _require_effective_patch(
    patched_log: Any, *, where: str, fire_ledger: Mapping[str, int] | None = None
) -> None:
    """Refuse a patched run whose positive fire ledger is empty.

    A patched rerun with ZERO effective replacements produces a metric row
    bitwise-equal to the corrupted baseline while looking like a measured
    causal effect (the exact silent no-op measured on real HF models, where
    facet homes land on ops the live-hook engine never fires at). The
    positive evidence required here is at least one fire record from THIS run
    with ``replaced=True``; fires that were all refused (``replaced=False``,
    e.g. an in-place site whose storage could not be safely rewritten) are
    named separately so the remedy is visible.
    """

    ctx = getattr(patched_log, "last_run", None)
    ctx = ctx if isinstance(ctx, dict) else {}
    started_at = ctx.get("started_at", ctx.get("timestamp"))
    records = []
    if isinstance(started_at, (int, float)):
        # Fire records are minted DURING the run, so they are filtered by the
        # run's start time; the run's end ``timestamp`` would exclude them all.
        for layer in getattr(patched_log, "layer_list", []) or []:
            for record in getattr(layer, "interventions", []) or []:
                record_timestamp = getattr(record, "timestamp", None)
                if isinstance(record_timestamp, (int, float)) and record_timestamp >= started_at:
                    records.append(record)
    if any(bool(getattr(record, "replaced", False)) for record in records):
        return
    fires = int(fire_ledger.get("fires", 0)) if fire_ledger is not None else None
    if fires is not None and fires > 0 and not records:
        # The hook demonstrably ran but its fire records are not readable
        # here; without replacement evidence either way, do not refuse a run
        # the hook itself witnessed.
        return
    if fires is None and not records:
        hooks_fired = ctx.get("hooks_fired")
        if isinstance(hooks_fired, int) and hooks_fired > 0:
            return
    if records:
        sites = sorted(
            {
                str(getattr(record, "site_label", None) or getattr(record, "target_label", ""))
                for record in records
            }
        )
        detail = (
            f"the hook fired {len(records)} time(s) but every fire was refused "
            f"(replaced=False) at site(s) {tuple(sites)!r}"
        )
    else:
        detail = "the hook never fired during the patched rerun"
    raise PatchApplicationError(
        f"Activation patch for {where} had no effect: {detail}. Publishing this row "
        "would silently report the corrupted baseline as a measured effect (a "
        "plausible-looking null result). This usually means the facet's home op is "
        "not a live-hookable site on this architecture -- for example the facet is "
        "anchored on an aliasing, mask-derived, or input-derived op. Remedy: inspect "
        "tl.facets.facet_coverage(trace) for this model, choose a facet whose home "
        "is a real computation site, or patch an explicit op site via fork.do().",
        code="patch_ineffective",
    )


def _model_input_home_role(log: Any, facet_name: str, address: str) -> str | None:
    """Return the recorded input address when a whole-tensor facet homes on a model input.

    Parameters
    ----------
    log:
        Trace holding the facet.
    facet_name:
        Facet to inspect.
    address:
        Module address exposing the facet.

    Returns
    -------
    str | None
        The home input op's hierarchical ``io_role`` address (for example
        ``"input.x.0.nested"``), or ``None`` when the home is a regular op
        (live hooks handle it) or no facet spec is reachable.

    Raises
    ------
    ValueError
        If a facet homed on a model input is not the identity view of the home
        tensor (patching the raw input would silently write outside the facet),
        or if the home input op cannot be bound to a recorded input address --
        falling back to the hook path would silently run an UNPATCHED
        counterfactual, because live hooks never fire at input sites.
    """

    try:
        spec = log.modules[address].facets[facet_name].spec
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return None
    home = getattr(spec, "home", None)
    if home is None or getattr(home, "layer_type", None) != "input":
        return None
    home_label = str(getattr(home, "label", ""))
    home_op = next(
        (op for op in getattr(log, "input_ops", ()) if str(op.label) == home_label),
        None,
    )
    if home_op is None:
        raise ValueError(
            f"Facet {facet_name!r} on module {address!r} homes on model input "
            f"{home_label!r}, which is not among this trace's input ops; the input "
            "patch cannot be bound and live hooks never fire at input sites."
        )
    if tuple(getattr(spec, "transforms", ()) or ()) or not bool(spec.write_mask().all()):
        raise ValueError(
            f"Facet {facet_name!r} on module {address!r} is a transformed or sliced view of "
            f"model input {home_label!r}. Sliced input facets cannot be patched: live hooks "
            "never fire at input sites, and whole-input replacement would write outside the "
            "facet."
        )
    io_role = getattr(home_op, "io_role", None)
    if not io_role:
        raise ValueError(
            f"Facet {facet_name!r} on module {address!r} homes on model input "
            f"{home_label!r}, but that op records no io_role input address; the input "
            "patch cannot be bound and live hooks never fire at input sites."
        )
    return str(io_role)


def _resolve_input_leaf_by_role(
    x: Any, model: nn.Module, io_role: str
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the leaf of ``x`` recorded at input address ``io_role``, plus all leaves.

    Mirrors capture-time input flattening exactly (see
    ``TorchBackend.fetch_label_move_input_tensors``): positional args are
    normalized against the model's forward signature, each arg's tensors are
    discovered with the same bounded traversal, addresses are
    ``input.<argname>[.<path>]``, and a repeated tensor object keeps only its
    first address -- so the recorded ``io_role`` binds by ADDRESS, never by
    ordinal position in some other walk order.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        The target leaf and every distinct tensor leaf of the tree.

    Raises
    ------
    ValueError
        If no leaf of ``x`` sits at the recorded address (fail closed: a
        silent fallback would rerun an unpatched counterfactual).
    """

    from ..backends.torch.backend import _get_input_arg_names
    from ..utils.arg_handling import normalize_input_args
    from ..utils.introspection import INPUT_SEARCH_DEPTH_LIMIT, get_vars_of_type_from_obj

    args = normalize_input_args(x, model)
    arg_names = _get_input_arg_names(model, args)
    target: torch.Tensor | None = None
    leaves: list[torch.Tensor] = []
    seen: set[int] = set()
    for arg, arg_name in zip(args, arg_names):
        for tensor, addr, _addr_full in get_vars_of_type_from_obj(
            arg,
            torch.Tensor,
            search_depth=INPUT_SEARCH_DEPTH_LIMIT,
            return_addresses=True,
        ):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            leaves.append(tensor)
            tensor_addr = f"input.{arg_name}" + (f".{addr}" if addr else "")
            if tensor_addr == io_role:
                target = tensor
    if target is None:
        raise ValueError(
            f"Input tree has no tensor leaf at recorded input address {io_role!r}; "
            "the input-homed facet patch cannot be bound to a rerun leaf."
        )
    return target, leaves


def _patch_input_leaf(
    x: Any,
    model: nn.Module,
    io_role: str,
    patch: Callable[[torch.Tensor], torch.Tensor],
) -> Any:
    """Return the input tree with the leaf at address ``io_role`` patched everywhere.

    The leaf is resolved by the capture-recorded hierarchical input address
    (``Op.io_role``), never by ordinal position: capture flattens inputs in
    BFS container order while a naive tree walk visits leaves in DFS order, so
    ordinal indexing silently patched the WRONG leaf on mixed-nesting inputs.
    Because capture dedupes a repeated tensor object into ONE input op,
    patching that op replaces the tensor at EVERY site it appears. Container
    types (Mapping subclasses, namedtuples, nested objects) are preserved by
    rebuilding through ``copy.deepcopy`` with every unpatched tensor leaf
    passed through by identity.

    Parameters
    ----------
    x:
        Model input tree exactly as passed to the baseline trace.
    model:
        Model whose forward signature determines positional-arg naming.
    io_role:
        Recorded hierarchical input address of the leaf to patch.
    patch:
        Callable producing the replacement tensor for the selected leaf.

    Returns
    -------
    Any
        Rebuilt input tree with the addressed leaf replaced at every site.

    Raises
    ------
    ValueError
        If no leaf sits at the recorded address, the patched tensor changes
        shape, or the tree cannot be rebuilt.
    """

    target, leaves = _resolve_input_leaf_by_role(x, model, io_role)
    patched = patch(target)
    if tuple(patched.shape) != tuple(target.shape):
        raise ValueError(
            f"Input patch changed the leaf shape from {tuple(target.shape)} to "
            f"{tuple(patched.shape)}; input patching must preserve shape."
        )
    memo: dict[int, Any] = {id(leaf): leaf for leaf in leaves}
    memo[id(target)] = patched
    try:
        return copy.deepcopy(x, memo)
    except Exception as exc:
        raise ValueError(
            f"Could not rebuild the input tree around patched leaf {io_role!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _modules_with_facet(log: Any, facet_name: str) -> list[str]:
    """Return module addresses with a facet available in the current capture."""

    return [
        str(module.address)
        for module in log.modules
        if getattr(module, "address", None) != "self" and module.facets.has(facet_name)
    ]


def _ensure_matching_modules(log: Any, modules: Sequence[str], *, facet_name: str) -> None:
    """Raise if a corrupted trace lacks an available clean-trace facet owner."""

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
