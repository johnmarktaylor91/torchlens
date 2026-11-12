"""Prebuilt activation and attribution patching helpers for facets."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .. import _user_public_impls
from .._selection.selectors import facet
from ..errors._base import TorchLensError, TorchLensWarning
from ..options import _CAPTURE_FIELDS, CaptureOptions
from ._input_patching import _model_input_home_role, _patch_input_leaf, _rerun_with_inputs
from ._state_guard import _CounterfactualStateGuard, _teardown
from .facets import Facet, MissingGrad
from .recipes._helpers import modules_with_facet as _modules_with_facet

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
                    ctx: Any,
                    position: Any = position,
                    clean_value: torch.Tensor = clean_value,
                ) -> torch.Tensor:
                    """Return ``out`` with one position replaced by the clean activation."""

                    del ctx
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
        P3 facet convention ``[batch, pos, head, d_model]``. ``result`` is a
        computed facet (``z_h @ W_O_h``) that no captured op holds, so the
        patch writes the head's ``z`` instead, which replaces exactly its
        ``result``. Under fused attention (``attn_implementation="sdpa"``)
        ``z`` is reconstructed read-only and the helper refuses with the
        eager-attention remedy; the pre-kernel ``"q"``/``"k"``/``"v"`` facets
        stay writable there.
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
        P3 facet convention ``[batch, pos, head, d_model]``. ``result`` is a
        computed facet (``z_h @ W_O_h``) that no captured op holds, so the
        patch writes the head's ``z`` instead, which replaces exactly its
        ``result``. Under fused attention (``attn_implementation="sdpa"``)
        ``z`` is reconstructed read-only and the helper refuses with the
        eager-attention remedy; the pre-kernel ``"q"``/``"k"``/``"v"`` facets
        stay writable there.
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
        clean_log.backward(clean_metric)
        corrupted_log.backward(corrupted_metric)
        modules = _modules_with_facet(clean_log, facet_name)
        _ensure_matching_modules(corrupted_log, modules, facet_name=facet_name)
        if not modules:
            raise ValueError(f"No modules expose facet {facet_name!r}.")
        return _attribution_patch_heads(
            clean_log, corrupted_log, modules, facet_name=facet_name, like=corrupted_metric
        )
    finally:
        _teardown(guard, clean_log, corrupted_log)


def _attribution_patch_heads(
    clean_log: Any,
    corrupted_log: Any,
    modules: Sequence[str],
    *,
    facet_name: str,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return the ``[layer, head]`` gradient-times-difference grid for ``facet_name``.

    Each cell is ``(grad * (clean - corrupted)).sum()`` over one head's facet
    slice at one module, with the corrupted run's gradient; the grid takes
    ``like``'s dtype and device.
    """

    n_heads = _num_heads(clean_log.modules[modules[0]], facet_name=facet_name)
    result = torch.empty((len(modules), n_heads), dtype=like.dtype, device=like.device)
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
    save_grads: bool | None = None,
    guard: _CounterfactualStateGuard,
) -> tuple[Any, Any]:
    """Return clean and corrupted traces with all layer activations saved.

    Both baselines start from the guard's pristine model/RNG snapshot so the
    corrupted run is a true counterfactual of the clean run rather than a run on
    a model already mutated by the clean forward.
    """

    kwargs = _trace_kwargs(trace_kwargs, save_grads=save_grads)
    guard.forward_kwargs = kwargs.get("input_kwargs")

    def _capture(inputs: Any) -> Callable[[], Any]:
        """Return a zero-argument capture that traces ``model`` on ``inputs``."""

        def _run() -> Any:
            """Reset the model/RNG state to the pristine snapshot and trace once."""
            guard.reset()
            return _user_public_impls.trace(model, inputs, **kwargs)  # looked up per call

        return _run

    clean_log = _capture_by_reference(model, guard, _capture(clean_input))
    try:
        corrupted_log = _capture_by_reference(model, guard, _capture(corrupted_input))
    except BaseException:
        clean_log.cleanup()
        raise
    return clean_log, corrupted_log


def _capture_by_reference(
    model: nn.Module, guard: _CounterfactualStateGuard, capture: Callable[[], Any]
) -> Any:
    """Run one helper capture with the model's parameter arguments kept by reference.

    ``save_args=True`` (required above) used to copy every weight argument into every helper trace;
    with a clean, a corrupted and a patched trace alive at once that was three extra copies of the
    weights (GPT-2's tied 50257 x 768 embedding twice per trace: the lookup and ``lm_head``). The
    helper traces never leave the helper, so a parameter argument is saved as an alias of the
    parameter's storage instead (see
    :func:`~torchlens._validation_argument_references.helper_parameter_references`). If the forward
    wrote a referenced parameter (its version counter moved, or its bytes left the guard's snapshot,
    which also catches ``param.data`` writes), the aliases no longer hold the call-time values: the
    trace is dropped and the capture runs again on the copy path. ``capture`` must start from
    :meth:`_CounterfactualStateGuard.reset` so both attempts run from the same state.
    """

    from .._validation_argument_references import helper_parameter_references

    with helper_parameter_references(model) as references:
        log = capture()
    if references.first_change() is None and not guard.parameters_drifted():
        return log
    log.cleanup()
    return capture()


def _trace_kwargs(
    trace_kwargs: Mapping[str, Any] | None,
    *,
    save_grads: bool | None = None,
) -> dict[str, Any]:
    """Return trace keyword arguments with the helper's REQUIRED fields applied.

    The patching helpers cannot work without ``save="all"`` (facet
    values must be readable on every candidate module) and
    ``save_args=True`` (reconstructed facets read saved op args), and
    attribution additionally needs ``save_grads``. These used to be supplied
    only when the caller passed NO ``capture=`` at all -- any user
    ``capture=`` silently dropped every one of them, so facets came up
    partially absent and grids were computed over a silently narrowed module
    set. The required fields now COMPOSE with the user's options: fields the
    user left unspecified are filled in, an explicitly matching value passes,
    and an explicitly CONFLICTING ``save``/``save_args``
    refuses with the requirement named (an explicit user ``save_grads``
    predicate is honored as-is -- a too-narrow one fails loudly at the
    gradient read). Flat capture spellings (``layers_to_save=`` etc.) were
    REMOVED from ``tl.trace``; this helper used to forward them verbatim into
    a guaranteed ``TypeError``, so it now refuses them with the grouped
    spelling named.
    """

    kwargs = dict(trace_kwargs or {})
    stale_flat = sorted(name for name in kwargs if name != "capture" and name in _CAPTURE_FIELDS)
    if stale_flat:
        raise ValueError(
            f"trace_kwargs contains removed flat capture kwargs {stale_flat!r}; tl.trace "
            "accepts capture knobs only through the grouped spelling "
            "capture=tl.options.CaptureOptions(...). Move these fields into capture=."
        )
    kwargs["capture"] = _compose_required_capture(
        kwargs.get("capture", CaptureOptions()),
        save_grads=save_grads,
        _save=kwargs.get("save", "all"),
    )
    kwargs["save"] = "all"
    return kwargs


def _compose_required_capture(capture: Any, *, save_grads: bool | None, _save: Any = "all") -> Any:
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
    required: dict[str, Any] = {"save_args": True}
    conflicts = [
        field_name
        for field_name, required_value in required.items()
        if capture._is_field_explicit(field_name) and getattr(capture, field_name) != required_value
    ]
    if not isinstance(_save, str) or _save != "all":
        conflicts.append("save")
    if conflicts:
        raise ValueError(
            f"Patching helpers require save='all' and capture options {required!r}, but the supplied "
            f"trace kwargs explicitly set {conflicts!r} to conflicting values. Facet values "
            "must be readable on every candidate module (save='all') and "
            "reconstructed facets read saved op args (save_args=True); a narrower "
            "capture silently shrinks or empties the patch table. Drop these fields from "
            "capture= (they are filled in automatically) or set them to the required "
            "values."
        )
    fills = {
        field_name: required_value
        for field_name, required_value in required.items()
        if not capture._is_field_explicit(field_name)
    }
    if save_grads is not None and not capture._is_field_explicit("save_grads"):
        fills["save_grads"] = save_grads
    if not fills:
        return capture
    values = capture.as_dict()
    values.update(fills)
    return CaptureOptions._from_values(
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
            out: torch.Tensor, *, ctx: Any, clean_value: torch.Tensor = clean_value
        ) -> torch.Tensor:
            """Return the clean activation for this facet slice."""

            del out, ctx
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
    facet_name = _head_write_facet(clean_log, modules, facet_name)
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
                out: torch.Tensor, *, ctx: Any, clean_value: torch.Tensor = clean_value
            ) -> torch.Tensor:
                """Return the clean activation for this head facet slice."""

                del out, ctx
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


def _head_write_facet(log: Any, modules: Sequence[str], facet_name: str) -> str:
    """Return the facet the per-head patch writes for ``facet_name``.

    ``result`` (a head's contribution to the attention output,
    ``z_h @ W_O_h``) is a COMPUTED facet: no captured op holds one head's
    projected output, so it is never writable, on eager attention too. It is
    linear in the head's ``z`` with the model's own fixed ``W_O`` slice, and
    the output bias is added after the heads are summed, so replacing head
    ``h``'s ``z`` with its clean value replaces exactly its ``result`` with the
    clean ``result``. The patch therefore writes ``z`` whenever ``result`` is
    read-only and every module exposes ``z``. Any other facet (and ``result``
    where it is writable) is written as named; under fused attention ``z`` is
    itself read-only, and the write refuses with the eager-attention remedy.
    """

    if facet_name != "result":
        return facet_name
    if all(_head_facet_writable(log.modules[address], "result") for address in modules):
        return facet_name
    if all(log.modules[address].facets.has("z") for address in modules):
        return "z"
    return facet_name


def _head_facet_writable(module: Any, facet_name: str) -> bool:
    """Return whether head 0's ``facet_name`` facet on ``module`` is write-safe."""

    try:
        value = module.facets.head(0)[facet_name]
        spec = value.spec if isinstance(value, Facet) else value
        spec._validate_intervention_safe(alias_policy=None)
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return False
    return True


def _counting_hook(
    hook: Callable[..., torch.Tensor], fire_ledger: dict[str, int]
) -> Callable[..., torch.Tensor]:
    """Wrap a patch hook so ``fire_ledger`` counts its fires and value-identical fires."""

    def _tracked_hook(
        out: torch.Tensor, *, ctx: Any, _inner: Callable[..., torch.Tensor] = hook
    ) -> torch.Tensor:
        """Run the patch hook while counting fires and value-identical fires."""

        result = _inner(out, ctx=ctx)
        fire_ledger["fires"] += 1
        if _replaced_identical(result, out):
            fire_ledger["identical"] += 1
        return result

    return _tracked_hook


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

    The model/RNG state is reset to the pristine snapshot before the rerun so the only difference
    between the corrupted baseline and this patched run is the injected clean activation. Every
    hook-path rerun must leave positive fire evidence (at least one ``replaced=True`` fire record);
    a run without it raises :class:`PatchApplicationError` instead of returning a trace whose metric
    would silently equal the corrupted baseline. ``where`` overrides the refusal's site description
    (used by the per-head caller, which must not enable the whole-facet input-patch route).

    Live hooks fire at wrapped-function and module-boundary sites only, so a facet homed on a MODEL
    INPUT op (e.g. ``resid_pre`` of a first block) has no site where a hook could ever fire. When
    the caller identifies the facet (``facet_name`` + ``address``) and its home is a model input,
    the patch is applied to the input tensor itself and the rerun proceeds without a hook —
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

    tracked_hook = _counting_hook(hook, fire_ledger)

    def _capture() -> Any:
        """Fork the corrupted trace, attach the counting hook, and rerun from the pristine state."""
        fire_ledger.update(fires=0, identical=0)
        log = corrupted_log.fork(name)
        log._attach_hooks(selector, tracked_hook)
        guard.reset()
        return _rerun_with_inputs(log, model, corrupted_input, guard.forward_kwargs)

    patched_log = _capture_by_reference(model, guard, _capture)
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

        result = hook(leaf.detach().clone(), ctx=None)
        fire_ledger["fires"] += 1
        if _replaced_identical(result, leaf):
            fire_ledger["identical"] += 1
        return result

    patched_input = _patch_input_leaf(corrupted_input, model, input_role, _tracked_input_patch)

    def _capture() -> Any:
        """Fork the corrupted trace and rerun it on the patched input from the pristine state."""
        log = corrupted_log.fork(name)
        guard.reset()
        return _rerun_with_inputs(log, model, patched_input, guard.forward_kwargs)

    return _capture_by_reference(model, guard, _capture)


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
                "tl.facets.coverage(trace) and re-anchor the facet before "
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
        for layer in patched_log._layer_list or []:
            for record in getattr(layer, "edits", []) or []:
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
        "tl.facets.coverage(trace) for this model, choose a facet whose home "
        "is a real computation site, or patch an explicit op site via fork.do().",
        code="patch_ineffective",
    )


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
    if isinstance(grad, MissingGrad):
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
