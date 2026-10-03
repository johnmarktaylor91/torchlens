"""Native layer-attribution methods for TorchLens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._core import (
    AttributionError,
    AttributionResult,
    InputKwargs,
    _AttributionTarget,
    _call_model,
    _completeness_extra,
    _interned_path_leaves,
    _make_input_leaves,
    _normalize_model_inputs,
    _PreparedInputs,
    _scalarize_output,
    _stacked_path_leaves,
    _target_repr,
    _temporarily_eval,
    _validate_baselines,
    _validate_positive_int,
)
from torchlens.attribution._steps import (
    _chunk_steps,
    _midpoint_alphas,
    _scalarize_stacked_output,
    _StepAuditor,
    _validate_step_audit,
    _validate_step_batch_size,
)
from torchlens.receptive_field._viz import _blend_heatmap
from torchlens.viz.node_plots import render_heatmap

LayerAttributionMethod: TypeAlias = Literal["activation_x_grad", "grad"]

# D21 (attrib memo): a layer-space completeness residual is expected near zero
# only when the selected layer output is a true bottleneck for the scored
# path; on non-bottleneck layers the residual measures the bypassed signal,
# not an implementation defect.
_LAYER_COMPLETENESS_CAVEAT = (
    "layer-space residual is expected near zero only when the selected layer "
    "output is a true bottleneck for the scored path"
)


@dataclass
class _LayerCapture:
    """Container for captured layer activations across every hook firing.

    Attributes
    ----------
    activations
        Distinct forward activations emitted by the target layer, one entry per
        distinct output tensor object, in firing order. A module that returns
        the very tensor object it received (``nn.Identity``) contributes one
        entry no matter how often it fires: autograd already accumulates every
        use of that node into one gradient, so a second entry would
        double-count its contribution.
    """

    activations: list[Tensor] = field(default_factory=list)

    def add(self, output: Any) -> None:
        """Record one hook firing.

        Parameters
        ----------
        output
            Value returned by the target layer for this firing.

        Raises
        ------
        AttributionError
            If the layer emitted a non-tensor output.
        """

        if not isinstance(output, Tensor):
            raise AttributionError("target layer must return a tensor activation")
        if not any(output is existing for existing in self.activations):
            self.activations.append(output)


def _layer_gradients(
    scalar: Tensor,
    activations: tuple[Tensor, ...],
    layer: str,
) -> tuple[Tensor, ...]:
    """Compute per-firing gradients of the scalar target for every captured activation.

    Firings that carry gradient tracking but provably do not reach the target
    contribute exact-zero gradients: the target does not depend on them, and
    zero is that statement, not a fallback. If NO firing reaches the target the
    request is a user error and the single-firing error contract is preserved.

    Parameters
    ----------
    scalar
        Scalarized model output.
    activations
        Distinct captured activations in firing order.
    layer
        Layer name used for error reporting.

    Returns
    -------
    tuple[Tensor, ...]
        Gradient of ``scalar`` with respect to each activation, in firing order.

    Raises
    ------
    AttributionError
        If no captured firing is differentiable or none reaches the target.
    """

    differentiable = [activation for activation in activations if activation.requires_grad]
    if not differentiable:
        raise AttributionError(
            f"layer {layer!r} activation is not differentiable with respect to target"
        )
    try:
        raw_gradients = torch.autograd.grad(scalar, differentiable, allow_unused=True)
    except RuntimeError as exc:
        raise AttributionError(
            f"target scalar is not differentiable with respect to layer {layer!r}"
        ) from exc
    if all(gradient is None for gradient in raw_gradients):
        raise AttributionError(
            f"target scalar is not differentiable with respect to layer {layer!r}"
        )
    gradient_by_id = {
        id(activation): gradient
        for activation, gradient in zip(differentiable, raw_gradients, strict=True)
    }
    return tuple(
        gradient
        if (gradient := gradient_by_id.get(id(activation))) is not None
        else torch.zeros_like(activation)
        for activation in activations
    )


def _validate_uniform_firing_shapes(activations: tuple[Tensor, ...], layer: str) -> None:
    """Require matching activation shapes across firings of a reused layer.

    Per-firing contributions are accumulated into one activation-shaped result,
    which is only well-defined when every firing produced the same shape.

    Parameters
    ----------
    activations
        Distinct captured activations in firing order.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If the layer produced differently shaped activations across firings.
    """

    shapes = {tuple(activation.shape) for activation in activations}
    if len(shapes) > 1:
        raise AttributionError(
            f"layer {layer!r} fired {len(activations)} times with mismatched activation "
            f"shapes {sorted(shapes)}; per-firing accumulation requires matching shapes"
        )


def _validate_matching_firings(
    reference: tuple[Tensor, ...],
    observed: tuple[Tensor, ...],
    layer: str,
) -> None:
    """Require consistent firing count and shapes across input-path points.

    Input-path layer methods pair the ``i``-th firing at one path point with the
    ``i``-th firing at every other; a count or shape change means the model took
    different control flow along the path and the pairing is meaningless.

    Parameters
    ----------
    reference
        Firing activations (or aligned gradients) at the reference path point.
    observed
        Firing activations (or aligned gradients) at another path point.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If the firing count or any per-firing shape differs.
    """

    if len(observed) != len(reference):
        raise AttributionError(
            f"layer {layer!r} fired {len(observed)} times at one input-path point and "
            f"{len(reference)} times at another; input-path layer attribution requires "
            "consistent control flow along the path"
        )
    for reference_item, observed_item in zip(reference, observed, strict=True):
        if tuple(reference_item.shape) != tuple(observed_item.shape):
            raise AttributionError(
                f"layer {layer!r} produced mismatched activation shapes across input-path "
                "points; input-path layer attribution requires consistent shapes"
            )


def _autograd_leaf_variable_ids(activations: tuple[Tensor, ...]) -> set[int]:
    """Collect ids of every autograd leaf tensor reachable from the activations.

    Parameters
    ----------
    activations
        Captured activations still connected to their autograd graphs.

    Returns
    -------
    set[int]
        ``id()`` values of leaf tensors (``AccumulateGrad`` variables) that are
        graph ancestors of any activation, plus the activations themselves so a
        passthrough layer that returns an input leaf unchanged still reports it.
    """

    found: set[int] = {id(activation) for activation in activations}
    # ``grad_fn.next_functions`` mints a FRESH Python wrapper around the
    # underlying autograd node on every access; nothing else keeps that
    # wrapper alive once it is popped off ``stack`` and only its ``id()`` is
    # retained. CPython is then free to recycle that exact address for the
    # NEXT node constructed during the same walk (observed on torch 2.7.1
    # for a 3+-hop chain such as Conv2d->ReLU->AvgPool2d: the freed
    # ``ReluBackward0`` wrapper's address was reused by the input leaf's own
    # ``AccumulateGrad`` wrapper), which makes an `id()`-keyed "seen" set
    # falsely treat the brand-new node as already visited and silently
    # prune the real leaf out of the walk. Keeping the node OBJECTS
    # themselves (not their bare ids) in ``seen`` fixes both problems at
    # once: it holds a strong reference for the rest of the traversal (no
    # address can be recycled out from under it) and still deduplicates
    # correctly, since these wrapper types use default identity-based
    # ``__eq__``/``__hash__``.
    seen: set[Any] = set()
    stack: list[Any] = [
        activation.grad_fn for activation in activations if activation.grad_fn is not None
    ]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        variable = getattr(node, "variable", None)
        if isinstance(variable, Tensor):
            found.add(id(variable))
        stack.extend(next_node for next_node, _ in node.next_functions)
    return found


def _feeding_original_leaves(
    activations: tuple[Tensor, ...],
    input_leaves: tuple[Tensor, ...],
    original_leaves: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Return the unique original input leaves that feed the captured activations.

    "Feeds" is autograd-graph ancestry of the substituted differentiable clone,
    not traversal order: an input that never reaches the target layer must not
    be treated as its coordinate system.

    Parameters
    ----------
    activations
        Captured activations still connected to their autograd graphs.
    input_leaves
        Substituted differentiable clones, slot-aligned with ``original_leaves``.
    original_leaves
        Original attributed leaves from the user's call.

    Returns
    -------
    tuple[Tensor, ...]
        Unique original leaves whose clones are graph ancestors of the
        activations, in traversal order.
    """

    reachable = _autograd_leaf_variable_ids(activations)
    feeding: list[Tensor] = []
    seen_clone_ids: set[int] = set()
    for clone, original in zip(input_leaves, original_leaves, strict=True):
        if id(clone) in seen_clone_ids:
            continue
        seen_clone_ids.add(id(clone))
        if id(clone) in reachable:
            feeding.append(original)
    return tuple(feeding)


def _format_layer_options(model: Module) -> str:
    """Return a compact list of useful layer-name suggestions.

    Parameters
    ----------
    model
        Model whose named modules should be searched.

    Returns
    -------
    str
        Comma-separated module names suitable for an error message.
    """

    named_modules = dict(model.named_modules())
    conv_like = [
        name
        for name, module in named_modules.items()
        if name and ("conv" in name.lower() or "conv" in type(module).__name__.lower())
    ]
    options = conv_like[:5]
    if not options:
        options = [name for name in named_modules if name][:5]
    if not options:
        return "<no named child modules>"
    return ", ".join(options)


def _resolve_named_layer(model: Module, layer: str) -> Module:
    """Resolve a user-specified module name.

    Parameters
    ----------
    model
        Model containing the target layer.
    layer
        Name from ``model.named_modules()``.

    Returns
    -------
    Module
        Resolved PyTorch module.

    Raises
    ------
    AttributionError
        If ``layer`` is not a named module string in ``model``.
    """

    if not isinstance(layer, str):
        raise AttributionError("layer must be a module name string")

    named_modules = dict(model.named_modules())
    if layer not in named_modules:
        options = _format_layer_options(model)
        raise AttributionError(
            f"layer {layer!r} was not found; available conv-like layers include: {options}"
        )
    return named_modules[layer]


def _capture_layer_activation(
    model: Module,
    inputs: _PreparedInputs,
    target: _AttributionTarget,
    layer: str,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Capture every distinct layer firing and its target gradient.

    A layer reused ``N`` times contributes ``N`` distinct activations; keeping
    only one would silently drop the other calls' contributions to the target.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name of the module whose activation should be captured.

    Returns
    -------
    tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]
        Detached per-firing activations, aligned per-firing gradients of the
        scalar target, and the unique original attributed input leaves that
        feed the captured activations through the autograd graph.

    Raises
    ------
    AttributionError
        If the target layer does not emit a differentiable tensor activation.
    """

    target_layer = _resolve_named_layer(model, layer)
    capture = _LayerCapture()
    hook_handles: list[RemovableHandle] = []

    def _forward_hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
        """Record the target layer output for each firing of the forward pass."""

        capture.add(output)

    hook_handles.append(target_layer.register_forward_hook(_forward_hook))
    try:
        with _temporarily_eval(model):
            input_leaves = _make_input_leaves(inputs)
            output = _call_model(model, inputs, input_leaves)
            activations = tuple(capture.activations)
            if not activations:
                raise AttributionError(f"layer {layer!r} did not run during the forward pass")
            if not any(activation.requires_grad for activation in activations):
                raise AttributionError(
                    f"layer {layer!r} activation is not differentiable with respect to target"
                )
            feeding_leaves = _feeding_original_leaves(
                activations, input_leaves, inputs.attributed_leaves
            )
            scalar = _scalarize_output(output, target)
            gradients = _layer_gradients(scalar, activations, layer)
    finally:
        for handle in hook_handles:
            handle.remove()

    return (
        tuple(activation.detach() for activation in activations),
        tuple(gradient.detach() for gradient in gradients),
        feeding_leaves,
    )


def _capture_layer_activation_for_leaves(
    model: Module,
    inputs: _PreparedInputs,
    target: _AttributionTarget,
    layer: str,
    input_leaves: tuple[Tensor, ...],
    *,
    require_gradient: bool,
    n_rows: int = 1,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None, Tensor]:
    """Capture every distinct layer firing and optionally the target gradients.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name of the module whose activation should be captured.
    input_leaves
        Differentiable leaves substituted into the model call.
    require_gradient
        Whether to compute ``dTarget / dActivation`` per firing.
    n_rows
        Number of path points stacked on the batch axis of ``input_leaves``;
        callable targets are applied per logical path point and summed.

    Returns
    -------
    tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None, Tensor]
        Detached per-firing activations, optional aligned per-firing gradients
        with respect to them, and the detached scalarized target value at this
        path point.

    Raises
    ------
    AttributionError
        If the target layer does not emit a differentiable tensor activation.
    """

    target_layer = _resolve_named_layer(model, layer)
    capture = _LayerCapture()
    hook_handles: list[RemovableHandle] = []

    def _forward_hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
        """Record the target layer output for each firing of the forward pass."""

        capture.add(output)

    hook_handles.append(target_layer.register_forward_hook(_forward_hook))
    try:
        output = _call_model(model, inputs, input_leaves, n_rows=n_rows)
        activations = tuple(capture.activations)
        if not activations:
            raise AttributionError(f"layer {layer!r} did not run during the forward pass")
        scalar = _scalarize_stacked_output(output, target, n_rows, _scalarize_output)
        if not require_gradient:
            return (
                tuple(activation.detach() for activation in activations),
                None,
                scalar.detach(),
            )
        if not any(activation.requires_grad for activation in activations):
            raise AttributionError(
                f"layer {layer!r} activation is not differentiable with respect to target"
            )
        gradients = _layer_gradients(scalar, activations, layer)
    finally:
        for handle in hook_handles:
            handle.remove()

    return (
        tuple(activation.detach() for activation in activations),
        tuple(gradient.detach() for gradient in gradients),
        scalar.detach(),
    )


def _split_stacked_firings(
    stacked: tuple[Tensor, ...],
    n_rows: int,
) -> list[tuple[Tensor, ...]]:
    """Split per-firing stacked tensors back into per-path-point firing tuples.

    Parameters
    ----------
    stacked
        Per-firing tensors whose leading dimension stacks ``n_rows`` path
        points.
    n_rows
        Number of stacked path points.

    Returns
    -------
    list[tuple[Tensor, ...]]
        Per-path-point tuples of per-firing tensors, in stacking order.
    """

    per_firing_rows = [tensor.chunk(n_rows, dim=0) for tensor in stacked]
    return [tuple(rows[row] for rows in per_firing_rows) for row in range(n_rows)]


def _validate_stacked_firings(
    reference: tuple[Tensor, ...],
    observed: tuple[Tensor, ...],
    layer: str,
    n_rows: int,
) -> None:
    """Require stacked firings to mirror the reference firing geometry.

    Parameters
    ----------
    reference
        Per-firing activations at an unstacked reference path point.
    observed
        Per-firing stacked tensors from one chunked path run.
    layer
        Layer name used for error reporting.
    n_rows
        Number of stacked path points.

    Raises
    ------
    AttributionError
        If the firing count or any per-firing stacked shape is inconsistent.
    """

    if len(observed) != len(reference):
        raise AttributionError(
            f"layer {layer!r} fired {len(observed)} times on a stacked path chunk and "
            f"{len(reference)} times at the path endpoints; step batching requires "
            "consistent control flow along the path. "
            "Remedy: run with step_batch_size=1.",
            code="step_batch_layer_firings_inconsistent",
        )
    for reference_item, observed_item in zip(reference, observed, strict=True):
        expected_shape = (reference_item.shape[0] * n_rows, *reference_item.shape[1:])
        if tuple(observed_item.shape) != expected_shape:
            raise AttributionError(
                f"layer {layer!r} produced a stacked activation of shape "
                f"{tuple(observed_item.shape)} where {expected_shape} was expected; "
                "the layer does not carry the stacked batch axis, so step batching "
                "cannot split its firings. Remedy: run with step_batch_size=1.",
                code="step_batch_layer_firings_inconsistent",
            )


def _layer_path_run(
    model: Module,
    inputs: _PreparedInputs,
    target: _AttributionTarget,
    layer: str,
    baseline_tensors: tuple[Tensor, ...],
    n_steps: int,
    *,
    chunk_size: int = 1,
    audit_mode: str = "off",
    audit_seed: int | None = None,
    capture_right_activations: bool = False,
) -> dict[str, Any]:
    """Capture endpoint activations and midpoint gradients along an input path.

    Every path point captures ALL distinct firings of the target layer. The
    firing count and per-firing shapes are validated to be consistent across
    path points (pairing firing ``i`` across points is otherwise meaningless)
    and uniform across firings (per-firing contributions are accumulated into
    one activation-shaped result). With ``chunk_size > 1`` path points are
    stacked on the ordinary batch axis (attrib memo D16) and the randomized
    audit ladder guards row coupling (D18).

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline_tensors
        Baseline leaves matching attributed inputs.
    n_steps
        Number of midpoint Riemann samples.
    chunk_size
        Validated number of path points stacked per forward/backward.
    audit_mode
        Resolved audit-ladder rung.
    audit_seed
        Optional deterministic seed for the audit's own (chunk, row) draw.
    capture_right_activations
        Whether to additionally capture the no-grad activations at the RIGHT
        interval edges ``(step + 1) / n_steps`` (layer conductance needs them).

    Returns
    -------
    dict[str, Any]
        ``baseline_activations``, ``input_activations``, ``gradients_by_step``,
        ``right_activations_by_step`` (``None`` unless requested), ``deltas``,
        ``endpoint_scalars`` (baseline, input), ``physical_forward_calls``,
        and the ``audit`` record.
    """

    deltas = tuple(
        input_leaf.detach() - baseline_tensor
        for input_leaf, baseline_tensor in zip(
            inputs.attributed_leaves, baseline_tensors, strict=True
        )
    )
    alphas = _midpoint_alphas(n_steps)
    chunks = _chunk_steps(alphas, chunk_size)
    auditor = _StepAuditor(
        audit_mode, len(chunks), [len(chunk) for chunk in chunks], seed=audit_seed
    )
    physical_calls = 2
    with _temporarily_eval(model):
        baseline_activations, _baseline_gradients, baseline_scalar = (
            _capture_layer_activation_for_leaves(
                model,
                inputs,
                target,
                layer,
                _interned_path_leaves(inputs, baseline_tensors, deltas, 0.0),
                require_gradient=False,
            )
        )
        _validate_uniform_firing_shapes(baseline_activations, layer)
        input_activations, _input_gradients, input_scalar = _capture_layer_activation_for_leaves(
            model,
            inputs,
            target,
            layer,
            _interned_path_leaves(inputs, baseline_tensors, deltas, 1.0),
            require_gradient=False,
        )
        _validate_matching_firings(baseline_activations, input_activations, layer)
        gradients_by_step: list[tuple[Tensor, ...]] = []
        for chunk_index, chunk_alphas in enumerate(chunks):
            if len(chunk_alphas) == 1:
                _activations, gradients, _scalar = _capture_layer_activation_for_leaves(
                    model,
                    inputs,
                    target,
                    layer,
                    _interned_path_leaves(inputs, baseline_tensors, deltas, chunk_alphas[0]),
                    require_gradient=True,
                )
                physical_calls += 1
                if gradients is None:
                    raise AttributionError(
                        "internal error: missing layer path gradient from a "
                        "require-gradient capture. This is a TorchLens contract "
                        "breach, not a user error. Remedy: report this as a bug.",
                        code="layer_path_gradient_missing",
                    )
                _validate_matching_firings(baseline_activations, gradients, layer)
                gradients_by_step.append(gradients)
                continue
            stacked_leaves = _stacked_path_leaves(inputs, baseline_tensors, deltas, chunk_alphas)
            _stacked_acts, stacked_gradients, _scalar = _capture_layer_activation_for_leaves(
                model,
                inputs,
                target,
                layer,
                stacked_leaves,
                require_gradient=True,
                n_rows=len(chunk_alphas),
            )
            physical_calls += 1
            if stacked_gradients is None:
                raise AttributionError(
                    "internal error: missing layer path gradient from a "
                    "require-gradient capture. This is a TorchLens contract "
                    "breach, not a user error. Remedy: report this as a bug.",
                    code="layer_path_gradient_missing",
                )
            _validate_stacked_firings(
                baseline_activations, stacked_gradients, layer, len(chunk_alphas)
            )
            chunk_gradients = _split_stacked_firings(stacked_gradients, len(chunk_alphas))
            audit_row = auditor.row_for_chunk(chunk_index)
            if audit_row is not None:
                _acts, sequential_gradients, _scalar = _capture_layer_activation_for_leaves(
                    model,
                    inputs,
                    target,
                    layer,
                    _interned_path_leaves(
                        inputs, baseline_tensors, deltas, chunk_alphas[audit_row]
                    ),
                    require_gradient=True,
                )
                physical_calls += 1
                if sequential_gradients is None:
                    raise AttributionError(
                        "internal error: missing layer path gradient from a "
                        "require-gradient capture. This is a TorchLens contract "
                        "breach, not a user error. Remedy: report this as a bug.",
                        code="layer_path_gradient_missing",
                    )
                auditor.check(
                    chunk_index,
                    audit_row,
                    chunk_gradients[audit_row],
                    sequential_gradients,
                )
            gradients_by_step.extend(chunk_gradients)
        right_activations_by_step: list[tuple[Tensor, ...]] | None = None
        if capture_right_activations:
            right_activations_by_step = []
            right_alphas = [(step + 1) / n_steps for step in range(n_steps)]
            for chunk_alphas in _chunk_steps(right_alphas, chunk_size):
                if len(chunk_alphas) == 1:
                    activations, _gradients, _scalar = _capture_layer_activation_for_leaves(
                        model,
                        inputs,
                        target,
                        layer,
                        _interned_path_leaves(inputs, baseline_tensors, deltas, chunk_alphas[0]),
                        require_gradient=False,
                    )
                    physical_calls += 1
                    _validate_matching_firings(baseline_activations, activations, layer)
                    right_activations_by_step.append(activations)
                    continue
                stacked_leaves = _stacked_path_leaves(
                    inputs, baseline_tensors, deltas, chunk_alphas
                )
                stacked_activations, _gradients, _scalar = _capture_layer_activation_for_leaves(
                    model,
                    inputs,
                    target,
                    layer,
                    stacked_leaves,
                    require_gradient=False,
                    n_rows=len(chunk_alphas),
                )
                physical_calls += 1
                _validate_stacked_firings(
                    baseline_activations, stacked_activations, layer, len(chunk_alphas)
                )
                right_activations_by_step.extend(
                    _split_stacked_firings(stacked_activations, len(chunk_alphas))
                )
    return {
        "baseline_activations": baseline_activations,
        "input_activations": input_activations,
        "gradients_by_step": gradients_by_step,
        "right_activations_by_step": right_activations_by_step,
        "deltas": deltas,
        "endpoint_scalars": (baseline_scalar, input_scalar),
        "physical_forward_calls": physical_calls,
        "audit": auditor.record(),
    }


def _spatial_reference_tensor(
    inputs: _PreparedInputs,
    feeding_leaves: tuple[Tensor, ...],
    layer: str,
) -> Tensor:
    """Return the input tensor whose grid the Grad-CAM should be upsampled onto.

    The CAM lives in the coordinate system of the input that actually feeds the
    target layer. Upsampling onto any other input would place the attribution
    in an unrelated coordinate space, so candidates are restricted to
    dependency-proven feeders and ambiguity between different grids is refused
    rather than resolved by traversal order.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    feeding_leaves
        Unique original attributed leaves proven to feed the target layer.
    layer
        Layer name used for error reporting.

    Returns
    -------
    Tensor
        Spatial (``ndim >= 4``) feeding leaf; when several feed the layer they
        must share one spatial grid.

    Raises
    ------
    AttributionError
        If no attributed leaf has spatial dimensions, no spatial leaf feeds the
        layer, or the feeding spatial grids are heterogeneous.
    """

    if not any(input_tensor.ndim >= 4 for input_tensor in inputs.attributed_leaves):
        raise AttributionError("grad_cam requires an input tensor with spatial dimensions")
    candidates = [input_tensor for input_tensor in feeding_leaves if input_tensor.ndim >= 4]
    if not candidates:
        raise AttributionError(
            f"grad_cam found no spatial (ndim >= 4) input tensor feeding layer {layer!r}; "
            "the CAM has no input coordinate system to upsample onto"
        )
    grids = {tuple(candidate.shape[-2:]) for candidate in candidates}
    if len(grids) > 1:
        raise AttributionError(
            f"grad_cam target layer {layer!r} is fed by spatial inputs with different "
            f"grids {sorted(grids)}; the upsampling target is ambiguous"
        )
    return candidates[0]


def _validate_conv_activation(activation: Tensor, layer: str) -> None:
    """Validate that an activation is a 2D convolution-style feature map.

    Parameters
    ----------
    activation
        Captured target-layer activation.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If ``activation`` is not shaped ``N, C, H, W``.
    """

    if activation.ndim != 4:
        raise AttributionError(
            f"grad_cam requires layer {layer!r} to produce a 4D N,C,H,W feature map; "
            f"got shape {tuple(activation.shape)}"
        )


def _cam_base_image(spatial_reference: Tensor, image: Image.Image | None) -> Image.Image:
    """Return an RGB base image for a CAM overlay.

    Parameters
    ----------
    spatial_reference
        Input tensor whose spatial grid was dependency-proven to feed the layer.
    image
        Optional user-provided source image.

    Returns
    -------
    PIL.Image.Image
        RGB image at the input tensor's spatial resolution.
    """

    height, width = (int(value) for value in spatial_reference.shape[-2:])
    if image is not None:
        return image.convert("RGB").resize((width, height))
    data = spatial_reference.detach().float().cpu()[0]
    if data.ndim == 3 and data.shape[0] in {1, 3, 4}:
        channels = data[:3]
        if channels.shape[0] == 1:
            channels = channels.expand(3, -1, -1)
        low = channels.amin()
        high = channels.amax()
        normalized = torch.zeros_like(channels) if high == low else (channels - low) / (high - low)
        array = (normalized.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
        return Image.fromarray(array, mode="RGB")
    reduced = data.abs().mean(dim=0) if data.ndim == 3 else data.abs()
    return render_heatmap(reduced.numpy(), width=width, height=height, cmap="gray")


def _cam_overlay(
    native_cam: Tensor,
    spatial_reference: Tensor,
    *,
    image: Image.Image | None,
    alpha: float,
    cmap: str,
) -> Image.Image:
    """Render CAM values through the receptive-field heatmap overlay path.

    Parameters
    ----------
    native_cam
        Channel-reduced map at the measured feature-map resolution.
    spatial_reference
        Dependency-proven input tensor defining the rendered grid.
    image
        Optional source image override.
    alpha
        Heatmap opacity.
    cmap
        Heatmap colormap.

    Returns
    -------
    PIL.Image.Image
        Overlay with a visible native-resolution disclosure footer.
    """

    if not 0.0 <= alpha <= 1.0:
        raise AttributionError("alpha must be between 0 and 1")
    base = _cam_base_image(spatial_reference, image)
    data = native_cam.detach().float().cpu().mean(dim=(0, 1))
    heatmap = render_heatmap(data.numpy(), width=base.width, height=base.height, cmap=cmap)
    native_height, native_width = (int(value) for value in native_cam.shape[-2:])
    disclosure = f"CAM native map: {native_height}x{native_width}; display interpolated"
    return _blend_heatmap(base, heatmap, alpha=alpha, disclosure=disclosure)


def grad_cam(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    layer: str,
    relu: bool = True,
    overlay: bool = True,
    image: Image.Image | None = None,
    alpha: float = 0.6,
    cmap: str = "magma",
) -> AttributionResult:
    """Compute Grad-CAM for a named convolution-style layer.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
        The layer must fire exactly once during the forward pass; Grad-CAM has
        no defined semantics for a reused layer, so multi-fire raises rather
        than silently attributing one arbitrary call.
    relu
        Whether to apply ReLU to the channel-reduced CAM.
    overlay
        Whether to render the CAM over the dependency-proven spatial input.
    image
        Optional source image override for the rendered overlay.
    alpha
        Heatmap opacity in ``[0, 1]``.
    cmap
        Heatmap colormap.

    Returns
    -------
    AttributionResult
        Grad-CAM values upsampled to the spatial size of the input that feeds
        the target layer, with shape ``N, 1, Hin, Win``. Metadata reports the
        measured native map resolution separately from the interpolated display
        resolution. When requested, ``extra["overlay"]`` is a PIL image whose
        visible footer discloses that native resolution.

    Raises
    ------
    AttributionError
        If the target layer fired more than once, no spatial input feeds it,
        or several feeding spatial inputs have different grids.
    """

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    activations, gradients, feeding_leaves = _capture_layer_activation(
        model, prepared_inputs, target, layer
    )
    if len(activations) != 1:
        raise AttributionError(
            f"grad_cam requires layer {layer!r} to fire exactly once during the forward "
            f"pass; it fired {len(activations)} times"
        )
    activation, gradient = activations[0], gradients[0]
    _validate_conv_activation(activation, layer)
    spatial_reference = _spatial_reference_tensor(prepared_inputs, feeding_leaves, layer)
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (channel_weights * activation).sum(dim=1, keepdim=True)
    if relu:
        cam = torch.relu(cam)
    native_resolution = tuple(int(value) for value in cam.shape[-2:])
    rendered_resolution = tuple(int(value) for value in spatial_reference.shape[-2:])
    upsampled_cam = F.interpolate(
        cam,
        size=spatial_reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return AttributionResult(
        method="grad_cam",
        values=upsampled_cam.detach(),
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "relu": relu,
            "native_map_resolution": native_resolution,
            "rendered_map_resolution": rendered_resolution,
            "upsampling": "bilinear_display_only",
            "overlay": (
                _cam_overlay(
                    cam,
                    spatial_reference,
                    image=image,
                    alpha=alpha,
                    cmap=cmap,
                )
                if overlay
                else None
            ),
        },
    )


def layer_integrated_gradients(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    layer: str,
    baseline: Any | None = None,
    n_steps: int = 50,
    step_batch_size: int | None = None,
    step_audit: str | None = None,
    step_audit_seed: int | None = None,
) -> AttributionResult:
    """Compute Layer Integrated Gradients for a named intermediate layer.

    The input path is the straight line from baseline leaves to input leaves.
    The returned values have the same shape as the target layer activation and
    use the Captum-style ``(A(input) - A(baseline)) * mean(dTarget / dA)`` rule.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    n_steps
        Number of midpoint Riemann samples along the straight input path.
    step_batch_size
        Optional number of path points stacked per forward/backward; ``None``
        (default) and ``1`` run sequentially. Strictly opt-in throughput.
    step_audit
        Audit-ladder rung when batching is on (``"per_call"`` default,
        ``"per_chunk"``, or the explicit expert ``"off"``).
    step_audit_seed
        Optional deterministic seed for the audit's own (chunk, row) draw;
        ``None`` draws a fresh disclosed seed riding ``extra["step_audit"]``.

    Returns
    -------
    AttributionResult
        Layer attribution values with the same shape as the captured activation.
        ``extra`` carries the layer-space completeness fields with the
        bottleneck caveat, plus cost and audit disclosure.
    """

    _validate_positive_int("n_steps", n_steps)
    chunk_size = _validate_step_batch_size(step_batch_size)
    audit_mode = _validate_step_audit(step_audit, chunk_size)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    run = _layer_path_run(
        model,
        prepared_inputs,
        target,
        layer,
        baseline_tensors,
        n_steps,
        chunk_size=chunk_size,
        audit_mode=audit_mode,
        audit_seed=step_audit_seed,
    )
    baseline_activations = run["baseline_activations"]
    input_activations = run["input_activations"]
    gradients_by_step = run["gradients_by_step"]
    # A layer reused N times contributes through every firing; the honest total
    # sums the per-firing (activation delta) x (mean path gradient) terms.
    values = torch.zeros_like(baseline_activations[0])
    for firing_index, (baseline_activation, input_activation) in enumerate(
        zip(baseline_activations, input_activations, strict=True)
    ):
        mean_gradient = torch.stack(
            [step_gradients[firing_index] for step_gradients in gradients_by_step], dim=0
        ).mean(dim=0)
        values = values + (input_activation - baseline_activation) * mean_gradient
    values = values.detach()
    baseline_scalar, input_scalar = run["endpoint_scalars"]
    target_delta = input_scalar - baseline_scalar
    return AttributionResult(
        method="layer_integrated_gradients",
        values=values,
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "n_steps": n_steps,
            "path_evaluations_logical": n_steps,
            "physical_forward_calls": run["physical_forward_calls"],
            "step_batch_size": chunk_size,
            "step_audit": run["audit"].to_extra(),
            "completeness_caveat": _LAYER_COMPLETENESS_CAVEAT,
            **_completeness_extra(values.sum(), target_delta),
        },
    )


def layer_conductance(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    layer: str,
    baseline: Any | None = None,
    n_steps: int = 50,
    step_batch_size: int | None = None,
    step_audit: str | None = None,
    step_audit_seed: int | None = None,
) -> AttributionResult:
    """Compute Layer Conductance for a named intermediate layer.

    Conductance decomposes input Integrated Gradients onto hidden units by
    integrating ``(dTarget / dA) * (dA / dalpha)`` along the input path. This
    implementation uses midpoint layer gradients and finite activation
    differences for each path interval.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    n_steps
        Number of midpoint Riemann samples along the straight input path.
    step_batch_size
        Optional number of path points stacked per forward/backward; ``None``
        (default) and ``1`` run sequentially. Strictly opt-in throughput.
    step_audit
        Audit-ladder rung when batching is on (``"per_call"`` default,
        ``"per_chunk"``, or the explicit expert ``"off"``).
    step_audit_seed
        Optional deterministic seed for the audit's own (chunk, row) draw;
        ``None`` draws a fresh disclosed seed riding ``extra["step_audit"]``.

    Returns
    -------
    AttributionResult
        Layer conductance values with the same shape as the captured activation.
        ``extra`` carries the layer-space completeness fields with the
        bottleneck caveat, plus cost and audit disclosure.
    """

    _validate_positive_int("n_steps", n_steps)
    chunk_size = _validate_step_batch_size(step_batch_size)
    audit_mode = _validate_step_audit(step_audit, chunk_size)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    run = _layer_path_run(
        model,
        prepared_inputs,
        target,
        layer,
        baseline_tensors,
        n_steps,
        chunk_size=chunk_size,
        audit_mode=audit_mode,
        audit_seed=step_audit_seed,
        capture_right_activations=True,
    )
    baseline_activations = run["baseline_activations"]
    gradients_by_step = run["gradients_by_step"]
    right_activations_by_step = run["right_activations_by_step"]
    if right_activations_by_step is None:
        raise AttributionError(
            "internal error: missing conductance right-edge activations from a "
            "capture_right_activations=True path run. This is a TorchLens "
            "contract breach, not a user error. Remedy: report this as a bug.",
            code="layer_conductance_edge_missing",
        )
    # A layer reused N times contributes through every firing; the honest total
    # sums the per-firing gradient x (activation interval) terms.
    activations_left = baseline_activations
    conductance = torch.zeros_like(baseline_activations[0])
    for gradients, activations_right in zip(
        gradients_by_step, right_activations_by_step, strict=True
    ):
        for gradient, activation_right, activation_left in zip(
            gradients, activations_right, activations_left, strict=True
        ):
            conductance = conductance + gradient * (activation_right - activation_left)
        activations_left = activations_right
    conductance = conductance.detach()
    baseline_scalar, input_scalar = run["endpoint_scalars"]
    target_delta = input_scalar - baseline_scalar
    return AttributionResult(
        method="layer_conductance",
        values=conductance,
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "n_steps": n_steps,
            "path_evaluations_logical": n_steps,
            "physical_forward_calls": run["physical_forward_calls"],
            "step_batch_size": chunk_size,
            "step_audit": run["audit"].to_extra(),
            "completeness_caveat": _LAYER_COMPLETENESS_CAVEAT,
            **_completeness_extra(conductance.sum(), target_delta),
        },
    )


def layer_attribution(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    layer: str,
    method: LayerAttributionMethod = "activation_x_grad",
) -> AttributionResult:
    """Compute attribution for a named intermediate layer.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    method
        Layer attribution method. ``"activation_x_grad"`` returns
        ``activation * gradient``. ``"grad"`` returns ``abs(gradient)``. A
        layer reused ``N`` times contributes through every firing, so both
        methods sum the per-firing terms.

    Returns
    -------
    AttributionResult
        Layer-attribution values with the same shape as the captured activation.

    Raises
    ------
    AttributionError
        If ``method`` is unsupported, or a reused layer produced mismatched
        activation shapes across firings.
    """

    if method not in ("activation_x_grad", "grad"):
        raise AttributionError("method must be 'activation_x_grad' or 'grad'")

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    activations, gradients, _feeding_leaves = _capture_layer_activation(
        model, prepared_inputs, target, layer
    )
    _validate_uniform_firing_shapes(activations, layer)
    values = torch.zeros_like(activations[0])
    for activation, gradient in zip(activations, gradients, strict=True):
        if method == "activation_x_grad":
            values = values + activation * gradient
        else:
            values = values + gradient.abs()
    return AttributionResult(
        method=f"layer_{method}",
        values=values.detach(),
        target_repr=_target_repr(target),
        extra={"layer": layer},
    )
