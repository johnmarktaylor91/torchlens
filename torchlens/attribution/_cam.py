"""Native Grad-CAM for TorchLens: feature maps, token grids and integer-input grids."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._cam_choice import _choose_cam_layer
from torchlens.attribution._cam_grid import (
    TokenReshape,
    _cam_feature_maps,
    _metadata_trace,
    _TraceGraph,
)
from torchlens.attribution._core import (
    _DOCS_POINTER as _INTEGER_DOCS_POINTER,
    AttributionError,
    AttributionResult,
    InputKwargs,
    _AttributionTarget,
    _normalize_model_inputs,
    _PreparedInputs,
    _target_repr,
)
from torchlens.attribution._entry import _EntryRoute
from torchlens.attribution._layer import (
    _cam_overlay,
    _capture_layer_activation,
    _integer_input_layers,
)


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

    if inputs.entry is not None:
        return _integer_spatial_reference(inputs.entry, feeding_leaves, layer)
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


def _integer_spatial_reference(
    entry: _EntryRoute,
    feeding_leaves: tuple[Tensor, ...],
    layer: str,
) -> Tensor:
    """Return the integer input whose last two axes are the grid of an integer-input Grad-CAM.

    Integer inputs laid out on a grid (token ids ``N, H, W``, or an integer
    image ``N, C, H, W``) carry the CAM's coordinate system. Candidates are the
    integer inputs consumed by the embeddings that feed the layer; with no
    embedding in the route every integer input with a grid is a candidate.

    Raises
    ------
    AttributionError
        If no candidate has at least three axes, or candidates disagree on
        the grid.
    """

    pool = entry.sources_of(feeding_leaves) if entry.sites else entry.integer_leaves
    candidates = [leaf for leaf in pool if leaf.ndim >= 3]
    grids = {tuple(candidate.shape[-2:]) for candidate in candidates}
    if len(grids) != 1:
        raise AttributionError(
            f"grad_cam found no single grid among the integer inputs feeding layer {layer!r} "
            f"(grids {sorted(grids)}); a CAM is laid on integer inputs shaped N, H, W or "
            "N, C, H, W. Remedy: attribute the layer without a map, e.g. "
            f"tl.attribution.layer_attribution(model, inputs, target=target, layer={layer!r}). "
            f"{_INTEGER_DOCS_POINTER}",
            code="grad_cam_integer_grid_invalid",
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


def grad_cam(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    layer: str | None = None,
    reshape: TokenReshape | None = None,
    relu: bool = True,
    overlay: bool = True,
    image: Image.Image | None = None,
    alpha: float = 0.6,
    cmap: str = "magma",
) -> AttributionResult:
    """Compute Grad-CAM for a convolution-style or token-grid layer.

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
        selected values, a callable ``output -> scalar tensor``, or a
        :class:`SiteTarget` naming a hidden layer's output (and unit) to
        attribute instead of the model output.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
        The layer must fire exactly once during the forward pass; Grad-CAM has
        no defined semantics for a reused layer, so multi-fire raises rather
        than silently attributing one arbitrary call. ``None`` chooses the layer
        from a metadata-only trace of the same call: the deepest once-firing
        module whose output is a spatial map or a provable token grid, lies on
        every input-to-output path, and carries a nonzero target gradient on
        its grid. ``extra["layer_source"]`` and ``extra["layer_reason"]``
        record the choice.
    reshape
        Optional callable mapping the layer output to ``N, C, H, W``. It must
        be a pure rearrangement (slicing, transposing, reshaping), because it
        is applied to the activation and to its gradient alike. Without it, a
        token-sequence output ``N, T, C`` is laid on the grid that the trace
        proves (the patch-embedding grid and the leading class, distillation
        or register tokens, or a Swin stage's grid), and the call refuses
        with ``grad_cam_token_grid_unprovable`` when the trace proves none.
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
        visible footer discloses that native resolution. ``extra["grid_route"]``
        is ``"feature_map"``, ``"derived"`` or ``"reshape"``,
        ``extra["prefix_tokens"]`` the count of leading tokens dropped, and
        ``extra["grid_evidence"]`` the trace ops that proved a derived grid.

    Raises
    ------
    AttributionError
        If the target layer fired more than once, no spatial input feeds it,
        several feeding spatial inputs have different grids, a token grid is
        not provable, or no layer can be chosen automatically.
    """

    prepared_inputs = _normalize_model_inputs(
        inputs, input_kwargs, model=model, target=target, method="grad_cam", integer_route=True
    )
    graph: _TraceGraph | None = None
    layer_reason: str | None = None
    if layer is None:
        graph = _TraceGraph(_metadata_trace(model, inputs, input_kwargs))
        choice = _choose_cam_layer(model, prepared_inputs, target, graph)
        layer, layer_reason = choice.layer, choice.reason
    activations, gradients, feeding_leaves = _capture_layer_activation(
        model, prepared_inputs, target, layer
    )
    if len(activations) != 1:
        raise AttributionError(
            f"grad_cam requires layer {layer!r} to fire exactly once during the forward "
            f"pass; it fired {len(activations)} times"
        )
    activation, gradient = activations[0], gradients[0]
    mapped = _cam_feature_maps(
        layer,
        (activation, gradient),
        target,
        reshape,
        lambda: (
            graph
            if graph is not None
            else _TraceGraph(_metadata_trace(model, inputs, input_kwargs))
        ),
    )
    if mapped is None:
        _validate_conv_activation(activation, layer)
        grid_extra: dict[str, Any] = {
            "grid_route": "feature_map",
            "prefix_tokens": 0,
            "grid_evidence": (),
        }
    else:
        activation, gradient, grid_extra = mapped
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
            "integer_input_layers": _integer_input_layers(prepared_inputs),
            "layer_source": "caller" if layer_reason is None else "trace",
            "layer_reason": layer_reason,
            **grid_extra,
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
