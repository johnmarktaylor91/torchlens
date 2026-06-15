"""Trace-aware activation feature-map node visualizations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
import tempfile
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from PIL import Image, ImageDraw

from .node_plots import _apply_colormap, _normalize_finite
from ..visualization.node_spec import NodeSpec, NodeSpecFn

FeatureMapEvolution: TypeAlias = "OrderedDict[str, torch.Tensor]"
FeatureMapChannels: TypeAlias = Sequence[int] | Literal["top"] | None
FeatureMapReduce: TypeAlias = Literal["mean", "max"]

_MODE_AGGREGATE = 0
_MODE_EXPLICIT = 1
_MODE_TOP = 2
_REDUCE_MEAN = 0
_REDUCE_MAX = 1
_TEXT_COLOR = (25, 28, 34)
_LABEL_FILL = (255, 255, 255)
_LABEL_OUTLINE = (95, 103, 117)


def feature_map_evolution(
    trace: Any,
    save: Any | None = None,
    *,
    stimuli: int | Sequence[int] | None = None,
    channels: FeatureMapChannels = None,
    top_k: int = 4,
    reduce: FeatureMapReduce = "mean",
    max_stimuli: int = 4,
    max_channels: int = 4,
) -> FeatureMapEvolution:
    """Compute and store per-site spatial activation maps for graph nodes.

    Parameters
    ----------
    trace:
        Captured TorchLens trace with saved activation payloads.
    save:
        Optional selector limiting which layers or pass-qualified ops to process.
    stimuli:
        Stimulus selection. ``None`` selects the first ``max_stimuli`` items,
        an integer selects the first that many items, and a sequence selects
        explicit indices.
    channels:
        Channel selection. ``None`` stores one channel-aggregate map per
        stimulus, a sequence stores explicit channels, and ``"top"`` stores
        per-stimulus top channels by mean absolute activation.
    top_k:
        Number of top channels to consider for ``channels="top"`` before the
        ``max_channels`` display cap is applied.
    reduce:
        Channel aggregation reduction for ``channels=None``: ``"mean"`` or
        ``"max"``.
    max_stimuli:
        Maximum number of stimuli to store.
    max_channels:
        Maximum number of explicit or top channels to store.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Stored map tensors keyed by ``layer:<label>`` or ``op:<label>``.

    Raises
    ------
    ValueError
        If caps, selectors, or reductions are invalid, or if no selected site
        has saved spatial ``[N, C, H, W]`` activations.
    """

    _validate_feature_map_options(
        top_k=top_k,
        reduce=reduce,
        max_stimuli=max_stimuli,
        max_channels=max_channels,
    )

    from ..repgeom import _selected_mds_sites, _store_annotation_tensor

    selected = _selected_mds_sites(trace, save)
    maps_by_key: FeatureMapEvolution = OrderedDict()
    non_spatial_shapes: list[str] = []
    for key, _site, activations in selected:
        tensor = _as_cpu_float_tensor(activations)
        if tensor.ndim != 4:
            non_spatial_shapes.append(f"{key}: {tuple(tensor.shape)}")
            continue

        stimulus_indices = _resolve_stimuli(stimuli, total_stimuli=tensor.shape[0], cap=max_stimuli)
        maps, channel_ids, mode_id = _select_maps(
            tensor,
            stimulus_indices=stimulus_indices,
            channels=channels,
            top_k=top_k,
            reduce=reduce,
            max_channels=max_channels,
        )
        reduce_id = _reduce_id(reduce)
        counts = torch.tensor(
            [
                int(tensor.shape[0]),
                int(tensor.shape[1]) if mode_id != _MODE_AGGREGATE else 1,
                int(maps.shape[0]),
                int(maps.shape[1]),
                mode_id,
                reduce_id,
            ],
            dtype=torch.int64,
        )
        _store_annotation_tensor(trace, f"featmap:{key}:maps", maps)
        _store_annotation_tensor(
            trace,
            f"featmap:{key}:stimuli",
            torch.tensor(stimulus_indices, dtype=torch.int64),
        )
        _store_annotation_tensor(trace, f"featmap:{key}:channels", channel_ids)
        _store_annotation_tensor(trace, f"featmap:{key}:counts", counts)
        maps_by_key[key] = maps

    if not maps_by_key:
        shape_text = "; ".join(non_spatial_shapes) if non_spatial_shapes else "no saved activations"
        raise ValueError(
            "feature_map_evolution requires spatial [N, C, H, W] activations; "
            f"saw {shape_text}. Capture with save= covering the conv layers before calling "
            "feature_map_evolution."
        )
    return maps_by_key


def feature_map_node_spec(
    *,
    overlay: bool = True,
    alpha: float = 0.55,
    cmap: str = "magma",
    cell_size: int = 72,
    thumbnail_size: int = 72,
    max_stimuli: int = 4,
    max_channels: int = 4,
) -> NodeSpecFn:
    """Return a draw-time node callback for stored feature-map annotations.

    Parameters
    ----------
    overlay:
        Whether to alpha-blend heatmaps over matching raw PIL stimuli when
        available.
    alpha:
        Heatmap opacity used for overlays.
    cmap:
        Colormap name passed to the node plot colormap helper.
    cell_size:
        Width and height of each rendered map cell in pixels.
    thumbnail_size:
        Required minimum size for raw stimulus thumbnails.
    max_stimuli:
        Maximum number of stimulus rows to display.
    max_channels:
        Maximum number of channel columns to display.

    Returns
    -------
    NodeSpecFn
        ``node_spec_fn`` suitable for ``Trace.draw(node_spec_fn=...)``.

    Raises
    ------
    ValueError
        If display parameters are invalid.
    """

    _validate_node_spec_options(
        overlay=overlay,
        alpha=alpha,
        cmap=cmap,
        cell_size=cell_size,
        thumbnail_size=thumbnail_size,
        max_stimuli=max_stimuli,
        max_channels=max_channels,
    )

    def node_spec_fn(layer: Any, spec: NodeSpec) -> NodeSpec | None:
        """Apply a feature-map image to a matching node spec.

        Parameters
        ----------
        layer:
            Layer or op-like render node passed by TorchLens.
        spec:
            Default ``NodeSpec`` to mutate by replacement.

        Returns
        -------
        NodeSpec | None
            Updated spec when feature maps are available, otherwise ``None``.
        """

        trace = getattr(layer, "source_trace", None)
        if trace is None:
            return None
        key, payload = _feature_map_payload_for_node(trace, layer)
        if key is None or payload is None:
            return None

        maps, stimulus_indices, channel_ids, counts = payload
        shown_rows = min(max_stimuli, maps.shape[0])
        shown_cols = min(max_channels, maps.shape[1])
        maps = maps[:shown_rows, :shown_cols]
        stimulus_indices = stimulus_indices[:shown_rows]
        channel_ids = channel_ids[:shown_rows, :shown_cols]
        more_count = max(0, int(counts[2].item()) - shown_rows) + max(
            0, int(counts[3].item()) - shown_cols
        )

        from ..repgeom import _matching_pil_image_batch

        raw_images = _matching_pil_image_batch(
            getattr(trace, "raw_input", None), int(counts[0].item())
        )
        overlay_available = overlay and raw_images is not None
        grid = _render_feature_map_grid(
            maps,
            stimulus_indices,
            channel_ids,
            raw_images=raw_images,
            overlay=overlay_available,
            alpha=alpha,
            cmap=cmap,
            cell_size=cell_size,
            more_count=more_count,
        )
        image_path = _write_feature_map_image(trace, key, grid)
        caption = str(getattr(layer, "layer_label", None) or getattr(layer, "label", key))
        mode_name = _mode_name(int(counts[4].item()))
        tooltip = f"Feature maps for {key}: {mode_name}, {int(counts[2].item())} stimuli"
        if overlay and not overlay_available:
            tooltip = f"{tooltip}; overlay unavailable"
        if more_count > 0:
            tooltip = f"{tooltip}; +{more_count} more"
        return spec.replace(
            lines=[caption],
            image=str(image_path),
            shape="box",
            tooltip=tooltip,
            extra_attrs={
                **getattr(spec, "extra_attrs", {}),
                "imagescale": "true",
                "labelloc": "b",
                "fixedsize": "false",
                "margin": "0.06,0.06",
            },
        )

    return node_spec_fn


def _validate_feature_map_options(
    *,
    top_k: int,
    reduce: str,
    max_stimuli: int,
    max_channels: int,
) -> None:
    """Validate feature-map selection options.

    Parameters
    ----------
    top_k:
        Requested top-channel count.
    reduce:
        Requested aggregate reduction name.
    max_stimuli:
        Stimulus cap.
    max_channels:
        Channel cap.

    Raises
    ------
    ValueError
        If an option is invalid.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if reduce not in {"mean", "max"}:
        raise ValueError("reduce must be 'mean' or 'max'.")
    if max_stimuli < 1:
        raise ValueError("max_stimuli must be at least 1.")
    if max_channels < 1:
        raise ValueError("max_channels must be at least 1.")


def _validate_node_spec_options(
    *,
    overlay: bool,
    alpha: float,
    cmap: str,
    cell_size: int,
    thumbnail_size: int,
    max_stimuli: int,
    max_channels: int,
) -> None:
    """Validate feature-map node-spec display options.

    Parameters
    ----------
    overlay:
        Overlay mode flag.
    alpha:
        Overlay opacity.
    cmap:
        Colormap name.
    cell_size:
        Cell side length.
    thumbnail_size:
        Requested thumbnail side length.
    max_stimuli:
        Stimulus row cap.
    max_channels:
        Channel column cap.

    Raises
    ------
    ValueError
        If an option is invalid.
    """

    _ = overlay
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    _apply_colormap(np.zeros((1, 1), dtype=np.float64), cmap)
    if cell_size < 8:
        raise ValueError("cell_size must be at least 8.")
    if thumbnail_size < 4:
        raise ValueError("thumbnail_size must be at least 4.")
    if max_stimuli < 1:
        raise ValueError("max_stimuli must be at least 1.")
    if max_channels < 1:
        raise ValueError("max_channels must be at least 1.")


def _as_cpu_float_tensor(value: Any) -> torch.Tensor:
    """Return an activation payload as a detached CPU float32 tensor.

    Parameters
    ----------
    value:
        Activation payload.

    Returns
    -------
    torch.Tensor
        Detached CPU float32 tensor.

    Raises
    ------
    TypeError
        If the payload is not a tensor.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("feature_map_evolution requires tensor activation payloads.")
    return value.detach().to(device="cpu", dtype=torch.float32)


def _resolve_stimuli(
    stimuli: int | Sequence[int] | None,
    *,
    total_stimuli: int,
    cap: int,
) -> list[int]:
    """Resolve user stimulus selection to bounded source indices.

    Parameters
    ----------
    stimuli:
        User stimulus selector.
    total_stimuli:
        Number of stimuli in the activation tensor.
    cap:
        Maximum number of stimuli to return.

    Returns
    -------
    list[int]
        Selected stimulus indices.

    Raises
    ------
    ValueError
        If the selector is empty, out of range, or exceeds the cap.
    """

    if total_stimuli < 1:
        raise ValueError("feature-map activations must have at least one stimulus.")
    if stimuli is None:
        return list(range(min(cap, total_stimuli)))
    if isinstance(stimuli, int):
        if stimuli < 1:
            raise ValueError("stimuli integer selector must be at least 1.")
        count = min(stimuli, cap, total_stimuli)
        return list(range(count))
    indices = [int(index) for index in stimuli]
    if not indices:
        raise ValueError("stimuli sequence must not be empty.")
    if len(indices) > cap:
        raise ValueError(f"stimuli sequence exceeds max_stimuli={cap}.")
    for index in indices:
        if index < 0 or index >= total_stimuli:
            raise ValueError(f"stimulus index {index} is out of range for {total_stimuli} stimuli.")
    return indices


def _select_maps(
    activations: torch.Tensor,
    *,
    stimulus_indices: Sequence[int],
    channels: FeatureMapChannels,
    top_k: int,
    reduce: FeatureMapReduce,
    max_channels: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select and reduce spatial maps from ``[N, C, H, W]`` activations.

    Parameters
    ----------
    activations:
        Activation tensor with shape ``[N, C, H, W]``.
    stimulus_indices:
        Source stimulus indices to keep.
    channels:
        Channel selection mode.
    top_k:
        Number of top channels for ``channels="top"``.
    reduce:
        Aggregate reduction name.
    max_channels:
        Maximum channel count.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        Maps ``[S, K, H, W]``, channel ids ``[S, K]``, and mode id.
    """

    selected = activations[list(stimulus_indices)]
    if channels is None:
        maps = _aggregate_channels(selected, reduce).unsqueeze(1)
        channel_ids = torch.full((selected.shape[0], 1), -1, dtype=torch.int64)
        return torch.nan_to_num(maps).contiguous(), channel_ids, _MODE_AGGREGATE
    if channels == "top":
        channel_ids = _top_channel_indices(selected, top_k=min(top_k, max_channels))
        maps = torch.stack(
            [selected[row_index, channel_ids[row_index]] for row_index in range(selected.shape[0])],
            dim=0,
        )
        return torch.nan_to_num(maps).contiguous(), channel_ids.to(dtype=torch.int64), _MODE_TOP

    explicit = _resolve_channels(channels, total_channels=activations.shape[1], cap=max_channels)
    explicit_tensor = torch.tensor(explicit, dtype=torch.int64)
    maps = selected[:, explicit_tensor]
    channel_ids = explicit_tensor.unsqueeze(0).expand(selected.shape[0], -1).contiguous()
    return torch.nan_to_num(maps).contiguous(), channel_ids, _MODE_EXPLICIT


def _aggregate_channels(activations: torch.Tensor, reduce: FeatureMapReduce) -> torch.Tensor:
    """Reduce activations over the channel dimension.

    Parameters
    ----------
    activations:
        Activation tensor with shape ``[S, C, H, W]``.
    reduce:
        Reduction name.

    Returns
    -------
    torch.Tensor
        Reduced maps with shape ``[S, H, W]``.
    """

    if reduce == "mean":
        return activations.mean(dim=1)
    return activations.max(dim=1).values


def _resolve_channels(channels: Sequence[int], *, total_channels: int, cap: int) -> list[int]:
    """Resolve explicit channel indices.

    Parameters
    ----------
    channels:
        User channel indices.
    total_channels:
        Number of channels in the activation tensor.
    cap:
        Maximum number of channels allowed.

    Returns
    -------
    list[int]
        Validated channel indices.

    Raises
    ------
    ValueError
        If no channels are supplied, the cap is exceeded, or an index is out of range.
    """

    indices = [int(index) for index in channels]
    if not indices:
        raise ValueError("channels sequence must not be empty.")
    if len(indices) > cap:
        raise ValueError(f"channels sequence exceeds max_channels={cap}.")
    for index in indices:
        if index < 0 or index >= total_channels:
            raise ValueError(
                f"channel index {index} is out of range for {total_channels} channels."
            )
    return indices


def _top_channel_indices(activations: torch.Tensor, *, top_k: int) -> torch.Tensor:
    """Return deterministic per-stimulus top-channel indices.

    Parameters
    ----------
    activations:
        Activation tensor with shape ``[S, C, H, W]``.
    top_k:
        Number of top channels to return.

    Returns
    -------
    torch.Tensor
        Channel ids with shape ``[S, top_k]``. Ties prefer lower channel index.
    """

    scores = activations.abs().mean(dim=(2, 3)).detach().cpu().numpy()
    selected = []
    for row in scores:
        indices = np.arange(row.shape[0], dtype=np.int64)
        order = np.lexsort((indices, -row))
        selected.append(order[: min(top_k, row.shape[0])])
    return torch.tensor(np.stack(selected, axis=0), dtype=torch.int64)


def _reduce_id(reduce: FeatureMapReduce) -> int:
    """Return the storage id for an aggregate reduction.

    Parameters
    ----------
    reduce:
        Reduction name.

    Returns
    -------
    int
        ``0`` for mean and ``1`` for max.
    """

    return _REDUCE_MEAN if reduce == "mean" else _REDUCE_MAX


def _feature_map_payload_for_node(
    trace: Any,
    node: Any,
) -> tuple[str | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Return stored feature-map tensors for a rendered node.

    Parameters
    ----------
    trace:
        Trace that owns annotation blobs.
    node:
        Rendered layer or op-like node.

    Returns
    -------
    tuple[str | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]
        Base key and map payload when present.
    """

    blobs = getattr(trace, "_annotation_blobs", None)
    if not isinstance(blobs, dict):
        return None, None
    candidates = []
    label = getattr(node, "label", None)
    if label is not None:
        candidates.append(f"op:{label}")
    layer_label = getattr(node, "layer_label", None)
    if layer_label is not None:
        candidates.append(f"layer:{layer_label}")
    for key in candidates:
        maps = blobs.get(f"featmap:{key}:maps")
        stimulus_indices = blobs.get(f"featmap:{key}:stimuli")
        channel_ids = blobs.get(f"featmap:{key}:channels")
        counts = blobs.get(f"featmap:{key}:counts")
        if not isinstance(maps, torch.Tensor):
            continue
        if not isinstance(stimulus_indices, torch.Tensor):
            continue
        if not isinstance(channel_ids, torch.Tensor):
            continue
        if not isinstance(counts, torch.Tensor):
            continue
        if (
            maps.ndim == 4
            and stimulus_indices.ndim == 1
            and channel_ids.ndim == 2
            and counts.shape == (6,)
            and maps.shape[:2] == channel_ids.shape
            and maps.shape[0] == stimulus_indices.shape[0]
        ):
            return key, (
                maps.detach().cpu(),
                stimulus_indices.detach().cpu(),
                channel_ids.detach().cpu(),
                counts.detach().cpu(),
            )
    return None, None


def _render_feature_map_grid(
    maps: torch.Tensor,
    stimulus_indices: torch.Tensor,
    channel_ids: torch.Tensor,
    *,
    raw_images: Sequence[Any] | None,
    overlay: bool,
    alpha: float,
    cmap: str,
    cell_size: int,
    more_count: int,
) -> Image.Image:
    """Render stored maps as one bounded small-multiples image.

    Parameters
    ----------
    maps:
        Feature maps with shape ``[S, K, H, W]``.
    stimulus_indices:
        Source stimulus indices with shape ``[S]``.
    channel_ids:
        Channel ids with shape ``[S, K]``.
    raw_images:
        Optional matching raw PIL stimulus batch.
    overlay:
        Whether to overlay heatmaps on raw stimulus thumbnails.
    alpha:
        Overlay heatmap opacity.
    cmap:
        Colormap name.
    cell_size:
        Cell side length in pixels.
    more_count:
        Number of omitted row/column entries to mark.

    Returns
    -------
    Image.Image
        RGB grid image.
    """

    rows, cols = int(maps.shape[0]), int(maps.shape[1])
    gap = 4
    width = cols * cell_size + max(0, cols - 1) * gap
    height = rows * cell_size + max(0, rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index in range(rows):
        for col_index in range(cols):
            cell = _render_feature_map_cell(
                maps[row_index, col_index],
                raw_images=raw_images,
                stimulus_index=int(stimulus_indices[row_index].item()),
                overlay=overlay,
                alpha=alpha,
                cmap=cmap,
                cell_size=cell_size,
            )
            x = col_index * (cell_size + gap)
            y = row_index * (cell_size + gap)
            canvas.paste(cell, (x, y))
            _draw_cell_label(
                draw,
                x=x,
                y=y,
                cell_size=cell_size,
                stimulus_index=int(stimulus_indices[row_index].item()),
                channel_id=int(channel_ids[row_index, col_index].item()),
            )
    if more_count > 0:
        _draw_more_marker(draw, width=width, height=height, text=f"+{more_count} more")
    return canvas


def _render_feature_map_cell(
    map_tensor: torch.Tensor,
    *,
    raw_images: Sequence[Any] | None,
    stimulus_index: int,
    overlay: bool,
    alpha: float,
    cmap: str,
    cell_size: int,
) -> Image.Image:
    """Render one feature-map cell.

    Parameters
    ----------
    map_tensor:
        Two-dimensional feature map.
    raw_images:
        Optional raw image batch.
    stimulus_index:
        Source stimulus index.
    overlay:
        Whether to alpha-blend over the raw image.
    alpha:
        Heatmap opacity.
    cmap:
        Colormap name.
    cell_size:
        Cell side length.

    Returns
    -------
    Image.Image
        RGB cell image.
    """

    heatmap = _map_to_heatmap_image(map_tensor, cmap=cmap, cell_size=cell_size)
    if not overlay or raw_images is None:
        return heatmap
    base = (
        raw_images[stimulus_index]
        .convert("RGB")
        .resize(
            (cell_size, cell_size),
            Image.Resampling.BILINEAR,
        )
    )
    return Image.blend(base, heatmap, alpha)


def _map_to_heatmap_image(map_tensor: torch.Tensor, *, cmap: str, cell_size: int) -> Image.Image:
    """Convert one map tensor to a colorized heatmap image.

    Parameters
    ----------
    map_tensor:
        Two-dimensional feature map tensor.
    cmap:
        Colormap name.
    cell_size:
        Output image side length.

    Returns
    -------
    Image.Image
        RGB heatmap image.
    """

    array = torch.nan_to_num(map_tensor.detach().to(device="cpu", dtype=torch.float32)).numpy()
    normalized = _normalize_finite(np.asarray(array, dtype=np.float64), None, None)
    colors = _apply_colormap(normalized, cmap)
    return Image.fromarray(colors, mode="RGB").resize(
        (cell_size, cell_size), Image.Resampling.BILINEAR
    )


def _draw_cell_label(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    cell_size: int,
    stimulus_index: int,
    channel_id: int,
) -> None:
    """Draw a compact stimulus/channel label when it fits.

    Parameters
    ----------
    draw:
        PIL drawing context.
    x:
        Cell left coordinate.
    y:
        Cell top coordinate.
    cell_size:
        Cell side length.
    stimulus_index:
        Source stimulus index.
    channel_id:
        Channel id, or ``-1`` for aggregate.
    """

    label = f"s{stimulus_index}/avg" if channel_id < 0 else f"s{stimulus_index}/c{channel_id}"
    bbox = draw.textbbox((0, 0), label)
    text_width = int(bbox[2] - bbox[0])
    text_height = int(bbox[3] - bbox[1])
    if text_width + 8 > cell_size or text_height + 6 > cell_size:
        return
    rect = (x + 2, y + 2, x + text_width + 8, y + text_height + 6)
    draw.rectangle(rect, fill=_LABEL_FILL, outline=_LABEL_OUTLINE)
    draw.text((x + 5, y + 4), label, fill=_TEXT_COLOR)


def _draw_more_marker(draw: ImageDraw.ImageDraw, *, width: int, height: int, text: str) -> None:
    """Draw one cap marker in the lower-right corner.

    Parameters
    ----------
    draw:
        PIL drawing context.
    width:
        Canvas width.
    height:
        Canvas height.
    text:
        Marker text.
    """

    bbox = draw.textbbox((0, 0), text)
    text_width = int(bbox[2] - bbox[0])
    text_height = int(bbox[3] - bbox[1])
    pad = 6
    x0 = max(0, width - text_width - 2 * pad - 5)
    y0 = max(0, height - text_height - 2 * pad - 5)
    x1 = width - 5
    y1 = height - 5
    draw.rectangle((x0, y0, x1, y1), fill=_LABEL_FILL, outline=_LABEL_OUTLINE)
    draw.text((x0 + pad, y0 + pad), text, fill=_TEXT_COLOR)


def _mode_name(mode_id: int) -> str:
    """Return a human-readable feature-map storage mode.

    Parameters
    ----------
    mode_id:
        Stored mode id.

    Returns
    -------
    str
        Mode name.
    """

    if mode_id == _MODE_AGGREGATE:
        return "aggregate"
    if mode_id == _MODE_EXPLICIT:
        return "explicit channels"
    if mode_id == _MODE_TOP:
        return "top channels"
    return "unknown mode"


def _write_feature_map_image(trace: Any, key: str, image: Image.Image) -> Path:
    """Write a draw-time feature-map image under the visualizer directory.

    Parameters
    ----------
    trace:
        Trace that owns the draw.
    key:
        Annotation key for the rendered payload.
    image:
        PIL image to save.

    Returns
    -------
    Path
        Local PNG path for ``NodeSpec.image``.
    """

    output_dir = getattr(trace, "_visualizer_dir", None)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="torchlens_visualizers_")
        trace._visualizer_dir = str(output_dir)
    plot_dir = Path(str(output_dir)) / "feature_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in key)
    image_path = plot_dir / f"{safe_key}.png"
    image.save(image_path)
    return image_path


__all__ = ["feature_map_evolution", "feature_map_node_spec"]
