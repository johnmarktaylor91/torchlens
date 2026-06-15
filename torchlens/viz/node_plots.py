"""PIL-only render primitives for compact node visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RGBColor: TypeAlias = tuple[int, int, int]

_COLORMAPS: dict[str, tuple[RGBColor, ...]] = {
    "viridis": (
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ),
    "magma": (
        (0, 0, 4),
        (73, 16, 110),
        (182, 54, 121),
        (251, 136, 97),
        (252, 253, 191),
    ),
    "gray": (
        (0, 0, 0),
        (255, 255, 255),
    ),
}

_AXIS_COLOR = (215, 219, 226)
_TEXT_COLOR = (35, 39, 47)
_POINT_COLOR = (48, 93, 170)
_POINT_OUTLINE = (20, 48, 100)
_MORE_FILL = (255, 255, 255)
_MORE_OUTLINE = (120, 128, 140)
_SCATTER_CAPTION_RESERVE = 56
_RANK_TOLERANCE = 1e-12
_DRAW_SCALE = 2


def render_heatmap(
    data: Any,
    *,
    width: int = 240,
    height: int = 240,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: RGBColor = (235, 235, 235),
    axis_images: Sequence[Image.Image] | None = None,
    axis_labels: Sequence[Any] | None = None,
    max_axis_items: int = 8,
) -> Image.Image:
    """Render a 2D numeric array as an RGB heatmap.

    Parameters
    ----------
    data:
        Two-dimensional array-like values to colorize.
    width:
        Output image width in pixels.
    height:
        Output image height in pixels.
    cmap:
        Colormap name: ``"viridis"``, ``"magma"``, or ``"gray"``.
    vmin:
        Optional lower normalization bound.
    vmax:
        Optional upper normalization bound.
    nan_color:
        RGB color used for non-finite data values.
    axis_images:
        Optional PIL thumbnails aligned with both heatmap axes.
    axis_labels:
        Optional text labels aligned with both heatmap axes.
    max_axis_items:
        Maximum number of axis thumbnails or labels to draw before adding a cap marker.

    Returns
    -------
    Image.Image
        RGB heatmap image of exactly ``(width, height)``.

    Raises
    ------
    ValueError
        If inputs have invalid shape, size, cap, or colormap.
    """

    _validate_size(width, height)
    if max_axis_items < 1:
        raise ValueError("max_axis_items must be at least 1.")
    array = np.asarray(_as_numpy(data), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("render_heatmap expects a 2D array.")

    normalized = _normalize_finite(array, vmin, vmax)
    colors = _apply_colormap(normalized, cmap)
    finite_mask = np.isfinite(array)
    colors[~finite_mask] = np.asarray(nan_color, dtype=np.uint8)
    heatmap = Image.fromarray(colors, mode="RGB")

    has_axis = axis_images is not None or axis_labels is not None
    if not has_axis:
        return heatmap.resize((width, height), Image.Resampling.BILINEAR)

    labels = _stringify_labels(axis_labels)
    top_count = min(max_axis_items, array.shape[1])
    left_count = min(max_axis_items, array.shape[0])
    measure_canvas = Image.new("RGB", (1, 1), "white")
    measure_draw = ImageDraw.Draw(measure_canvas)
    plot_left, plot_top = _heatmap_axis_margins(
        measure_draw,
        width=width,
        height=height,
        labels=labels,
        images=axis_images,
        top_count=top_count,
        left_count=left_count,
    )
    plot_width = max(1, width - plot_left - 6)
    plot_height = max(1, height - plot_top - 6)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        heatmap.resize((plot_width, plot_height), Image.Resampling.BILINEAR), (plot_left, plot_top)
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [(plot_left, plot_top), (plot_left + plot_width - 1, plot_top + plot_height - 1)],
        outline=_AXIS_COLOR,
    )
    _draw_heatmap_axis_items(
        canvas,
        draw,
        images=axis_images,
        labels=labels,
        count=top_count,
        total=array.shape[1],
        max_axis_items=max_axis_items,
        axis="top",
        plot_left=plot_left,
        plot_top=plot_top,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    _draw_heatmap_axis_items(
        canvas,
        draw,
        images=axis_images,
        labels=labels,
        count=left_count,
        total=array.shape[0],
        max_axis_items=max_axis_items,
        axis="left",
        plot_left=plot_left,
        plot_top=plot_top,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    return canvas


def render_lineplot(
    series: Any,
    *,
    x_values: Any | None = None,
    labels: Sequence[Any] | None = None,
    width: int = 320,
    height: int = 180,
    y_min: float | None = None,
    y_max: float | None = None,
    colors: Sequence[RGBColor] | None = None,
    show_legend: bool = True,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Image.Image:
    """Render one or more series as a compact PIL line plot.

    Parameters
    ----------
    series:
        One-dimensional ``[K]`` or two-dimensional ``[S, K]`` numeric values.
    x_values:
        Optional ``[K]`` x coordinates. Defaults to ``0..K-1``.
    labels:
        Optional series labels for the legend.
    width:
        Output image width in pixels.
    height:
        Output image height in pixels.
    y_min:
        Optional lower y-axis bound.
    y_max:
        Optional upper y-axis bound.
    colors:
        Optional RGB colors for each series.
    show_legend:
        Whether to draw a legend when labels are supplied.
    x_label:
        Optional x-axis label.
    y_label:
        Optional y-axis label.

    Returns
    -------
    Image.Image
        RGB line plot image of exactly ``(width, height)``.

    Raises
    ------
    ValueError
        If shapes, sizes, or bounds are invalid.
    """

    _validate_size(width, height)
    values = np.asarray(_as_numpy(series), dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("render_lineplot expects shape [K] or [S, K] with K > 0.")
    n_series, n_points = values.shape
    xs = np.arange(n_points, dtype=np.float64)
    if x_values is not None:
        xs = np.asarray(_as_numpy(x_values), dtype=np.float64)
        if xs.ndim != 1 or xs.shape[0] != n_points:
            raise ValueError("x_values must have shape [K].")

    finite_y = values[np.isfinite(values)]
    finite_x = xs[np.isfinite(xs)]
    if finite_y.size == 0 or finite_x.size == 0:
        raise ValueError("render_lineplot requires at least one finite point.")
    low_y = float(np.min(finite_y)) if y_min is None else float(y_min)
    high_y = float(np.max(finite_y)) if y_max is None else float(y_max)
    if not np.isfinite(low_y) or not np.isfinite(high_y):
        raise ValueError("y_min and y_max must be finite when provided.")
    if high_y <= low_y:
        pad = 1.0 if high_y == 0.0 else abs(high_y) * 0.05
        low_y -= pad
        high_y += pad
    low_x = float(np.min(finite_x))
    high_x = float(np.max(finite_x))
    if high_x <= low_x:
        low_x -= 0.5
        high_x += 0.5

    palette = _line_colors(colors, n_series)
    canvas = Image.new("RGB", (width * _DRAW_SCALE, height * _DRAW_SCALE), "white")
    draw = ImageDraw.Draw(canvas)
    margin_left = 38 * _DRAW_SCALE
    margin_right = 12 * _DRAW_SCALE
    margin_top = 12 * _DRAW_SCALE
    margin_bottom = (30 if x_label is not None else 22) * _DRAW_SCALE
    plot_left = margin_left
    plot_top = margin_top
    plot_right = max(plot_left + 1, width * _DRAW_SCALE - margin_right)
    plot_bottom = max(plot_top + 1, height * _DRAW_SCALE - margin_bottom)
    draw.rectangle([(plot_left, plot_top), (plot_right, plot_bottom)], outline=_AXIS_COLOR, width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=_TEXT_COLOR, width=1)
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=_TEXT_COLOR, width=1)

    for series_index, row in enumerate(values):
        points = [
            _lineplot_point(
                x_value,
                y_value,
                low_x=low_x,
                high_x=high_x,
                low_y=low_y,
                high_y=high_y,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
            )
            for x_value, y_value in zip(xs, row, strict=True)
        ]
        segment: list[tuple[float, float]] = []
        for point in points:
            if point is None:
                if len(segment) >= 2:
                    draw.line(
                        segment, fill=palette[series_index], width=2 * _DRAW_SCALE, joint="curve"
                    )
                segment = []
                continue
            segment.append(point)
        if len(segment) >= 2:
            draw.line(segment, fill=palette[series_index], width=2 * _DRAW_SCALE, joint="curve")
        elif len(segment) == 1:
            _draw_scaled_point(draw, segment[0], palette[series_index])

    _draw_text(
        draw, (4 * _DRAW_SCALE, plot_top), f"{high_y:.3g}", fill=_TEXT_COLOR, scale=_DRAW_SCALE
    )
    _draw_text(
        draw,
        (4 * _DRAW_SCALE, plot_bottom - 8 * _DRAW_SCALE),
        f"{low_y:.3g}",
        fill=_TEXT_COLOR,
        scale=_DRAW_SCALE,
    )
    if x_label is not None:
        _draw_text(
            draw,
            (
                (plot_left + plot_right) / 2.0 - 16 * _DRAW_SCALE,
                height * _DRAW_SCALE - 16 * _DRAW_SCALE,
            ),
            x_label,
            fill=_TEXT_COLOR,
            scale=_DRAW_SCALE,
        )
    if y_label is not None:
        _draw_text(
            draw, (4 * _DRAW_SCALE, 4 * _DRAW_SCALE), y_label, fill=_TEXT_COLOR, scale=_DRAW_SCALE
        )
    if show_legend and labels is not None:
        _draw_legend(draw, labels=labels, colors=palette, plot_right=plot_right, plot_top=plot_top)
    return canvas.resize((width, height), Image.Resampling.LANCZOS)


def render_image_scatter(
    coords: Any,
    *,
    images: Sequence[Image.Image] | None = None,
    labels: Sequence[Any] | None = None,
    max_items: int = 16,
    thumbnail_size: int = 36,
    canvas_size: int = 420,
    min_distance: float | None = None,
    show_axes: bool = True,
    background: RGBColor = (255, 255, 255),
) -> Image.Image:
    """Render a bounded 2D scatter with optional PIL thumbnails.

    Parameters
    ----------
    coords:
        ``[N, 2]`` NumPy-like or Torch tensor coordinates.
    images:
        Optional PIL images to paste at scatter positions.
    labels:
        Optional labels for point fallback rendering.
    max_items:
        Maximum number of coordinates to draw before adding a cap marker.
    thumbnail_size:
        Maximum width and height for pasted thumbnails.
    canvas_size:
        Width and height of the square output image.
    min_distance:
        Optional minimum center-to-center distance in pixels.
    show_axes:
        Whether to draw faint central guide axes.
    background:
        RGB canvas background.

    Returns
    -------
    Image.Image
        RGB scatter image of exactly ``(canvas_size, canvas_size)``.

    Raises
    ------
    ValueError
        If coordinates or sizing parameters are invalid.
    """

    if max_items < 1:
        raise ValueError("max_items must be at least 1.")
    if thumbnail_size < 4:
        raise ValueError("thumbnail_size must be at least 4.")
    if canvas_size <= thumbnail_size * 2:
        raise ValueError("canvas_size must be larger than twice thumbnail_size.")
    array = np.asarray(_as_numpy(coords), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("render_image_scatter expects coords with shape [N, 2].")
    if array.shape[0] == 0:
        raise ValueError("render_image_scatter requires at least one coordinate.")
    if not np.all(np.isfinite(array)):
        raise ValueError("coords must contain only finite values.")
    if images is not None and len(images) < min(max_items, array.shape[0]):
        raise ValueError("images must contain at least the number of drawn coordinates.")

    shown_count = min(max_items, array.shape[0])
    margin = thumbnail_size / 2.0 + _SCATTER_CAPTION_RESERVE
    centers = _coords_to_pixel_centers(array[:shown_count], canvas_size=canvas_size, margin=margin)
    spacing = (
        min_distance if min_distance is not None else (float(thumbnail_size) if images else 14.0)
    )
    centers = _spread_close_centers(
        centers, canvas_size=canvas_size, margin=margin, min_distance=spacing
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), background)
    draw = ImageDraw.Draw(canvas)
    if show_axes:
        _draw_scatter_axes(draw, canvas_size=canvas_size, margin=int(round(margin)))
    if images is None:
        _draw_point_fallback(draw, centers, labels=labels)
    else:
        _paste_scatter_thumbnails(
            canvas,
            centers,
            images[:shown_count],
            thumbnail_size=thumbnail_size,
        )
    more_count = array.shape[0] - shown_count
    if more_count > 0:
        _draw_more_indicator(draw, canvas_size=canvas_size, text=f"+{more_count} more")
    return canvas


def _as_numpy(value: Any) -> np.ndarray:
    """Return ``value`` as a CPU NumPy array, accepting Torch-like tensors.

    Parameters
    ----------
    value:
        NumPy-like object or tensor with ``detach``/``cpu``/``numpy`` methods.

    Returns
    -------
    np.ndarray
        NumPy view or copy of the input.
    """

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return cast(Any, value).detach().cpu().numpy()
    return np.asarray(value)


def _validate_size(width: int, height: int) -> None:
    """Validate positive image dimensions.

    Parameters
    ----------
    width:
        Candidate image width.
    height:
        Candidate image height.

    Raises
    ------
    ValueError
        If either dimension is non-positive.
    """

    if width < 1 or height < 1:
        raise ValueError("image width and height must be positive.")


def _normalize_finite(array: np.ndarray, vmin: float | None, vmax: float | None) -> np.ndarray:
    """Normalize finite array values into ``[0, 1]`` without producing NaNs.

    Parameters
    ----------
    array:
        Numeric array to normalize.
    vmin:
        Optional lower bound.
    vmax:
        Optional upper bound.

    Returns
    -------
    np.ndarray
        Float64 array with non-finite values set to zero and finite values clipped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If explicit bounds are not finite or are reversed.
    """

    finite = np.isfinite(array)
    normalized = np.zeros(array.shape, dtype=np.float64)
    if not np.any(finite):
        return normalized
    low = float(np.min(array[finite])) if vmin is None else float(vmin)
    high = float(np.max(array[finite])) if vmax is None else float(vmax)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("vmin and vmax must be finite when provided.")
    if high <= low:
        return normalized
    normalized[finite] = np.clip((array[finite] - low) / (high - low), 0.0, 1.0)
    return normalized


def _apply_colormap(norm_array: np.ndarray, cmap: str) -> np.ndarray:
    """Apply a hard-coded RGB colormap to normalized data.

    Parameters
    ----------
    norm_array:
        Numeric array whose values are interpreted in ``[0, 1]``.
    cmap:
        Colormap name.

    Returns
    -------
    np.ndarray
        ``uint8`` RGB array with shape ``norm_array.shape + (3,)``.

    Raises
    ------
    ValueError
        If ``cmap`` is unknown.
    """

    if cmap not in _COLORMAPS:
        raise ValueError(f"Unknown colormap {cmap!r}.")
    stops = np.asarray(_COLORMAPS[cmap], dtype=np.float64)
    scaled = np.clip(norm_array, 0.0, 1.0) * (len(stops) - 1)
    low_indices = np.floor(scaled).astype(np.int64)
    high_indices = np.clip(low_indices + 1, 0, len(stops) - 1)
    weights = (scaled - low_indices)[..., np.newaxis]
    colors = stops[low_indices] * (1.0 - weights) + stops[high_indices] * weights
    return colors.astype(np.uint8)


def _measure_text(draw: ImageDraw.ImageDraw, text: str, *, scale: int = 1) -> tuple[int, int]:
    """Measure text using Pillow's default font.

    Parameters
    ----------
    draw:
        PIL drawing context.
    text:
        Text to measure.
    scale:
        Font scale factor.

    Returns
    -------
    tuple[int, int]
        Text width and height in pixels.
    """

    font = _font(scale)
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(round(bbox[2] - bbox[0])), int(round(bbox[3] - bbox[1]))


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    fill: RGBColor,
    scale: int = 1,
) -> None:
    """Draw text using Pillow's default font.

    Parameters
    ----------
    draw:
        PIL drawing context.
    xy:
        Text origin.
    text:
        Text to draw.
    fill:
        RGB text color.
    scale:
        Font scale factor.
    """

    draw.text(xy, text, fill=fill, font=_font(scale))


def _font(scale: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Return a deterministic default Pillow font.

    Parameters
    ----------
    scale:
        Font scale factor.

    Returns
    -------
    ImageFont.ImageFont | ImageFont.FreeTypeFont
        Pillow default font.
    """

    return ImageFont.load_default(size=10 * scale)


def _thumbnail(img: Image.Image, size: int) -> Image.Image:
    """Return an RGB thumbnail copy bounded by ``size``.

    Parameters
    ----------
    img:
        Source PIL image.
    size:
        Maximum thumbnail side length.

    Returns
    -------
    Image.Image
        Resized RGB thumbnail.
    """

    thumb = img.convert("RGB")
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    return thumb


def _stringify_labels(labels: Sequence[Any] | None) -> list[str] | None:
    """Convert optional labels to strings.

    Parameters
    ----------
    labels:
        Optional label sequence.

    Returns
    -------
    list[str] | None
        String labels when provided.
    """

    if labels is None:
        return None
    return [str(label) for label in labels]


def _heatmap_axis_margins(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    labels: list[str] | None,
    images: Sequence[Image.Image] | None,
    top_count: int,
    left_count: int,
) -> tuple[int, int]:
    """Return reserved left and top margins for heatmap axis decoration.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    width:
        Output image width.
    height:
        Output image height.
    labels:
        Optional string labels.
    images:
        Optional axis thumbnails.
    top_count:
        Number of capped top-axis items.
    left_count:
        Number of capped left-axis items.

    Returns
    -------
    tuple[int, int]
        Left and top plot offsets in pixels.
    """

    base_pad = max(24, min(width, height) // 8)
    if images is not None:
        margin = min(32, max(24, base_pad)) + 8
        return min(margin, max(1, width - 24)), min(margin, max(1, height - 24))
    top_height = _max_label_height(draw, labels, top_count)
    left_width = _max_label_width(draw, labels, left_count)
    left_margin = max(24, left_width + 10)
    top_margin = max(20, top_height + 10)
    return min(left_margin, max(1, width - 24)), min(top_margin, max(1, height - 24))


def _max_label_width(
    draw: ImageDraw.ImageDraw,
    labels: list[str] | None,
    count: int,
) -> int:
    """Return the maximum measured label width among capped axis labels.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    labels:
        Optional string labels.
    count:
        Maximum number of labels to inspect.

    Returns
    -------
    int
        Maximum label width in pixels.
    """

    if labels is None or count <= 0:
        return 0
    return max((_measure_text(draw, label)[0] for label in labels[:count]), default=0)


def _max_label_height(
    draw: ImageDraw.ImageDraw,
    labels: list[str] | None,
    count: int,
) -> int:
    """Return the maximum measured label height among capped axis labels.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    labels:
        Optional string labels.
    count:
        Maximum number of labels to inspect.

    Returns
    -------
    int
        Maximum label height in pixels.
    """

    if labels is None or count <= 0:
        return 0
    return max((_measure_text(draw, label)[1] for label in labels[:count]), default=0)


def _draw_heatmap_axis_items(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    images: Sequence[Image.Image] | None,
    labels: list[str] | None,
    count: int,
    total: int,
    max_axis_items: int,
    axis: str,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
) -> None:
    """Draw capped labels or thumbnails for one heatmap axis.

    Parameters
    ----------
    canvas:
        Output image.
    draw:
        PIL drawing context.
    images:
        Optional axis thumbnails.
    labels:
        Optional axis labels.
    count:
        Number of items to draw.
    total:
        Total axis item count.
    max_axis_items:
        Cap used for marker text.
    axis:
        ``"top"`` or ``"left"``.
    plot_left:
        Heatmap left coordinate.
    plot_top:
        Heatmap top coordinate.
    plot_width:
        Heatmap width.
    plot_height:
        Heatmap height.
    """

    del max_axis_items
    if count <= 0:
        return
    thumb_size = max(10, min(24, plot_top - 8 if axis == "top" else plot_left - 8))
    more_count = total - count
    more_text = f"+{more_count} more" if more_count > 0 else None
    indices = _select_heatmap_axis_indices(
        draw,
        labels=labels,
        has_images=images is not None,
        count=count,
        total=total,
        axis=axis,
        plot_left=plot_left,
        plot_top=plot_top,
        plot_width=plot_width,
        plot_height=plot_height,
        canvas_width=canvas.width,
        canvas_height=canvas.height,
        thumb_size=thumb_size,
        more_text=more_text,
    )
    for index in indices:
        item_width, item_height = _heatmap_axis_item_size(
            draw,
            has_images=images is not None,
            labels=labels,
            index=index,
            thumb_size=thumb_size,
        )
        if axis == "top":
            center_x = plot_left + (index + 0.5) * plot_width / total
            x = center_x - item_width / 2.0
            y = max(2.0, plot_top - item_height - 4.0)
        else:
            center_y = plot_top + (index + 0.5) * plot_height / total
            x = max(2.0, plot_left - item_width - 4.0)
            y = center_y - item_height / 2.0
        _draw_axis_item(canvas, draw, images, labels, index, x, y, thumb_size)
    if more_text is not None:
        _draw_heatmap_more_text(
            draw,
            text=more_text,
            axis=axis,
            plot_left=plot_left,
            plot_top=plot_top,
            canvas_width=canvas.width,
            canvas_height=canvas.height,
        )


def _select_heatmap_axis_indices(
    draw: ImageDraw.ImageDraw,
    *,
    labels: list[str] | None,
    has_images: bool,
    count: int,
    total: int,
    axis: str,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
    canvas_width: int,
    canvas_height: int,
    thumb_size: int,
    more_text: str | None,
) -> list[int]:
    """Select a deterministic non-overlapping subset of axis item indices.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    labels:
        Optional string labels.
    has_images:
        Whether items are rendered as thumbnails.
    count:
        Number of capped items available for drawing.
    total:
        Total number of axis positions.
    axis:
        ``"top"`` or ``"left"``.
    plot_left:
        Heatmap left coordinate.
    plot_top:
        Heatmap top coordinate.
    plot_width:
        Heatmap width.
    plot_height:
        Heatmap height.
    canvas_width:
        Output image width.
    canvas_height:
        Output image height.
    thumb_size:
        Thumbnail side length.
    more_text:
        Optional cap marker text.

    Returns
    -------
    list[int]
        Capped item indices that fit without overlap.
    """

    for step in range(1, count + 1):
        selected = list(range(0, count, step))
        if _heatmap_axis_selection_fits(
            draw,
            selected,
            labels=labels,
            has_images=has_images,
            total=total,
            axis=axis,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_width=plot_width,
            plot_height=plot_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            thumb_size=thumb_size,
            more_text=more_text,
        ):
            return selected
    return []


def _heatmap_axis_selection_fits(
    draw: ImageDraw.ImageDraw,
    indices: Sequence[int],
    *,
    labels: list[str] | None,
    has_images: bool,
    total: int,
    axis: str,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
    canvas_width: int,
    canvas_height: int,
    thumb_size: int,
    more_text: str | None,
) -> bool:
    """Return whether selected heatmap axis items fit without overlap.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    indices:
        Candidate item indices.
    labels:
        Optional string labels.
    has_images:
        Whether items are rendered as thumbnails.
    total:
        Total number of axis positions.
    axis:
        ``"top"`` or ``"left"``.
    plot_left:
        Heatmap left coordinate.
    plot_top:
        Heatmap top coordinate.
    plot_width:
        Heatmap width.
    plot_height:
        Heatmap height.
    canvas_width:
        Output image width.
    canvas_height:
        Output image height.
    thumb_size:
        Thumbnail side length.
    more_text:
        Optional cap marker text.

    Returns
    -------
    bool
        Whether all selected item intervals fit the axis margin.
    """

    gap = 4.0
    marker_width, marker_height = _heatmap_more_text_size(draw, more_text)
    intervals: list[tuple[float, float]] = []
    if axis == "top":
        limit_high = (
            float(canvas_width - marker_width - 2 * gap)
            if more_text is not None
            else canvas_width - 1.0
        )
        for index in indices:
            item_width, _item_height = _heatmap_axis_item_size(
                draw, has_images=has_images, labels=labels, index=index, thumb_size=thumb_size
            )
            center = plot_left + (index + 0.5) * plot_width / total
            intervals.append((center - item_width / 2.0, center + item_width / 2.0))
    else:
        limit_high = (
            float(canvas_height - marker_height - 2 * gap)
            if more_text is not None
            else canvas_height - 1.0
        )
        for index in indices:
            _item_width, item_height = _heatmap_axis_item_size(
                draw, has_images=has_images, labels=labels, index=index, thumb_size=thumb_size
            )
            center = plot_top + (index + 0.5) * plot_height / total
            intervals.append((center - item_height / 2.0, center + item_height / 2.0))
    if any(start < 0.0 or end > limit_high for start, end in intervals):
        return False
    return all(
        intervals[index][0] - intervals[index - 1][1] >= gap for index in range(1, len(intervals))
    )


def _heatmap_axis_item_size(
    draw: ImageDraw.ImageDraw,
    *,
    has_images: bool,
    labels: list[str] | None,
    index: int,
    thumb_size: int,
) -> tuple[int, int]:
    """Return rendered width and height for one heatmap axis item.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    has_images:
        Whether the item is rendered as a thumbnail.
    labels:
        Optional string labels.
    index:
        Item index.
    thumb_size:
        Thumbnail side length.

    Returns
    -------
    tuple[int, int]
        Item width and height in pixels.
    """

    if has_images:
        return thumb_size, thumb_size
    if labels is None or index >= len(labels):
        return 0, 0
    return _measure_text(draw, labels[index])


def _heatmap_more_text_size(
    draw: ImageDraw.ImageDraw,
    text: str | None,
) -> tuple[int, int]:
    """Measure optional heatmap cap marker text.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    text:
        Optional cap marker text.

    Returns
    -------
    tuple[int, int]
        Text width and height in pixels.
    """

    if text is None:
        return 0, 0
    return _measure_text(draw, text)


def _draw_heatmap_more_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    axis: str,
    plot_left: int,
    plot_top: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Draw a plain heatmap cap marker inside the reserved axis margin.

    Parameters
    ----------
    draw:
        PIL drawing context.
    text:
        Marker text.
    axis:
        ``"top"`` or ``"left"``.
    plot_left:
        Heatmap left coordinate.
    plot_top:
        Heatmap top coordinate.
    canvas_width:
        Output image width.
    canvas_height:
        Output image height.
    """

    del plot_left
    text_width, text_height = _measure_text(draw, text)
    if axis == "top":
        x = canvas_width - text_width - 4
        y = max(2, plot_top - text_height - 4)
    else:
        x = 4
        y = canvas_height - text_height - 4
    _draw_text(draw, (x, y), text, fill=_TEXT_COLOR)


def _draw_axis_item(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    images: Sequence[Image.Image] | None,
    labels: list[str] | None,
    index: int,
    x: float,
    y: float,
    thumb_size: int,
) -> None:
    """Draw one heatmap axis item.

    Parameters
    ----------
    canvas:
        Output image.
    draw:
        PIL drawing context.
    images:
        Optional thumbnails.
    labels:
        Optional labels.
    index:
        Item index.
    x:
        Item left coordinate.
    y:
        Item top coordinate.
    thumb_size:
        Maximum thumbnail size.
    """

    if images is not None and index < len(images):
        thumb = _thumbnail(images[index], thumb_size)
        canvas.paste(thumb, (int(round(x)), int(round(y))))
        return
    if labels is None or index >= len(labels):
        return
    _draw_text(draw, (x, y), labels[index], fill=_TEXT_COLOR)


def _line_colors(colors: Sequence[RGBColor] | None, n_series: int) -> list[RGBColor]:
    """Return a color for each line series.

    Parameters
    ----------
    colors:
        Optional caller-provided colors.
    n_series:
        Number of series.

    Returns
    -------
    list[RGBColor]
        RGB colors.
    """

    default = [(48, 93, 170), (214, 91, 61), (82, 145, 85), (129, 82, 161)]
    if colors is None:
        return [default[index % len(default)] for index in range(n_series)]
    if len(colors) < n_series:
        raise ValueError("colors must contain at least one color per series.")
    return list(colors[:n_series])


def _lineplot_point(
    x_value: float,
    y_value: float,
    *,
    low_x: float,
    high_x: float,
    low_y: float,
    high_y: float,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
) -> tuple[float, float] | None:
    """Map one data point into line-plot pixel space.

    Parameters
    ----------
    x_value:
        Data-space x value.
    y_value:
        Data-space y value.
    low_x:
        Minimum x-axis value.
    high_x:
        Maximum x-axis value.
    low_y:
        Minimum y-axis value.
    high_y:
        Maximum y-axis value.
    plot_left:
        Plot area left coordinate.
    plot_right:
        Plot area right coordinate.
    plot_top:
        Plot area top coordinate.
    plot_bottom:
        Plot area bottom coordinate.

    Returns
    -------
    tuple[float, float] | None
        Pixel point, or ``None`` for non-finite values.
    """

    if not np.isfinite(x_value) or not np.isfinite(y_value):
        return None
    x_frac = (x_value - low_x) / (high_x - low_x)
    y_frac = (y_value - low_y) / (high_y - low_y)
    x = plot_left + x_frac * (plot_right - plot_left)
    y = plot_bottom - y_frac * (plot_bottom - plot_top)
    return x, y


def _draw_scaled_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    color: RGBColor,
) -> None:
    """Draw a single scaled line-plot point.

    Parameters
    ----------
    draw:
        PIL drawing context.
    point:
        Pixel-space point.
    color:
        RGB marker color.
    """

    radius = 3 * _DRAW_SCALE
    draw.ellipse(
        [(point[0] - radius, point[1] - radius), (point[0] + radius, point[1] + radius)],
        fill=color,
    )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    labels: Sequence[Any],
    colors: Sequence[RGBColor],
    plot_right: int,
    plot_top: int,
) -> None:
    """Draw a compact line-plot legend.

    Parameters
    ----------
    draw:
        PIL drawing context.
    labels:
        Series labels.
    colors:
        Series colors.
    plot_right:
        Plot area right coordinate.
    plot_top:
        Plot area top coordinate.
    """

    for index, label in enumerate(labels[: len(colors)]):
        text = str(label)
        text_width, text_height = _measure_text(draw, text, scale=_DRAW_SCALE)
        x0 = plot_right - text_width - 28 * _DRAW_SCALE
        y0 = plot_top + 5 * _DRAW_SCALE + index * (text_height + 5 * _DRAW_SCALE)
        draw.line(
            [(x0, y0 + text_height / 2.0), (x0 + 18 * _DRAW_SCALE, y0 + text_height / 2.0)],
            fill=colors[index],
            width=2 * _DRAW_SCALE,
        )
        _draw_text(draw, (x0 + 22 * _DRAW_SCALE, y0), text, fill=_TEXT_COLOR, scale=_DRAW_SCALE)


def _coords_to_pixel_centers(
    coords: np.ndarray,
    *,
    canvas_size: int,
    margin: float,
) -> list[tuple[float, float]]:
    """Normalize coordinates into drawable pixel centers.

    Parameters
    ----------
    coords:
        ``[N, 2]`` coordinate matrix.
    canvas_size:
        Output image side length.
    margin:
        Minimum distance from any center to the canvas edge.

    Returns
    -------
    list[tuple[float, float]]
        Pixel-space centers.
    """

    if coords.shape[0] == 0:
        return []
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = maxs - mins
    drawable = max(1.0, float(canvas_size) - 2.0 * margin)
    centers = []
    for row in coords:
        values = []
        for axis in range(2):
            if spans[axis] <= _RANK_TOLERANCE:
                values.append(float(canvas_size) / 2.0)
            else:
                values.append(float(margin + ((row[axis] - mins[axis]) / spans[axis]) * drawable))
        centers.append((values[0], float(canvas_size) - values[1]))
    return centers


def _spread_close_centers(
    centers: list[tuple[float, float]],
    *,
    canvas_size: int,
    margin: float,
    min_distance: float,
) -> list[tuple[float, float]]:
    """Deterministically move close centers apart within canvas bounds.

    Parameters
    ----------
    centers:
        Initial pixel centers.
    canvas_size:
        Output image side length.
    margin:
        Minimum edge margin for centers.
    min_distance:
        Required center spacing.

    Returns
    -------
    list[tuple[float, float]]
        Adjusted centers.
    """

    if len(centers) < 2:
        return [_clamp_center(center, canvas_size=canvas_size, margin=margin) for center in centers]

    target_distance = _feasible_grid_distance(
        n_centers=len(centers),
        canvas_size=canvas_size,
        margin=margin,
        requested_distance=min_distance,
    )
    placed: list[tuple[float, float]] = []
    for index, center in enumerate(centers):
        candidate = _clamp_center(center, canvas_size=canvas_size, margin=margin)
        if _is_far_enough(candidate, placed, min_distance=target_distance):
            placed.append(candidate)
            continue
        for offset in _spiral_offsets(index=index, step=max(2.0, target_distance)):
            candidate = _clamp_center(
                (center[0] + offset[0], center[1] + offset[1]),
                canvas_size=canvas_size,
                margin=margin,
            )
            if _is_far_enough(candidate, placed, min_distance=target_distance):
                break
        placed.append(candidate)
    if _all_far_enough(placed, min_distance=target_distance):
        return placed

    relaxed = _relax_center_overlaps(
        placed,
        canvas_size=canvas_size,
        margin=margin,
        min_distance=target_distance,
    )
    if _all_far_enough(relaxed, min_distance=target_distance):
        return relaxed
    return _grid_spread_centers(
        centers,
        canvas_size=canvas_size,
        margin=margin,
        min_distance=target_distance,
    )


def _spiral_offsets(index: int, *, step: float) -> list[tuple[float, float]]:
    """Return deterministic candidate offsets for overlap resolution.

    Parameters
    ----------
    index:
        Stimulus index, used only to rotate tie-break ordering.
    step:
        Base radial step.

    Returns
    -------
    list[tuple[float, float]]
        Candidate offsets ordered from near to far.
    """

    directions = [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.7071, 0.7071),
        (-0.7071, 0.7071),
        (-0.7071, -0.7071),
        (0.7071, -0.7071),
    ]
    rotated = directions[index % len(directions) :] + directions[: index % len(directions)]
    offsets = []
    for radius in range(1, 9):
        for dx, dy in rotated:
            offsets.append((dx * step * radius, dy * step * radius))
    return offsets


def _feasible_grid_distance(
    *,
    n_centers: int,
    canvas_size: int,
    margin: float,
    requested_distance: float,
) -> float:
    """Return the largest grid spacing no larger than the requested spacing.

    Parameters
    ----------
    n_centers:
        Number of centers to place.
    canvas_size:
        Output image side length.
    margin:
        Minimum edge margin.
    requested_distance:
        Preferred center spacing.

    Returns
    -------
    float
        Spacing that can fit all centers in the bounded square.
    """

    if n_centers < 2:
        return requested_distance
    side = max(0.0, float(canvas_size) - 2.0 * margin)
    if side <= 0.0:
        return 0.0
    best = 0.0
    for columns in range(1, n_centers + 1):
        rows = int(np.ceil(n_centers / columns))
        x_gap = requested_distance if columns == 1 else side / float(columns - 1)
        y_gap = requested_distance if rows == 1 else side / float(rows - 1)
        best = max(best, min(requested_distance, x_gap, y_gap))
    return best


def _relax_center_overlaps(
    centers: list[tuple[float, float]],
    *,
    canvas_size: int,
    margin: float,
    min_distance: float,
) -> list[tuple[float, float]]:
    """Iteratively push overlapping centers apart inside the canvas.

    Parameters
    ----------
    centers:
        Current pixel centers.
    canvas_size:
        Output image side length.
    margin:
        Minimum edge margin.
    min_distance:
        Required center spacing.

    Returns
    -------
    list[tuple[float, float]]
        Relaxed centers.
    """

    relaxed = np.asarray(centers, dtype=np.float64)
    low = margin
    high = float(canvas_size) - margin
    for _pass_index in range(80):
        moved = False
        for left in range(relaxed.shape[0]):
            for right in range(left + 1, relaxed.shape[0]):
                delta = relaxed[right] - relaxed[left]
                distance = float(np.linalg.norm(delta))
                if distance >= min_distance:
                    continue
                if distance <= _RANK_TOLERANCE:
                    angle = (left * 31 + right * 17) * np.pi / 8.0
                    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
                    distance = 0.0
                else:
                    direction = delta / distance
                push = (min_distance - distance) / 2.0
                relaxed[left] -= direction * push
                relaxed[right] += direction * push
                moved = True
        if not moved:
            break
        relaxed[:, 0] = np.clip(relaxed[:, 0], low, high)
        relaxed[:, 1] = np.clip(relaxed[:, 1], low, high)
    return [(float(x), float(y)) for x, y in relaxed]


def _grid_spread_centers(
    centers: list[tuple[float, float]],
    *,
    canvas_size: int,
    margin: float,
    min_distance: float,
) -> list[tuple[float, float]]:
    """Place centers on deterministic bounded grid slots.

    Parameters
    ----------
    centers:
        Initial pixel centers.
    canvas_size:
        Output image side length.
    margin:
        Minimum edge margin.
    min_distance:
        Grid spacing to preserve.

    Returns
    -------
    list[tuple[float, float]]
        Centers assigned to unique grid slots.
    """

    n_centers = len(centers)
    columns, rows = _best_grid_shape(
        n_centers=n_centers,
        canvas_size=canvas_size,
        margin=margin,
        min_distance=min_distance,
    )
    low = margin
    high = float(canvas_size) - margin
    xs = np.asarray([(low + high) / 2.0]) if columns == 1 else np.linspace(low, high, columns)
    ys = np.asarray([(low + high) / 2.0]) if rows == 1 else np.linspace(low, high, rows)
    slots = [(float(x), float(y)) for y in ys for x in xs]
    remaining = slots[:]
    assigned: list[tuple[float, float]] = []
    for center in centers:
        clamped = _clamp_center(center, canvas_size=canvas_size, margin=margin)
        best_index = min(
            range(len(remaining)),
            key=lambda slot_index: (
                (remaining[slot_index][0] - clamped[0]) ** 2
                + (remaining[slot_index][1] - clamped[1]) ** 2,
                slot_index,
            ),
        )
        assigned.append(remaining.pop(best_index))
    return assigned


def _best_grid_shape(
    *,
    n_centers: int,
    canvas_size: int,
    margin: float,
    min_distance: float,
) -> tuple[int, int]:
    """Return a compact grid shape that preserves the requested spacing.

    Parameters
    ----------
    n_centers:
        Number of centers to place.
    canvas_size:
        Output image side length.
    margin:
        Minimum edge margin.
    min_distance:
        Required center spacing.

    Returns
    -------
    tuple[int, int]
        Column and row counts.
    """

    side = max(0.0, float(canvas_size) - 2.0 * margin)
    best_shape = (n_centers, 1)
    best_key = (-1.0, n_centers, n_centers)
    for columns in range(1, n_centers + 1):
        rows = int(np.ceil(n_centers / columns))
        x_gap = min_distance if columns == 1 else side / float(columns - 1)
        y_gap = min_distance if rows == 1 else side / float(rows - 1)
        gap = min(x_gap, y_gap)
        if gap + _RANK_TOLERANCE < min_distance:
            continue
        imbalance = abs(columns - rows)
        area = columns * rows
        key = (gap, -imbalance, -area)
        if key > best_key:
            best_shape = (columns, rows)
            best_key = key
    return best_shape


def _clamp_center(
    center: tuple[float, float],
    *,
    canvas_size: int,
    margin: float,
) -> tuple[float, float]:
    """Clamp a center to the drawable area.

    Parameters
    ----------
    center:
        Candidate center.
    canvas_size:
        Output image side length.
    margin:
        Minimum edge margin.

    Returns
    -------
    tuple[float, float]
        Clamped center.
    """

    low = margin
    high = float(canvas_size) - margin
    return (min(high, max(low, center[0])), min(high, max(low, center[1])))


def _is_far_enough(
    center: tuple[float, float],
    placed: list[tuple[float, float]],
    *,
    min_distance: float,
) -> bool:
    """Return whether a center clears all existing placements.

    Parameters
    ----------
    center:
        Candidate center.
    placed:
        Existing centers.
    min_distance:
        Required center spacing.

    Returns
    -------
    bool
        Whether the candidate is sufficiently separated.
    """

    min_squared = min_distance * min_distance
    return all((center[0] - x) ** 2 + (center[1] - y) ** 2 >= min_squared for x, y in placed)


def _all_far_enough(centers: list[tuple[float, float]], *, min_distance: float) -> bool:
    """Return whether all center pairs clear the requested spacing.

    Parameters
    ----------
    centers:
        Pixel centers to compare.
    min_distance:
        Required center spacing.

    Returns
    -------
    bool
        Whether all pairwise distances are at least ``min_distance``.
    """

    for left, center in enumerate(centers):
        if not _is_far_enough(center, centers[:left], min_distance=min_distance):
            return False
    return True


def _draw_scatter_axes(draw: ImageDraw.ImageDraw, *, canvas_size: int, margin: int) -> None:
    """Draw unobtrusive scatter guide axes.

    Parameters
    ----------
    draw:
        PIL drawing context.
    canvas_size:
        Output image side length.
    margin:
        Drawable margin.
    """

    mid = canvas_size // 2
    draw.line([(margin, mid), (canvas_size - margin, mid)], fill=_AXIS_COLOR, width=1)
    draw.line([(mid, margin), (mid, canvas_size - margin)], fill=_AXIS_COLOR, width=1)


def _draw_point_fallback(
    draw: ImageDraw.ImageDraw,
    centers: list[tuple[float, float]],
    *,
    labels: Sequence[Any] | None,
) -> None:
    """Draw point markers for coords-only fallback rendering.

    Parameters
    ----------
    draw:
        PIL drawing context.
    centers:
        Pixel centers to draw.
    labels:
        Optional point labels.
    """

    radius = 5
    for index, (x, y) in enumerate(centers):
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=_POINT_COLOR,
            outline=_POINT_OUTLINE,
        )
        label = str(labels[index]) if labels is not None and index < len(labels) else str(index)
        _draw_text(draw, (x + radius + 2, y - radius - 1), label, fill=_TEXT_COLOR)


def _paste_scatter_thumbnails(
    canvas: Image.Image,
    centers: list[tuple[float, float]],
    images: Sequence[Image.Image],
    *,
    thumbnail_size: int,
) -> None:
    """Paste resized thumbnails onto the scatter canvas.

    Parameters
    ----------
    canvas:
        PIL canvas image.
    centers:
        Pixel centers.
    images:
        PIL images to paste.
    thumbnail_size:
        Maximum thumbnail side length.
    """

    for center, image in zip(centers, images, strict=True):
        thumb = _thumbnail(image, thumbnail_size)
        x = int(round(center[0] - thumb.width / 2.0))
        y = int(round(center[1] - thumb.height / 2.0))
        canvas.paste(thumb, (x, y))


def _draw_more_indicator(draw: ImageDraw.ImageDraw, *, canvas_size: int, text: str) -> None:
    """Draw a ``+K more`` cap indicator in the scatter image.

    Parameters
    ----------
    draw:
        PIL drawing context.
    canvas_size:
        Output image side length.
    text:
        Indicator text.
    """

    text_width, text_height = _measure_text(draw, text)
    pad = 6
    x0 = canvas_size - text_width - 2 * pad - 8
    y0 = canvas_size - text_height - 2 * pad - 8
    x1 = canvas_size - 8
    y1 = canvas_size - 8
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        fill=_MORE_FILL,
        outline=_MORE_OUTLINE,
    )
    _draw_text(draw, (x0 + pad, y0 + pad), text, fill=_TEXT_COLOR)
