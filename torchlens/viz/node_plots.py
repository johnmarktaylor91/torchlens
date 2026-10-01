"""PIL-only render primitives for compact node visualizations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from PIL import Image, ImageDraw

# Historical import homes: the axis-selection internals moved to
# _heatmap_axes.py in the file-size split; existing tests import them here.
from ._heatmap_axes import (
    _axis_selection_with_marker as _axis_selection_with_marker,
    _draw_heatmap_axis_items,
    _draw_text,
    _heatmap_axis_margins,
    _measure_text,
    _select_heatmap_axis_indices as _select_heatmap_axis_indices,
    _thumbnail,
)

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
    row_labels: Sequence[Any] | None = None,
    col_labels: Sequence[Any] | None = None,
    max_axis_items: int = 8,
) -> Image.Image:
    """Render a 2D numeric array as an RGB heatmap of discrete cells.

    Cells are drawn with NEAREST-neighbor scaling (tviz FIX-H): the data is a
    grid of discrete values, so no interpolated colors are ever introduced
    between cells.

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
        Optional text labels applied to BOTH axes (convenience for square
        symmetric data). Per-axis ``row_labels``/``col_labels`` take
        precedence on their axis when also given.
    row_labels:
        Optional text labels for the row axis (drawn on the left). For an
        attention picture these are the QUERY tokens (rows attend from).
    col_labels:
        Optional text labels for the column axis (drawn on top). For an
        attention picture these are the KEY tokens (columns attend to).
    max_axis_items:
        Maximum number of axis thumbnails or labels to draw before adding a
        cap marker. The rendered ``+N more`` count is computed AFTER the
        final overlap selection, so it always states exactly how many axis
        items are not shown (tviz FIX-H).

    Returns
    -------
    Image.Image
        RGB heatmap image of exactly ``(width, height)``.

    Raises
    ------
    ValueError
        If inputs have invalid shape, size, cap, or colormap, or if explicit
        ``vmin``/``vmax`` bounds are non-finite or reversed (``vmax < vmin``).
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

    has_axis = (
        axis_images is not None
        or axis_labels is not None
        or row_labels is not None
        or col_labels is not None
    )
    if not has_axis:
        return heatmap.resize((width, height), Image.Resampling.NEAREST)

    left_labels = _stringify_labels(row_labels if row_labels is not None else axis_labels)
    top_labels = _stringify_labels(col_labels if col_labels is not None else axis_labels)
    top_count = min(max_axis_items, array.shape[1])
    left_count = min(max_axis_items, array.shape[0])
    measure_canvas = Image.new("RGB", (1, 1), "white")
    measure_draw = ImageDraw.Draw(measure_canvas)
    plot_left, plot_top = _heatmap_axis_margins(
        measure_draw,
        width=width,
        height=height,
        left_labels=left_labels,
        top_labels=top_labels,
        images=axis_images,
        top_count=top_count,
        left_count=left_count,
    )
    plot_width = max(1, width - plot_left - 6)
    plot_height = max(1, height - plot_top - 6)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        heatmap.resize((plot_width, plot_height), Image.Resampling.NEAREST), (plot_left, plot_top)
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
        labels=top_labels,
        count=top_count,
        total=array.shape[1],
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
        labels=left_labels,
        count=left_count,
        total=array.shape[0],
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
    xs: np.ndarray[Any, np.dtype[np.float64]] = np.arange(n_points, dtype=np.float64)
    if x_values is not None:
        xs = np.asarray(_as_numpy(x_values), dtype=np.float64)
        if xs.ndim != 1 or xs.shape[0] != n_points:
            raise ValueError("x_values must have shape [K].")

    # A point is drawable only when BOTH its x and y are finite. Checking the
    # two axes independently accepted inputs with finite y's and finite x's at
    # DISJOINT indices -- zero drawable pairs -- and then rendered a blank plot.
    finite_x_mask = np.isfinite(xs)
    finite_x = xs[finite_x_mask]
    finite_pairs = np.isfinite(values) & finite_x_mask[np.newaxis, :]
    if not np.any(finite_pairs):
        raise ValueError("render_lineplot requires at least one finite (x, y) point.")
    # The auto y-range must reflect only points that will actually be drawn, so
    # a y-value sitting at a non-finite x column does not stretch the axis.
    plotted_y = values[finite_pairs]
    low_y = float(np.min(plotted_y)) if y_min is None else float(y_min)
    high_y = float(np.max(plotted_y)) if y_max is None else float(y_max)
    if not np.isfinite(low_y) or not np.isfinite(high_y):
        raise ValueError("y_min and y_max must be finite when provided.")
    if high_y < low_y:
        low_y, high_y = high_y, low_y
    if high_y == low_y:
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
    legend_height = _legend_height(labels, palette) if show_legend and labels is not None else 0
    margin_top = (12 * _DRAW_SCALE) + legend_height
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
        _draw_legend(
            draw,
            labels=labels,
            colors=palette,
            plot_left=plot_left,
            plot_right=plot_right,
            legend_top=12 * _DRAW_SCALE,
        )
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
        Optional minimum center-to-center distance in pixels. Pass ``0`` to
        disable overlap-avoidance and draw every item at its exact
        projected coordinate.
    show_axes:
        Whether to draw faint central guide axes.
    background:
        RGB canvas background.

    Returns
    -------
    Image.Image
        RGB scatter image of exactly ``(canvas_size, canvas_size)``. Close
        centers are deterministically spread apart for legibility; whenever
        any item is moved off its true projected coordinate, the image
        carries a quantified ``spread <=Npx`` marker in the lower-left
        corner.

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
    true_centers = _coords_to_pixel_centers(
        array[:shown_count], canvas_size=canvas_size, margin=margin
    )
    spacing = (
        min_distance if min_distance is not None else (float(thumbnail_size) if images else 14.0)
    )
    centers = _spread_close_centers(
        true_centers, canvas_size=canvas_size, margin=margin, min_distance=spacing
    )
    # Position is the datum in a scatter: when overlap-avoidance moves any
    # item off its true projected coordinate, the image itself must say so.
    max_displacement = max(
        (
            math.hypot(moved[0] - original[0], moved[1] - original[1])
            for original, moved in zip(true_centers, centers)
        ),
        default=0.0,
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
    if max_displacement > 0.5:
        _draw_spread_indicator(
            draw,
            canvas_size=canvas_size,
            text=f"spread <={max(1, int(math.ceil(max_displacement)))}px",
        )
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
        If explicit bounds are not finite, or are reversed (``vmax < vmin``).
        Degenerate equal explicit bounds (``vmax == vmin``) are not an error and
        yield an all-zero (uniform) array, matching an empty value range.
    """

    finite = np.isfinite(array)
    normalized = np.zeros(array.shape, dtype=np.float64)
    if not np.any(finite):
        return normalized
    low = float(np.min(array[finite])) if vmin is None else float(vmin)
    high = float(np.max(array[finite])) if vmax is None else float(vmax)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("vmin and vmax must be finite when provided.")
    # Reversed explicit bounds are a caller error and were silently swallowed
    # here (returning a uniform array), contradicting this function's documented
    # contract. Surface it. Data-derived bounds (vmin/vmax=None) are always
    # ordered, so this can only trip on explicit reversed vmin/vmax.
    if high < low:
        raise ValueError("vmax must be greater than or equal to vmin.")
    if high == low:
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
    # Clamp to the plot rectangle. Values outside the (possibly caller-set)
    # y_min/y_max or x range would otherwise be drawn on top of the axes,
    # labels, and title. The rectangle is convex, so clamping both endpoints
    # keeps every drawn segment inside the plot area.
    x = min(max(x, float(plot_left)), float(plot_right))
    y = min(max(y, float(plot_top)), float(plot_bottom))
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
    plot_left: int,
    plot_right: int,
    legend_top: int,
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
    plot_left:
        Plot area left coordinate.
    plot_right:
        Plot area right coordinate.
    legend_top:
        Top coordinate of the reserved legend band.
    """

    if (
        _legend_bbox(
            draw,
            labels=labels,
            colors=colors,
            plot_left=plot_left,
            plot_right=plot_right,
            legend_top=legend_top,
        )
        is None
    ):
        return
    for index, label in enumerate(labels[: len(colors)]):
        text = str(label)
        text_width, text_height = _measure_text(draw, text, scale=_DRAW_SCALE)
        x0 = max(plot_left, plot_right - text_width - 28 * _DRAW_SCALE)
        y0 = legend_top + index * (text_height + 5 * _DRAW_SCALE)
        draw.line(
            [(x0, y0 + text_height / 2.0), (x0 + 18 * _DRAW_SCALE, y0 + text_height / 2.0)],
            fill=colors[index],
            width=2 * _DRAW_SCALE,
        )
        _draw_text(draw, (x0 + 22 * _DRAW_SCALE, y0), text, fill=_TEXT_COLOR, scale=_DRAW_SCALE)


def _legend_height(labels: Sequence[Any], colors: Sequence[RGBColor]) -> int:
    """Return the scaled height needed for a vertically stacked legend.

    Parameters
    ----------
    labels:
        Series labels.
    colors:
        Series colors.

    Returns
    -------
    int
        Height in supersampled pixels, including the gap below the legend.
    """

    if not labels or not colors:
        return 0
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_heights = [
        _measure_text(draw, str(label), scale=_DRAW_SCALE)[1] for label in labels[: len(colors)]
    ]
    row_height = max(text_heights) + 5 * _DRAW_SCALE
    return len(text_heights) * row_height + 4 * _DRAW_SCALE


def _legend_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    labels: Sequence[Any],
    colors: Sequence[RGBColor],
    plot_left: int,
    plot_right: int,
    legend_top: int,
) -> tuple[int, int, int, int] | None:
    """Return the pixel bounds of a compact stacked legend.

    Parameters
    ----------
    draw:
        PIL drawing context used to measure legend labels.
    labels:
        Series labels.
    colors:
        Series colors.
    plot_left:
        Plot area left coordinate.
    plot_right:
        Plot area right coordinate.
    legend_top:
        Top coordinate of the reserved legend band.

    Returns
    -------
    tuple[int, int, int, int] | None
        ``(left, top, right, bottom)`` bounds, or ``None`` for no entries.
    """

    entries = labels[: len(colors)]
    if not entries:
        return None
    bounds: list[tuple[int, int, int, int]] = []
    for index, label in enumerate(entries):
        text_width, text_height = _measure_text(draw, str(label), scale=_DRAW_SCALE)
        x0 = max(plot_left, plot_right - text_width - 28 * _DRAW_SCALE)
        y0 = legend_top + index * (text_height + 5 * _DRAW_SCALE)
        bounds.append((x0, y0, x0 + text_width + 22 * _DRAW_SCALE, y0 + text_height))
    left = min(bound[0] for bound in bounds)
    top = min(bound[1] for bound in bounds)
    right = max(bound[2] for bound in bounds)
    bottom = max(bound[3] for bound in bounds)
    return left, top, right, bottom


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


def _draw_spread_indicator(draw: ImageDraw.ImageDraw, *, canvas_size: int, text: str) -> None:
    """Draw the displacement disclosure in the scatter's lower-left corner.

    Parameters
    ----------
    draw:
        PIL drawing context.
    canvas_size:
        Output image side length.
    text:
        Quantified displacement text (e.g. ``"spread <=24px"``).
    """

    text_width, text_height = _measure_text(draw, text)
    pad = 6
    x0 = 8
    y0 = canvas_size - text_height - 2 * pad - 8
    x1 = 8 + text_width + 2 * pad
    y1 = canvas_size - 8
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        fill=_MORE_FILL,
        outline=_MORE_OUTLINE,
    )
    _draw_text(draw, (x0 + pad, y0 + pad), text, fill=_TEXT_COLOR)
