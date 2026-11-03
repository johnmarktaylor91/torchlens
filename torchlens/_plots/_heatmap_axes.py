"""Heatmap axis-label layout: margins, overlap selection, honest omission.

Split from ``node_plots.py`` (file-size ratchet): the tviz FIX-H machinery --
per-axis label margins, deterministic non-overlapping selection, and the
``+N more`` omission marker computed AFTER the final overlap selection so the
count states exactly how many axis items are not drawn.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

_TEXT_COLOR = (35, 39, 47)


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
    fill: tuple[int, int, int],
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


def _heatmap_axis_margins(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    left_labels: list[str] | None,
    top_labels: list[str] | None,
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
    left_labels:
        Optional row-axis string labels.
    top_labels:
        Optional column-axis string labels.
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
    top_height = _max_label_height(draw, top_labels, top_count)
    left_width = _max_label_width(draw, left_labels, left_count)
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

    if count <= 0:
        return
    thumb_size = max(10, min(24, plot_top - 8 if axis == "top" else plot_left - 8))
    indices, more_count = _axis_selection_with_marker(
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
    )
    more_text = f"+{more_count} more" if more_count > 0 else None
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


def _axis_selection_with_marker(  # noqa: PLR0913 -- the axis-geometry tuple is explicit by design
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
) -> tuple[list[int], int]:
    """Return the final axis item selection and its honest omission count.

    tviz FIX-H: the omission count is computed AFTER the final overlap
    selection -- the ``+N more`` marker states exactly how many axis items
    are NOT drawn, including those dropped by decimation, not just those
    over the cap. Reserving marker space can itself shrink the selection
    (which can widen the marker text), so this iterates to a fixpoint; the
    count is monotonically non-decreasing across iterations, so it
    terminates.

    Parameters
    ----------
    draw:
        PIL drawing context used for text measurement.
    labels:
        Optional string labels for this axis.
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

    Returns
    -------
    tuple[list[int], int]
        Drawn item indices and the count of axis items not drawn.
    """

    def select(more_text: str | None) -> list[int]:
        """Run overlap selection with space reserved for ``more_text``."""

        return _select_heatmap_axis_indices(
            draw,
            labels=labels,
            has_images=has_images,
            count=count,
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
        )

    indices = select(None)
    more_count = total - len(indices)
    while more_count > 0:
        indices = select(f"+{more_count} more")
        new_count = total - len(indices)
        if new_count == more_count:
            break
        more_count = new_count
    return indices, more_count


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
