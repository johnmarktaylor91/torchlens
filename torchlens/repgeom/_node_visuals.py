"""Repgeom node visuals: MDS/RDM/scree node-spec callbacks (C01 item 18).

PIL-only render-time images composed from ``tl.viz.render_*`` primitives,
returned as NodeSpec callbacks for ``Trace.draw(node_spec_fn=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..utils._multipass_access import get_multipass_attr
from ..utils.display import ensure_trace_visualizer_dir
from ..viz.node_plots import render_heatmap, render_image_scatter, render_lineplot
from ._geometry import (
    _SCATTER_CANVAS_SIZE,
    EffectiveDimensionalityInfo,
    _as_numpy_array,
    _check_square_distances,
    _effective_dimensionality_from_eigenvalues,
    _validate_finite,
    _validate_variance_threshold,
    rank_transform_rdm,
)

__tl_layer__ = "L6"


def mds_scatter_node_spec(
    *,
    max_thumbnails: int = 16,
    thumbnail_size: int = 36,
    canvas_size: int = _SCATTER_CANVAS_SIZE,
    min_distance: float | None = None,
) -> Callable[[Any, Any], Any | None]:
    """Return a draw-time node callback for stored MDS scatter annotations.

    The returned callback reads ``layer:<layer_label>`` or ``op:<op.label>``
    coordinate tensors from ``trace._annotation_blobs`` and composes a fresh
    PIL PNG every time a graph is drawn. When ``trace.raw_input`` is a PIL image
    batch with the same leading count as the coordinates, thumbnails are pasted
    at the normalized MDS positions. Otherwise, the callback renders an explicit
    point-cloud fallback image.

    Parameters
    ----------
    max_thumbnails:
        Maximum number of stimuli to draw before adding a ``+K more`` indicator.
    thumbnail_size:
        Maximum width and height for each pasted thumbnail.
    canvas_size:
        Width and height of the rendered scatter image in pixels.
    min_distance:
        Minimum center-to-center distance in pixels. Defaults to the thumbnail
        size for thumbnail scatters and a smaller point spacing for fallbacks.

    Returns
    -------
    Callable[[Any, Any], Any | None]
        ``node_spec_fn`` suitable for ``Trace.draw(node_spec_fn=...)``.

    Raises
    ------
    ValueError
        If sizing or cap parameters are invalid.
    """

    if max_thumbnails < 1:
        raise ValueError("max_thumbnails must be at least 1.")
    if thumbnail_size < 4:
        raise ValueError("thumbnail_size must be at least 4.")
    if canvas_size <= thumbnail_size * 2:
        raise ValueError("canvas_size must be larger than twice thumbnail_size.")

    def node_spec_fn(layer: Any, spec: Any) -> Any | None:
        """Apply an MDS scatter image to a matching node spec.

        Parameters
        ----------
        layer:
            Layer or op-like render node passed by TorchLens.
        spec:
            Default ``NodeSpec`` to mutate by replacement.

        Returns
        -------
        Any | None
            Updated spec when coordinates are available, otherwise ``None``.
        """

        trace = getattr(layer, "source_trace", None)
        if trace is None:
            return None
        key, coords = _mds_scatter_coords_for_node(trace, layer)
        if key is None or coords is None:
            return None

        images = _matching_pil_image_batch(getattr(trace, "raw_input", None), coords.shape[0])
        shown_count = min(max_thumbnails, coords.shape[0])
        more_count = max(0, coords.shape[0] - shown_count)
        fallback_reason = (
            None if images is not None else "points fallback: raw PIL image batch unavailable"
        )
        scatter = render_image_scatter(
            coords,
            images=images,
            max_items=max_thumbnails,
            thumbnail_size=thumbnail_size,
            canvas_size=canvas_size,
            min_distance=min_distance,
        )
        image_path = _write_mds_scatter_image(trace, key, scatter)
        tooltip = f"MDS thumbnail scatter for {key}: {coords.shape[0]} stimuli"
        if fallback_reason is not None:
            tooltip = f"MDS thumbnail scatter for {key} ({fallback_reason})"
        if more_count > 0:
            tooltip = f"{tooltip}; +{more_count} more"

        caption = str(getattr(layer, "layer_label", None) or getattr(layer, "label", key))
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


def rdm_node_spec(  # noqa: PLR0913 -- draw-time producer: six all-defaulted presentation dials ARE the spec'd surface
    *,
    max_stimuli: int = 8,
    thumbnail_size: int = 24,
    canvas_size: int = 360,
    cmap: str = "viridis",
    show_axis_thumbnails: bool = True,
    display: str = "raw",
) -> Callable[[Any, Any], Any | None]:
    """Return a draw-time node callback for stored RDM heatmap annotations.

    Parameters
    ----------
    max_stimuli:
        Maximum number of stimulus rows/columns to label before adding a cap
        marker.
    thumbnail_size:
        Maximum width and height for axis thumbnails.
    canvas_size:
        Width and height of the rendered heatmap image in pixels.
    cmap:
        Colormap passed to :func:`torchlens.viz.render_heatmap`.
    show_axis_thumbnails:
        Whether to use matching raw PIL image stimuli as symmetric axis
        thumbnails when available.
    display:
        Value transform applied before rendering: ``"raw"`` (default,
        historical behavior), ``"rank"``, or ``"percentile"`` -- the
        standard RSA display convention (Nili et al. 2014's
        percentile-ranked RDMs; thingsvision ships the same rank-scaled
        display) via :func:`torchlens.repgeom.rank_transform_rdm`.
        Display-only: stored annotations are untouched and the tooltip
        discloses the transform.

    Returns
    -------
    Callable[[Any, Any], Any | None]
        ``node_spec_fn`` suitable for ``Trace.draw(node_spec_fn=...)``.

    Raises
    ------
    ValueError
        If sizing or cap parameters are invalid, or ``display`` is not one
        of ``"raw"``, ``"rank"``, ``"percentile"``.
    """

    if max_stimuli < 1:
        raise ValueError("max_stimuli must be at least 1.")
    if thumbnail_size < 4:
        raise ValueError("thumbnail_size must be at least 4.")
    if canvas_size <= thumbnail_size * 2:
        raise ValueError("canvas_size must be larger than twice thumbnail_size.")
    if display not in ("raw", "rank", "percentile"):
        raise ValueError(
            f"Unsupported rdm_node_spec display: {display!r} "
            "(expected 'raw', 'rank', or 'percentile')."
        )

    def node_spec_fn(layer: Any, spec: Any) -> Any | None:
        """Apply an RDM heatmap image to a matching node spec.

        Parameters
        ----------
        layer:
            Layer or op-like render node passed by TorchLens.
        spec:
            Default ``NodeSpec`` to mutate by replacement.

        Returns
        -------
        Any | None
            Updated spec when an RDM is available, otherwise ``None``.
        """

        trace = getattr(layer, "source_trace", None)
        if trace is None:
            return None
        key, matrix = _rdm_matrix_for_node(trace, layer)
        if key is None or matrix is None:
            return None

        shown_matrix = matrix
        if display != "raw":
            shown_matrix = rank_transform_rdm(matrix, output=display)  # type: ignore[arg-type]
        images = None
        if show_axis_thumbnails:
            images = _matching_pil_image_batch(getattr(trace, "raw_input", None), matrix.shape[0])
        labels = None if images is not None else [f"s{index}" for index in range(matrix.shape[0])]
        heatmap = render_heatmap(
            shown_matrix,
            width=canvas_size,
            height=canvas_size,
            cmap=cmap,
            axis_images=images,
            axis_labels=labels,
            max_axis_items=max_stimuli,
        )
        image_path = _write_node_plot_image(trace, "rdm", key, heatmap)
        shown_count = min(max_stimuli, matrix.shape[0])
        more_count = max(0, matrix.shape[0] - shown_count)
        more_text = f"; +{more_count} more" if more_count > 0 else ""
        display_text = "" if display == "raw" else f", display={display}"
        tooltip = (
            f"RDM heatmap for {key}: metric=precomputed{display_text}, "
            f"N={matrix.shape[0]} stimuli{more_text}"
        )
        caption = str(getattr(layer, "layer_label", None) or getattr(layer, "label", key))
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


def scree_node_spec(
    *,
    max_components: int = 24,
    width: int = 320,
    height: int = 180,
    variance_threshold: float = 0.90,
) -> Callable[[Any, Any], Any | None]:
    """Return a draw-time node callback for stored scree annotations.

    Parameters
    ----------
    max_components:
        Maximum number of leading components to plot.
    width:
        Width of the rendered scree image in pixels.
    height:
        Height of the rendered scree image in pixels.
    variance_threshold:
        Cumulative variance fraction used for the threshold line and callout.

    Returns
    -------
    Callable[[Any, Any], Any | None]
        ``node_spec_fn`` suitable for ``Trace.draw(node_spec_fn=...)``.

    Raises
    ------
    ValueError
        If sizing, cap, or threshold parameters are invalid.
    """

    if max_components < 1:
        raise ValueError("max_components must be at least 1.")
    if width < 120:
        raise ValueError("width must be at least 120.")
    if height < 90:
        raise ValueError("height must be at least 90.")
    _validate_variance_threshold(variance_threshold)

    def node_spec_fn(layer: Any, spec: Any) -> Any | None:
        """Apply a scree plot image to a matching node spec.

        Parameters
        ----------
        layer:
            Layer or op-like render node passed by TorchLens.
        spec:
            Default ``NodeSpec`` to mutate by replacement.

        Returns
        -------
        Any | None
            Updated spec when scree eigenvalues are available, otherwise
            ``None``.
        """

        trace = getattr(layer, "source_trace", None)
        if trace is None:
            return None
        key, eigenvalues = _scree_eigenvalues_for_node(trace, layer)
        if key is None or eigenvalues is None:
            return None

        info = _effective_dimensionality_from_eigenvalues(
            eigenvalues,
            variance_threshold=variance_threshold,
        )
        image = _render_scree_image(
            info,
            max_components=max_components,
            width=width,
            height=height,
            variance_threshold=variance_threshold,
        )
        image_path = _write_node_plot_image(trace, "scree", key, image)
        caption = str(getattr(layer, "layer_label", None) or getattr(layer, "label", key))
        threshold_percent = int(round(variance_threshold * 100.0))
        tooltip = (
            f"Scree plot for {key}: PR={info['participation_ratio']:.2f}, "
            f"{info['n_components_for_threshold']} comps for {threshold_percent}% variance, "
            f"rank {info['effective_rank']}"
        )
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


def _scree_eigenvalues_for_node(trace: Any, node: Any) -> tuple[str | None, np.ndarray | None]:
    """Return stored scree eigenvalues for a rendered layer or op node.

    Parameters
    ----------
    trace:
        Trace that owns the annotation blobs.
    node:
        Rendered layer or op-like object.

    Returns
    -------
    tuple[str | None, np.ndarray | None]
        Base annotation key and one-dimensional eigenvalue vector when present.
    """

    try:
        blobs = trace._annotation_blobs
    except AttributeError:
        blobs = None
    if not isinstance(blobs, dict):
        return None, None
    candidates = []
    # A rolled multi-pass Layer has no single per-pass label; plain getattr
    # would leak the multi-pass ValueError tripwire and kill the whole draw.
    label = get_multipass_attr(node, "label", None, multipass=None)
    if label is not None:
        candidates.append(f"op:{label}")
    layer_label = get_multipass_attr(node, "layer_label", None, multipass=None)
    if layer_label is not None:
        candidates.append(f"layer:{layer_label}")
    for key in candidates:
        value = blobs.get(f"scree:{key}")
        if value is None:
            continue
        eigenvalues = _as_numpy_array(value)
        if eigenvalues.ndim == 1 and eigenvalues.shape[0] > 0:
            _validate_finite(eigenvalues, "scree eigenvalues")
            return key, np.clip(eigenvalues, 0.0, None)
    return None, None


def _render_scree_image(
    info: EffectiveDimensionalityInfo,
    *,
    max_components: int,
    width: int,
    height: int,
    variance_threshold: float,
) -> Image.Image:
    """Render a scree line plot with a compact statistics callout.

    Parameters
    ----------
    info:
        Effective-dimensionality statistics derived from stored eigenvalues.
    max_components:
        Maximum number of leading components to draw.
    width:
        Output image width in pixels.
    height:
        Output image height in pixels.
    variance_threshold:
        Cumulative variance threshold for the reference line and callout.

    Returns
    -------
    Image.Image
        RGB image containing the scree plot and text callout.
    """

    eigenvalues = info["eigenvalues"]
    n_components = min(max_components, max(1, eigenvalues.shape[0]))
    x_values = np.arange(1, n_components + 1, dtype=np.float64)
    variance = info["variance_explained"][:n_components]
    cumulative = info["cumulative_variance"][:n_components]
    if variance.shape[0] < n_components:
        variance = np.pad(variance, (0, n_components - variance.shape[0]))
        cumulative = np.pad(cumulative, (0, n_components - cumulative.shape[0]))
    threshold = np.full(n_components, variance_threshold, dtype=np.float64)
    plot = render_lineplot(
        np.vstack([variance, cumulative, threshold]),
        x_values=x_values,
        labels=("variance", "cumulative", "threshold"),
        width=width,
        height=height,
        y_min=0.0,
        y_max=1.0,
        colors=((48, 93, 170), (32, 128, 90), (160, 72, 72)),
        show_legend=True,
        x_label="component",
        y_label="fraction",
    )
    callout_height = 34
    canvas = Image.new("RGB", (width, height + callout_height), "white")
    canvas.paste(plot, (0, 0))
    draw = ImageDraw.Draw(canvas)
    threshold_percent = int(round(variance_threshold * 100.0))
    callout = (
        f"PR={info['participation_ratio']:.2f}, "
        f"{info['n_components_for_threshold']} comps for {threshold_percent}% var, "
        f"rank {info['effective_rank']}"
    )
    draw.rectangle([(0, height), (width - 1, height + callout_height - 1)], fill=(255, 255, 255))
    draw.line([(0, height), (width - 1, height)], fill=(215, 219, 226))
    draw.text((8, height + 10), callout, fill=(35, 39, 47))
    return canvas


def _rdm_matrix_for_node(trace: Any, node: Any) -> tuple[str | None, np.ndarray | None]:
    """Return a stored RDM matrix for a rendered layer or op node.

    Parameters
    ----------
    trace:
        Trace that owns the annotation blobs.
    node:
        Rendered layer or op-like object.

    Returns
    -------
    tuple[str | None, np.ndarray | None]
        Base annotation key and ``[N, N]`` matrix when present.
    """

    try:
        blobs = trace._annotation_blobs
    except AttributeError:
        blobs = None
    if not isinstance(blobs, dict):
        return None, None
    candidates = []
    # A rolled multi-pass Layer has no single per-pass label; plain getattr
    # would leak the multi-pass ValueError tripwire and kill the whole draw.
    label = get_multipass_attr(node, "label", None, multipass=None)
    if label is not None:
        candidates.append(f"op:{label}")
    layer_label = get_multipass_attr(node, "layer_label", None, multipass=None)
    if layer_label is not None:
        candidates.append(f"layer:{layer_label}")
    for key in candidates:
        value = blobs.get(f"rdm:{key}")
        if value is None:
            continue
        matrix = _as_numpy_array(value)
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] > 0:
            _check_square_distances(matrix)
            return key, matrix
    return None, None


def _mds_scatter_coords_for_node(trace: Any, node: Any) -> tuple[str | None, np.ndarray | None]:
    """Return stored MDS coordinates for a rendered layer or op node.

    Parameters
    ----------
    trace:
        Trace that owns the annotation blobs.
    node:
        Rendered layer or op-like object.

    Returns
    -------
    tuple[str | None, np.ndarray | None]
        Annotation key and ``[N, 2]`` coordinates when present.
    """

    try:
        blobs = trace._annotation_blobs
    except AttributeError:
        blobs = None
    if not isinstance(blobs, dict):
        return None, None
    candidates = []
    # A rolled multi-pass Layer has no single per-pass label; plain getattr
    # would leak the multi-pass ValueError tripwire and kill the whole draw.
    label = get_multipass_attr(node, "label", None, multipass=None)
    if label is not None:
        candidates.append(f"op:{label}")
    layer_label = get_multipass_attr(node, "layer_label", None, multipass=None)
    if layer_label is not None:
        candidates.append(f"layer:{layer_label}")
    for key in candidates:
        value = blobs.get(f"mds:{key}")
        if value is None:
            continue
        coords = _as_numpy_array(value)
        if coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0:
            _validate_finite(coords, "MDS scatter coordinates")
            return key, coords
    return None, None


def _matching_pil_image_batch(raw_input: Any, n_coords: int) -> Sequence[Any] | None:
    """Return a matching raw PIL image batch if one is available.

    Parameters
    ----------
    raw_input:
        Trace raw input payload.
    n_coords:
        Required number of stimuli.

    Returns
    -------
    Sequence[Any] | None
        PIL image sequence when the length exactly matches the coordinates.
    """

    if raw_input is None or isinstance(raw_input, str | bytes | bytearray):
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    if isinstance(raw_input, Sequence):
        sequence = raw_input
    elif hasattr(raw_input, "__len__") and hasattr(raw_input, "__iter__"):
        sequence = tuple(raw_input)
    else:
        return None
    if len(sequence) != n_coords:
        return None
    if not all(isinstance(item, Image.Image) for item in sequence):
        return None
    return sequence


def _write_mds_scatter_image(trace: Any, key: str, image: Any) -> Path:
    """Write a draw-time scatter image under the trace visualizer directory.

    Parameters
    ----------
    trace:
        Trace that owns the draw.
    key:
        Annotation key for the rendered coordinates.
    image:
        PIL image to save.

    Returns
    -------
    Path
        Local PNG path for ``NodeSpec.image``.
    """

    return _write_node_plot_image(trace, "mds_scatter", key, image)


def _write_node_plot_image(trace: Any, namespace: str, key: str, image: Any) -> Path:
    """Write a draw-time node plot image under the trace visualizer directory.

    Parameters
    ----------
    trace:
        Trace that owns the draw.
    namespace:
        Subdirectory namespace for the rendered image.
    key:
        Annotation key for the rendered payload.
    image:
        PIL image to save.

    Returns
    -------
    Path
        Local PNG path for ``NodeSpec.image``.
    """

    plot_dir = ensure_trace_visualizer_dir(trace) / namespace
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in key)
    image_path = plot_dir / f"{safe_key}.png"
    image.save(image_path)
    return image_path
