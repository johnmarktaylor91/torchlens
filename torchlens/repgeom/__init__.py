"""Import-clean representation geometry helpers.

This provisional module intentionally depends only on NumPy and Torch. The
helpers are for visualization-oriented representation geometry, not inference.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
import tempfile
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch

DistanceMetric: TypeAlias = Literal["euclidean", "cosine", "correlation"]
MDSInfo: TypeAlias = dict[str, int | float | bool | str]
MDSEvolution: TypeAlias = "OrderedDict[str, np.ndarray]"

_RANK_TOLERANCE = 1e-12
_SYMMETRY_TOLERANCE = 1e-10
_SCATTER_CANVAS_SIZE = 420
_SCATTER_BACKGROUND = (255, 255, 255)
_SCATTER_AXIS_COLOR = (215, 219, 226)
_SCATTER_POINT_COLOR = (48, 93, 170)
_SCATTER_TEXT_COLOR = (35, 39, 47)
_SCATTER_MORE_FILL = (255, 255, 255)
_SCATTER_MORE_OUTLINE = (120, 128, 140)
_SCATTER_CAPTION_RESERVE = 56


def _as_numpy_array(value: Any) -> np.ndarray:
    """Return ``value`` as a CPU float64 NumPy array.

    Parameters
    ----------
    value:
        NumPy-like or Torch tensor input.

    Returns
    -------
    np.ndarray
        Float64 NumPy array.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _validate_finite(array: np.ndarray, name: str) -> None:
    """Raise when ``array`` contains NaN or Inf values.

    Parameters
    ----------
    array:
        Array to check.
    name:
        Human-readable value name for errors.

    Raises
    ------
    ValueError
        If any array value is not finite.
    """

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")


def _looks_like_distance_matrix(array: np.ndarray) -> bool:
    """Return whether ``array`` has distance-matrix structure.

    Parameters
    ----------
    array:
        Candidate input array.

    Returns
    -------
    bool
        True when the input is square, symmetric, and has a zero diagonal.
    """

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False
    return bool(
        np.allclose(array, array.T, atol=_SYMMETRY_TOLERANCE, rtol=0.0)
        and np.allclose(np.diag(array), 0.0, atol=_SYMMETRY_TOLERANCE, rtol=0.0)
    )


def _check_square_distances(distances: np.ndarray) -> None:
    """Validate a square pairwise distance matrix.

    Parameters
    ----------
    distances:
        Pairwise distance matrix.

    Raises
    ------
    ValueError
        If the matrix is not a valid finite symmetric distance matrix.
    """

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square pairwise distance matrix.")
    _validate_finite(distances, "distances")
    if not np.allclose(distances, distances.T, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise ValueError("distances must be symmetric.")
    if not np.allclose(np.diag(distances), 0.0, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise ValueError("distances must have a zero diagonal.")
    if np.any(distances < -_SYMMETRY_TOLERANCE):
        raise ValueError("distances must be non-negative.")


def _check_stimulus_count(n_stimuli: int, min_n: int) -> None:
    """Validate the minimum number of stimuli for display-oriented MDS.

    Parameters
    ----------
    n_stimuli:
        Number of rows in the representation.
    min_n:
        User-facing minimum stimulus gate.

    Raises
    ------
    ValueError
        If the input has too few stimuli.
    """

    if n_stimuli < 3:
        raise ValueError("classical_mds requires at least 3 stimuli.")
    if n_stimuli < min_n:
        raise ValueError(
            f"classical_mds has too few stimuli for a stable visualization: "
            f"got {n_stimuli}, need at least {min_n}."
        )


def _has_duplicate_distances(distances: np.ndarray) -> bool:
    """Return whether any distinct stimulus pair has zero distance.

    Parameters
    ----------
    distances:
        Square pairwise distance matrix.

    Returns
    -------
    bool
        True when off-diagonal distances indicate duplicate stimuli.
    """

    off_diagonal_zero = np.isclose(distances, 0.0, atol=_SYMMETRY_TOLERANCE, rtol=0.0)
    np.fill_diagonal(off_diagonal_zero, False)
    return bool(np.any(off_diagonal_zero))


def _positive_rank_tolerance(eigenvalues: np.ndarray) -> float:
    """Return the numerical threshold for positive eigenvalues.

    Parameters
    ----------
    eigenvalues:
        Eigenvalues from the centered Gram matrix.

    Returns
    -------
    float
        Scale-aware positivity threshold.
    """

    scale = max(1.0, float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 1.0)
    return _RANK_TOLERANCE * scale


def _canonicalize_axis_signs(embedding: np.ndarray) -> np.ndarray:
    """Apply a deterministic sign convention to embedding axes.

    Parameters
    ----------
    embedding:
        Row-wise MDS coordinates.

    Returns
    -------
    np.ndarray
        Coordinates with the largest-magnitude loading on each axis positive.
    """

    result = embedding.copy()
    for axis in range(result.shape[1]):
        column = result[:, axis]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            result[:, axis] *= -1.0
    return result


def activation_distance_matrix(
    activations: Any,
    metric: DistanceMetric = "euclidean",
) -> np.ndarray:
    """Return pairwise dissimilarities for an activation batch.

    The first dimension is treated as the stimulus/item dimension and all
    remaining dimensions are flattened per item.

    Parameters
    ----------
    activations:
        Activation array or tensor with shape ``[N, ...]``.
    metric:
        Dissimilarity metric. Supported values are ``"euclidean"``,
        ``"cosine"``, and ``"correlation"``.

    Returns
    -------
    np.ndarray
        Square ``N x N`` pairwise dissimilarity matrix.

    Raises
    ------
    ValueError
        If activations are non-finite, empty, or the metric is unsupported.
    """

    array = _as_numpy_array(activations)
    _validate_finite(array, "activations")
    if array.ndim < 1 or array.shape[0] == 0:
        raise ValueError("activations must have a non-empty leading stimulus dimension.")

    features = array.reshape(array.shape[0], -1)
    if metric == "euclidean":
        differences = features[:, None, :] - features[None, :, :]
        distances = np.sqrt(np.sum(differences * differences, axis=-1))
    elif metric == "cosine":
        distances = _angular_dissimilarity(features, center_rows=False)
    elif metric == "correlation":
        distances = _angular_dissimilarity(features, center_rows=True)
    else:
        raise ValueError(f"Unsupported activation distance metric: {metric!r}.")

    distances = (distances + distances.T) * 0.5
    np.fill_diagonal(distances, 0.0)
    return distances


def _angular_dissimilarity(features: np.ndarray, *, center_rows: bool) -> np.ndarray:
    """Return cosine or correlation dissimilarities for row-wise features.

    Parameters
    ----------
    features:
        Two-dimensional row-wise feature matrix.
    center_rows:
        Whether to subtract each row's feature mean before normalization.

    Returns
    -------
    np.ndarray
        Square ``1 - similarity`` dissimilarity matrix.

    Raises
    ------
    ValueError
        If a row has zero norm under the requested normalization.
    """

    working = features - features.mean(axis=1, keepdims=True) if center_rows else features.copy()
    norms = np.linalg.norm(working, axis=1, keepdims=True)
    if np.any(norms <= _SYMMETRY_TOLERANCE):
        metric_name = "correlation" if center_rows else "cosine"
        raise ValueError(f"{metric_name} distance is undefined for zero-norm stimuli.")
    normalized = working / norms
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return 1.0 - similarities


def classical_mds(
    data: Any,
    n_components: int = 2,
    *,
    min_n: int = 8,
) -> tuple[np.ndarray, MDSInfo]:
    """Embed pairwise distances or row-wise features with classical MDS.

    Square symmetric inputs with zero diagonal are interpreted as precomputed
    distances. Other inputs are treated as ``[N, ...]`` features and converted
    to Euclidean distances first. Negative centered-Gram eigenvalues are clipped
    to zero and reported because non-PSD dissimilarities are expected for some
    visualization metrics.

    Parameters
    ----------
    data:
        Pairwise distance matrix or row-wise coordinate/feature data.
    n_components:
        Number of embedding axes to return.
    min_n:
        Minimum number of stimuli required for visualization-oriented MDS.

    Returns
    -------
    tuple[np.ndarray, MDSInfo]
        Embedding with shape ``[N, n_components]`` and diagnostic metadata.

    Raises
    ------
    ValueError
        If inputs are non-finite, too small, duplicate, or rank-deficient.
    """

    if n_components < 1:
        raise ValueError("n_components must be at least 1.")
    if min_n < 3:
        raise ValueError("min_n must be at least 3.")

    array = _as_numpy_array(data)
    _validate_finite(array, "data")
    distances = (
        array.copy() if _looks_like_distance_matrix(array) else activation_distance_matrix(array)
    )
    _check_square_distances(distances)
    n_stimuli = distances.shape[0]
    _check_stimulus_count(n_stimuli, min_n)
    if _has_duplicate_distances(distances):
        raise ValueError("classical_mds input is rank-deficient or contains duplicate stimuli.")

    squared = distances * distances
    centering = np.eye(n_stimuli) - np.full((n_stimuli, n_stimuli), 1.0 / n_stimuli)
    gram = -0.5 * centering @ squared @ centering
    gram = (gram + gram.T) * 0.5

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive_tolerance = _positive_rank_tolerance(eigenvalues)
    positive_mask = eigenvalues > positive_tolerance
    effective_rank = int(np.count_nonzero(positive_mask))
    if effective_rank < n_components:
        raise ValueError(
            f"classical_mds input is rank-deficient for {n_components} components: "
            f"effective rank is {effective_rank}."
        )

    negative_mask = eigenvalues < -positive_tolerance
    negative_mass = float(np.sum(np.abs(eigenvalues[negative_mask])))
    positive_mass = float(np.sum(eigenvalues[positive_mask]))
    total_reported_mass = positive_mass + negative_mass
    discarded_fraction = negative_mass / total_reported_mass if total_reported_mass > 0.0 else 0.0

    selected_values = np.clip(eigenvalues[:n_components], 0.0, None)
    embedding = eigenvectors[:, :n_components] * np.sqrt(selected_values)
    embedding = _canonicalize_axis_signs(embedding)
    info: MDSInfo = {
        "n_stimuli": n_stimuli,
        "n_components": n_components,
        "effective_rank": effective_rank,
        "min_n": min_n,
        "negative_eigenvalue_count": int(np.count_nonzero(negative_mask)),
        "discarded_variance_fraction": discarded_fraction,
        "input_kind": "distances" if _looks_like_distance_matrix(array) else "features",
    }
    return embedding, info


def procrustes_align(source_2d: Any, target_2d: Any) -> np.ndarray:
    """Align ``source_2d`` to ``target_2d`` with rotation-only Procrustes.

    The alignment centers both point clouds and applies an orthogonal rotation
    without scaling. Reflections are forbidden by forcing ``det(R) = +1`` so
    layer-to-layer representation displays do not flip left and right.

    Parameters
    ----------
    source_2d:
        Source coordinates with shape ``[N, 2]``.
    target_2d:
        Target coordinates with shape ``[N, 2]``.

    Returns
    -------
    np.ndarray
        Source coordinates rotated and translated into the target frame.

    Raises
    ------
    ValueError
        If inputs are non-finite, malformed, or rank-deficient.
    """

    source = _as_numpy_array(source_2d)
    target = _as_numpy_array(target_2d)
    _validate_procrustes_points(source, "source_2d")
    _validate_procrustes_points(target, "target_2d")
    if source.shape != target.shape:
        raise ValueError("source_2d and target_2d must have the same shape.")

    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("procrustes_align requires non-rank-deficient 2D point clouds.")

    u, _, vt = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return source_centered @ rotation + target_mean


def mds_evolution(
    trace: Any,
    save: Any | None = None,
    *,
    metric: DistanceMetric = "euclidean",
    min_n: int = 8,
    align: bool = True,
) -> MDSEvolution:
    """Compute and annotate per-layer 2D MDS coordinates.

    Parameters
    ----------
    trace:
        Captured TorchLens trace with saved activation payloads.
    save:
        Optional selector limiting which layers or pass-qualified ops to
        process. When omitted, all layers with saved activations are used.
    metric:
        Activation distance metric passed to :func:`activation_distance_matrix`.
    min_n:
        Minimum number of stimuli required by :func:`classical_mds`.
    align:
        Whether to Procrustes-align each processed embedding to the previous
        processed embedding.

    Returns
    -------
    OrderedDict[str, np.ndarray]
        Coordinate arrays keyed by ``layer:<layer_label>`` for single-pass
        layers and ``op:<op.label>`` for pass-qualified recurrent selections.

    Raises
    ------
    ValueError
        If a selected site has no saved activation, is recurrent without a
        pass-qualified selection, or fails MDS preconditions.
    """

    selected = _selected_mds_sites(trace, save)
    coords_by_key: MDSEvolution = OrderedDict()
    previous_coords: np.ndarray | None = None
    for key, site, activations in selected:
        distances = activation_distance_matrix(activations, metric=metric)
        coords, _info = classical_mds(distances, n_components=2, min_n=min_n)
        if align and previous_coords is not None:
            coords = procrustes_align(coords, previous_coords)
        _annotate_mds_coords(trace, site, coords)
        coords_by_key[key] = coords
        previous_coords = coords
    return coords_by_key


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
        scatter = _render_mds_scatter_image(
            coords,
            images=images,
            max_thumbnails=max_thumbnails,
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
        value = blobs.get(key)
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


def _render_mds_scatter_image(
    coords: np.ndarray,
    *,
    images: Sequence[Any] | None,
    max_thumbnails: int,
    thumbnail_size: int,
    canvas_size: int,
    min_distance: float | None,
) -> Any:
    """Compose a PIL scatter image for MDS coordinates.

    Parameters
    ----------
    coords:
        ``[N, 2]`` coordinate matrix.
    images:
        Optional matching PIL image stimuli.
    max_thumbnails:
        Maximum number of stimuli to draw.
    thumbnail_size:
        Maximum thumbnail side length.
    canvas_size:
        Output image side length.
    min_distance:
        Optional minimum center spacing in pixels.

    Returns
    -------
    Any
        PIL ``Image`` containing the scatter.
    """

    from PIL import Image, ImageDraw

    shown_count = min(max_thumbnails, coords.shape[0])
    margin = thumbnail_size / 2.0 + _SCATTER_CAPTION_RESERVE
    centers = _coords_to_pixel_centers(coords[:shown_count], canvas_size=canvas_size, margin=margin)
    spacing = min_distance if min_distance is not None else (thumbnail_size if images else 14.0)
    centers = _spread_close_centers(
        centers, canvas_size=canvas_size, margin=margin, min_distance=spacing
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), _SCATTER_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    _draw_scatter_axes(draw, canvas_size=canvas_size, margin=int(round(margin)))
    if images is None:
        _draw_point_fallback(draw, centers)
    else:
        _paste_scatter_thumbnails(
            canvas,
            centers,
            images[:shown_count],
            thumbnail_size=thumbnail_size,
        )
    more_count = coords.shape[0] - shown_count
    if more_count > 0:
        _draw_more_indicator(draw, canvas_size=canvas_size, text=f"+{more_count} more")
    return canvas


def _coords_to_pixel_centers(
    coords: np.ndarray,
    *,
    canvas_size: int,
    margin: float,
) -> list[tuple[float, float]]:
    """Normalize MDS coordinates into drawable pixel centers.

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

    placed: list[tuple[float, float]] = []
    for index, center in enumerate(centers):
        candidate = _clamp_center(center, canvas_size=canvas_size, margin=margin)
        if _is_far_enough(candidate, placed, min_distance=min_distance):
            placed.append(candidate)
            continue
        for offset in _spiral_offsets(index=index, step=max(2.0, min_distance)):
            candidate = _clamp_center(
                (center[0] + offset[0], center[1] + offset[1]),
                canvas_size=canvas_size,
                margin=margin,
            )
            if _is_far_enough(candidate, placed, min_distance=min_distance):
                break
        placed.append(candidate)
    return placed


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


def _draw_scatter_axes(draw: Any, *, canvas_size: int, margin: int) -> None:
    """Draw unobtrusive MDS guide axes.

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
    draw.line([(margin, mid), (canvas_size - margin, mid)], fill=_SCATTER_AXIS_COLOR, width=1)
    draw.line([(mid, margin), (mid, canvas_size - margin)], fill=_SCATTER_AXIS_COLOR, width=1)


def _draw_point_fallback(draw: Any, centers: list[tuple[float, float]]) -> None:
    """Draw point markers for coords-only fallback rendering.

    Parameters
    ----------
    draw:
        PIL drawing context.
    centers:
        Pixel centers to draw.
    """

    radius = 5
    for index, (x, y) in enumerate(centers):
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=_SCATTER_POINT_COLOR,
            outline=(20, 48, 100),
        )
        draw.text((x + radius + 2, y - radius - 1), str(index), fill=_SCATTER_TEXT_COLOR)


def _paste_scatter_thumbnails(
    canvas: Any,
    centers: list[tuple[float, float]],
    images: Sequence[Any],
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

    from PIL import Image

    for center, image in zip(centers, images, strict=True):
        thumb = cast(Any, image).convert("RGB")
        thumb.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        x = int(round(center[0] - thumb.width / 2.0))
        y = int(round(center[1] - thumb.height / 2.0))
        canvas.paste(thumb, (x, y))


def _draw_more_indicator(draw: Any, *, canvas_size: int, text: str) -> None:
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

    try:
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text)
    pad = 6
    x0 = canvas_size - text_width - 2 * pad - 8
    y0 = canvas_size - text_height - 2 * pad - 8
    x1 = canvas_size - 8
    y1 = canvas_size - 8
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        fill=_SCATTER_MORE_FILL,
        outline=_SCATTER_MORE_OUTLINE,
    )
    draw.text((x0 + pad, y0 + pad), text, fill=_SCATTER_TEXT_COLOR)


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

    output_dir = getattr(trace, "_visualizer_dir", None)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="torchlens_visualizers_")
        trace._visualizer_dir = str(output_dir)
    scatter_dir = Path(str(output_dir)) / "mds_scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in key)
    image_path = scatter_dir / f"{safe_key}.png"
    image.save(image_path)
    return image_path


def _selected_mds_sites(trace: Any, save: Any | None) -> list[tuple[str, Any, Any]]:
    """Resolve the layer or op payloads that should receive MDS coordinates.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    save:
        Optional selector passed by the user.

    Returns
    -------
    list[tuple[str, Any, Any]]
        Tuples of annotation key, resolved annotation site, and activation
        payload.
    """

    if save is None:
        return _default_saved_mds_sites(trace)

    sites = list(trace.resolve_sites(save, max_fanout=1_000_000))
    selected_by_layer: dict[str, list[Any]] = OrderedDict()
    for site in sites:
        selected_by_layer.setdefault(str(getattr(site, "layer_label")), []).append(site)

    selected: list[tuple[str, Any, Any]] = []
    for layer_label, layer_sites in selected_by_layer.items():
        layer = trace.layer_logs[layer_label]
        if int(getattr(layer, "num_passes", 1)) > 1:
            if len(layer_sites) != 1:
                _raise_recurrent_layer_requires_pass(layer)
            site = layer_sites[0]
            if not _site_label_is_pass_qualified(site):
                _raise_recurrent_layer_requires_pass(layer)
            selected.append(_op_mds_site(site))
        else:
            selected.append(_single_pass_layer_mds_site(layer))
    return selected


def _default_saved_mds_sites(trace: Any) -> list[tuple[str, Any, Any]]:
    """Return saved single-pass layer payloads for default MDS evolution.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.

    Returns
    -------
    list[tuple[str, Any, Any]]
        Tuples of annotation key, resolved site, and activation payload.
    """

    selected: list[tuple[str, Any, Any]] = []
    for layer in trace.layers:
        if int(getattr(layer, "num_passes", 1)) > 1:
            saved_ops = [op for op in layer.ops if bool(getattr(op, "has_saved_activation", False))]
            if saved_ops:
                _raise_recurrent_layer_requires_pass(layer)
            continue
        if bool(getattr(layer, "has_saved_activation", False)):
            selected.append(_single_pass_layer_mds_site(layer))
    if not selected:
        raise ValueError(
            "mds_evolution requires saved activations; capture with save= covering "
            "the MDS layers before calling mds_evolution."
        )
    return selected


def _single_pass_layer_mds_site(layer: Any) -> tuple[str, Any, Any]:
    """Return the MDS payload tuple for a single-pass layer.

    Parameters
    ----------
    layer:
        Aggregate single-pass layer.

    Returns
    -------
    tuple[str, Any, Any]
        Annotation key, annotation site, and activation payload.
    """

    layer_label = str(getattr(layer, "layer_label"))
    if not bool(getattr(layer, "has_saved_activation", False)):
        _raise_unsaved_activation(layer_label)
    out = getattr(layer, "out", None)
    if out is None:
        _raise_unsaved_activation(layer_label)
    return f"layer:{layer_label}", layer.ops[0], out


def _op_mds_site(op: Any) -> tuple[str, Any, Any]:
    """Return the MDS payload tuple for a pass-qualified op.

    Parameters
    ----------
    op:
        Pass-qualified op selected by the caller.

    Returns
    -------
    tuple[str, Any, Any]
        Annotation key, annotation site, and activation payload.
    """

    op_label = str(getattr(op, "label"))
    if not bool(getattr(op, "has_saved_activation", False)):
        _raise_unsaved_activation(op_label)
    out = getattr(op, "out", None)
    if out is None:
        _raise_unsaved_activation(op_label)
    return f"op:{op_label}", op, out


def _site_label_is_pass_qualified(site: Any) -> bool:
    """Return whether a resolved site carries a pass-qualified op label.

    Parameters
    ----------
    site:
        Resolved layer-pass record.

    Returns
    -------
    bool
        Whether the final op label differs from the aggregate layer label.
    """

    return str(getattr(site, "label", "")) != str(getattr(site, "layer_label", ""))


def _annotate_mds_coords(trace: Any, site: Any, coords: np.ndarray) -> None:
    """Persist MDS coordinates as a Torch tensor annotation blob.

    Parameters
    ----------
    trace:
        Trace to annotate.
    site:
        Resolved layer-pass site.
    coords:
        ``N x 2`` coordinate array.

    Returns
    -------
    None
        The trace is mutated in place through ``Trace.annotate``.
    """

    from ..intervention.selectors import label

    trace.annotate(
        label(str(getattr(site, "label"))),
        data=torch.from_numpy(coords),
        max_fanout=1_000_000,
    )


def _raise_recurrent_layer_requires_pass(layer: Any) -> None:
    """Raise the public recurrent-layer selector error for MDS evolution.

    Parameters
    ----------
    layer:
        Aggregate recurrent layer.

    Raises
    ------
    ValueError
        Always raised with a pass-selection diagnostic.
    """

    layer_label = str(getattr(layer, "layer_label"))
    num_passes = int(getattr(layer, "num_passes", 0))
    raise ValueError(
        f"mds_evolution cannot compute aggregate MDS for recurrent layer "
        f"{layer_label!r} with {num_passes} passes; select a pass "
        f"(layer is recurrent), for example tl.label('{layer_label}:1')."
    )


def _raise_unsaved_activation(label: str) -> None:
    """Raise the public unsaved-activation error for MDS evolution.

    Parameters
    ----------
    label:
        Layer or op label selected for MDS.

    Raises
    ------
    ValueError
        Always raised with capture guidance.
    """

    raise ValueError(
        f"mds_evolution requires saved activations for {label!r}; capture with "
        "save= covering the MDS layers before calling mds_evolution."
    )


def _validate_procrustes_points(points: np.ndarray, name: str) -> None:
    """Validate a 2D point cloud for Procrustes alignment.

    Parameters
    ----------
    points:
        Candidate point matrix.
    name:
        Human-readable value name for errors.

    Raises
    ------
    ValueError
        If the point cloud is not finite ``[N, 2]`` data with at least 3 rows.
    """

    _validate_finite(points, name)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2].")
    if points.shape[0] < 3:
        raise ValueError(f"{name} must contain at least 3 points.")


__all__ = [
    "activation_distance_matrix",
    "classical_mds",
    "mds_evolution",
    "mds_scatter_node_spec",
    "procrustes_align",
]
