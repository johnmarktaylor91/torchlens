"""Representation-geometry math kernel (repgeom/ promotion, C01 item 18).

NumPy/Torch-only geometry: distances, RDMs, classical MDS, scree,
effective dimensionality, Procrustes. No Trace access, no rendering.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import torch

__tl_layer__ = "L5"

DistanceMetric: TypeAlias = Literal["euclidean", "manhattan", "cosine", "correlation", "gaussian"]
MDSInputKind: TypeAlias = Literal["auto", "distances", "features"]
MDSInfo: TypeAlias = dict[str, int | float | bool | str]
MDSEvolution: TypeAlias = "OrderedDict[str, np.ndarray]"
RDMEvolution: TypeAlias = "OrderedDict[str, np.ndarray]"
ScreeEvolution: TypeAlias = "OrderedDict[str, np.ndarray]"


class EffectiveDimensionalityInfo(TypedDict):
    """Summary statistics for a representation scree spectrum."""

    eigenvalues: np.ndarray
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    participation_ratio: float
    n_components_for_threshold: int
    total_positive_variance: float
    effective_rank: int


_RANK_TOLERANCE = 1e-12
_SYMMETRY_TOLERANCE = 1e-10
_SCATTER_CANVAS_SIZE = 420


def _symmetry_tolerance(array: np.ndarray) -> float:
    """Return the scale-aware absolute tolerance for symmetry-family gates.

    ``_SYMMETRY_TOLERANCE`` is a RELATIVE budget measured against the
    largest magnitude in the matrix (the ``_positive_rank_tolerance``
    idiom). The former fixed absolute ``1e-10`` was broken in both
    directions: 66% relative asymmetry at scale ``1e-10`` read as symmetric,
    while ``1e-15``-relative float64 round-off at scale ``1e7`` was
    rejected. Scaling by ``max|x|`` keeps the gate at ~``1e-10`` relative at
    every scale: strictly tighter than the old absolute gate below O(1)
    scale (fail-toward-strict) and no longer false-failing float64 noise
    above it. Used by the symmetry, zero-diagonal, non-negativity, and
    ambiguity-disclosure comparisons.

    Parameters
    ----------
    array:
        Matrix whose magnitude sets the tolerance scale.

    Returns
    -------
    float
        Absolute tolerance proportional to the matrix's largest magnitude.
    """

    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    return _SYMMETRY_TOLERANCE * max_abs


def _near_zero_tolerance(array: np.ndarray) -> float:
    """Return the scale-aware tolerance for closeness-to-zero REFUSAL gates.

    Duplicate-distance and zero-norm detection REFUSE input when a value
    sits within tolerance of zero, so a LARGER tolerance is the strict
    direction (more refusals) and a smaller one widens acceptance. This
    tolerance therefore keeps the O(1) floor -- ``max(1.0, max_abs)`` --
    so behavior below O(1) scale is unchanged from the historical absolute
    ``1e-10`` (never widening acceptance there), while above O(1) scale a
    value that is ~``1e-10``-relative-to-max close to zero is now correctly
    refused instead of slipping past a vanishing absolute gate.

    Parameters
    ----------
    array:
        Values whose magnitude sets the tolerance scale.

    Returns
    -------
    float
        Absolute tolerance with an O(1) scale floor.
    """

    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    return _SYMMETRY_TOLERANCE * max(1.0, max_abs)


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
    tolerance = _symmetry_tolerance(array)
    return bool(
        np.allclose(array, array.T, atol=tolerance, rtol=0.0)
        and np.allclose(np.diag(array), 0.0, atol=tolerance, rtol=0.0)
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
    tolerance = _symmetry_tolerance(distances)
    if not np.allclose(distances, distances.T, atol=tolerance, rtol=0.0):
        raise ValueError("distances must be symmetric.")
    if not np.allclose(np.diag(distances), 0.0, atol=tolerance, rtol=0.0):
        raise ValueError("distances must have a zero diagonal.")
    if np.any(distances < -tolerance):
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

    off_diagonal_zero = np.isclose(distances, 0.0, atol=_near_zero_tolerance(distances), rtol=0.0)
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


def _centered_gram_eigh(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return descending eigensystem of the double-centered distance Gram.

    Parameters
    ----------
    distances:
        Square symmetric pairwise distance matrix with zero diagonal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Descending eigenvalues and column-aligned eigenvectors.
    """

    _check_square_distances(distances)
    n_stimuli = distances.shape[0]
    squared = distances * distances
    centering = np.eye(n_stimuli) - np.full((n_stimuli, n_stimuli), 1.0 / n_stimuli)
    gram = -0.5 * centering @ squared @ centering
    gram = (gram + gram.T) * 0.5

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _as_distance_matrix_or_activations(data: Any, metric: DistanceMetric) -> np.ndarray:
    """Return a distance matrix from precomputed distances or activations.

    Parameters
    ----------
    data:
        Square distance matrix or activation representation with leading
        stimulus dimension.
    metric:
        Metric used when ``data`` is not already a distance matrix.

    Returns
    -------
    np.ndarray
        Square pairwise distance matrix.
    """

    array = _as_numpy_array(data)
    _validate_finite(array, "data")
    distances = (
        array.copy()
        if _looks_like_distance_matrix(array)
        else activation_distance_matrix(
            array,
            metric=metric,
        )
    )
    _check_square_distances(distances)
    return distances


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
        ``"cosine"``, ``"correlation"``, and ``"gaussian"``.

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
        feature_tensor = torch.as_tensor(features, dtype=torch.float64)
        distances = torch.cdist(feature_tensor, feature_tensor, p=2).cpu().numpy()
    elif metric == "cosine":
        distances = _angular_dissimilarity(features, center_rows=False)
    elif metric == "correlation":
        distances = _angular_dissimilarity(features, center_rows=True)
    elif metric == "gaussian":
        distances = _gaussian_dissimilarity(features)
    elif metric in _TORCH_METRIC_KERNELS:
        feature_tensor = torch.as_tensor(features, dtype=torch.float64)
        distances = (
            _TORCH_METRIC_KERNELS[metric](feature_tensor, feature_tensor, None).cpu().numpy()
        )
    else:
        raise ValueError(f"Unsupported activation distance metric: {metric!r}.")

    distances = (distances + distances.T) * 0.5
    np.fill_diagonal(distances, 0.0)
    return distances


def _kernel_minkowski(p: float) -> Any:
    """Return a row-block cdist kernel for a Minkowski order ``p``."""

    def kernel(rows: torch.Tensor, features: torch.Tensor, _chunk: int | None) -> torch.Tensor:
        """Compute pairwise distances between ``rows`` and all ``features``."""

        return torch.cdist(rows, features, p=p)

    return kernel


def _kernel_angular(*, center_rows: bool) -> Any:
    """Return a row-block angular (cosine/correlation) dissimilarity kernel."""

    def kernel(rows: torch.Tensor, features: torch.Tensor, _chunk: int | None) -> torch.Tensor:
        """Compute ``1 - similarity`` between ``rows`` and all ``features``."""

        def normalize(block: torch.Tensor) -> torch.Tensor:
            """Row-normalize (optionally row-centered), refusing zero norms."""

            working = block - block.mean(dim=1, keepdim=True) if center_rows else block
            norms = torch.linalg.vector_norm(working, dim=1, keepdim=True)
            tolerance = torch.finfo(working.dtype).eps * max(working.shape[1], 1) * 100
            if bool((norms <= tolerance).any()):
                metric_name = "correlation" if center_rows else "cosine"
                raise ValueError(f"{metric_name} distance is undefined for zero-norm stimuli.")
            return working / norms

        similarities = torch.clamp(normalize(rows) @ normalize(features).T, -1.0, 1.0)
        return 1.0 - similarities

    return kernel


# D-5 metric dispatch table: callable metrics and a streaming-Gram backend
# land later as new rows here, without an API change.
_TORCH_METRIC_KERNELS: dict[str, Any] = {
    "euclidean": _kernel_minkowski(2.0),
    "manhattan": _kernel_minkowski(1.0),
    "cosine": _kernel_angular(center_rows=False),
    "correlation": _kernel_angular(center_rows=True),
}


def _condensed_upper(distances: torch.Tensor) -> torch.Tensor:
    """Return the strict upper triangle of a square matrix, row-major."""

    n = distances.shape[0]
    index = torch.triu_indices(n, n, offset=1, device=distances.device)
    return distances[index[0], index[1]]


def _rdm_torch_path(  # noqa: PLR0913 -- mirrors the D-5 public rdm() keyword set one-to-one
    features: torch.Tensor,
    metric: str,
    *,
    compute_device: Any,
    row_chunk_size: int | None,
    output_device: Any,
    output: str,
) -> Any:
    """Blocked torch RDM path behind the D-5 keyword-only options.

    Parameters
    ----------
    features:
        ``[N, D]`` feature matrix (already dtype-resolved).
    metric:
        Key into the metric dispatch table.
    compute_device:
        Device the pairwise kernel runs on.
    row_chunk_size:
        Rows per block; ``None`` computes in one block.
    output_device:
        Device rows land on as they finish (``"cpu"`` streams GPU blocks
        straight to host memory).
    output:
        ``"square"`` or ``"condensed"``.

    Returns
    -------
    Any
        ``np.ndarray`` for CPU outputs, ``torch.Tensor`` otherwise.
    """

    kernel = _TORCH_METRIC_KERNELS[metric]
    n = features.shape[0]
    work = features.to(compute_device) if compute_device is not None else features
    out_device = torch.device(output_device) if output_device is not None else torch.device("cpu")
    chunk = int(row_chunk_size) if row_chunk_size else n
    if chunk <= 0:
        raise ValueError("row_chunk_size must be a positive integer or None.")
    result = torch.empty((n, n), dtype=work.dtype, device=out_device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = kernel(work[start:stop], work, row_chunk_size)
        result[start:stop] = block.to(out_device)
    result = (result + result.T) * 0.5
    result.fill_diagonal_(0.0)
    if output == "condensed":
        result = _condensed_upper(result)
    elif output != "square":
        raise ValueError(f"Unsupported RDM output form: {output!r} (use 'square'/'condensed').")
    if out_device.type == "cpu":
        return result.numpy()
    return result


def rdm(  # noqa: PLR0913 -- the D-5-widened public signature (brainpipe memo): each keyword is a spec'd GPU-scale knob, documented verbatim
    activations: Any,
    metric: DistanceMetric = "euclidean",
    *,
    compute_device: Any = None,
    row_chunk_size: int | None = None,
    output_device: Any = None,
    dtype: Any = None,
    output: str = "square",
    input_kind: str = "activations",
) -> Any:
    """Return a representational dissimilarity matrix for activations.

    The first activation dimension is treated as the stimulus dimension and
    all remaining dimensions are flattened per stimulus. The keyword-only
    options (F20, brainpipe memo D-5; spellings DOCUMENTED-UNSTABLE) widen
    the existing name for GPU-scale work: the no-new-keyword call is
    bit-identical to the historical behavior.

    Parameters
    ----------
    activations:
        Activation array or tensor with shape ``[N, ...]`` -- or, with
        ``input_kind="batched"``, ``[B, N, ...]`` for one RDM per batch
        element (3-D input already MEANS one flattened stimulus set today,
        so the batched reading is explicit, never guessed).
    metric:
        ``"euclidean"``, ``"manhattan"``, ``"cosine"``, ``"correlation"``,
        or ``"gaussian"`` (dispatch is table-driven; callable metrics are a
        later table row). ``"gaussian"`` computes a data-derived bandwidth
        over the full matrix and runs on the historical numpy path only.
        The default deliberately differs from
        :func:`torchlens.neuro.rdms`, whose source mode defaults to
        ``"correlation"`` (field canon for the RSA audience); both
        docstrings cross-reference the divergence.
    compute_device:
        Device for the pairwise kernel (e.g. ``"cuda"``). ``None`` keeps the
        historical CPU compute.
    row_chunk_size:
        Row-block size for memory-bounded computation; each finished block
        lands on ``output_device`` before the next is computed.
    output_device:
        Where the result lives. ``None``/CPU returns ``np.ndarray``
        (historical); a non-CPU device returns a ``torch.Tensor`` there.
    dtype:
        Computation dtype (default ``float64``, the historical behavior).
    output:
        ``"square"`` (default) or ``"condensed"`` (strict upper triangle,
        row-major -- the rsatoolbox/scipy vector form).
    input_kind:
        ``"activations"`` (default) or ``"batched"`` for ``[B, N, ...]``.

    Returns
    -------
    Any
        Square ``N x N`` matrix (or condensed vector / batched stack).

    Raises
    ------
    ValueError
        If activations are non-finite, empty, zero-norm for angular metrics,
        or an option value is unsupported.
    """

    modern = (
        compute_device is not None
        or row_chunk_size is not None
        or output_device is not None
        or dtype is not None
        or output != "square"
        or input_kind != "activations"
        or metric == "manhattan"
    )
    if not modern:
        # Bit-identical legacy path (D-5: the no-new-keyword call must not
        # move a single existing caller's numbers).
        return activation_distance_matrix(activations, metric=metric)

    if metric not in _TORCH_METRIC_KERNELS:
        if metric == "gaussian":
            raise ValueError(
                "metric='gaussian' derives its bandwidth from the full pairwise "
                "matrix and runs on the historical numpy path only; it does not "
                "support the GPU-path keywords (compute_device / row_chunk_size / "
                "output_device / dtype / output / input_kind). Call "
                "rdm(activations, metric='gaussian') with no other options."
            )
        raise ValueError(f"Unsupported activation distance metric: {metric!r}.")
    if input_kind not in {"activations", "batched"}:
        raise ValueError(
            f"Unsupported input_kind: {input_kind!r} (use 'activations' or 'batched')."
        )
    array = torch.as_tensor(_as_numpy_array(activations))
    _validate_finite(
        array.numpy() if array.device.type == "cpu" else array.cpu().numpy(), "activations"
    )
    resolved_dtype = dtype if dtype is not None else torch.float64
    if input_kind == "batched":
        if array.ndim < 3:
            raise ValueError(
                "input_kind='batched' requires [B, N, ...] input with at least 3 dimensions."
            )
        stacked = [
            _rdm_torch_path(
                array[b].reshape(array.shape[1], -1).to(resolved_dtype),
                metric,
                compute_device=compute_device,
                row_chunk_size=row_chunk_size,
                output_device=output_device,
                output=output,
            )
            for b in range(array.shape[0])
        ]
        if isinstance(stacked[0], np.ndarray):
            return np.stack(stacked)
        return torch.stack(stacked)
    if array.ndim < 1 or array.shape[0] == 0:
        raise ValueError("activations must have a non-empty leading stimulus dimension.")
    features = array.reshape(array.shape[0], -1).to(resolved_dtype)
    return _rdm_torch_path(
        features,
        metric,
        compute_device=compute_device,
        row_chunk_size=row_chunk_size,
        output_device=output_device,
        output=output,
    )


def _kendall_tau_a(a: np.ndarray, b: np.ndarray) -> float:
    """Return Kendall tau-a over paired vectors (O(n^2), ties count zero)."""

    n = a.shape[0]
    sign_a = np.sign(a[:, None] - a[None, :])
    sign_b = np.sign(b[:, None] - b[None, :])
    index = np.triu_indices(n, k=1)
    concordance = float(np.sum(sign_a[index] * sign_b[index]))
    n_pairs = n * (n - 1) / 2.0
    return concordance / n_pairs if n_pairs else math.nan


def rdm_compare(rdm_a: Any, rdm_b: Any, method: str = "spearman") -> float:
    """Return a descriptive rank correlation between two model RDMs.

    F20, brainpipe memo D-6 (spelling DOCUMENTED-UNSTABLE): the same class
    of descriptive object as CKA. Both RDMs are aligned on their strict
    upper triangles with the diagonal EXCLUDED -- the two silent hand-rolled
    errors this function exists to prevent are including the diagonal and
    correlating full symmetric matrices (which double-counts every pair).
    Deliberately descriptive-only: no p-values, no noise ceilings, no
    subject aggregation -- inferential RSA belongs to rsatoolbox or
    Net2Brain's ``RSA.evaluate``.

    Parameters
    ----------
    rdm_a:
        Square RDM (``N x N``) or condensed upper-triangle vector.
    rdm_b:
        Square RDM or condensed vector over the SAME stimuli in the same
        order.
    method:
        ``"spearman"`` (default), ``"pearson"``, or ``"kendall"`` (tau-a).

    Returns
    -------
    float
        The requested correlation over aligned upper triangles.

    Raises
    ------
    ValueError
        On shape mismatch, a non-square non-condensed input, or an unknown
        method.
    """

    def triangle(value: Any, name: str) -> np.ndarray:
        """Return the strict upper triangle of a square RDM or a condensed vector."""

        array = _as_numpy_array(value).astype(np.float64)
        _validate_finite(array, name)
        if array.ndim == 1:
            return array.copy()
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(
                f"{name} must be a square RDM or a condensed upper-triangle "
                f"vector; got shape {array.shape}."
            )
        index = np.triu_indices(array.shape[0], k=1)
        return array[index]

    vector_a = triangle(rdm_a, "rdm_a")
    vector_b = triangle(rdm_b, "rdm_b")
    if vector_a.shape[0] != vector_b.shape[0]:
        raise ValueError(
            f"rdm_compare requires RDMs over the same stimuli: upper triangles "
            f"have {vector_a.shape[0]} and {vector_b.shape[0]} entries."
        )
    if vector_a.shape[0] < 2:
        raise ValueError("rdm_compare needs at least 2 upper-triangle entries (3+ stimuli).")

    if method == "pearson":
        pass
    elif method == "spearman":
        vector_a = _average_ranks(vector_a)
        vector_b = _average_ranks(vector_b)
    elif method == "kendall":
        return _kendall_tau_a(vector_a, vector_b)
    else:
        raise ValueError(
            f"Unknown rdm_compare method: {method!r} (use 'pearson', 'spearman', "
            "or 'kendall'). For inferential RSA -- p-values, noise ceilings, "
            "subject aggregation -- use rsatoolbox or Net2Brain's RSA.evaluate."
        )
    centered_a = vector_a - vector_a.mean()
    centered_b = vector_b - vector_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(centered_a, centered_b) / denominator)


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
    zero_rows = np.flatnonzero(norms.ravel() <= _near_zero_tolerance(norms))
    if zero_rows.size:
        metric_name = "correlation" if center_rows else "cosine"
        raise ValueError(
            f"{metric_name} distance is undefined for zero-norm stimuli "
            f"(stimulus rows {zero_rows.tolist()})."
        )
    normalized = working / norms
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return 1.0 - similarities


def _gaussian_dissimilarity(features: np.ndarray) -> np.ndarray:
    """Return Gaussian/RBF-kernel dissimilarities for row-wise features.

    Convention (thingsvision's, credited -- ``thingsvision.core.rsa``):
    with squared euclidean pairwise distances ``D``, the bandwidth is
    ``mean(D)`` over the FULL ``N x N`` matrix (diagonal zeros included),
    the similarity kernel is ``exp(-D / (2 * mean(D)))``, and the
    dissimilarity is ``1 - similarity``. The bandwidth choice is a
    convention, not a law of nature; it is stated here so the number is
    reproducible.

    Parameters
    ----------
    features:
        Two-dimensional row-wise feature matrix.

    Returns
    -------
    np.ndarray
        Square ``1 - exp(-D / (2 * mean(D)))`` dissimilarity matrix.

    Raises
    ------
    ValueError
        If all stimuli are identical: the data-derived bandwidth is zero
        and the kernel is undefined (a typed refusal where the reference
        convention would silently return NaN).
    """

    feature_tensor = torch.as_tensor(features, dtype=torch.float64)
    squared = torch.cdist(feature_tensor, feature_tensor, p=2).cpu().numpy() ** 2
    bandwidth = float(squared.mean())
    if bandwidth <= 0.0:
        raise ValueError(
            "gaussian distance is undefined when all stimuli are identical: "
            "the data-derived bandwidth (mean squared pairwise distance, "
            "thingsvision's convention) is zero."
        )
    return 1.0 - np.exp(-squared / (2.0 * bandwidth))


def rank_transform_rdm(
    distances: Any,
    *,
    output: Literal["rank", "percentile"] = "percentile",
) -> np.ndarray:
    """Return a rank- or percentile-scaled copy of an RDM for display.

    The standard RSA display convention (Nili et al. 2014's
    percentile-ranked RDMs; thingsvision ships the same rank-scaled
    display): each unordered off-diagonal stimulus pair is replaced by its
    rank among all pairs (ties get their average rank, 1-based), or by
    ``rank / n_pairs * 100`` for ``output="percentile"`` (the largest
    dissimilarity maps to exactly 100). The transform is display-only:
    it destroys metric information and its output must never re-enter
    distance arithmetic.

    Parameters
    ----------
    distances:
        Square symmetric zero-diagonal dissimilarity matrix.
    output:
        ``"percentile"`` (default) or ``"rank"``.

    Returns
    -------
    np.ndarray
        Symmetric zero-diagonal matrix of ranks or percentiles.

    Raises
    ------
    ValueError
        If the matrix is not a valid finite symmetric distance matrix, is
        smaller than 2 x 2, or ``output`` is not a supported token.
    """

    if output not in ("rank", "percentile"):
        raise ValueError(
            f"Unsupported rank_transform_rdm output: {output!r} (expected 'rank' or 'percentile')."
        )
    array = _as_numpy_array(distances)
    _validate_finite(array, "distances")
    _check_square_distances(array)
    n_stimuli = array.shape[0]
    if n_stimuli < 2:
        raise ValueError("rank_transform_rdm requires at least 2 stimuli.")
    rows, cols = np.triu_indices(n_stimuli, k=1)
    ranks = _average_ranks(array[rows, cols])
    values = ranks / ranks.size * 100.0 if output == "percentile" else ranks
    result = np.zeros_like(array)
    result[rows, cols] = values
    result[cols, rows] = values
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks (ties share their mean rank).

    Parameters
    ----------
    values:
        One-dimensional array to rank.

    Returns
    -------
    np.ndarray
        Float64 ranks with tied values averaged.
    """

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def classical_mds(
    data: Any,
    n_components: int = 2,
    *,
    min_n: int = 8,
    input_kind: MDSInputKind = "auto",
) -> tuple[np.ndarray, MDSInfo]:
    """Embed pairwise distances or row-wise features with classical MDS.

    Under the default ``input_kind="auto"``, square symmetric inputs with zero
    diagonal are interpreted as precomputed distances and other inputs are
    treated as ``[N, ...]`` features and converted to Euclidean distances first.
    Because a square symmetric zero-diagonal *feature* matrix is
    indistinguishable from a distance matrix by content alone, ``"auto"`` warns
    whenever it has to make that guess; declare ``input_kind="distances"`` or
    ``input_kind="features"`` to state the intent and silence the guess.
    Negative centered-Gram eigenvalues are clipped to zero and reported because
    non-PSD dissimilarities are expected for some visualization metrics.

    Parameters
    ----------
    data:
        Pairwise distance matrix or row-wise coordinate/feature data.
    n_components:
        Number of embedding axes to return.
    min_n:
        Minimum number of stimuli required for visualization-oriented MDS.
    input_kind:
        How to read ``data``: ``"auto"`` detects a distance matrix from
        structure (and warns when the structure is ambiguous), ``"distances"``
        declares a precomputed pairwise distance matrix, and ``"features"``
        declares row-wise feature data even when it is square and symmetric.

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
    if input_kind not in ("auto", "distances", "features"):
        raise ValueError(
            "input_kind must be one of 'auto', 'distances', or 'features'; "
            f"received {input_kind!r}."
        )

    array = _as_numpy_array(data)
    _validate_finite(array, "data")
    if input_kind == "auto":
        input_is_distances = _looks_like_distance_matrix(array)
        if input_is_distances and not np.allclose(
            activation_distance_matrix(array, metric="euclidean"),
            array,
            atol=_symmetry_tolerance(array),
            rtol=0.0,
        ):
            warnings.warn(
                (
                    "classical_mds received an ambiguous square input with symmetric zero "
                    "diagonal; treating it as a precomputed distance matrix. Pass "
                    "input_kind='distances' or input_kind='features' to declare the intent "
                    "and avoid ambiguous square input handling."
                ),
                UserWarning,
                stacklevel=2,
            )
    else:
        input_is_distances = input_kind == "distances"
    if input_is_distances:
        distances = array.copy()
        _check_square_distances(distances)
    else:
        distances = activation_distance_matrix(array, metric="euclidean")
        _check_square_distances(distances)
    n_stimuli = distances.shape[0]
    _check_stimulus_count(n_stimuli, min_n)
    if _has_duplicate_distances(distances):
        raise ValueError("classical_mds input is rank-deficient or contains duplicate stimuli.")

    eigenvalues, eigenvectors = _centered_gram_eigh(distances)

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
        "input_kind": "distances" if input_is_distances else "features",
    }
    return embedding, info


def scree(
    activations: Any,
    *,
    metric: DistanceMetric = "euclidean",
    min_n: int = 3,
) -> np.ndarray:
    """Return sorted non-negative centered-Gram eigenvalues for activations.

    Parameters
    ----------
    activations:
        Activation representation with leading stimulus dimension, or a square
        precomputed distance matrix.
    metric:
        Distance metric used when activations are not already distances.
    min_n:
        Minimum number of stimuli required to compute the scree spectrum.

    Returns
    -------
    np.ndarray
        Descending non-negative eigenvalues.

    Raises
    ------
    ValueError
        If inputs are malformed, non-finite, or have too few stimuli.
    """

    if min_n < 3:
        raise ValueError("min_n must be at least 3 for scree.")
    distances = _as_distance_matrix_or_activations(activations, metric=metric)
    _check_stimulus_count(distances.shape[0], min_n)
    eigenvalues, _eigenvectors = _centered_gram_eigh(distances)
    return np.clip(eigenvalues, 0.0, None)


def effective_dimensionality(
    activations: Any,
    *,
    metric: DistanceMetric = "euclidean",
    min_n: int = 3,
    variance_threshold: float = 0.90,
) -> EffectiveDimensionalityInfo:
    """Return scree-derived effective-dimensionality statistics.

    Parameters
    ----------
    activations:
        Activation representation with leading stimulus dimension, or a square
        precomputed distance matrix.
    metric:
        Distance metric used when activations are not already distances.
    min_n:
        Minimum number of stimuli required to compute the scree spectrum.
    variance_threshold:
        Cumulative variance fraction used for the component-count callout.

    Returns
    -------
    EffectiveDimensionalityInfo
        Eigenvalues, variance fractions, cumulative fractions, participation
        ratio, threshold component count, total positive variance, and rank.

    Raises
    ------
    ValueError
        If inputs are malformed or ``variance_threshold`` is outside ``[0, 1]``.
    """

    _validate_variance_threshold(variance_threshold)
    eigenvalues = scree(activations, metric=metric, min_n=min_n)
    return _effective_dimensionality_from_eigenvalues(
        eigenvalues,
        variance_threshold=variance_threshold,
    )


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


def _validate_variance_threshold(variance_threshold: float) -> None:
    """Validate a cumulative variance threshold.

    Parameters
    ----------
    variance_threshold:
        Candidate threshold value.

    Raises
    ------
    ValueError
        If the threshold is non-finite or outside ``[0, 1]``.
    """

    if not np.isfinite(variance_threshold) or not 0.0 <= variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be finite and between 0 and 1.")


def _effective_dimensionality_from_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    variance_threshold: float,
) -> EffectiveDimensionalityInfo:
    """Return effective-dimensionality statistics from scree eigenvalues.

    Parameters
    ----------
    eigenvalues:
        Descending centered-Gram eigenvalues.
    variance_threshold:
        Cumulative variance fraction used for the component-count callout.

    Returns
    -------
    EffectiveDimensionalityInfo
        Scree-derived statistics.
    """

    _validate_variance_threshold(variance_threshold)
    clipped = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    _validate_finite(clipped, "scree eigenvalues")
    total_positive = float(np.sum(clipped))
    if total_positive > 0.0:
        variance = clipped / total_positive
        cumulative = np.cumsum(variance)
        denominator = float(np.sum(clipped * clipped))
        participation_ratio = (
            (total_positive * total_positive) / denominator if denominator > 0.0 else 0.0
        )
        n_components = min(
            int(np.searchsorted(cumulative, variance_threshold, side="left") + 1),
            clipped.size,
        )
    else:
        variance = np.zeros_like(clipped)
        cumulative = np.zeros_like(clipped)
        participation_ratio = 0.0
        n_components = 0
    tolerance = _positive_rank_tolerance(clipped)
    return {
        "eigenvalues": clipped,
        "variance_explained": variance,
        "cumulative_variance": cumulative,
        "participation_ratio": float(participation_ratio),
        "n_components_for_threshold": n_components,
        "total_positive_variance": total_positive,
        "effective_rank": int(np.count_nonzero(clipped > tolerance)),
    }


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
