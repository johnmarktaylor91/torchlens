"""Trace-facing repgeom views: evolution tables + node visuals (C01 item 18).

Consumes the math kernel in ``_geometry`` and the PIL node-plot primitives;
holds every Trace-reading and image-writing helper.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from ..intervention.errors import MultiMatchWarning
from ._annotation_gate import (
    _VERB_VOCABULARY,
    _commit_annotation_tensors,
    _expected_stimulus_counts,
    _raise_ineligible_site,
    _raise_recurrent_layer_requires_pass,
    _raise_unsaved_activation,
    _site_ineligibility,
    _warn_skipped_sites,
)
from ._geometry import (
    DistanceMetric,
    MDSEvolution,
    RDMEvolution,
    ScreeEvolution,
    _validate_variance_threshold,
    activation_distance_matrix,
    classical_mds,
    procrustes_align,
    scree,
)

__tl_layer__ = "L6"


class _PayloadBasisDict(OrderedDict):
    """Ordered result mapping with a payload-provenance side table.

    F20 D-19 disclosure: geometry computed from a TRANSFORMED payload is a
    different scientific object than raw-payload geometry, so every
    evolution result records which payload fed each key on
    ``result.payload_basis`` (``key -> "raw" | "transformed"``). Plain
    ``OrderedDict`` semantics are unchanged; session-time only.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.payload_basis: dict[str, str] = {}


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

    selected = _selected_mds_sites(trace, save, verb="mds_evolution")
    coords_by_key: _PayloadBasisDict = _PayloadBasisDict()
    staged: OrderedDict[str, torch.Tensor] = OrderedDict()
    previous_coords: np.ndarray | None = None
    for key, _site, activations, payload_kind in selected:
        coords_by_key.payload_basis[key] = payload_kind
        try:
            distances = activation_distance_matrix(activations, metric=metric)
            coords, _info = classical_mds(
                distances, n_components=2, min_n=min_n, input_kind="distances"
            )
        except ValueError as exc:
            raise ValueError(f"mds_evolution failed for site {key!r}: {exc}") from exc
        if align and previous_coords is not None:
            coords = procrustes_align(coords, previous_coords)
        staged[f"mds:{key}"] = torch.from_numpy(coords.copy())
        coords_by_key[key] = coords.copy()
        previous_coords = coords
    # Annotations commit only after the WHOLE sweep succeeds; a failed site
    # leaves zero blobs behind (gated ATOMIC write, neuro MEMO D4).
    _commit_annotation_tensors(trace, staged)
    return coords_by_key


def rdm_evolution(
    trace: Any,
    save: Any | None = None,
    *,
    metric: DistanceMetric = "euclidean",
    min_n: int = 2,
) -> RDMEvolution:
    """Compute and annotate per-layer representational dissimilarity matrices.

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
        Minimum number of stimuli required for each RDM. Must be at least 2.

    Returns
    -------
    OrderedDict[str, np.ndarray]
        RDM arrays keyed by ``layer:<layer_label>`` for single-pass layers and
        ``op:<op.label>`` for pass-qualified recurrent selections.

    Raises
    ------
    ValueError
        If a selected site has no saved activation, has too few stimuli, is
        recurrent without a pass-qualified selection, or fails metric
        preconditions.
    """

    if min_n < 2:
        raise ValueError("min_n must be at least 2 for rdm_evolution.")

    selected = _selected_activation_sites(trace, save, verb="rdm_evolution")
    matrices_by_key: _PayloadBasisDict = _PayloadBasisDict()
    staged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, _site, activations, payload_kind in selected:
        matrices_by_key.payload_basis[key] = payload_kind
        try:
            matrix = activation_distance_matrix(activations, metric=metric)
        except ValueError as exc:
            raise ValueError(f"rdm_evolution failed for site {key!r}: {exc}") from exc
        if matrix.shape[0] < min_n:
            raise ValueError(
                f"rdm_evolution has too few stimuli for {key!r}: "
                f"got {matrix.shape[0]}, need at least {min_n}."
            )
        staged[f"rdm:{key}"] = torch.from_numpy(matrix.copy())
        matrices_by_key[key] = matrix.copy()
    # Annotations commit only after the WHOLE sweep succeeds; a failed site
    # leaves zero blobs behind (gated ATOMIC write, neuro MEMO D4).
    _commit_annotation_tensors(trace, staged)
    return matrices_by_key


def scree_evolution(
    trace: Any,
    save: Any | None = None,
    *,
    metric: DistanceMetric = "euclidean",
    min_n: int = 3,
    variance_threshold: float = 0.90,
) -> ScreeEvolution:
    """Compute and annotate per-layer scree eigenvalue spectra.

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
        Minimum number of stimuli required for each scree spectrum.
    variance_threshold:
        Validated threshold for consistency with render-time callouts. The
        returned and stored payloads remain eigenvalue vectors only.

    Returns
    -------
    OrderedDict[str, np.ndarray]
        Eigenvalue arrays keyed by ``layer:<layer_label>`` for single-pass
        layers and ``op:<op.label>`` for pass-qualified recurrent selections.

    Raises
    ------
    ValueError
        If a selected site has no saved activation, has too few stimuli, is
        recurrent without a pass-qualified selection, or fails metric
        preconditions.
    """

    _validate_variance_threshold(variance_threshold)
    selected = _selected_activation_sites(trace, save, verb="scree_evolution")
    eigenvalues_by_key: _PayloadBasisDict = _PayloadBasisDict()
    staged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, _site, activations, payload_kind in selected:
        eigenvalues_by_key.payload_basis[key] = payload_kind
        try:
            eigenvalues = scree(activations, metric=metric, min_n=min_n)
        except ValueError as exc:
            raise ValueError(f"scree_evolution failed for site {key!r}: {exc}") from exc
        staged[f"scree:{key}"] = torch.from_numpy(eigenvalues)
        eigenvalues_by_key[key] = eigenvalues
    # Annotations commit only after the WHOLE sweep succeeds; a failed site
    # leaves zero blobs behind (gated ATOMIC write, neuro MEMO D4).
    _commit_annotation_tensors(trace, staged)
    return eigenvalues_by_key


def _selected_mds_sites(
    trace: Any, save: Any | None, *, verb: str = "mds_evolution"
) -> list[tuple[str, Any, Any, str]]:
    """Resolve the layer or op payloads that should receive annotations.

    Only stimulus-indexed sites qualify: a default sweep SKIPS ineligible
    sites with one summarized disclosure, while an explicit ``save=``
    selection REFUSES them actionably (neuro MEMO D4/D5 -- previously a
    stock resnet18 sweep fabricated 78 buffer pseudo-RDMs). See
    :func:`_site_ineligibility` for the evidence order and the named
    coincidental-equality residual.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    save:
        Optional selector passed by the user.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    list[tuple[str, Any, Any]]
        Tuples of annotation key, resolved annotation site, and activation
        payload.
    """

    if save is None:
        return _default_saved_mds_sites(trace, verb=verb)

    expected_counts = _expected_stimulus_counts(trace)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MultiMatchWarning)
        sites = list(trace.resolve_sites(save, max_fanout=1_000_000))
    selected_by_layer: dict[str, list[Any]] = OrderedDict()
    for site in sites:
        selected_by_layer.setdefault(str(getattr(site, "layer_label")), []).append(site)

    selected: list[tuple[str, Any, Any, str]] = []
    for layer_label, layer_sites in selected_by_layer.items():
        layer = trace.layer_logs[layer_label]
        if int(getattr(layer, "num_passes", 1)) > 1:
            if len(layer_sites) != 1:
                _raise_recurrent_layer_requires_pass(layer, verb=verb)
            site = layer_sites[0]
            if not _site_label_is_pass_qualified(site):
                _raise_recurrent_layer_requires_pass(layer, verb=verb)
            reason = _site_ineligibility(site, expected_counts)
            if reason is not None:
                _raise_ineligible_site(verb, str(getattr(site, "label", layer_label)), reason)
            selected.append(_op_mds_site(site, verb=verb))
        else:
            reason = _site_ineligibility(layer, expected_counts)
            if reason is not None:
                _raise_ineligible_site(verb, layer_label, reason)
            selected.append(_single_pass_layer_mds_site(layer, verb=verb))
    return selected


_selected_activation_sites = _selected_mds_sites


def _default_saved_mds_sites(
    trace: Any, *, verb: str = "mds_evolution"
) -> list[tuple[str, Any, Any, str]]:
    """Return saved, stimulus-indexed single-pass layer payloads.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    list[tuple[str, Any, Any]]
        Tuples of annotation key, resolved site, and activation payload.
    """

    expected_counts = _expected_stimulus_counts(trace)
    selected: list[tuple[str, Any, Any, str]] = []
    skipped: OrderedDict[str, str] = OrderedDict()
    for layer in trace.layers:
        layer_label = str(getattr(layer, "layer_label"))
        saved_ops = [
            op for op in layer.ops.values() if bool(getattr(op, "has_saved_activation", False))
        ]
        reason = _site_ineligibility(layer, expected_counts)
        if reason is not None:
            if saved_ops:
                skipped[layer_label] = reason
            continue
        if int(getattr(layer, "num_passes", 1)) > 1:
            if saved_ops:
                _raise_recurrent_layer_requires_pass(layer, verb=verb)
            continue
        if bool(getattr(layer, "has_saved_activation", False)):
            selected.append(_single_pass_layer_mds_site(layer, verb=verb))
    _warn_skipped_sites(verb, skipped)
    if not selected:
        vocabulary = _VERB_VOCABULARY.get(verb, _VERB_VOCABULARY["mds_evolution"])
        skip_note = (
            f" ({len(skipped)} saved site(s) were excluded as not stimulus-indexed)"
            if skipped
            else ""
        )
        raise ValueError(
            f"{verb} requires saved activations{skip_note}; capture with save= "
            f"covering the {vocabulary['layers']} layers before calling {verb}."
        )
    return selected


def _saved_payload_with_kind(record: Any, label: str, *, verb: str) -> tuple[Any, str]:
    """Return a site's retained payload plus its provenance kind (F20 D-19).

    Geometry collectors historically read ``.out`` only, so the composition
    reduce-then-geometry was broken on exactly the traces the sweep
    produces. When the raw payload was dropped in favor of a transform, the
    TRANSFORMED payload feeds the geometry -- with the kind recorded, since
    a raw RDM and a post-reduction RDM are different scientific objects --
    guarded by a stimulus-axis check: a transform that consumed the
    stimulus axis (batch pooling) cannot feed stimulus-indexed geometry and
    refuses with the remedy named.

    Parameters
    ----------
    record:
        Layer or op record carrying payload fields.
    label:
        Display label for diagnostics.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    tuple[Any, str]
        ``(payload, kind)`` with kind ``"raw"`` or ``"transformed"``.
    """

    out = getattr(record, "out", None)
    if out is not None:
        return out, "raw"
    transformed = getattr(record, "transformed_out", None)
    if transformed is None:
        _raise_unsaved_activation(label, verb=verb)
    raw_shape = getattr(record, "shape", None)
    transformed_shape = tuple(getattr(transformed, "shape", ()) or ())
    if raw_shape and transformed_shape and transformed_shape[0] != raw_shape[0]:
        raise ValueError(
            f"{verb} cannot use the transformed payload for {label!r}: the "
            f"transform changed the stimulus axis (raw batch {raw_shape[0]}, "
            f"transformed leading dim {transformed_shape[0]}), so rows no "
            "longer index stimuli. Remedy: keep the raw payload for this "
            "site (save_raw_activations=True) or use a transform that "
            "preserves the batch axis."
        )
    return transformed, "transformed"


def _single_pass_layer_mds_site(
    layer: Any, *, verb: str = "mds_evolution"
) -> tuple[str, Any, Any, str]:
    """Return the annotation payload tuple for a single-pass layer.

    Parameters
    ----------
    layer:
        Aggregate single-pass layer.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    tuple[str, Any, Any, str]
        Annotation key, annotation site, activation payload, and payload
        provenance kind (``"raw"`` / ``"transformed"``).
    """

    layer_label = str(getattr(layer, "layer_label"))
    if not bool(getattr(layer, "has_saved_activation", False)):
        _raise_unsaved_activation(layer_label, verb=verb)
    payload, kind = _saved_payload_with_kind(layer, layer_label, verb=verb)
    return f"layer:{layer_label}", layer.ops[0], payload, kind


def _op_mds_site(op: Any, *, verb: str = "mds_evolution") -> tuple[str, Any, Any, str]:
    """Return the annotation payload tuple for a pass-qualified op.

    Parameters
    ----------
    op:
        Pass-qualified op selected by the caller.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    tuple[str, Any, Any, str]
        Annotation key, annotation site, activation payload, and payload
        provenance kind (``"raw"`` / ``"transformed"``).
    """

    op_label = str(getattr(op, "label"))
    if not bool(getattr(op, "has_saved_activation", False)):
        _raise_unsaved_activation(op_label, verb=verb)
    payload, kind = _saved_payload_with_kind(op, op_label, verb=verb)
    return f"op:{op_label}", op, payload, kind


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


def _annotate_mds_coords(trace: Any, key: str, coords: np.ndarray) -> None:
    """Persist MDS coordinates as a Torch tensor annotation blob.

    Parameters
    ----------
    trace:
        Trace to annotate.
    key:
        Annotation key for the selected layer or pass-qualified op.
    coords:
        ``N x 2`` coordinate array.

    Returns
    -------
    None
        The trace is mutated in place through ``_annotation_blobs``.

    Notes
    -----
    Stored under the ``mds:`` namespace: bare ``layer:``/``op:`` keys belong
    to USER ``annotate(data=...)`` blobs, and an unprefixed store both
    collided with them and let a user ``[N, 2]`` payload be misread as MDS
    coordinates by the scatter reader. Derived-prefix keys are also what the
    rerun refresh invalidates while preserving user blobs.
    """

    _store_annotation_tensor(trace, f"mds:{key}", torch.from_numpy(coords.copy()))


def _store_annotation_tensor(trace: Any, key: str, tensor: torch.Tensor) -> None:
    """Store a validated tensor annotation blob on a torch trace.

    Parameters
    ----------
    trace:
        Trace to annotate.
    key:
        Exact annotation blob key to write.
    tensor:
        Tensor payload to store.

    Returns
    -------
    None
        The trace is mutated in place through ``_annotation_blobs``.

    Raises
    ------
    ValueError
        If the trace is not a torch trace or the payload is not portable.
    TypeError
        If ``tensor`` is not a torch tensor.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("_store_annotation_tensor requires a torch.Tensor payload.")
    backend_name = str(getattr(trace, "backend", "torch"))
    if backend_name != "torch":
        raise ValueError(
            "Tensor annotation blobs are supported only for torch traces in this "
            f"release; this trace uses backend={backend_name!r}."
        )
    validate_tensor = getattr(trace, "_validate_annotation_tensor", None)
    if not callable(validate_tensor):
        raise ValueError("trace does not support validated tensor annotation blobs.")
    validate_tensor(tensor)
    if getattr(trace, "_annotation_blobs", None) is None:
        trace._annotation_blobs = {}
    trace._annotation_blobs[key] = tensor
    mark_mutated = getattr(trace, "_mark_annotations_mutated", None)
    if callable(mark_mutated):
        mark_mutated()
