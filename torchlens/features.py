"""The ONE named, recorded feature-matrix shaping operation (tvscope B7/D15).

Every consumer that needs "stimuli x features" 2-D matrices -- the export
contract, the rsatoolbox/xarray adapters, bring-your-own-alignment file
round trips -- shares THIS operation, so no consumer ever writes its own
reshape ("write your own flatten" is how row-order bugs enter). The
operation is recorded: callers get a :class:`ShapingRecord` naming the op,
the shapes, and the batch axis, suitable for sidecar provenance.

Row order is inviolable: row ``i`` of the returned matrix is stimulus ``i``
of the input's batch axis, and a caller-supplied row-id list must agree on
cardinality or the shaping refuses typed (the manifest, not a text file, is
the sole row-order authority).

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint; import as
``import torchlens.features``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, cast

import torch

from torchlens._errors import _actionable_message, _ActionableErrorMixin
from torchlens.errors._base import ConfigurationError

__tl_layer__ = "L5"

#: The shaping-op identity token recorded on every record.
SHAPING_OP = "flatten_features_v1"


class FeatureShapingError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Typed refusal from the shaping operation (tvscope B7).

    Codes: ``feature_matrix_scalar_input`` (a 0-dim tensor has no stimulus
    axis) and ``feature_rows_ids_mismatch`` (row/id cardinality
    disagreement -- proceeding would mislabel every row after the shorter
    of the two).
    """

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize a typed shaping refusal.

        Parameters
        ----------
        problem:
            What made the shaping unsafe.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


@dataclass(frozen=True)
class ShapingRecord:
    """Provenance of one shaping application (rides sidecars and adapters).

    Attributes
    ----------
    op:
        The shaping-op identity token (:data:`SHAPING_OP`).
    input_shape:
        Shape of the tensor as supplied.
    output_shape:
        Shape of the returned 2-D matrix.
    batch_axis:
        Which input axis was the stimulus axis.
    """

    op: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    batch_axis: int

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-portable dict."""

        return {
            "op": self.op,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "batch_axis": self.batch_axis,
        }


def as_matrix(
    tensor: torch.Tensor,
    *,
    batch_axis: int = 0,
    row_ids: list[str] | None = None,
) -> tuple[torch.Tensor, ShapingRecord]:
    """Shape one tensor into the documented "stimuli x features" matrix (B7).

    The stimulus axis moves to axis 0 (when it is not already there) and
    every remaining axis flattens, C-contiguously, into the feature axis --
    an explicit, named, recorded operation, never an implicit ``.flatten``.

    Parameters
    ----------
    tensor:
        Activation tensor with a stimulus axis.
    batch_axis:
        Which axis indexes stimuli (default 0, the TorchLens convention).
    row_ids:
        Optional per-row identifiers; supplied, their cardinality must equal
        the stimulus-axis extent or the shaping refuses typed.

    Returns
    -------
    tuple[torch.Tensor, ShapingRecord]
        The 2-D matrix (rows = stimuli, in axis order) and the shaping
        provenance record.

    Raises
    ------
    FeatureShapingError
        ``feature_matrix_scalar_input`` on a 0-dim tensor;
        ``feature_rows_ids_mismatch`` on a row/id cardinality disagreement.
    """

    if tensor.ndim == 0:
        raise FeatureShapingError(
            "a 0-dim tensor has no stimulus axis to shape into matrix rows.",
            code="feature_matrix_scalar_input",
            remedy="pass an activation tensor with a leading stimulus axis",
            shape=[],
        )
    moved = tensor if batch_axis == 0 else torch.movedim(tensor, batch_axis, 0)
    matrix = moved.reshape(moved.shape[0], -1)
    if row_ids is not None and len(row_ids) != int(matrix.shape[0]):
        raise FeatureShapingError(
            f"{len(row_ids)} row ids were supplied for {int(matrix.shape[0])} "
            "stimulus rows; proceeding would mislabel every row after the "
            "shorter of the two.",
            code="feature_rows_ids_mismatch",
            remedy="pass exactly one identifier per stimulus row, in row order",
            n_ids=len(row_ids),
            n_rows=int(matrix.shape[0]),
        )
    record = ShapingRecord(
        op=SHAPING_OP,
        input_shape=tuple(int(d) for d in tensor.shape),
        output_shape=tuple(int(d) for d in matrix.shape),
        batch_axis=batch_axis,
    )
    return matrix, record


@dataclass(frozen=True)
class SiteMatrix:
    """One site's feature matrix with its row identity and provenance.

    Attributes
    ----------
    matrix:
        The 2-D "stimuli x features" tensor.
    record:
        The shaping provenance.
    site:
        The selector/output key the matrix came from.
    row_ids:
        Per-row stimulus identifiers when the source carries them
        (extraction artifacts with a recorded id sidecar), else ``None``.
    input_preprocessing:
        The source's input-preprocessing provenance block when it carries
        one (extraction manifests; legacy reads as unknown), else ``None``.
    """

    matrix: torch.Tensor
    record: ShapingRecord
    site: str
    row_ids: list[str] | None
    input_preprocessing: dict[str, Any] | None


def _matrix_from_trace(trace: Any, site: str) -> SiteMatrix:
    """Shape one saved Trace site (adapters' in-memory route)."""

    layer = trace[site]
    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor):
        raise FeatureShapingError(
            f"site {site!r} holds no saved tensor activation on this trace.",
            code="feature_site_payload_unavailable",
            remedy=(
                "capture with save= selecting this site (or layers_to_save), "
                "then shape it; metadata-only captures hold no payloads"
            ),
            site=site,
        )
    matrix, record = as_matrix(out.detach().cpu())
    return SiteMatrix(
        matrix=matrix,
        record=record,
        site=site,
        row_ids=None,
        input_preprocessing=None,
    )


def _extraction_row_ids(loaded: Any) -> list[str] | None:
    """Read the recorded stimulus ids of a loaded extraction, if any."""

    from torchlens._data_substrate import STIMULUS_IDS_FILENAME
    from torchlens._io import _json

    manifest = loaded.manifest
    if not (manifest.get("stimulus_provenance") or {}).get("ids_recorded"):
        return None
    if not loaded.batch_paths:
        return None
    sidecar = loaded.batch_paths[0].parent / STIMULUS_IDS_FILENAME
    payload: Any = None
    with contextlib.suppress(Exception):  # unreadable sidecar = ids unavailable
        payload = _json.read_bounded(sidecar)
    ids = payload.get("ids") if isinstance(payload, dict) else None
    return [str(item) for item in ids] if isinstance(ids, list) else None


def _matrix_from_extraction(loaded: Any, site: str) -> SiteMatrix:
    """Shape one extraction-artifact site (adapters' file route)."""

    if site not in loaded.activations:
        raise FeatureShapingError(
            f"output key {site!r} is not in this extraction artifact "
            f"(available: {sorted(loaded.activations)}).",
            code="feature_site_payload_unavailable",
            remedy="request an output key recorded in the artifact's manifest",
            site=site,
            available=sorted(loaded.activations),
        )
    ids = _extraction_row_ids(loaded)
    matrix, record = as_matrix(loaded.activations[site], row_ids=ids)
    from torchlens.dataset_extraction import input_preprocessing_of

    return SiteMatrix(
        matrix=matrix,
        record=record,
        site=site,
        row_ids=ids,
        input_preprocessing=input_preprocessing_of(loaded.manifest),
    )


def site_matrix(source: Any, site: str) -> SiteMatrix:
    """Shape one site from a Trace or an extraction artifact (B7/B8 door).

    The file route and the in-memory route run the SAME shaping op, so the
    two are numerically identical for the same capture.

    Parameters
    ----------
    source:
        A TorchLens ``Trace`` (in-memory route), a ``LoadedExtraction``, or
        an extraction-artifact directory path (file route).
    site:
        Site selector: a Trace lookup (module address, label) or an
        extraction output key.

    Returns
    -------
    SiteMatrix
        Matrix + shaping record + row identity + provenance block.

    Raises
    ------
    FeatureShapingError
        ``feature_site_payload_unavailable`` when the site holds no payload;
        ``feature_rows_ids_mismatch`` on row/id disagreement.
    """

    from pathlib import Path

    from torchlens.dataset_extraction import LoadedExtraction, load_extraction

    if isinstance(source, LoadedExtraction):
        return _matrix_from_extraction(source, site)
    if isinstance(source, (str, Path)):
        return _matrix_from_extraction(load_extraction(source, layers=[site]), site)
    return _matrix_from_trace(source, site)


__all__ = [
    "SHAPING_OP",
    "FeatureShapingError",
    "ShapingRecord",
    "SiteMatrix",
    "as_matrix",
    "site_matrix",
]
