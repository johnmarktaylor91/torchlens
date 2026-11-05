"""Resolve the source bundle a portable-loaded trace was rehydrated from."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .._utils._record_group_table import record_member
from . import ArtifactError
from .lazy import LazyActivationRef

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _source_bundle_path_for_trace(trace: Trace) -> Path:
    """Resolve the source bundle path recorded on a portable-loaded ``Trace``.

    Parameters
    ----------
    trace:
        Model log whose source bundle should be resolved.

    Returns
    -------
    Path
        Source bundle directory.

    Raises
    ------
    ArtifactError
        If the model log does not retain a source bundle reference.
    """

    bundle_path = getattr(trace, "_source_bundle_path", None)
    if isinstance(bundle_path, Path):
        return bundle_path

    for layer in trace._layer_list:
        out_ref = record_member(layer, "out_ref", None)
        if isinstance(out_ref, LazyActivationRef):
            return out_ref.source_bundle_path
        grad_ref = record_member(layer, "grad_ref", None)
        if isinstance(grad_ref, LazyActivationRef):
            return grad_ref.source_bundle_path

    raise ArtifactError("Trace does not retain a source bundle path for nested blob rehydration.")
