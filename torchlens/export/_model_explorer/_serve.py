"""Serve one-liner over the public pinned vendor API (memo D18).

``model_explorer.visualize(..., reuse_server=...)`` with supported watch
semantics does the whole job; this module deliberately depends on NO private
attribute of the pinned dependency (the in-memory ``graphs_list`` fast path
was checked and cut). Answers Model Explorer issue #365 from the TorchLens
side: the graph the code ACTUALLY RAN, served in one line.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ..._errors import MissingDependencyError
from ._files import model_explorer

__tl_layer__ = "L8"


def model_explorer_serve(
    source: Any,
    *,
    path: str | Path | None = None,
    **visualize_kwargs: Any,
) -> Path:
    """Export a capture and open it in a local Model Explorer server.

    Parameters
    ----------
    source:
        A TorchLens ``Trace`` (exported first), or a path to an
        already-exported Model Explorer collection JSON.
    path:
        Optional destination for the exported JSON when ``source`` is a
        trace; defaults to a temporary file that outlives the call.
    **visualize_kwargs:
        Forwarded verbatim to the public ``model_explorer.visualize`` API
        (``host=``, ``port=``, ``reuse_server=``, ...).

    Returns
    -------
    Path
        The served collection JSON path.
    """

    try:
        import model_explorer as vendor
    except ImportError as exc:
        raise MissingDependencyError(
            "Serving requires the Model Explorer app package, which is not installed",
            code="model_explorer_serve_unavailable",
            remedy="pip install ai-edge-model-explorer==0.1.32",
            dependency="ai-edge-model-explorer",
            install="pip install ai-edge-model-explorer==0.1.32",
        ) from exc
    if isinstance(source, (str, Path)):
        destination = Path(source)
    else:
        if path is None:
            # The exported JSON must OUTLIVE the call (the vendor server reads
            # it), so this is a deliberate persistent temp path, not a leak.
            descriptor, path = tempfile.mkstemp(prefix="torchlens-model-explorer-", suffix=".json")
            os.close(descriptor)
        destination = model_explorer(source, path)
    vendor.visualize(model_paths=[str(destination)], **visualize_kwargs)
    return destination
