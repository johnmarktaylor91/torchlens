"""Google Model Explorer export: a writer for Model Explorer's file-ingest schema with module-hierarchy namespaces, op-type labels, ported edges, shapes, node-data overlays, rolled and unrolled graph collections, per-step episode collections and bundle diffs, plus a serve helper over the pinned vendor API.

The vendor schema of record is ``model_explorer.graph_builder`` / ``node_data_builder`` from ``ai-edge-model-explorer`` 0.1.33; the executed oracle is the pinned ``dist/worker.js`` from ``ai-edge-model-explorer-visualizer`` (version 0.1.2)."""

from __future__ import annotations

from ._collection import to_model_explorer_dict
from ._diff import model_explorer_diff
from ._files import model_explorer
from ._serve import model_explorer_serve
from ._validate import validate_model_explorer_payload

__tl_layer__ = "L8"

__all__ = [
    "model_explorer",
    "model_explorer_diff",
    "model_explorer_serve",
    "to_model_explorer_dict",
    "validate_model_explorer_payload",
]
