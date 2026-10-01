"""Google Model Explorer export family (F15; modelexplorer memo B0-B12).

The panel-converged schema v3 writer for Model Explorer's file-ingest
contract: module-hierarchy namespaces (D1), the universal site-key+ordinal
node-id rule (D2), short op-type labels (D3), occurrence-preserving ported
edges through an explicit two-spelling map (D4), shapes on edges (D5),
curated attrs (D6), group rows with the ``""`` provenance/disclosure block
(D7), node-data overlays over the existing overlay vocabulary (D8),
rolled/unrolled multi-graph collections with the recurrent-feedback overlay
(D10), EPISODE per-step collections (D11-D14), bundle value diff (D15), and
the serve one-liner over the public pinned vendor API (D18).

Every public spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification. The vendor schema of record is ``model_explorer.graph_builder``
/ ``node_data_builder`` from ``ai-edge-model-explorer`` 0.1.32; the executed
oracle is the pinned ``dist/worker.js`` from
``ai-edge-model-explorer-visualizer`` 0.1.2 (tests/test_modelexplorer_assets).
"""

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
