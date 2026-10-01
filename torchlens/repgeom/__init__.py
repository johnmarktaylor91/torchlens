"""Import-clean representation geometry helpers (real subpackage; C01 item 18).

This provisional surface intentionally depends only on NumPy, Torch, and PIL.
The math kernel lives in ``_geometry`` and the Trace-facing evolution/node
visuals in ``_trace_views``; this facade re-exports every historical spelling.
"""

from __future__ import annotations

from ._geometry import (  # noqa: F401 -- public type aliases re-exported
    DistanceMetric,
    EffectiveDimensionalityInfo,
    MDSEvolution,
    MDSInfo,
    MDSInputKind,
    RDMEvolution,
    ScreeEvolution,
    activation_distance_matrix,
    classical_mds,
    effective_dimensionality,
    procrustes_align,
    rdm,
    scree,
)
from ._node_visuals import (
    mds_scatter_node_spec,
    rdm_node_spec,
    scree_node_spec,
)
from ._trace_views import (
    mds_evolution,
    rdm_evolution,
    scree_evolution,
)

__tl_layer__ = "FACADE"

__all__ = [
    "activation_distance_matrix",
    "classical_mds",
    "effective_dimensionality",
    "mds_evolution",
    "mds_scatter_node_spec",
    "procrustes_align",
    "rdm",
    "rdm_evolution",
    "rdm_node_spec",
    "scree",
    "scree_evolution",
    "scree_node_spec",
]

# Private-but-consumed helpers: viz.feature_maps, receptive_field._viz, and
# the validation-edge suites import these through the package facade.
from ._annotation_gate import _commit_annotation_tensors  # noqa: E402,F401
from ._geometry import _check_square_distances  # noqa: E402,F401
from ._node_visuals import (  # noqa: E402,F401
    _matching_pil_image_batch,
    _mds_scatter_coords_for_node,
    _rdm_matrix_for_node,
    _scree_eigenvalues_for_node,
)
from ._trace_views import _selected_mds_sites, _store_annotation_tensor  # noqa: E402,F401
