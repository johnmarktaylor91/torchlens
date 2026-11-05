"""The mechanistic-interpretability analysis kit: residual decomposition, direct logit attribution, patching grids, head and neuron scores, and circuit edge graphs, all derived from a captured trace.

It attributes logits to internal writers; ``torchlens.attribution`` attributes model outputs to inputs."""

from __future__ import annotations

from .._semantic import patching
from .._semantic.logit_lens import LogitLens, LogitLensPrediction, LogitLensPredictions
from ..stats._trajectory import TrajectoryStats, measure_trajectory
from ._aliases import AliasResolution, alias_report, resolve_alias, translation_table
from ._compose import apply_norm_scale, full_decomposition, stack_facet
from ._controls import AblationControlsReport, ablation_controls
from ._dla import direct_logit_contributions, token_directions
from ._eap import edge_scores
from ._edge_patch import patch_edges
from ._edges import (
    ComponentEdge,
    ComponentGraph,
    ComponentReceiver,
    ComponentSender,
    component_graph,
)
from ._factored import FactoredMatrix
from ._faithfulness import circuit_model_difference, circuit_performance_ratio
from ._feature import visualize_feature
from ._grids import PatchGrid, patch_grid, patch_heads_grid, patch_residual_grid
from ._heads import attention_head_contributions
from ._kv_heads import patch_query_heads_grid
from ._lowered import PatchedRun, patch_head
from ._neurons import mlp_neuron_contributions
from ._params import HeadWeightViews, ProjectionSource, head_weight_views, ov_circuit, qk_circuit
from ._prompts import PromptInspection, inspect_prompt, test_prompt
from ._records import ComponentRow, ComponentStack, ContributionScores, Coordinate
from ._residual import residual_accumulation, residual_decomposition
from ._retention import SavePlan, retention_plan
from ._scores import FixedPeriodInduction, HeadScores, head_scores

__all__ = [
    "patching",
    "TrajectoryStats",
    "measure_trajectory",
    "LogitLens",
    "LogitLensPrediction",
    "LogitLensPredictions",
    "AblationControlsReport",
    "AliasResolution",
    "ComponentEdge",
    "ComponentGraph",
    "ComponentSender",
    "ComponentReceiver",
    "ComponentRow",
    "ComponentStack",
    "ContributionScores",
    "Coordinate",
    "FactoredMatrix",
    "FixedPeriodInduction",
    "HeadScores",
    "HeadWeightViews",
    "PatchedRun",
    "PatchGrid",
    "ProjectionSource",
    "PromptInspection",
    "SavePlan",
    "ablation_controls",
    "alias_report",
    "apply_norm_scale",
    "attention_head_contributions",
    "circuit_model_difference",
    "circuit_performance_ratio",
    "component_graph",
    "direct_logit_contributions",
    "edge_scores",
    "full_decomposition",
    "patch_grid",
    "head_scores",
    "head_weight_views",
    "inspect_prompt",
    "patch_head",
    "mlp_neuron_contributions",
    "ov_circuit",
    "patch_edges",
    "patch_heads_grid",
    "patch_query_heads_grid",
    "patch_residual_grid",
    "qk_circuit",
    "residual_accumulation",
    "residual_decomposition",
    "resolve_alias",
    "retention_plan",
    "stack_facet",
    "test_prompt",
    "token_directions",
    "translation_table",
    "visualize_feature",
]
