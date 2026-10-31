"""torchlens.mechinterp -- the mech-interp analysis kit (mikit memo).

TransformerLens's daily-driver analysis tools rebuilt on torchlens: residual
decomposition, direct logit attribution, per-head results, patching grids,
head-taxonomy scores, per-neuron MLP rows, the component edge graph (heads
and MLPs as nodes, q/k/v split receivers) with edge attribution patching
scores (EAP, EAP-IG) and circuit faithfulness areas (CPR, CMD), QK/OV circuits (``FactoredMatrix``,
vendored from TransformerLens with credit), prompt utilities, an alias layer, and
retention plans
-- derived from the captured graph on ANY architecture with a certifiable
additive residual topology, with verification built in (identities run IN
the API, not only in CI).

Axis disambiguation (mikit D1): ``torchlens.attribution`` attributes model
OUTPUTS to INPUTS; this kit attributes logits to internal WRITERS. No kit
name contains the word "attribution".

The facade name (and every spelling below) is a PLACEHOLDER, DOCUMENTED-
UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from ._aliases import AliasResolution, alias_report, resolve_alias, translation_table
from ._compose import apply_norm_scale, full_decomposition, stack_facet
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
]
