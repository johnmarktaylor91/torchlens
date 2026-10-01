"""torchlens.mechinterp -- the mech-interp analysis kit (mikit memo).

TransformerLens's daily-driver analysis tools rebuilt on torchlens: residual
decomposition, direct logit attribution, per-head results, patching grids,
head-taxonomy scores, prompt utilities, an alias layer, and retention plans
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
from ._errors import MechInterpError
from ._grids import PatchGrid, grid, patch_heads_grid, patch_residual_grid
from ._heads import attention_head_contributions
from ._lowered import LoweredResult, lowered_counterfactual
from ._params import HeadWeightViews, ProjectionSource, head_weight_views
from ._prompts import PromptInspection, inspect_prompt, test_prompt
from ._records import ComponentRow, ComponentStack, ContributionScores, Coordinate
from ._residual import residual_accumulation, residual_decomposition
from ._retention import RetentionPlan, retention_plan
from ._scores import HeadScoreResult, head_scores

__all__ = [
    "AliasResolution",
    "ComponentRow",
    "ComponentStack",
    "ContributionScores",
    "Coordinate",
    "HeadScoreResult",
    "HeadWeightViews",
    "LoweredResult",
    "MechInterpError",
    "PatchGrid",
    "ProjectionSource",
    "PromptInspection",
    "RetentionPlan",
    "alias_report",
    "apply_norm_scale",
    "attention_head_contributions",
    "direct_logit_contributions",
    "full_decomposition",
    "grid",
    "head_scores",
    "head_weight_views",
    "inspect_prompt",
    "lowered_counterfactual",
    "patch_heads_grid",
    "patch_residual_grid",
    "residual_accumulation",
    "residual_decomposition",
    "resolve_alias",
    "retention_plan",
    "stack_facet",
    "test_prompt",
    "token_directions",
    "translation_table",
]
