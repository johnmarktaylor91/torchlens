"""torchlens.tviz -- transformer pictures (tviz memo).

The CircuitsVis/BertViz/Ecco/inspectus picture families rebuilt on torchlens:
token-labeled attention heatmaps and grids, the attention atlas, colored-token
strips, logit-lens prediction pictures, per-token loss/entropy strips, the
term-complete score decomposition, and the one picture none of them shipped --
attention annotated with MEASURED ablation effects (causal receipts).

Static is the product (memo D1): every picture saves genuine paper-ready
PNG/SVG/PDF through matplotlib (resolved at call time; absent, the typed
refusal prints the install command, and the zero-dependency SVG/HTML token
emitters still work). One rendering stack, no new frameworks (D2/D3): typed
display records underneath, rect meshes never ``imshow`` (D4), honesty
wording centralized in ``_wording.py``.

Import as ``import torchlens.tviz`` -- the root-name budget is frozen; the
root spelling awaits the F35 registration sweep. Records are session-only
display values, never persisted. Every spelling is DOCUMENTED-UNSTABLE
pending the naming session.
"""

from __future__ import annotations

from ._bridge import bertviz_tuple, circuitsvis_attention, circuitsvis_payload
from ._decomp import render_neuron_card, score_decomposition
from ._errors import TvizError
from ._extract import attention_view, attention_views
from ._metrics import metric_strip, render_metric_strip, token_metrics
from ._predictions import (
    prediction_table,
    prediction_table_html,
    prediction_trajectory,
    render_answer_trajectory,
    render_prediction_ribbon,
    render_prediction_table,
)
from ._receipts import receipt_from_patch_grid, render_receipt_grid
from ._records import (
    ANNOTATION_KINDS,
    Annotation,
    Artifact,
    AttentionView,
    CausalReceipt,
    CropInfo,
    EpisodeCoordinates,
    GqaInfo,
    MaskInfo,
    PredictionTable,
    PredictionTrajectory,
    ScoreDecomposition,
    TokenAxis,
    TokenMetrics,
    TokenScoreRow,
    TokenScores,
)
from ._render_attention import render_attention, render_attention_atlas
from ._strip import render_token_strip, token_strip_html, token_strip_svg

__all__ = [
    "ANNOTATION_KINDS",
    "Annotation",
    "Artifact",
    "AttentionView",
    "CausalReceipt",
    "CropInfo",
    "EpisodeCoordinates",
    "GqaInfo",
    "MaskInfo",
    "PredictionTable",
    "PredictionTrajectory",
    "ScoreDecomposition",
    "TokenAxis",
    "TokenMetrics",
    "TokenScoreRow",
    "TokenScores",
    "TvizError",
    "attention_view",
    "attention_views",
    "bertviz_tuple",
    "circuitsvis_attention",
    "circuitsvis_payload",
    "metric_strip",
    "prediction_table",
    "prediction_table_html",
    "prediction_trajectory",
    "receipt_from_patch_grid",
    "render_answer_trajectory",
    "render_attention",
    "render_attention_atlas",
    "render_metric_strip",
    "render_neuron_card",
    "render_prediction_ribbon",
    "render_prediction_table",
    "render_receipt_grid",
    "render_token_strip",
    "score_decomposition",
    "token_metrics",
    "token_strip_html",
    "token_strip_svg",
]
