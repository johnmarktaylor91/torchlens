"""Static transformer pictures: token-labeled attention heatmaps, logit-lens and per-token loss strips, and attention annotated with measured ablation effects.

Pictures render through matplotlib resolved at call time; the SVG and HTML token emitters need no extra dependency."""

from __future__ import annotations

from ._bridge import bertviz_tuple, circuitsvis_attention, circuitsvis_payload
from ._decomp import render_neuron_card, score_decomposition
from ._extract import attention_view, attention_views
from ._metrics import metric_strip, token_metrics
from ._predictions import (
    prediction_table,
    prediction_table_html,
    prediction_trajectory,
    render_answer_trajectory,
    render_prediction_ribbon,
)
from ._receipts import receipt_from_patch_grid
from ._records import (
    ANNOTATION_KINDS,
    Annotation,
    AttentionView,
    CausalReceipt,
    CropInfo,
    EpisodeCoordinates,
    GqaInfo,
    MaskInfo,
    PredictionTable,
    PredictionTrajectory,
    SavedFigure,
    ScoreDecomposition,
    TokenAxis,
    TokenMetrics,
    TokenScoreRow,
    TokenScores,
)
from ._strip import token_strip_html

__all__ = [
    "ANNOTATION_KINDS",
    "Annotation",
    "SavedFigure",
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
    "render_neuron_card",
    "render_prediction_ribbon",
    "score_decomposition",
    "token_metrics",
    "token_strip_html",
]
