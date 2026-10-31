"""Streaming statistics for out aggregation (real subpackage; C01 item 18).

Implementation lives in ``_streaming`` (stat kernels), ``_per_unit``
(``Quantile`` and ``TopK``, pooled or per unit), ``_aggregate`` (the
dataloader aggregation face, with ``_aggregate_cache`` behind ``cache=``)
and ``_fitted`` / ``_state_io`` (``save_fitted`` / ``load_fitted`` for fitted
PCA and accumulator state); this facade re-exports every historical
spelling unchanged.
"""

from __future__ import annotations

from ._aggregate import Aggregator, aggregate
from ._streaming import (
    CKA,
    PCA,
    Covariance,
    CrossCovariance,
    Mean,
    Norm,
    SimilarityScore,
    StreamingStat,
    cka,
)

__tl_layer__ = "FACADE"

# C02 numbers substrate (lovely items 2-4): the frozen TensorStats record,
# the sound kernel, and the pure stats-line renderers. Spellings
# DOCUMENTED-UNSTABLE pending naming-session ratification; F10 builds the
# public lovely surfaces on these.
# C06 observability substrate (explorer D18): the always-on scalar Spine and
# the fixed signed-log2 Histogram implement StreamingStat, closing the
# tl.stats Histogram gap -- tl.aggregate gets dataset-mode histograms free.
# One implementation home (torchlens/_observability/_kernels.py); spellings
# DOCUMENTED-UNSTABLE pending naming-session ratification.
from .._observability._kernels import (  # noqa: E402
    Histogram,
    HistogramDescriptor,
    HistogramResult,
    Spine,
    SpineResult,
)
from ..intervention._metrics import cosine_distance, relative_l1, relative_l2  # noqa: E402

# Reduce-and-discard capture (tl.trace(..., save="stats")) reads back through
# site_spines; measure_trajectory folds logit-lens trajectory statistics over
# batches. Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
from ._capture_stats import site_spines  # noqa: E402
from ._fitted import (  # noqa: E402
    FittedArtifactError,
    FittedPCA,
    load_fitted,
    save_fitted,
)
from ._glance import glance  # noqa: E402

# Per-unit statistics (neurons, channels, SAE latents): ``unit_axis=`` on
# Quantile and TopK, with UnitTopK carrying example ids and positions.
# Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
from ._per_unit import Quantile, TopK, UnitTopK  # noqa: E402
from ._seams import EchoObserver, echo, summary_cell, treescope_card_fields  # noqa: E402
from ._stats_render import (  # noqa: E402
    GLYPH_RAMP_ASCII,
    GLYPH_RAMP_UNICODE,
    degrade,
    format_sig,
    render_core_line,
)
from ._tensor_stats import (  # noqa: E402
    TENSOR_STATS_SCHEMA_VERSION,
    FamilyEvidence,
    TensorStats,
    tensor_stats,
)
from ._trajectory import TrajectoryStats, measure_trajectory  # noqa: E402

# F10 lovely surfaces (items 10-11): glance (FORK-C function form; submodule
# spelling, root budget frozen), the summary-cell/treescope seam adapters,
# and the echo observer for the shipped hooks=/tap surface. Spellings
# DOCUMENTED-UNSTABLE pending naming-session ratification.

__all__ = [
    "Aggregator",
    "CKA",
    "Covariance",
    "CrossCovariance",
    "EchoObserver",
    "FamilyEvidence",
    "FittedArtifactError",
    "FittedPCA",
    "GLYPH_RAMP_ASCII",
    "GLYPH_RAMP_UNICODE",
    "Histogram",
    "HistogramDescriptor",
    "HistogramResult",
    "Mean",
    "Norm",
    "PCA",
    "Quantile",
    "SimilarityScore",
    "Spine",
    "SpineResult",
    "StreamingStat",
    "TENSOR_STATS_SCHEMA_VERSION",
    "TensorStats",
    "TopK",
    "TrajectoryStats",
    "UnitTopK",
    "aggregate",
    "cka",
    "cosine_distance",
    "degrade",
    "echo",
    "format_sig",
    "glance",
    "load_fitted",
    "measure_trajectory",
    "relative_l1",
    "relative_l2",
    "render_core_line",
    "save_fitted",
    "site_spines",
    "summary_cell",
    "tensor_stats",
    "treescope_card_fields",
]
