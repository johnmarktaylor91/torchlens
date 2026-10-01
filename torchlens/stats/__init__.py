"""Streaming statistics for out aggregation (real subpackage; C01 item 18).

Implementation lives in ``_streaming`` (stat kernels) and ``_aggregate``
(the dataloader aggregation face); this facade re-exports every historical
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
    Quantile,
    StreamingStat,
    TopK,
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
# One implementation home (torchlens/observability/_kernels.py); spellings
# DOCUMENTED-UNSTABLE pending naming-session ratification.
from ..observability._kernels import (  # noqa: E402
    Histogram,
    HistogramDescriptor,
    HistogramResult,
    Spine,
    SpineResult,
)
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

__all__ = [
    "Aggregator",
    "CKA",
    "Covariance",
    "CrossCovariance",
    "FamilyEvidence",
    "GLYPH_RAMP_ASCII",
    "GLYPH_RAMP_UNICODE",
    "Histogram",
    "HistogramDescriptor",
    "HistogramResult",
    "Mean",
    "Norm",
    "PCA",
    "Quantile",
    "Spine",
    "SpineResult",
    "StreamingStat",
    "TENSOR_STATS_SCHEMA_VERSION",
    "TensorStats",
    "TopK",
    "aggregate",
    "cka",
    "degrade",
    "format_sig",
    "render_core_line",
    "tensor_stats",
]
