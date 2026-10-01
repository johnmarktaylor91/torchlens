"""TorchLens semantic lenses: the themes roster and its honesty machinery.

The two-axis theme system (themes trilab memo): a semantic LENS answers a
question (what is slow, what is big, what is broken) and a cosmetic SKIN
styles the ink; they compose freely. This subpackage carries the lens
roster v1 (nine rows + three composition recipes), view-aware source
families (N16), the visible-detail budget resolver (N10), the six-state
nonfinite status channel (N6), the general display filter (N9), and the
resolution front door that enforces the refusal taxonomy and rendered
disclosure.

The public ``draw(theme=..., skin=...)`` spelling is a named [UI-SPRINT]
fork; every name here is DOCUMENTED-UNSTABLE until the naming session
ratifies it, and the package is imported explicitly
(``torchlens.visualization.lenses``), never eagerly.

The validation harness (corpus runner, Stage-0 deterministic audit,
RenderIR answer keys, naive-evaluator battery) lives in
``torchlens.visualization.lenses.audit``.
"""

from __future__ import annotations

from ._budget import (
    BAND_CEILING,
    BAND_FLOOR,
    BAND_TARGET,
    COVERAGE_FLOOR,
    BudgetResolution,
    resolve_budget,
)
from ._families import (
    SOURCE_FAMILIES,
    ResolvedSource,
    SourceCoverage,
    SourceFamily,
    resolve_source_family,
    source_coverage,
)
from ._filter import (
    BRIDGED_EDGE_LEGEND_LINE,
    FILTER_TOKENS,
    CompiledDisplayFilter,
    DisplayFilter,
    compile_display_filter,
)
from ._nonfinite import (
    NONFINITE_STATES,
    NonfiniteChannel,
    derive_nonfinite_channel,
    nonfinite_spec_fn,
)
from ._resolve import (
    LEGEND_DEPENDENT_LENSES,
    LENS_DETAIL_CEILING,
    LensResolution,
    draw_with_lens,
    resolve_lens,
)
from ._roster import (
    COMPOSITIONS,
    CORE_LENSES,
    EXTENDED_LENSES,
    LENS_SUBJECTS,
    PERF_FAMILIES,
    ROSTER,
)

__all__ = [
    "BAND_CEILING",
    "BAND_FLOOR",
    "BAND_TARGET",
    "BRIDGED_EDGE_LEGEND_LINE",
    "COMPOSITIONS",
    "CORE_LENSES",
    "COVERAGE_FLOOR",
    "EXTENDED_LENSES",
    "FILTER_TOKENS",
    "LEGEND_DEPENDENT_LENSES",
    "LENS_DETAIL_CEILING",
    "LENS_SUBJECTS",
    "NONFINITE_STATES",
    "PERF_FAMILIES",
    "ROSTER",
    "SOURCE_FAMILIES",
    "BudgetResolution",
    "CompiledDisplayFilter",
    "DisplayFilter",
    "LensResolution",
    "NonfiniteChannel",
    "ResolvedSource",
    "SourceCoverage",
    "SourceFamily",
    "compile_display_filter",
    "derive_nonfinite_channel",
    "draw_with_lens",
    "nonfinite_spec_fn",
    "resolve_budget",
    "resolve_lens",
    "resolve_source_family",
    "source_coverage",
]
