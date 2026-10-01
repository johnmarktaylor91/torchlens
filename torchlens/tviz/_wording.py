"""Exact rendered disclosure wording for transformer pictures (tviz memo).

Every honesty line a picture renders lives HERE, once: renderers and emitters
import these constants so the wording can never fork between the matplotlib
path, the zero-dependency HTML emitter, and the bridge payloads. The precise
phrasing of every line is a named [UI-SPRINT] handoff (tviz memo section 11);
until then these are the panel's placeholder texts, DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

__all__ = [
    "AXES_WORDING",
    "JOINT_EFFECT_WORDING",
    "KV_GROUP_WORDING",
    "MEASURED_N_OF_M",
    "NOT_ADDITIVE_WORDING",
    "OUTPUT_EQUIVALENT_WORDING",
    "PANELS_NOT_COMPARABLE",
    "PROVENANCE_RECONSTRUCTED",
    "PROVENANCE_USER_SUPPLIED",
    "ROUTING_FOOTER",
    "UNMARKED_MASK_WORDING",
]

#: Permanent footer on every attention picture (tviz memo D5).
ROUTING_FOOTER = "attention weights are routing measurements, not causal importance"

#: Fixed axis semantics printed on every attention artifact (tviz memo D5).
AXES_WORDING = "rows attend from (query); columns attend to (key)"

#: Rendered when no mask provenance source is available (tviz memo D6) --
#: masked-position inference from zeros is BANNED, so absent provenance
#: renders unmarked with this exact wording.
UNMARKED_MASK_WORDING = "zero and masked positions are not distinguished"

#: Visible label whenever per-panel rescale replaces the fixed shared
#: [0, 1] probability domain (expert-only mode, tviz memo D5).
PANELS_NOT_COMPARABLE = "panels not comparable: per-panel color rescale"

#: Printed on every receipt grid beside the sum of cells (tviz memo D13).
NOT_ADDITIVE_WORDING = "single-head effects are not additive"

#: The joint-measurement line template for receipt grids (tviz memo D13).
JOINT_EFFECT_WORDING = (
    "joint effect of ablating all {n} measured heads together: {joint:+.4g} "
    "(sum of single-head cells: {cell_sum:+.4g}; " + NOT_ADDITIVE_WORDING + ")"
)

#: Coverage disclosure on annotated figures (tviz memo D16).
MEASURED_N_OF_M = "measured {n} of {m}"

#: The qualified fused-lowering disclosure (tviz memo D11): a lowered
#: counterfactual is labeled this, never "wrote the reconstructed facet".
OUTPUT_EQUIVALENT_WORDING = "output-equivalent counterfactual"

#: GQA header disclosure (tviz memo D15): shared KV storage is never
#: presented as independent per-query-head measurement.
KV_GROUP_WORDING = "kv group {group} of {n_groups}, shared by query heads {first}-{last}"

#: Provenance wording for reconstructed (fused-path) attention patterns.
PROVENANCE_RECONSTRUCTED = "reconstructed from fused-kernel inputs (read-only)"

#: Provenance wording for raw-tensor constructor inputs (composition row 12):
#: user-supplied values never earn a reconstruction badge.
PROVENANCE_USER_SUPPLIED = "user supplied; unvalidated"
