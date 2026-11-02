# bundle/ - Implementation Guide

Package whose `__init__.py` holds the `Bundle` record and thin methods that delegate to the functional sibling modules (`_labels`, `_members`, `_deltas`, `_comparisons`, `_relation_carriage`, `_token_spans`, `_draw`, ...), exporting `Bundle` and `AmbiguousLabelError`
(`__all__` at the bottom of the file).

- `Bundle` is a named collection of aligned Traces for cross-run comparison,
  constructed by `tl.Bundle(...)` (top-level convenience in `torchlens/__init__.py`)
  or directly as `tl.Bundle(...)`.
- `AmbiguousLabelError` fires on cross-member label lookups that match more than one
  aligned site.
- The bundle namespace is dual-homed by design: `Bundle.draw` (the merged bundle graph; `Bundle.show()`
  defaults to it) is a thin method here, while its keyword checks and the graph construction live in
  `_visualization/_bundle_graph.py` (`draw_bundle`, the `_add_bundle_forward_nodes` / supergraph-walk
  helpers), and `_visualization/bundle_diff.py` is the paired-trace DIFF renderer. Keep graph/render
  logic out of this package either way.
- Public-name changes here must update the glossary and root docs in the same change
  (see the lockstep rule in the root `AGENTS.md`).
