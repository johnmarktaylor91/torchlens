# bundle/ - Implementation Guide

Package whose `__init__.py` holds the `Bundle` record and thin methods that delegate to the functional sibling modules (`_labels`, `_members`, `_deltas`, `_comparisons`, `_relation_carriage`, `_token_spans`, `_draw`, ...), exporting `Bundle` and `AmbiguousLabelError`
(`__all__` at the bottom of the file).

- `Bundle` is a named collection of aligned Traces for cross-run comparison,
  constructed by `tl.Bundle(...)` (top-level convenience in `torchlens/__init__.py`)
  or directly as `tl.Bundle(...)`.
- `AmbiguousLabelError` fires on cross-member label lookups that match more than one
  aligned site.
- The bundle namespace is dual-homed by design: the bundle GRAPH construction
  behind `torchlens._visualization.show_bundle_graph` lives in `_user_public_impls.py` (the entry
  point, routed via `user_funcs.py`) and `_visualization/_bundle_graph.py` (the
  `_add_bundle_forward_nodes` / supergraph-walk helpers), while `_visualization/bundle_diff.py` is the paired-trace
  DIFF renderer. Keep graph/render logic out of this package either way.
- Public-name changes here must update the glossary and root docs in the same change
  (see the lockstep rule in the root `AGENTS.md`).

## Module map (record rebuild, lane G)

`Bundle` keeps its stored attributes, dunders, field-derived properties and one thin method per
verb; each logic body is a function `f(self: Bundle, ...)` in a sibling module:

| Module | Holds |
|---|---|
| `_card.py` | `repr`, bounded `str`, the outcome distribution and the derived worst-of-members outcome |
| `_members.py` | member parsing and coercion, baseline resolution and auto-detection, capacity and eviction, `add`/`remove`/`remove_except`/`clear`, per-kind shared and divergent name lists |
| `_lookup.py` | `at`, the cached supergraph, structural consistency |
| `_aligned_sites.py` | aligned-site tables and the selector-site resolution behind `at` |
| `_operations.py` | `do`, `fork`, `run`, `map`, `show`, the chronology ledger rows, the effect-table read |
| `_comparisons.py` | `compare`, `rank_changes`, the comparability matrix, per-site diffs, relationship evidence; stored comparisons |
| `_deltas.py` | metric and lookup primitives (`_distance_value`, `_tensor_field`, which joins a multi-pass layer's passes in pass order) and the per-pass sites behind `draw(graph="diff")` (`site_deltas`, `SiteDeltas`) |
| `_relation_carriage.py` | relation table construction, removal cascades, `relate`, episode status, upstream/downstream |
| `_save.py` | `save` (the package's one deferred `_io` import) |
