# bundle/ - Implementation Guide

Single-module package (`__init__.py`) exporting `Bundle` and `AmbiguousLabelError`
(`__all__` at the bottom of the file).

- `Bundle` is a named collection of aligned Traces for cross-run comparison,
  constructed by `tl.bundle(...)` (top-level convenience in `torchlens/__init__.py`)
  or directly as `tl.Bundle(...)`.
- `AmbiguousLabelError` fires on cross-member label lookups that match more than one
  aligned site.
- The bundle namespace is dual-homed by design: the bundle GRAPH construction
  behind `tl.show_bundle_graph` lives in `_user_public_impls.py` (the entry
  point, routed via `user_funcs.py`) and `visualization/_bundle_graph.py` (the
  `_add_bundle_forward_nodes` / supergraph-walk helpers), while `visualization/bundle_diff.py` is the paired-trace
  DIFF renderer. Keep graph/render logic out of this package either way.
- Public-name changes here must update the glossary and root docs in the same change
  (see the lockstep rule in the root `AGENTS.md`).
