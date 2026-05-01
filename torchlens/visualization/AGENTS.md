# visualization/ - Implementation Guide

## Key Internal Functions

### `render_graph()` in `rendering.py`
Main forward graph entry point. It normalizes buffer visibility, applies module focus, skip
and collapse decisions, builds Graphviz nodes/edges, styles modules, optionally adds legends
and code panels, and writes/renders output.

### `render_backward_graph()` in `rendering.py`
Renders `GradFnLog` nodes and gradient edges captured by `capture/backward.py`.

### Collapse and Focus
- `_should_collapse_module()` is the central collapse decision.
- `_is_collapsed_module()` protects indexing into `containing_modules`; keep its guard strict.
- `_build_module_focus_entries()` inserts boundary nodes for module-scoped renders.
- Focus runs before skip/collapse.

### Edge Logic
- `_build_skip_filtered_edge_map()` chains edges through skipped nodes.
- `_compute_edge_label()` combines arg labels, conditional arm labels, and pass labels.
- `_get_arm_edge_entries()` reads cond-id-aware conditional metadata.
- Duplicate-edge filtering can affect labels; check conditional and collapsed-module tests
  when editing edge construction.

### NodeSpec and Modes
- `compute_default_node_lines()` builds default label rows.
- `_apply_node_spec_fn()` applies mode presets and user callbacks.
- `node_spec.py` owns `NodeSpec`, `render_lines_to_html()`, and intervention node specs.
- `modes.py` owns default/profiling/vision/attention preset functions.

### ELK
`_elk_internal/layout.py` exposes `render_elk_direct()`, `render_with_elk()`,
`render_with_sfdp()`, and `get_node_placement_engine()`. It uses Node.js workers, V8 heap
sizing, topological seeding, and fallback layout for unavailable ELK.

## Other Modules
- `themes.py`: theme registry and semantic attrs.
- `overlays.py`: overlay lines and borders for score maps/nonfinite markers.
- `bundle_diff.py`: multi-log SVG diff renderer for `torchlens.viz.bundle_diff`.
- `fastlog_preview.py`: overlays predicate decisions on a full log.
- `fastlog_live.py`: live fastlog preview helpers.
- `code_panel.py`: Graphviz code side panel.

## Gotchas
- Graphviz render writes a DOT source file alongside rendered output.
- ELK paths do not support every Graphviz feature; verify conditional labels, module clusters,
  and code panels after layout changes.
- `show_model_graph()` should cleanup temporary logs in `finally`.
- Buffer visibility has multiple modes; use `_normalize_buffer_visibility()`.
- Intervention node rendering depends on `intervention_ready` metadata.
- Bundle diff rendering is SVG-string based; compare snapshots after visual changes.

## Tests to Run After Changes
- `pytest tests/test_conditional_rendering.py -x --tb=short`
- `pytest tests/test_node_spec_api.py tests/test_node_modes.py -x --tb=short`
- `pytest tests/test_bundle_diff_renderer.py -x --tb=short`
- `pytest tests/test_large_graphs.py -x --tb=short` for layout backend changes
