# torchlens/visualization architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

## Forward Rendering Pipeline

Forward `Trace.draw()` resolves its request once, then follows one renderer-neutral pipeline:

```
SourceGraph -> NodeUniverse -> RenderIR -> renderers/{base,graphviz}
```

- `source_graph.py` normalizes the trace walk: focus, buffer visibility, `skip_fn`, and edge occurrences.
- `node_universe.py` projects that source graph into visible structural units and projected endpoints.
  Collapse planning uses this same universe through `collapse_plan.py`.
- `render_ir.py` decorates those units with resolved nodes, edges, regions, ordering constraints, and
  backend-ready statements. Renderers receive this immutable IR, not TorchLens trace objects.
- `renderers/base.py` defines the renderer protocol and capability checks; `renderers/graphviz.py`
  serializes and executes Graphviz. The rank layout backend also consumes the resolved IR.

`Trace.draw()` dispatches directly to `_render_dot.py`; backward and combined entrypoints dispatch
to `_render_entrypoints.py`. The graphviz renderer is the primary backend;
`vis_node_placement="auto"` selects dot or the rank
layout according to the resolved graph cost. Forward sibling ordering is a Graphviz-only post-layout
operation and conservatively no-ops outside its supported forward/unrolled/dot cases.

## Related Surfaces

`node_spec.py`, `themes.py`, `modes.py`, and `overlays.py` provide node presentation decisions.
`code_panel.py` adds captured source beside Graphviz output. `bundle_diff.py`, `fastlog_preview.py`,
and `fastlog_live.py` support specialized visualization workflows.

## Internal Helper Modules

The `_render_dot.py` entry point is split across sibling helper modules; all are internal:

| Module | Purpose |
|--------|---------|
| `request.py` | Resolved visualization requests and output targets (`ResolvedRenderRequest`, `RenderTarget`, `RenderContext`) |
| `_render_common.py` | Shared render types, constants, and imports for Graphviz rendering |
| `_render_flow.py` | Focus, skip, container, and sibling setup helpers |
| `_render_nodes.py` | Node construction and raw value helpers |
| `_render_edges.py` | Edge and endpoint helpers |
| `_render_leaf.py` | Backward/grad-fn leaf placement and module inference |
| `_render_ordering.py` | Sibling-ordering scope decision, plain-layout verification, and the DOT rank-group post-pass |
| `_render_regions.py` | Nested module-cluster (region) subgraph emission and empty-subtree pruning |
| `_svg_compose.py` | SVG post-processing (image inlining, viewBox normalization) and code-panel composition |
| `_render_utils.py` | Internal Graphviz helpers shared across rendering paths (subprocess execution, HTML escaping) |
| `_label_format.py` | Node label formatting helpers |
| `_typography.py` | The one typography record (vizmech D29): pinned font family + semantic size roles consumed by every label builder |
| `_legend.py` | The one compact HTML-table legend node in a dedicated rank (role legend, channel disclosures, backward key as sections; vizmech D28/D30) |
| `_geometry_audit.py` | Geometry audit v2 + usability envelopes: widened element classes, grid-indexed pairing, engine-attributed parsing (vizmech wave 3; test-support instrument) |
| `_edge_multiplicity.py` | Rendered-edge multiplicity disclosure (dedupe registry, honest `xN` edge labels) and the D9 argument-position edge-label builders; at visible fan-in >= `_ARG_LABEL_MIDPOINT_FANIN` argument labels ride reserved-space midpoint labels and same-pair parallel arg edges merge into one row-listing edge (D03-R4) |
| `_buffer_visibility.py` | Tri-state `show_buffer_layers` visibility predicates (R43 split from `_render_edges.py`) |
| `_condensed_flow.py` | Child condensed-flow-graph construction for smart module collapse |
| `_segment_descriptors.py` | Segment descriptor construction and label derivation for collapse plans |
| `_collapse_disclosures.py` | Human-facing collapse disclosure warnings (no-silent-floor, memo D4) |
| `_collapse_frontier.py` | Collapse-optimizer frontier records (`_DecisionPoint` and the decision witnesses), `K_CAP`/`FRONTIER_CAP`, and the pure frontier-merge helpers |
| `_bundle_graph.py` | Graphviz node/cluster/edge construction helpers behind `tl.show_bundle_graph` |
| `_rank_layout_internal/`, `_summary_internal/` | Rank-layout backend internals and `summary()` internals |
| `renderers/` | Renderer protocol (`base.py`) and the Graphviz backend (`graphviz.py`) |

Smart collapse: `auto_collapse.py` (analysis + fold discovery), `collapse_optimizer.py` (v2
frontier selection; owns the `COLLAPSE_OPTIMIZER_MAX_OPS` defensive constant),
`collapse_plan.py` (plan/schedule records), and the F11 modules --
`_collapse_signatures.py` (B1 member-fingerprint memo), `_collapse_runs.py` (B2 indexed
run-fold legality), `collapse_estimator.py` ((U, W) admission estimator, budget, watchdog),
`collapse_fallback.py` (deterministic significance-greedy fallback planner),
`collapse_ladder.py` (typed event ladder: auto, float levels, and the public schedule),
`collapse_patterns.py` (declarative user-named pattern folding) -- form the engine.

Backward and combined graph entrypoints live in `_render_entrypoints.py`. Their grad-function source
normalizers produce the same `RenderIRNode`, `RenderIREdge`, and `RenderIRRegion` records as the forward
pipeline, including backward-pass regions for unrolled graphs, and dispatch through the same renderer
protocol and Graphviz backend. Experimental Dagua remains opt-in under `torchlens.experimental.dagua`.

## Key Internal Functions

### `draw()` in `_render_dot.py`
Main forward graph entry point. It normalizes buffer visibility, applies module focus, skip
and collapse decisions, builds Graphviz nodes/edges, styles modules, optionally adds legends
and code panels, and writes/renders output.

### `render_backward_graph()` in `_render_entrypoints.py`
Renders `GradFn` nodes and grad edges captured by `backends/torch/backward.py`.

### Collapse and Focus
- The v2 smart-collapse ENGINE behind `draw(collapse="auto"/"max")` is
  `auto_collapse.py` + `collapse_optimizer.py` + `collapse_plan.py` (public
  schedule via `Trace.collapse_plan()`/`collapse_schedule()`); change policy
  THERE. `_should_collapse_module()` (`_render_leaf.py`) is only the legacy
  per-leaf decision path.
- `_is_collapsed_module()` protects indexing into `modules`; keep its guard strict.
- `_build_module_focus_entries()` inserts boundary nodes for module-scoped renders.
- Focus runs before skip/collapse.

### Edge Logic
- `_build_skip_filtered_edge_map()` chains edges through skipped nodes.
- `_compute_edge_label()` combines arg labels, conditional arm labels, and pass labels.
- `_get_arm_edge_entries()` reads cond-id-aware conditional metadata.
- Duplicate-edge filtering can affect labels; check conditional and collapsed-module tests
  when editing edge construction.
- `order_siblings=True` captures rendered forward edges at emission time, queues same-rank
  invisible chains for sole-parent sibling fanouts, verifies local edge-stretch with
  `dot -Tplain`, and drops chains above the cap.

### NodeSpec and Modes
- `compute_default_node_lines()` builds default label rows.
- `_apply_node_spec_fn()` applies mode presets and user callbacks.
- `node_spec.py` owns `NodeSpec`, `render_lines_to_html()`, and intervention node specs.
- `modes.py` owns default/profiling/vision/attention preset functions.

### Rank Layout
`_rank_layout_internal/layout.py` exposes `render_rank_layout()`,
`estimate_rank_layout_cost()`, and `get_node_placement_engine()`. Auto layout uses a
cost trigger based on topological rank spans and switches from Graphviz `dot` to the
pure-Python rank layout above 20,000 cost units.

## Other Modules

- `themes.py`: theme registry and semantic attrs.
- `overlays.py`: overlay lines and borders for score maps/nonfinite markers.
- `bundle_diff.py`: multi-log SVG diff renderer for `torchlens.viz.bundle_diff`.
- `fastlog_preview.py`: overlays predicate decisions on a full log.
- `fastlog_live.py`: live fastlog preview helpers.
- `code_panel.py`: Graphviz code side panel.
- `_render_common.py`: shared render types, constants, and imports for Graphviz rendering.
- `_render_nodes.py`: node construction and raw-value helpers for Graphviz rendering.
- `_render_edges.py`: edge and endpoint helpers for Graphviz rendering.
- `_render_flow.py`: focus, skip, container, and sibling setup helpers.
- `_render_ordering.py`: sibling-ordering scope decision, plain-layout verification, and the DOT rank-group post-pass.
- `_render_regions.py`: nested module-cluster (region) subgraph emission and empty-subtree pruning.
- `_svg_compose.py`: SVG post-processing (image inlining, viewBox normalization) and code-panel composition.
- `_render_utils.py`: internal Graphviz helpers shared across rendering paths.
- `_label_format.py`: node-label formatting helpers.
- `_typography.py`: the one typography record (vizmech D29) -- pinned font family +
  semantic size roles consumed by every label builder.
- `_legend.py`: the one compact HTML-table legend node in a dedicated rank; the role
  legend, channel disclosures, and the backward key are sections of it (D28/D30).
- `_geometry_audit.py`: geometry audit v2 + usability envelopes -- widened element
  classes, grid-indexed pairing, engine-attributed parsing (vizmech wave 3;
  test-support instrument).
- `_edge_multiplicity.py`: rendered-edge multiplicity disclosure (r19 dedupe registry)
  plus the D9 argument-position edge-label builders; at visible fan-in >=
  `_ARG_LABEL_MIDPOINT_FANIN` argument labels relocate to reserved-space midpoint
  labels and same-pair parallel arg edges merge into one row-listing edge (D03-R4).
- `_buffer_visibility.py`: tri-state `show_buffer_layers` predicates (R43 split from
  `_render_edges.py`).
- `_condensed_flow.py`: child condensed-flow-graph construction for smart collapse.
- `_segment_descriptors.py`: segment descriptor construction and label derivation for
  collapse plans (R43 split from `collapse_optimizer.py`).
- `_collapse_disclosures.py`: human-facing collapse disclosure warnings (no-silent-floor,
  collapse memo D4; R43 split from `auto_collapse.py`).
- `_collapse_frontier.py`: frontier records, `K_CAP`/`FRONTIER_CAP` and the pure
  frontier-merge helpers (R43 split from `collapse_optimizer.py`).
- `_bundle_graph.py`: Graphviz construction helpers for `tl.show_bundle_graph` (R43 split
  from `_user_public_impls.py`).
- `request.py`: resolved visualization requests and output targets.

## Gotchas

- Graphviz render writes a DOT source file alongside rendered output.
- Sibling ordering is intentionally scoped to forward unrolled Graphviz dot renders under
  the node cap. The exact in-scope predicate is `_should_order_siblings`
  (`_render_ordering.py`): dot engine AND unrolled mode AND node count under
  `SIBLING_ORDER_NODE_CAP` AND no `module=` focus AND
  `vis_intervention_mode == "node_mark"` AND `vis_call_depth >= 1000` — so
  rolled, focused, rank-layout, capped-depth, and large graphs no-op there
  (backward/combined renders never reach it; they dispatch through
  `_render_entrypoints.py`). It has NO collapse or conditional term:
  `collapse_fn` is accepted but unused by the predicate.
- The rank-layout path is a direct DOT writer rendered through `neato -n`; verify module
  clusters and edge labels after layout changes.
- `show_model_graph()` (implemented in `torchlens/_user_public_impls.py`, not
  package-local) should cleanup temporary logs in `finally`.
- Buffer visibility has multiple modes; use `_normalize_buffer_visibility()`.
- Intervention node rendering keys on the trace's `_intervention_spec`
  (`node_spec.py`) and the `vis_intervention_mode` render option
  (`_render_dot.py`) — NOT on the deprecated `intervention_ready` name.
- Bundle diff rendering is SVG-string based; compare snapshots after visual changes.

## Tests to Run After Changes

- `pytest tests/test_conditional_rendering.py -x --tb=short`
- `pytest tests/test_node_spec_api.py tests/test_node_modes.py -x --tb=short`
- `pytest tests/test_bundle_diff_renderer.py -x --tb=short`
- `pytest tests/test_large_graphs.py -x --tb=short` for layout backend changes
