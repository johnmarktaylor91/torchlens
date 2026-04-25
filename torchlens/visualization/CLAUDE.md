# visualization/ — Computational Graph Rendering

## What This Does
Renders computational graphs using Python's `graphviz` library, an optional ELK
layout engine for large graphs, and an optional dagua renderer. Called via
`show_model_graph()` or `log_forward_pass(..., vis_mode='rolled')`.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `rendering.py` | 1900+ | Graphviz rendering: nodes, skip-aware edges, module subgraphs, IF/THEN labels, NodeSpec callbacks |
| `elk_layout.py` | 1300+ | ELK-based layout engine for large graphs (150k+ nodes), Worker thread, sfdp fallback |
| `code_panel.py` | n/a | Source-code capture and Graphviz side-panel rendering helpers |
| `node_spec.py` | — | Public NodeSpec dataclass and HTML-label row renderer |
| `modes.py` | — | NodeSpec preset registry for default/profiling/vision/attention node modes |
| `dagua_bridge.py` | — | ModelLog → DaguaGraph conversion for dagua renderer (opt-in) |

## How It Connects

Called by `user_funcs.py:show_model_graph()`. Reads LayerLog/LayerPassLog from ModelLog
to build graph nodes and edges. Two independent rendering paths (Graphviz vs dagua),
with Graphviz using an optional ELK layout backend for large graphs.

## Code Panel
`ModelLog.render_graph(code_panel=...)` and `show_model_graph(..., code_panel=...)`
can add a right-side source-code panel next to the computational graph.

- `False`: no panel.
- `True` / `"forward"`: use captured `model.forward` source.
- `"class"`: use captured full model class source.
- `"init+forward"`: use captured `__init__` plus `forward` source.
- Callable: called as `fn(model)` at render time. This only works while the
  original model object is still alive; saved ModelLogs should use built-in modes.

Built-in source snippets are captured at `log_forward_pass()` time and stored on
`ModelLog._source_code_blob`. The panel is pure Graphviz: `code_panel.py` adds a
`cluster_torchlens_code_panel` subgraph with an HTML-like monospace table label,
expands tabs to 4 spaces, escapes `<`, `>`, and `&`, and truncates after
`MAX_CODE_PANEL_LINES` lines. Because ELK direct rendering bypasses `Digraph`
construction, code-panel renders stay on the Graphviz path.

## Visualization Modes
- **`vis_mode='unrolled'`**: Shows every pass separately (uses LayerPassLog entries)
- **`vis_mode='rolled'`**: Collapses repeated layers into single nodes (uses LayerLog)

## Module Focus
`ModelLog.render_graph(module=...)` and `show_model_graph(..., module=...)` can
render one module's subgraph in the same visual format as the full model.

- `module=None`: render the whole model.
- `module="block.address"`: focus the ModuleLog with that address.
- `module=module_log`: focus that ModuleLog; it must belong to the rendered ModelLog.
- `ModuleLog.show_graph(**kwargs)`: convenience wrapper for
  `module_log._source_model_log.render_graph(module=module_log, **kwargs)`.

The focus pass runs before `skip_fn` and `collapse_fn`. It keeps layers whose
`containing_modules` path includes the target module address, drops outside
layers, and inserts synthetic green/red boundary nodes for external upstreams
and downstreams. Child module clusters inside the focused module are still
available for normal collapse behavior.

## Node Mode Presets
`VisualizationOptions.node_mode` applies a built-in NodeSpec preset before any
user-supplied `node_spec_fn`, so user callbacks always win. Public flat alias:
`vis_node_mode`.

- **`default`**: identity preset; Phase 1 default important-args rows remain.
- **`profiling`**: appends available timing, output bytes, call site, and function
  name rows. Collapsed modules get aggregate timing/output rows.
- **`vision`**: appends input/output shape rows for spatial layers such as conv,
  pooling, adaptive pooling, upsample, interpolate, and resize.
- **`attention`**: appends heads/embed/head_dim/dropout details for attention ops
  and role annotations for attention projection Linear layers.

## Node Styling
| Type | Color |
|------|-------|
| Input layers | Green (#98FB98) |
| Output layers | Red (#ff9999) |
| Parameter nodes | Gray (#E6E6E6) |
| Boolean layers | Yellow (#F7D460) |
| Regular layers | White/default |
| Collapsed modules | box3d shape |

## Override System
Node labels and node-level attributes use the callable NodeSpec API:

- `VisualizationOptions.node_spec_fn`: receives `(layer_log, default_spec)` and
  returns a `NodeSpec` or `None`. In unrolled mode the callback still receives
  the parent aggregate `LayerLog`, not the per-pass `LayerPassLog`.
- `VisualizationOptions.collapsed_node_spec_fn`: receives
  `(module_log, default_spec)` for collapsed module nodes.
- `NodeSpec.lines` are plain text rows; `render_lines_to_html()` is the only
  place that converts them to Graphviz HTML-like table labels.

Graph, edge, gradient-edge, and module styling remain dict/callable based:
`vis_graph_overrides`, `vis_edge_overrides`, `vis_gradient_edge_overrides`,
`vis_module_overrides`.

`VisualizationOptions.collapse_fn` receives a `ModuleLog` and replaces the
legacy `max_module_depth`/`vis_nesting_depth` collapse decision when supplied.
`VisualizationOptions.skip_fn` receives a `LayerLog`, may not skip input or
output layers, and chains edges through skipped nodes before both Graphviz and
ELK layout consume the graph.

## ELK Layout Engine
- Worker thread runs Node.js ELK subprocess (prevents stack overflow)
- Stress algorithm for >150k nodes with Kahn's topological sort for initial positions
- **>100k nodes bypass ELK entirely** → Python topological layout (O(n+m))
- sfdp fallback when Node.js unavailable (loses module cluster structure)
