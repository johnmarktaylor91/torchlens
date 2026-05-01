# visualization/ - Graph Rendering and Visual Helpers

## What This Does
Renders TorchLens graph structures and related previews. Graphviz is the primary public
renderer. Large graph layout can use the internal ELK backend. Experimental Dagua rendering
lives under `torchlens.experimental.dagua` and requires explicit opt-in.

## Files

| File | Purpose |
|------|---------|
| `rendering.py` | Main Graphviz forward/backward rendering, module focus, skip/collapse, intervention nodes |
| `_elk_internal/layout.py` | ELK/sfdp/topological layout backend for large graphs |
| `code_panel.py` | Source-code side panel helpers for Graphviz |
| `node_spec.py` | Public `NodeSpec`, HTML label rendering, intervention node specs |
| `modes.py` | Node mode presets: default, profiling, vision, attention |
| `themes.py` | Built-in themes and semantic node/edge attrs |
| `overlays.py` | Overlay score normalization, node lines, border attrs |
| `bundle_diff.py` | SVG bundle diff renderer used by demos/tests |
| `fastlog_preview.py` | Predicate preview graph overlay for fastlog |
| `fastlog_live.py` | Live fastlog helper rendering |
| `_render_utils.py` | Shared render helpers |
| `_summary_internal/` | Internal summary builders |

## Entry Points
- `ModelLog.render_graph()` and `show_model_graph()` call `rendering.render_graph()`.
- `show_backward_graph()` calls `rendering.render_backward_graph()`.
- `torchlens.viz.bundle_diff()` calls `visualization.bundle_diff.bundle_diff()`.
- `torchlens.fastlog.preview()` uses `visualization.fastlog_preview.preview_fastlog()`.

## Code Panel
`code_panel=True`, `"forward"`, `"class"`, or `"init+forward"` adds a Graphviz side panel
from source captured at logging time. Callable code panels require a live model. ELK direct
rendering bypasses this path.

## Visualization Modes
- `vis_mode="unrolled"` renders per-pass `LayerPassLog` nodes.
- `vis_mode="rolled"` renders aggregate `LayerLog` nodes.
- `module=...` focuses a submodule and inserts synthetic boundary nodes.
- `skip_fn` can hide layers while chaining edges through them.
- `collapse_fn` or module-depth options collapse module subgraphs.

## Node Customization
`VisualizationOptions.node_spec_fn` receives `(layer_log, default_spec)` and returns a
`NodeSpec` or `None`. `collapsed_node_spec_fn` customizes collapsed module nodes. Node mode
presets are applied before user callbacks, so callbacks win.

Graph, edge, gradient-edge, and module styling use dict/callable override options.

## Conditional and Intervention Rendering
Graphviz renders IF/THEN/ELIF/ELSE arm labels from cond-id-aware metadata. Intervention-ready
logs can render hook nodes and cone/site styling through `node_spec.py` helpers.

## ELK Backend
`_elk_internal/layout.py` uses Node.js ELK when available, falls back to sfdp/topological
layout when needed, and bypasses ELK for very large graphs. Some Graphviz-only features
remain unavailable or less complete on the ELK path.
