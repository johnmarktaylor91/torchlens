# visualization/ — Computational Graph Rendering

## What This Does
Renders computational graphs using Python's `graphviz` library, an optional ELK
layout engine for large graphs, and an optional dagua renderer. Called via
`show_model_graph()` or `log_forward_pass(..., vis_mode='rolled')`.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `rendering.py` | 1501 | Graphviz rendering: nodes, edges, module subgraphs, IF/THEN labels, override system |
| `elk_layout.py` | 1276 | ELK-based layout engine for large graphs (150k+ nodes), Worker thread, sfdp fallback |
| `dagua_bridge.py` | — | ModelLog → DaguaGraph conversion for dagua renderer (opt-in) |

## How It Connects

Called by `user_funcs.py:show_model_graph()`. Reads LayerLog/LayerPassLog from ModelLog
to build graph nodes and edges. Two independent rendering paths (Graphviz vs dagua),
with Graphviz using an optional ELK layout backend for large graphs.

## Visualization Modes
- **`vis_mode='unrolled'`**: Shows every pass separately (uses LayerPassLog entries)
- **`vis_mode='rolled'`**: Collapses repeated layers into single nodes (uses LayerLog)

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
All visual properties overridable via dicts or callables:
`vis_graph_overrides`, `vis_node_overrides`, `vis_nested_node_overrides`,
`vis_edge_overrides`, `vis_gradient_edge_overrides`, `vis_module_overrides`.

## ELK Layout Engine
- Worker thread runs Node.js ELK subprocess (prevents stack overflow)
- Stress algorithm for >150k nodes with Kahn's topological sort for initial positions
- **>100k nodes bypass ELK entirely** → Python topological layout (O(n+m))
- sfdp fallback when Node.js unavailable (loses module cluster structure)
