# visualization/ — Computational Graph Rendering

## What This Does
Renders computational graphs using Python's `graphviz` library and an optional ELK
layout engine for large graphs. Called via `show_model_graph()` or
`log_forward_pass(..., vis_opt='rolled')`.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `rendering.py` | 1501 | Graphviz rendering: nodes, edges, module subgraphs, IF/THEN labels, override system |
| `elk_layout.py` | 1276 | ELK-based layout engine for large graphs (150k+ nodes), Worker thread, sfdp fallback |

## Visualization Modes
- **`vis_opt='unrolled'`**: Shows every pass separately (uses LayerPassLog entries)
- **`vis_opt='rolled'`**: Collapses repeated layers into single nodes (uses LayerLog)
- **`vis_nesting_depth`**: How many module levels to show as subgraphs. Deeper modules
  rendered as collapsed "box3d" nodes.
- **`vis_direction`**: `'bottomup'` (default), `'topdown'`, or `'leftright'`
- **`vis_fileformat`**: `'pdf'`, `'png'`, `'svg'`, etc.
- **`vis_node_placement`**: `'dot'` (default), `'elk'` (large graphs), auto-selected above 3500 nodes

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
- `vis_graph_overrides` — graph-level attrs
- `vis_node_overrides` — per-node (callable receives ModelLog + node)
- `vis_nested_node_overrides` — subgraph node attrs
- `vis_edge_overrides` — edge attrs
- `vis_gradient_edge_overrides` — gradient edge attrs
- `vis_module_overrides` — module subgraph attrs

## Key Internal Functions

### `render_graph()` (rendering.py)
Main entry. Creates graphviz.Digraph, iterates entries, builds nodes with appropriate
styling, adds edges from parent layers, sets up module subgraphs. Guard at line 71:
`_all_layers_logged` check prevents rendering when `keep_unsaved_layers=False` and
not all layers are saved.

### `_is_collapsed_module()` (rendering.py, lines 226-234)
**THE single guard point for IndexError safety.** Returns True when module nesting
exceeds `vis_nesting_depth`. All downstream indexing into
`containing_modules_origin_nested[vis_nesting_depth - 1]` relies on this. Do not weaken.

### `_set_up_subgraphs()` (rendering.py)
Recursive context managers creating graphviz subgraphs for module nesting.

### Edge Logic (rendering.py)
- **Deduplication**: `edges_used` set prevents duplicate edges (lines 956-958)
- **Collapsed module intra-edge skip**: When both parent and child are in same collapsed
  module, edge is hidden
- **Edge labels**: Arg position numbers, "IF" markers for `cond_branch_start_children`,
  "THEN" markers for `cond_branch_then_children` (lines 970-982), pass numbers for rolled
  graphs when edges vary across passes

### `render_elk_direct()` (elk_layout.py)
Alternative rendering path for large graphs. Uses Node.js ELK subprocess via Worker thread
(prevents stack overflow). Kahn's topological sort for stress seeding. Falls back to sfdp
when Node.js/ELK unavailable.

## ELK Layout Engine (elk_layout.py)
- **Worker thread**: Runs Node.js ELK subprocess in a separate thread to prevent stack overflow
- **Stress algorithm**: For graphs >150k nodes, uses stress layout with Kahn's topological sort for initial positions
- **Heap sizing**: 48x node count with 16GB floor for V8 heap
- **Spline cutoff**: Curved splines (`splines=true`) up to 1000 nodes, straight lines above
- **sfdp fallback**: When Node.js unavailable, loses module cluster structure

## Known Bugs
- **ELK-IF-THEN**: ELK path (elk_layout.py:954-1013) has ZERO conditional branch code.
  IF/THEN edge labels only appear in the Graphviz path (rendering.py:970-982), not ELK.
- **ELK-EDGE-LABEL-DEDUP**: Edge deduplication (elk_layout.py:990-992) keeps first edge
  between a pair, dropping later edges' arg labels. Label loss when first edge lacks label.
- **Edge dedup before labels**: rendering.py:956-958 dedup happens BEFORE IF/THEN label
  assignment at 970-982. Can silently drop labels for collapsed modules. Fix: move dedup after labels.

## Gotchas
- **Extra file**: graphviz `render()` creates a DOT source file alongside the rendered
  output (e.g., PDF). Known issue.
- **Substring false positive risk**: `parent_node.layer_label in arg_label` for unrolled
  nodes does substring check. `"add_1_5" in "add_1_50"` is a potential false match.
  Relies on `type_num` uniqueness to avoid it. Fragile.
- **Module addresses assumed to never contain colons**: `split(":")` in rendering.py
  assumes no embedded colons. PyTorch module names typically don't, but implicit assumption.
- **`show_buffer_layers` not forwarded** to rolled edge check (line 778) — always
  assumes False. Cosmetic only.
- **`show_model_graph` memory leak (FIXED)**: Now calls `model_log.cleanup()` in a
  try/finally block in user_funcs.py.

## Related
- [data_classes/](../data_classes/CLAUDE.md) — LayerLog and LayerPassLog provide the data
- `user_funcs.py` — `show_model_graph()` is the public API
- [postprocess/](../postprocess/CLAUDE.md) — Builds the graph structure that gets rendered
