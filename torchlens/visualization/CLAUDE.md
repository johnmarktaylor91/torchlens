# visualization/ — Computational Graph Rendering

## What This Does
Renders computational graphs using Python's `graphviz` library. Called via
`show_model_graph()` or `log_forward_pass(..., vis_opt='rolled')`.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `rendering.py` | 1225 | All rendering logic: nodes, edges, module subgraphs, override system |

## Visualization Modes
- **`vis_opt='unrolled'`**: Shows every pass separately (uses LayerPassLog entries)
- **`vis_opt='rolled'`**: Collapses repeated layers into single nodes (uses LayerLog)
- **`vis_nesting_depth`**: How many module levels to show as subgraphs. Deeper modules
  rendered as collapsed "box3d" nodes.
- **`vis_direction`**: `'bottomup'` (default), `'topdown'`, or `'leftright'`
- **`vis_fileformat`**: `'pdf'`, `'png'`, `'svg'`, etc.

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

### `render_graph()`
Main entry. Creates graphviz.Digraph, iterates entries, builds nodes with appropriate
styling, adds edges from parent layers, sets up module subgraphs. Guard at line 71:
`_all_layers_logged` check prevents rendering when `keep_unsaved_layers=False` and
not all layers are saved.

### `_check_if_collapsed_module()` (lines 226-234)
**THE single guard point for IndexError safety.** Returns True when module nesting
exceeds `vis_nesting_depth`. All downstream indexing into
`containing_modules_origin_nested[vis_nesting_depth - 1]` relies on this. Do not weaken.

### `_set_up_subgraphs()`
Recursive context managers creating graphviz subgraphs for module nesting.

### Edge Logic
- **Deduplication**: `edges_used` set prevents duplicate edges
- **Collapsed module intra-edge skip**: When both parent and child are in same collapsed
  module, edge is hidden
- **Edge labels**: Arg position numbers, "IF" markers for conditional branches, pass
  numbers for rolled graphs when edges vary across passes

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
- **`show_model_graph` memory leak**: Never calls `model_log.cleanup()`. Circular refs
  leak until Python's gen-2 GC runs.

## Related
- [data_classes/](../data_classes/CLAUDE.md) — LayerLog and LayerPassLog provide the data
- `user_funcs.py` — `show_model_graph()` is the public API
- [postprocess/](../postprocess/CLAUDE.md) — Builds the graph structure that gets rendered
