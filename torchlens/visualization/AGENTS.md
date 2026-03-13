# visualization/ — Implementation Guide

## Key Internal Functions

### `render_graph()` (rendering.py)
Main entry. Creates graphviz.Digraph, iterates entries, builds nodes with appropriate
styling, adds edges from parent layers, sets up module subgraphs. Guard at line 71:
`_all_layers_logged` check prevents rendering when `keep_unsaved_layers=False` and
not all layers are saved.

### `_is_collapsed_module()` (rendering.py, lines 226-234)
**THE single guard point for IndexError safety.** Returns True when module nesting
exceeds `vis_nesting_depth`. All downstream indexing into
`containing_modules[vis_nesting_depth - 1]` relies on this. Do not weaken.

### Edge Logic (rendering.py)
- **Deduplication**: `edges_used` set prevents duplicate edges (lines 956-958)
- **Collapsed module intra-edge skip**: When both parent and child are in same collapsed
  module, edge is hidden
- **Edge labels**: Arg position numbers, "IF" markers for `cond_branch_start_children`,
  "THEN" markers for `cond_branch_then_children`, pass numbers for rolled graphs

### `render_elk_direct()` (elk_layout.py)
Alternative rendering path for large graphs. Node.js ELK subprocess via Worker thread.
Kahn's topological sort for stress seeding. sfdp fallback when Node.js/ELK unavailable.

### ELK Layout Details
- **Heap sizing**: 48x node count with 16GB floor for V8 heap
- **Spline cutoff**: Curved splines up to 1000 nodes, straight lines above
- `vis_node_placement`: `'dot'` (default), `'elk'` (large graphs), auto-selected above 3500 nodes
- `vis_direction`: `'bottomup'` (default), `'topdown'`, or `'leftright'`

## Known Bugs
- **ELK-IF-THEN**: ELK path has ZERO conditional branch code. IF/THEN edge labels only
  appear in the Graphviz path (rendering.py:970-982), not ELK.
- **ELK-EDGE-LABEL-DEDUP**: Edge dedup (elk_layout.py:990-992) keeps first edge between a
  pair, dropping later edges' arg labels.
- **Edge dedup before labels**: rendering.py:956-958 dedup happens BEFORE IF/THEN label
  assignment at 970-982. Can silently drop labels for collapsed modules.

## Gotchas
- **Extra file**: graphviz `render()` creates a DOT source file alongside the rendered output
- **Substring false positive risk**: `parent_node.layer_label in arg_label` for unrolled
  nodes does substring check. Relies on `type_num` uniqueness. Fragile.
- **Module addresses assumed to never contain colons**: `split(":")` assumes no embedded colons
- **`show_buffer_layers` not forwarded** to rolled edge check — always assumes False (cosmetic)
- **`show_model_graph` memory leak (FIXED)**: Now calls `model_log.cleanup()` in try/finally
- **ELK stress O(n^2)**: allocates two n^2 × 8-byte distance matrices. 100k nodes = 160GB.
