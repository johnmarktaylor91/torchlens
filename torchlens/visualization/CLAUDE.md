# visualization/ - Graph Rendering and Visual Helpers

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

`rendering.py` remains the compatibility facade used by `Trace.draw()` and `show_model_graph()`.
The graphviz renderer is the primary backend; `vis_node_placement="auto"` selects dot or the rank
layout according to the resolved graph cost. Forward sibling ordering is a Graphviz-only post-layout
operation and conservatively no-ops outside its supported forward/unrolled/dot cases.

## Related Surfaces

`node_spec.py`, `themes.py`, `modes.py`, and `overlays.py` provide node presentation decisions.
`code_panel.py` adds captured source beside Graphviz output. `bundle_diff.py`, `fastlog_preview.py`,
and `fastlog_live.py` support specialized visualization workflows.

Backward and combined graph entrypoints live in `_render_entrypoints.py`; they retain their dedicated
rendering path. Experimental Dagua remains opt-in under `torchlens.experimental.dagua`.
