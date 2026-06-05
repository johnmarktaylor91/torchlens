# Edge-count + Edge-object proposal (mini-sprint)

## Grounding (how TL models edges today)
- Edge = directed `parent_label -> child_label`. Stored on every `Op` as `parents` /
  `children` (bare label lists) + `parent_arg_positions` (arg slot the parent fills).
- In-degree exists: `Op.num_parents`. No out-degree count, no trace/module edge count.
- Node categories: boundary (`is_input`/`is_output`/`is_buffer`), `is_compute_op =
  not(input|output|buffer)`, plus `is_orphan` / `is_internal_source` / `is_internal_sink`.
- Backward graph = `GradFn.parents`/`children` (own connectivity), gated on gradient capture.
- Op = per-pass record; Layer = aggregate (recurrent ops rolled). Same multi-pass nuance as nodes.
- Edge-as-data precedent: `conditional_branch_edges`, `conditional_arm_entry_edges` (tuple lists).
- Module scoping: `ModuleCall.ops` (pass-qualified labels), `input_ops`/`output_ops`; each Op has
  `address` (module path) + `input_to_module_calls` / `output_of_module_calls`.

## The counting decisions to lock (the "what to include" riff)
1. **Boundary edges:** count the input->first-op and last-op->output sentinel edges, or only
   compute<->compute edges? -> REC: expose BOTH (a raw total + a `compute_edges` count).
2. **Buffer edges:** count `producer -> buffer -> reader` edges? -> REC: yes, counted; also a
   separate `num_buffer_edges` so they can be isolated (consistent with `compute_ops` excluding buffers).
3. **Orphan / internal-source edges:** orphans are pruned by default; internal-source (`torch.ones`)
   edges are real. -> REC: count real edges only; orphans already removed, so naturally excluded.
4. **Per-pass vs rolled:** Op-graph edges (per-pass, what actually executed) vs Layer-graph edges
   (rolled/aggregate). -> REC: default `num_edges` = per-pass Op graph (matches `num_ops`); add a
   rolled count keyed off layers if cheap (`num_layer_edges`).
5. **Forward vs backward:** -> REC: `num_edges` = forward; `num_backward_edges` separate, `None`
   when no gradients captured (mirrors `num_backward_passes` / `num_saved_grad_fns`).
6. **Multi-edges:** a parent feeding a child in two arg slots = 1 edge or 2? -> REC: count DISTINCT
   (parent,child) pairs as the edge count; expose arg-multiplicity only on the Edge object (Tier 4).

## Tier 1 — Trace-level counts (ship; trivial)
- `Trace.num_edges` -> int. Distinct forward (parent,child) pairs over the per-pass Op graph.
- `Trace.num_compute_edges` -> int. Edges where both endpoints are compute ops (no boundary/buffer).
- `Trace.num_buffer_edges` -> int. Edges incident to a buffer-version node.
- `Trace.num_backward_edges` -> int | None. GradFn-graph edges; `None` if no gradients captured.
- (optional) `Trace.num_layer_edges` -> int. Rolled aggregate-graph edges (Layer level).
- Surfaced in `summary()` and `__repr__` (e.g. `... 412 ops, 530 edges`).

## Tier 2 — Module / ModuleCall / Layer scoping (ship; the useful part)
Per `Module` and `ModuleCall`:
- `num_internal_edges` -> edges with BOTH endpoints inside this module(call).
- `num_input_edges` / `num_output_edges` -> edges crossing the module boundary inward / outward
  (the module's fan-in / fan-out). Parallels existing `input_ops`/`output_ops`.
- (optional) `num_edges` -> internal + boundary (the module's total edge footprint).
Per `Op`/`Layer`: add `num_children` (out-degree) to complement the existing `num_parents`
(in-degree); optionally `in_degree`/`out_degree` aliases if we want graph-theoretic names.

## Tier 3 — Graph-shape / analytics (cheap derived; optional)
- `Trace.edge_density` -> edges / nodes (branching factor).
- `Trace.max_in_degree` / `max_out_degree`.
- These are one-liners off Tier 1/2; include only if they earn their name. REC: skip for v1,
  revisit if users ask (avoid surface bloat).

## Tier 4 — Edges as first-class objects (the JMT riff)
An `Edge` view + `trace.edges` accessor, sibling of ops/layers/modules/params/buffers.
What an `Edge` would carry (projection over existing data, like `Buffer` over version nodes):
- `source` / `target` (Op or Layer endpoints), `source_label` / `target_label`.
- `arg_position` (which child arg slot; from `parent_arg_positions`), `arg_multiplicity`.
- `direction` (`"forward"` / `"backward"`).
- `activation` (the tensor on the edge == `source.out`; shape/dtype/memory), `has_saved_activation`.
- `crosses_module_boundary` + `modules_entered` / `modules_exited`.
- `kind`: `compute` / `buffer_read` / `buffer_write` / `input` / `output` / `conditional`.
- `is_recurrent` (loop back-edge), `span` (output_distance(target) - output_distance(source) ->
  long-range/residual detection), `conditional_arm` membership.
- `.draw()` hook / render hints (edges already have viz styling).

**Why it could be useful**
- Interventions: an edge IS a natural intervention site ("ablate the edge attn->mlp", "scale the
  residual edge"). Today you select nodes; edges would let you target a specific data path.
- Analysis/RSA: edge list is the canonical form for graph algorithms; `trace.edges` makes TL a
  clean graph substrate. "All edges into module X", "the residual-stream edge".
- Viz: edges carry labels/styling; an `Edge` object is the natural owner of per-edge render hints.
- Symmetry: TL already made Param and Buffer first-class projections; Edge completes the graph.

**Why to be cautious**
- Surface cost: violates "don't add top-level names casually"; `Edge` + accessor + Super-family
  (`SuperEdge`?) + save/load is a real expansion, not a mini-sprint.
- Scale: edges >> nodes; a 10M-node graph has more edges. Must be LAZY/on-demand views (construct
  on access, never stored/serialized) or it blows up memory + `.tlspec` size.
- Mostly projection: an Edge's data is derivable from endpoints (`activation == source.out`), so
  it's convenience, not new info — the bar for a new class is "does it unlock something nodes can't".
  Interventions-on-edges is the one genuinely-new capability; the rest is ergonomics.

**REC:** Do NOT ship the full `Edge` class in this mini-sprint. Ship Tiers 1-2 now (counts +
module scoping) — that's the easy, clearly-useful 90%. File `Edge`-as-object as its own design
todo, gated on a concrete driver (edge-targeted interventions). IF we want a taste now, a minimal
lazy `trace.edges` returning lightweight `(source,target,kind,arg_position)` views is cheap and
non-committal — but flag it as provisional, not part of the locked 2.0 surface.

## Naming summary (matches `num_*` convention: `num_ops`, `num_layers`, `num_backward_passes`)
Trace: `num_edges`, `num_compute_edges`, `num_buffer_edges`, `num_backward_edges`,
`num_layer_edges?`. Module/ModuleCall: `num_internal_edges`, `num_input_edges`,
`num_output_edges`, `num_edges?`. Op/Layer: `num_children` (+ existing `num_parents`).

## Open questions for JMT
- Q1: rolled `num_layer_edges` in v1, or per-pass only?
- Q2: `in_degree`/`out_degree` aliases, or just `num_parents`/`num_children`?
- Q3: Tier 3 analytics now or defer?
- Q4: minimal provisional `trace.edges` lazy view now, or fully defer the Edge object?
