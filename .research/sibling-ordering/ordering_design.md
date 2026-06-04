# Design: deterministic horizontal ordering of parallel siblings in TorchLens Graphviz output

## Problem (empirically established)

When one tensor fans out to N parallel consumers (e.g. an Inception block feeding
1x1 / 3x3 / 5x5 / pool branches), the Graphviz DOT rendering does NOT place those
branches left-to-right in execution order. The order is effectively arbitrary.

Evidence (measured, not theorized), `torchvision.models.googlenet`, traced with
`tl.trace`, rendered with `log.draw()`, coordinates read via `dot -Tplain`:
- **9 of 9 Inception fan-out points** render their branches OUT of execution order.
- The *first-executed* branch lands *rightmost* every time.

## Root cause (confirmed)

It is NOT that children are unsorted. Three facts already hold in current code:
1. `parent_node.children` is already in execution order (capture appends children as
   consumer ops are discovered, which is execution order). Verified: a fan-out's
   children come out sorted ascending by `Op.step_index`.
2. Edges are emitted in that order in the DOT source.
3. `ordering=out` is ALREADY set on the graph, node_attr, and edge_attr (rendering.py
   lines ~733/747/1146).

The breakage: **Graphviz's crossing-minimization (mincross) overrides `ordering=out`
for multi-op branches that live inside module-box clusters.** `ordering=out` is a soft
hint that mincross is free to violate, and it does so once branches are more than a
single node deep and/or sit inside `cluster_*` subgraphs. Single-op, cluster-free
branches happen to survive; real modules do not.

## Proven fix (POC measured on googlenet)

For each fan-out point, inject an invisible same-rank ordering chain among its children,
in execution order:

```
{ rank=same; "child1" -> "child2" -> ... -> "childK" [style=invis] }
```

POC results (manipulating the emitted DOT string, then re-running `dot -Tplain`):
- Baseline: 9/9 violations.
- **rank=same invis chains: 0/9 violations.** Canvas bbox 25.5x162 -> 22.3x160
  (slightly MORE compact; no distortion).
- `newrank=true` added: re-broke it (9/9). DO NOT use newrank.
- `constraint=false` weighted flat edges (no rank=same): no effect (9/9).

So the ONLY working recipe is the rank=same invisible chain, WITHOUT newrank.

## Critical correctness guard: independence (empirically validated)

A naive "rank=same over all children" is WRONG and would mangle every residual/skip
network (ResNet, Transformers). Example (validated): residual block where `t = relu(x)`
feeds both `self.a` (a sublayer) AND a downstream `add` (`return sublayer(t) + t`).
Then `t.children = [linear_1_2 (step 2), add_1_5 (step 5)]`, and `add_1_5` is REACHABLE
from `linear_1_2` (it is downstream of the sublayer). These two are SEQUENTIAL, not
parallel; forcing them same-rank would distort badly.

Guard: only order children that are **pairwise mutually unreachable** (true parallel
siblings). Concretely, drop any child T that is reachable, via forward adjacency, from
another child T' in the same sibling set. In the residual case this drops `add_1_5`,
leaving a single child -> no chain emitted -> no distortion. Correct.

## Proposed algorithm (v1 -- THIS IS WHAT YOU ARE ATTACKING)

Run a post-pass AFTER all nodes and forward edges have been added to the graphviz
object, for each relevant forward render mode (unrolled, rolled, focused, collapsed):

```
for each rendered source node S with >= 2 outgoing forward edges:
    raw_children = the same child layer-labels _add_edges_for_node iterates for S
    # Map each child layer-label to its RENDERED DOT node name, reusing the EXACT
    # same remap _add_edges_for_node uses: collapse -> module-box name, skip-chaining,
    # focus boundaries, ":N" -> "passN" suffix. Multiple children may map to the same
    # rendered target (e.g. collapsed into one box).
    targets = distinct rendered node names, each with sort_key = min(step_index) over
              the raw children that map to it
    # Independence guard (forward reachability over the RENDERED topology):
    independent = [T in targets such that no other T' in targets can reach T]
    if len(independent) >= 2:
        order independent ascending by (sort_key, name)   # name = deterministic tiebreak
        emit into the CORRECT (sub)graph scope:
            { rank=same; T_1 -> T_2 -> ... -> T_k [style=invis] }
```

Keep existing `ordering=out`. Do NOT add `newrank`.

## Known attack surface (find MORE; break these and beyond)

1. **rank=same across clusters generalization.** The POC worked on googlenet's specific
   cluster nesting. rank=same with members spread across different `cluster_*` subgraphs
   is documented to be unreliable without `newrank=true` -- but newrank BROKE our POC.
   Why did it work without newrank on googlenet? Does it generalize to other cluster
   nestings (deeply nested modules, siblings in DIFFERENT top-level clusters, a mix of
   clustered and non-clustered siblings)? THIS IS THE #1 RISK.
2. **Reachability scope + cost.** Reachability must be computed over the RENDERED
   topology (post-skip/collapse), not the raw op graph, or guard mismatches. Naive
   per-fan-out BFS is O(V*E) worst case; need memoization. Acceptable for the DOT path
   (only used below ~3500 nodes) but confirm.
3. **Siblings not naturally same-rank.** An "independent" sibling may still be forced to
   a different rank by an EXTRA upstream parent (not a sibling). Forcing rank=same then
   could create a rank conflict -> dot distorts or silently drops the constraint. The
   independence guard does NOT catch this. Is it a real failure? Construct one.
4. **Loops / recurrence (RNNs).** Back-edges can make two siblings mutually reachable ->
   both dropped -> conservative no-op (safe?), or could the BFS loop without a visited
   set? Recurrent/equivalent ops, `:N` passes.
5. **Conditionals (if/then/else).** `conditional_*_children` arms are mutually exclusive
   but laid out in parallel. Do they appear as fan-out children? Will the pass break or
   help them? Should arm order be honored?
6. **Multi-parent / shared descendants / diamonds.** A child reachable from a sibling via
   a LONG path vs being a direct co-sibling. Diamond reconvergence.
7. **Collapsed-source / collapsed-target combinations.** Source collapsed, some children
   collapsed to same box, some not. Dedup correctness.
8. **Determinism.** step_index is unique among compute ops (sequential) but boundary ops
   (input/output/buffer) are step_index 0 -- multiple zeros. Tiebreak.
9. **Does the injected rank=same ever create a rank CYCLE** (A same-rank-as B, but a real
   edge forces A above B)? Independent siblings have no real edges among them -- but is
   that guaranteed after skip/collapse remap?
10. **Backward graph** and **bundle/diff** render paths -- in scope or not? Default: keep
    forward-only; confirm no shared code path silently affected.
11. **Regression**: ~1000 existing layout/aesthetic tests. Could the pass change layouts
    that were previously fine (even if not "wrong")? Acceptable cosmetic drift vs real
    breakage.
12. **Empty/degenerate**: 1 child, 0 children, self-loop, child == source after collapse.

## Efficiency requirement (HARD constraint)

The fix must NOT materially increase render time. Specifics:
- The Graphviz `dot` layout subprocess already dominates render cost. The Python
  post-pass must be small relative to it -- target ~O(V + E) amortized total, NOT
  O(V*E) and NOT a per-fan-out full-graph traversal.
- The independence/reachability guard is the only nontrivial cost. It must be bounded:
  reachability among a fan-out's k children should be a BOUNDED search (e.g. confined to
  the cone between the fan-out and the deepest sibling by `step_index`, since the
  rendered forward graph is a step-ordered DAG), with memoization so shared subpaths are
  not re-walked. No global transitive-closure matrix (O(V^2) memory is unacceptable).
- The injected invisible same-rank edges add only O(sum of sibling-set sizes) edges --
  must not increase the number of RANKS, and must not slow the `dot` subprocess itself.
  Validation must confirm `dot` wall-time does not regress (measure before/after on
  googlenet + a larger model).
- This pass only runs on the DOT path (graphs below the ~3500-node engine threshold), so
  worst-case V is already bounded -- but do not rely on that as an excuse for a sloppy
  super-linear algorithm.

## Validation plan (programmatic, NOT subjective)

- Harness exists: trace model -> `log.draw(vis_save_only=True)` returns DOT source ->
  `dot -Tplain` -> parse `node NAME x y ...` -> for each fan-out, compare exec-order vs
  x-order. Count violations. Also parse `graph SCALE W H` for canvas bbox.
- Gate: 0 violations on (a) toy 5-branch fan-out, (b) googlenet (all 9 inception splits).
- Anti-distortion gate: residual toy + a resnet + a small transformer -> canvas bbox
  within tolerance of baseline AND no NEW violations introduced AND dot exits 0 with no
  new warnings.
- Existing `pytest -m smoke` + tier-2 visualization tests stay green.

## POC reference

`/tmp/poc_fix.py` (the injection experiment), `/tmp/googlenet_diag.py`,
`/tmp/residual_guard.py`, `/tmp/cluster_isolate.py` -- all runnable in the repo
(`/home/jtaylor/projects/torchlens`, env `py311`, graphviz + torchvision installed).
