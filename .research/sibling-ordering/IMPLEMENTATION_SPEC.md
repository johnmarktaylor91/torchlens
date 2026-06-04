# Implementation spec: deterministic sibling ordering in Graphviz draw + distance-flood fix

Self-contained build spec. The DESIGN was validated over 3 rounds of dual-lab (Claude +
Codex) adversarial review; see the `ordering_design_v*.md` and `ordering_*_findings_v*.md`
in this directory for the full rationale. Implement EXACTLY the v4 decisions below.

Repo: `/home/jtaylor/projects/torchlens`, env `py311` (`python`), `dot` on PATH,
torchvision installed. Branch off local `main` (do NOT push; merge to LOCAL main when
green). NO version bump. NO AI attribution in commits. Conventional commits, commit
INCREMENTALLY (per phase) so work is never lost. Validation is a TRIPWIRE -- never weaken
a check to pass a test.

## Background

TorchLens renders model graphs via Graphviz. When one tensor fans out to parallel
consumers (e.g. an Inception block's 4 branches), dot's crossing-minimizer places them in
nondeterministic left-to-right order, ignoring `ordering=out`. We add a post-pass that
orders true parallel siblings by execution order, VERIFIED to never produce a worse layout.

## Validated v4 algorithm (forward / unrolled mode only)

Scope: forward graph, `vis_mode="unrolled"` only. Skip rolled, collapsed, backward,
bundle-diff, and conditional branch-point fanouts (detect via `conditional_*_children`).

```
0. Cheap gate O(V): any fanout (node with >=2 forward children)? else no-op, render normally.

1. Candidate chains:
   for each fanout source S with >=2 forward children:
     # rendered names: CAPTURE the names emitted by _add_edges_for_node as it writes into
     # module_cluster_dict[key]["edges"] (do NOT string-build "child+pass1" -> ghost nodes).
     targets = distinct rendered targets of S's children
     # SOLE-PARENT GUARD: keep only targets whose sole rendered parent is S
     #   (uses default-on parent lists; drops residual merges + ancestor-drag children).
     kept = [t for t in targets if rendered_parents(t) == {S}]
     if len(kept) >= 2:
         order kept by (step_index, rendered_name)         # deterministic; step_index NOT unique
         record chain c = (S, kept, lca_recursion_key)     # see LCA below

   if no chains -> render normally.

2. Baseline layout L0 = `dot -Tplain` on baseline DOT.

3. Inject ALL candidate chains, layout once:
   each chain -> { rank=same; "t1" -> "t2" -> ... [style=invis, comment="tl:sibling-order"] }
   emitted at its LCA scope (see below). L1 = `dot -Tplain` on injected DOT.

4. PER-CHAIN verify on local edge-stretch RATIO (metric decided empirically -- see below):
   for each chain c at fanout S:
     incident = baseline real forward edges touching S or any kept child
     ratio(c) = max over incident of  flow_span_L1(edge) / max(eps, flow_span_L0(edge))
     c.bad = ratio(c) > STRETCH_CAP
   # flow_span = |y_tail - y_head| for TB/BT ; |x_tail - x_head| for LR.

5. FIXED-POINT drop loop (bounded): drop bad chains, relayout survivors, recheck; repeat
   until no chain exceeds cap, with a HARD CEILING (max 2 extra layout passes; if still
   over after the ceiling, ship the current survivors -- they are never worse than dropping
   all, and the guard already removed severe distorters).

6. Render the final image from the surviving-chains DOT.

Backstops (asserts): (a) no emitted same-rank pair has a real forward edge between them;
(b) every chain member resolves to an existing rendered node (no ghosts); (c) rendered node
count unchanged vs baseline.
```

### METRIC (the one calibration item -- confirm on real models during build)

Use RATIO ONLY. Do NOT use absolute displacement -- it is REFUTED: GoogLeNet's legit fix
has abs-span-increase 5.82 (LARGER than a real distorter's 4.0), confounded by graph scale.
Measured separation: GoogLeNet chains ratio ~3.94 (KEEP); moderate distorter (L=5) ratio
5.0 (DROP). Set `STRETCH_CAP` so GoogLeNet-class fixes pass and moderate distorters fail.
Starting value **4.5**. DURING THE BUILD, measure per-chain ratio across googlenet,
resnet18/50, a small transformer, and the unequal-depth distorter family, and pick the cap
that keeps all legit real-model fixes and drops the distorters. If a legit real-model fix
exceeds 4.5, RAISE the cap to fit it and report; a wrongly-dropped chain is a safe no-op,
a wrongly-kept one is a MILD nudge (guard removed the severe cases). Document the cap as a
CONSERVATIVE QUALITY THRESHOLD, not a safety proof (the stretch distribution is continuous).

### LCA emission scope

Emit each chain's rank-group into the deepest module cluster containing ALL its members,
keyed by the SAME recursion path `_setup_subgraphs_recurse` uses (NOT raw
`module_addr:call_index`). Compute the LCA over the RENDERED placement key (post-collapse /
post-`_collapse_address_for_node`), not raw `node.modules`. If the LCA is top-level, emit at
top level. ASSERT queued rank-groups == emitted rank-groups (catches a wrong/unvisited key).

### Direction comparator (for the violation notion only; metric is flow-axis)

TB/BT: execution order = increasing x. LR: increasing-exec = decreasing y (first-exec
topmost). No public RL direction exists.

### DROPPED from earlier designs (do NOT implement)

- The distance-based STATIC PRE-FILTER: REFUTED unsafe (equal max_distance_from_input AND
  max_distance_to_output siblings still distort 7x under rank=same, because dot uses
  network-simplex not longest-path). Do not gate verify on distances.
- Absolute-displacement metric: refuted (see METRIC).
- Global bbox-area gate: replaced by per-chain ratio.

## Public API / toggle

Add a visualization option to enable/disable (default ON): e.g. `order_siblings: bool =
True` in the visualization options group, threaded into `draw()`. If you add a PUBLIC name,
you MUST update the glossary (canonical: vault, but at least the in-repo glossary
`.research/glossary_v9_working.md`), `CLAUDE.md`/`AGENTS.md`, and any examples per the
docs-lockstep rule. Also CAP the feature off above a node threshold (~2000) where dot
passes get expensive.

## Part B: distance-flood fix + default-on (decoupled from the feature; JMT-approved)

This is independent of the ordering feature (the pre-filter was dropped). Do it as Phase 1.

1. FIX the superlinear flood in `torchlens/postprocess/graph_traversal.py`
   (`_flood_graph_from_input_or_output_nodes` / `_check_whether_to_add_node_to_flood_stack`
   ~line 430). Current code re-pushes a node on every new min OR max OR ancestor, giving
   O(V^1.5..V^2) on dense-reconverging graphs (measured: densenet201 587ms; 3402-layer
   dense graph = 36% of trace time). Rework to a single topological-order pass: process
   nodes in topological order (forward pass for from-input distances, reverse for
   to-output), each node finalized once from already-finalized predecessors -> true O(V+E).
   VERIFY equivalence: the new distances must EQUAL the old ones on a battery of models
   (googlenet, resnet, a reconverging graph) -- this is a correctness-preserving perf fix,
   NOT a semantics change. Measure densenet201 before/after (expect ~linear now).
2. Flip `torchlens/options.py:631` `compute_input_output_distances: bool = False` -> `True`.
3. Update the `tl.trace` docstring that calls distance computation "expensive" (it's now
   cheap). The glossary already documents `=True` -> this is a conformance fix.
4. Update tests that assert distances are None by default; allow `None` for nodes with no
   input/output reach (verified: ~106/283 resnet50 ops legitimately have None
   max_distance_from_input). Confirm `validation/invariants.py:2112` distance checks PASS
   when on (they run min<=max + input/output-zero asserts; do NOT weaken them).

## Tests (programmatic; this is the real final validation)

New `tests/visualization/test_sibling_ordering.py` (or nearest existing convention):
- googlenet: all 9 inception fanouts render in execution order L->R (parse `dot -Tplain`,
  assert exec-order == x-order); 0 ghost nodes; dot exit 0.
- mixed googlenet + unequal-depth distorter: all 9 googlenet chains kept, distorter dropped.
- residual toy + densenet: feature is a safe no-op (sole-parent guard skips; rendered layout
  == baseline, 0 violations introduced).
- moderate distorter (L=5): dropped by the ratio cap.
- determinism: two renders -> identical chosen DOT / decision.
- LR direction: members placed first-exec-topmost deterministically.
- perf: post-pass time << dot time; flood-fix makes densenet201 distances ~linear (assert
  not 30x the googlenet-per-node rate). Mark slow tests `@pytest.mark.slow` if >5min.

## Quality gates

`ruff check . --fix`; `mypy torchlens/`; `pytest tests/ -m smoke -x`; `pytest tests/ -m
"not slow" -x` for the viz + boundary changes. Inspect a rendered googlenet PDF visually
(the layout should look clean, branches L->R in execution order).

## Constraints recap

No version bump. No push. Merge to LOCAL main when green. No AI attribution. Conventional
commits, per-phase. Never weaken validation. Keep distances semantics identical across the
flood-fix. Surface the final chosen STRETCH_CAP + the real-model ratios you measured.
```
