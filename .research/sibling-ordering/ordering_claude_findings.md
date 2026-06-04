# Adversarial critique: deterministic horizontal sibling ordering (rank=same invis chains)

Reviewer: Claude (Anthropic). Repo: /home/jtaylor/projects/torchlens, env py311.
graphviz dot 7.0.5, torch 2.8.0, torchvision 0.23.0.

Baseline reproduced: `python /tmp/poc_fix.py` -> baseline 9/9 violations, rank=same 0/9,
newrank re-breaks 9/9, flatweight no-op 9/9. So the POC's core claim holds on googlenet.
This critique attacks generalization, the guard, efficiency, scope, and the primitive.

---

## ROOT-CAUSE CORRECTION (the design mis-states the trigger)

Design says mincross breaks "multi-op branches inside clusters". TESTED -- that is NOT
the trigger. Measured (`/tmp/adv_clusters.py`, `/tmp/adv_inception.py`):
- 4 siblings in DIFFERENT top-level clusters, each a 3-op `Sequential` branch:
  baseline **0/1 violations** (no breakage). Clusters + multi-op alone do NOT break.
- 4 EQUAL-length cat branches each in own cluster: baseline **0/1** (no breakage).
- 4 UNEQUAL-length cat branches (inception replica, 1 vs 2 conv ops):
  baseline **2/2 violations**, rank=same -> 0/2. THIS breaks.

=> The real trigger is **unequal branch depth (differing rank counts), feeding a
common merge (cat/add)** -- mincross reorders to reduce crossings when branches have
different lengths. The design's mental model ("clusters + multi-op") is wrong, which
matters because the guard/scope reasoning is built on that wrong model. Not blocking by
itself, but it means the "why did googlenet work without newrank" question is
mis-framed: googlenet worked because its branches happen to nest as they do AND because
the chain members are the per-branch ENTRY ops (all naturally on the first post-split
rank), not because rank=same-across-clusters is robust. See Finding 1.

---
## FINDINGS

### [BLOCKING] F1: rank=same on independent siblings at different natural ranks inflates width up to +58%
Construction `/tmp/adv_offrank.py`: `relu` fans to 3 INDEPENDENT siblings of unequal
depth (1 op / 3 ops / 1 op), merging only at the end. All pairwise mutually unreachable
=> the independence guard KEEPS all of them and emits the chain. Measured:
- baseline canvas W=4.96; sigmoid at y=5.04, cos at y=6.44 (DIFFERENT natural ranks).
- rank=same canvas W=**7.84 (+58%)**; cos dragged from y=6.44 up to y=5.04.
The guard cannot catch this -- the siblings ARE independent; they just don't naturally
share a rank. Forcing rank=same drags the deeper sibling up and stretches the canvas.
This is design item 3, EMPIRICALLY REPRODUCED, and it is the central correctness/quality
hole: the fix trades "wrong L->R order" for "correct order but materially distorted
layout" precisely in the unequal-depth case that is ALSO the case the fix targets
(unequal-depth branches are the trigger -- see root-cause correction). On googlenet the
chain members are branch ENTRY ops which happen to be co-rank, so it looked clean; that
is luck of inception's symmetric first rank, NOT a general property.
FIX DIRECTION: only emit the chain over siblings that ALREADY share a rank in the
baseline layout (requires a pre-layout pass to read ranks, i.e. run dot twice -- doubles
dot cost, see F5), OR restrict to single-op / co-rank siblings, OR abandon rank=same in
favor of a flat-edge ordering primitive that does not force same-rank (tested in F8).

### [BLOCKING] F2: naive child->rendered-name remap silently no-ops on recurrent/`:N` ops
Construction `/tmp/adv_offrank.py`, `/tmp/adv_item3.py`: a sibling that is an
equivalent/recurrent op renders as e.g. `tanh_1_3:1` -> the POC name `tanh_1_3:1pass1`
does NOT match any rendered node (rendered name uses `pass`-join differently). Measured:
that chain member resolves to `None` (absent from `dot -Tplain`), so dot SILENTLY accepts
a chain referencing a non-existent node and the ordering is NOT applied to it -- a no-op
that the validation harness (which only counts violations among nodes it can find) would
score as PASS. The design hand-waves "reuse the EXACT same remap _add_edges_for_node
uses" but that remap (rendering.py:3543-3577) spans collapse_address, atomic-module
parent adjustment, skip-chaining, and `:`->`pass` join order -- re-deriving it in a
second code site is a guaranteed drift point. dot emits NO warning for an edge to an
undeclared node (it auto-creates an invisible orphan), so the failure is invisible.
This is items 4/7/8 collapsing into one silent-failure class.
FIX DIRECTION: the post-pass MUST reuse the actual emitted target names captured DURING
edge emission (record (source_render_name -> [child_render_name,...]) inside
_add_edges_for_node), never re-derive. And dot should be invoked with a check that every
chain member was a declared node (assert membership before injecting).
ESCALATION (worse than no-op): a mangled chain member is not silently dropped -- dot
AUTO-CREATES it as a VISIBLE orphan node. `{ rank=same; b -> ghost [style=invis] }`
makes `ghost` a real lightgrey ellipse (only the EDGE is invis, not the node). Verified
both in raw dot (/tmp/undeclared.dot, node ghost_node rendered) AND end-to-end: OffRank
model rendered node count 10 -> 11, spurious node `"tanh_1_3:1pass1"` injected. So the
remap-drift bug VISIBLY CORRUPTS the graph, it does not merely fail to order.

### [MAJOR] F5: naive reachability guard is quadratic; 1.5s on a 2760-op graph (under threshold)
Construction `/tmp/adv_blowup.py` (StackedDiamonds: D levels x W-wide fan-out, each
reconverging by sum, so every fan-out child's descendant cone is ~the whole remaining
graph). Measured per-fan-out DFS over `children` adjacency with NO memo:
  levels=20 ops=460 -> 50ms;  40 ops=920 -> 191ms;  80 ops=1840 -> 690ms;
  120 ops=2760 -> **1521ms**, descendant-visits 36k -> 1.32M (clean quadratic).
Memoized version: 690-op graph 90ms -> 13.2ms (7x). At 2760 ops (under the 3500 dot
threshold, i.e. IN SCOPE) the naive guard alone (~1.5s) exceeds dot's own layout time.
The design DOES call for memoization + a bounded cone, so this is a HARD IMPLEMENTATION
CONSTRAINT not a design-intent flaw -- but the POC and any straightforward children-DFS
implementation hit it. dot wall-time itself is unaffected by the injected invis edges
(+1.4% googlenet, -1.9% resnet50, +2.8% on the 690-op diamond -- all within noise), so
the ONLY efficiency risk is the Python guard, and it MUST be memoized + cone-bounded.
Note the cone bound "confined between fan-out and deepest sibling by step_index" is
sound for a step-ordered DAG ONLY in forward graphs without back-edges; see F6 (RNN).

### [MINOR/INFO] F6: children adjacency is a forward DAG even for RNN/LSTM/GRU -- cone-bound is SOUND
Construction: manual-RNN, nn.LSTM, nn.GRU, conditional (/tmp probes). The only edges
with child.step_index <= parent.step_index are `final_op -> output_1` where output is
step-COINCIDENT (equal), never a true back-edge. No cycles in children adjacency =>
naive DFS with a visited set terminates (verified: RNN reach() bounded, no cap hit), and
the cone-bound `[fanout_step, max_sibling_step]` cannot miss a sibling-to-sibling path
(reconvergence below all siblings cannot climb back to a sibling in a DAG). This is a
POINT IN THE DESIGN'S FAVOR -- the efficiency cone-bound is correct. BUT the bound must
be INCLUSIVE of equal-step output nodes or it could clip the final merge.

### [MAJOR] F3: collapsed-mode + conditional are pure no-ops with the POC remap, and ghost-inject if mis-mapped
Construction `/tmp` collapsed probe: `collapse_fn=lambda L: True` renders boxes
`m1pass1..m4pass1`; the naive POC chain references branch ENTRY ops
(`linear_1_2pass1` etc.) -- ALL FOUR absent from the rendered graph (False), so naive
injection is a 100% no-op AND injects 4 ghost nodes. With the CORRECT box-name chain
(`m1pass1->m2pass1->...`) there are zero ghosts and order is honored. Independently,
collapsed boxes ALREADY render in-order at baseline (single-rank boxes survive mincross,
same reason single-op branches do) -- so in collapsed mode the pass is best-case a
redundant no-op and worst-case (mis-mapped) a ghost-injector; it buys nothing.
Conditional (/tmp): only the TAKEN arm is traced; arms never appear as simultaneous
fan-out children, so there is nothing to order -- design item 5 is moot in unrolled mode
(safe). Net: the pass's VALUE is confined to unrolled multi-op unequal-depth branches;
in collapsed/rolled/conditional it ranges from no-op to corrupting, depending entirely on
flawless remap reuse (F2).

### [MAJOR] F7: guard over-include => SILENT distortion, dot emits NO error (no safety net)
Construction `/tmp` SeqSib: relu fans to `sigmoid_1_2`(step2) and `add_1_4`(step4) where
`add` IS reachable from `sigmoid` (sequential). Simulating a guard MISS, forcing
`{rank=same; sigmoid -> add [invis]}` while the real forward edge sigmoid->tanh->add
exists: dot exit 0, **stderr empty (no warning)**, canvas width 2.39 -> **3.72 (+56%)**.
So whenever the independence guard or the remap is even slightly wrong (F1/F2/F3), the
failure mode is SILENT layout distortion, never a loud dot error. There is no backstop.
This raises the bar on the guard: it must be provably correct, because nothing downstream
will catch a mistake. Recommend an assertion in the post-pass: for every emitted
same-rank pair, assert no real forward edge connects them in the rendered topology
(cheap: they're already in the reachability structure), and assert all members are
declared nodes (F2). Fail the render loudly in tests rather than ship a silent distortion.

### [INFO] F8: primitive comparison -- design's recipe is best on googlenet, but ALT2 is the safer fallback for off-rank siblings
Measured on googlenet (`/tmp`):
- design (rank=same + invis CHAIN): 0/9, canvas 22.3 x 160.2 (most compact). CONFIRMED.
- rank=same WITHOUT the chain edges (bare same-rank): **6/9** -- proves the invis CHAIN
  edges, not rank=same, are what enforce intra-rank L->R order. rank=same alone is
  insufficient.
- ALT2 (invisible ordering edges, weight=100, constraint=TRUE, NO rank=same): 0/9 but
  canvas 24.6 x **205 (+28% height vs 160)**. Orders correctly without forcing same-rank,
  so it does NOT drag off-rank siblings horizontally (avoids F1's width blow-up) -- at the
  cost of added height. There is a real primitive tradeoff: rank=same compresses width but
  blows up width when siblings are off-rank (F1); constraint=true ordering edges never
  force same-rank but add height. A robust design likely needs BOTH: rank=same+chain ONLY
  for siblings already co-rank in the baseline; constraint=true ordering edges (or nothing)
  otherwise. That requires knowing baseline ranks -> a pre-layout dot pass (F5 cost).
  UPDATE (2-sibling off-rank test, /tmp): with genuinely off-rank siblings (sigmoid step2
  vs cos step4) BOTH primitives distort -- rank=same 4.96->5.66 (+14%), ALT2 4.96->6.37
  (+28%). Neither is safe off-rank. CONCLUSION: the only safe rule is DO NOT order off-rank
  siblings (skip the chain when baseline ranks differ). Confirms F1's fix is mandatory and
  a pre-layout rank read (double dot) is the only fully-correct gate -- OR accept the pass
  fires only for co-rank sibling sets (exactly the googlenet inception-entry case).

### [MAJOR] F10: the only fully-correct fix for F1 (read baseline ranks) DOUBLES dot time -- violates the efficiency hard-constraint
F1 needs to know each sibling's baseline RANK to decide whether forcing same-rank is safe.
Ranks are assigned by dot's network-simplex, not predictable from step_index (mincross
runs first). The only way to read them is to run `dot` once for ranks, then inject, then
run `dot` again. Measured: resnet50 single `dot -Tplain` = 63ms; double = ~127ms (+100%).
The design states the fix "must NOT materially increase render time" and "dot already
dominates render cost" -- a second dot pass is the single largest cost in the whole
render, so the correct-by-construction F1 guard is itself a hard-constraint violation.
This is the real bind: either (a) accept off-rank distortion (F1 unfixed), (b) pay +100%
dot (F10), or (c) restrict the pass to sibling sets that are PROVABLY co-rank without
running dot -- but no such proof exists from the op graph alone. Option (c) collapses the
feature to "order only siblings whose branch sub-DAGs are isomorphic in depth", which is
a much narrower, harder-to-specify feature than the design claims.

### [INFO] F11: #1 risk (cross-top-cluster rank=same) GENERALIZES for co-rank entry ops -- partial defense
Construction `/tmp` CrossClusterUnequal: 4 unequal-depth branches (1op/3op/1op/3op),
each its OWN top-level cluster. baseline 1/1 violation; rank=same -> 0/1, dot exit 0,
NO warning, canvas 12.2->11.6 wide / 16->18 tall (mild). Members are branch ENTRY ops,
all co-rank directly after relu -> rank=same across different top-level clusters works
WITHOUT newrank. Also googlenet: injecting all 9 chains adds ZERO ranks (82->82 distinct
y-levels) and compacts canvas (25.5->22.3). So the design's stated #1 risk is LESS
dangerous than feared: cross-cluster placement per se does not break rank=same; what
breaks is OFF-RANK membership (F1). The two findings are the same coin: as long as the
chained targets are co-rank, cross-cluster is fine and no ranks are added (efficiency
claim holds); the moment a target is off-rank, you get F1 distortion regardless of
clustering. Net: re-scope the design's risk list -- the danger is RANK PARITY of members,
not cluster topology.

### [MINOR] F9: design's "step_index is unique among compute ops" is FALSE; tiebreak is load-bearing
Measured: WithBuffers has step 5 shared by `add_1_3:2` AND `output_1`; googlenet step 197
shared by `linear_1_197:1` AND `output_1`; step 0 shared by input + all buffers (114 on
googlenet). The design's `(sort_key, name)` tiebreak rescues determinism, so this is not a
live bug -- BUT the stated justification ("unique among compute ops, sequential") is wrong,
and an implementer who trusts it and drops the `name` tiebreak gets non-deterministic L->R
order. Keep `name` tiebreak; delete the false uniqueness claim from the spec.

---
## SUMMARY OF FIXES REQUIRED (in priority order)

1. (F2) Capture child rendered-names DURING `_add_edges_for_node` emission; never re-derive
   the collapse/skip/`:N`->pass remap. Assert every chain member is a declared node before
   injecting -- else dot silently injects a VISIBLE ghost node.
2. (F1/F7/F10) Do NOT force rank=same on siblings that are not co-rank. Either (a) restrict
   the pass to sibling sets provably co-rank from graph structure (very narrow: equal-depth
   branch sub-DAGs), or (b) gate on a baseline-rank read (which costs +100% dot -- F10 --
   and so likely must be rejected). Add a hard assertion: no emitted same-rank pair has a
   real forward edge between them in the rendered topology (silent distortion has no dot
   backstop -- F7).
3. (F5) Implement reachability with memoization + step_index cone-bound (sound per F6).
   Naive children-DFS is quadratic: 1.5s on a 2760-op graph under the dot threshold.
4. (F3) In collapsed/rolled mode the pass is a redundant no-op at best and a ghost-injector
   at worst; either prove the remap is exact there or skip those modes entirely.
5. (F9) Keep the `name` tiebreak; delete the false "step_index unique among compute ops"
   claim. step 0 (input+buffers) and the final-op/output step collide routinely.

## WHAT ACTUALLY WORKS (verified, in the design's favor)
- Core POC reproduces: googlenet 9/9 -> 0/9, newrank re-breaks, canvas compacts.
- rank=same across DIFFERENT top-level clusters works without newrank FOR CO-RANK members
  (F11). Adds zero ranks on googlenet (82->82). Efficiency of the dot step is fine
  (+1.4% gn, -1.9% r50, +2.8% on a 690-op diamond -- all noise).
- children adjacency is a forward DAG even for RNN/LSTM/GRU (F6) -> visited-set DFS
  terminates and the cone-bound is sound. No infinite-loop risk.
- Conditionals trace only the taken arm -> no parallel arms to mis-order (safe).

## ROOT MISDIAGNOSIS
The design says mincross breaks "multi-op branches inside clusters." TESTED FALSE. The
trigger is UNEQUAL branch depth feeding a common merge. Clusters + multi-op with EQUAL
depth do NOT break (0 violations). This matters because the guard and risk analysis are
built on the wrong mental model, and the genuinely dangerous axis (rank parity of chained
members) is not the one the design's #1 risk names (cluster topology).

VERDICT: NOT BULLETPROOF -- 2 blocking issues
- BLOCKING F1: forcing rank=same on independent-but-off-rank siblings inflates canvas
  width up to +58% (measured); the guard cannot catch it (they ARE independent); this is
  exactly the unequal-depth case the fix targets. The "proven" 0-distortion result is an
  artifact of googlenet's co-rank inception entry ops.
- BLOCKING F2: the naive child->rendered-name remap silently mismatches recurrent/`:N`,
  collapsed, and skip-chained targets, and dot then AUTO-CREATES the mangled name as a
  VISIBLE ghost node (verified end-to-end, node count 10->11; collapsed mode 4/4 members
  unrendered + 4 ghosts). No dot warning. The design's "reuse the EXACT remap" is the
  whole ballgame and the POC does not do it.
Major: F3 (collapsed/rolled no-op-or-corrupt), F5 (quadratic guard, 1.5s/2760 ops),
F7 (silent distortion, no dot backstop), F10 (correct F1 fix doubles dot time).
Minor: F6/F9/F11 (mostly in design's favor; fix the false uniqueness claim).
