# Design v2: deterministic sibling ordering in TorchLens Graphviz output

Supersedes v1 (`/tmp/ordering_design.md`). v1 was critiqued by a Claude (Opus) and a
Codex adversarial reviewer; both returned NOT BULLETPROOF. Their findings are in
`/tmp/ordering_claude_findings.md` and `/tmp/ordering_codex_findings.md`. This v2 is
built to resolve every blocking + major issue. ATTACK IT AGAIN.

## What changed since v1 (the two big lessons)

1. **Root misdiagnosis corrected.** v1 claimed mincross breaks "multi-op branches inside
   clusters." FALSE (proven). Equal-depth cluster branches render fine. The real
   nondeterminism is among parallel siblings that dot's network-simplex ranker is free
   to permute; the trigger is unequal-depth branches reconverging at a common merge.
2. **No cheap structural predictor exists** for "is `rank=same` safe on these siblings."
   Proven: GoogLeNet's inception entries are NOT co-rank (entries land on 2-3 distinct
   dot ranks, e.g. y=[28.6, 22.7, 22.7, 26.6]) yet `rank=same` fixes them with ZERO
   distortion (canvas shrinks). The adversarial 1/3/1 toy is ALSO non-co-rank but there
   `rank=same` distorts +58%. So co-rank / independence / depth all fail to separate
   safe from unsafe. Only the actual dot layout knows.

## Core approach (decided): measure -> verify -> fallback

Because no cheap predictor exists, we do NOT predict. We inject the ordering constraints,
re-run layout, and KEEP the injected layout ONLY IF it is measurably better and not
distorted; otherwise we fall back to the baseline layout. **This makes the output
provably never worse than today's** -- the literal "won't break anything" guarantee.

## Algorithm v2

Scope: FORWARD graph, UNROLLED (`vis_mode="unrolled"`) only. Rolled and collapsed modes
are skipped (Claude F3: collapsed boxes already render in order; Codex #3: rolled
introduces recurrence cycles). Backward graph and bundle-diff: out of scope. Conditional
branch-point fan-outs: skipped in v1 (Codex #6) -- detected via `conditional_*` metadata.

```
# 0. Cheap gate (O(V)): does the graph have any fan-out (node with >= 2 forward
#    children) in unrolled mode? If NOT -> no-op, ZERO extra cost, render normally.

# 1. Baseline layout (only reached when fan-outs exist):
L0 = dot -Tplain on the baseline DOT          # gives coords + bbox0
violations0 = count_cross_axis_order_violations(L0)   # see "direction" below
if violations0 == 0:
    render final from baseline; DONE            # nothing to fix

# 2. Build candidate ordering chains:
build forward adjacency ONCE from the rendered unrolled topology (a DAG).
for each fan-out source S:
    targets = distinct RENDERED node names of S's children, obtained by REUSING the
              exact RenderEdge / _collapse_address_for_node / _render_node_label remap
              that _add_edges_for_node uses (NOT string "+pass1"). ASSERT each target
              has a real node definition in the graph (no ghost nodes).
    # cheap independence guard (unrolled DAG; memoized descendant sets, cone-bounded by
    # max target step_index): drop any target reachable from another target (sequential
    # children, e.g. residual add). Removes rank-CONFLICT chains up front.
    independent = drop_sequential(targets)
    if len(independent) >= 2:
        order independent by (step_index, name)   # name tiebreak: step_index is NOT
                                                  # unique (boundary ops share 0) -- F9
        record chain (S, independent, LCA_scope(independent))

# 3. Inject chains into the CORRECT scope (Codex #1): each chain is emitted into the
#    deepest module cluster that contains ALL its members (their common module-path
#    prefix); top level only if no common cluster. Emit as:
#       { rank=same; "t1" -> "t2" -> ... [style=invis, <layout-edge marker>] }
#    The marker attribute (e.g. comment="tl:sibling-order" or a custom key) tags these
#    as layout-only, not model edges (Codex #7 -- DOT pollution).

# 4. Verify:
L1 = dot -Tplain on the injected DOT          # bbox1, violations1
accept = (violations1 < violations0) AND (area(bbox1) <= area(bbox0) * (1 + TOL))
         # TOL ~ 0.10. By construction rank=same drives violations1 -> 0.

# 5. Render final from the WINNER (injected if accept else baseline).
```

## How each round-1 blocker is resolved

- **Codex #1 (rank-group scope):** chains emitted at the members' LCA module cluster via
  the deferred cluster builder, never blind top-level append. PLUS verify catches any
  residual scope-induced distortion and falls back.
- **Codex #2 / Claude F1 (off-rank distortion):** not predicted, MEASURED. The verify
  step's bbox-area gate rejects any injection that distorts; we fall back. Bulletproof by
  construction. (The independence guard additionally removes rank-conflict chains early.)
- **Codex #3 (rolled recurrence no-op/cycles):** rolled mode is OUT OF SCOPE; unrolled
  forward is a DAG (Claude F6 confirms), so reachability + cone-bound are well-defined.
- **Codex #4 / Claude F5 (quadratic guard):** adjacency built ONCE; descendant sets
  memoized; reachability cone-bounded by max sibling step_index. Amortized O(V+E). No
  O(V^2) matrix.
- **Codex #5 (leftright/RL):** ordering is along the CROSS-FLOW axis (horizontal for
  TB/BT, vertical for LR/RL). `rank=same` + chain order yields a DETERMINISTIC
  execution-order placement along that axis in all four directions (for LR, first-exec
  sibling is topmost). The violation counter measures the cross-axis coordinate
  (x for TB/BT, y for LR/RL). Documented as "cross-flow-axis execution order," not
  literally "left-to-right." Determinism -- the user's actual goal -- holds in all dirs.
- **Codex #6 (conditional arms):** conditional branch-point fan-outs are SKIPPED in v1
  (detected via `conditional_*_children` metadata). Documented as future work; the pass
  must not mis-order or distort arms (verified by a conditional-model test).
- **Codex #7 (DOT pollution):** invisible edges carry a layout-only marker attribute and
  live in anonymous rank subgraphs; documented as non-model edges for `return_graph=True`
  / `-Tdot` consumers.
- **Codex #8 / Claude F2 (ghost nodes from naive remap):** MANDATORY reuse of the real
  RenderEdge/_collapse_address_for_node/_render_node_label remap + an ASSERT that every
  chain member resolves to an existing rendered node. No `+"pass1"`.
- **Claude F7 (silent distortion has no backstop):** TWO backstops -- (a) the verify
  step (never ship a worse layout), (b) an assertion that no emitted same-rank pair has a
  real forward edge between them (catches a broken independence guard at build time).
- **Claude F9 (step_index not unique):** deterministic `(step_index, name)` ordering key.

## Efficiency (the hard constraint)

- Graphs with NO fan-outs: zero overhead (O(V) gate, then normal single render).
- Graphs WITH fan-outs but already 0 violations: 1 `-Tplain` + 1 final render.
- Graphs with fixable fan-outs: up to baseline `-Tplain` + injected `-Tplain` + 1 final
  render = up to +2 dot layout passes. Measured dot layout cost on real models:
  googlenet 0.069s, resnet50 0.068s, tiny_transformer 0.057s -> +~0.07-0.14s, on the
  (interactive, non-hot) viz path only.
- CAP: above a node threshold (e.g. 2000) where a dot pass approaches ~1s+ (Codex
  measured 2.3s at 5400 nodes), the feature self-disables (the marginal value of
  sibling ordering on a near-unreadable large graph is low; consistent with the
  "render less / dagua for huge graphs" doctrine). Threshold configurable.
- The Python guard is amortized O(V+E) (built-once adjacency + memoized cone-bounded
  reachability); it must stay << the dot pass. Perf test with a shared-cone graph
  (Codex's 5000-source/5000-cone shape: memoized 0.008s vs naive 4.7s).

## Open questions for round-2 critique (find MORE; break these)

1. Is GLOBAL accept/reject too coarse? One subtle-distortion chain among many good ones
   could blow the bbox gate and revert ALL chains (losing the googlenet fix). Does it in
   practice? Real models dilute local distortion in global bbox; toys (1 fan-out = whole
   graph) correctly reject. CONSTRUCT a real-ish model where one bad chain sinks many
   good ones, and decide whether per-cluster granularity is needed.
2. Is bbox-area the right distortion metric? Could injection keep area constant yet make
   the layout materially worse (more crossings, node overlaps, longer edges)? Propose +
   test a better/cheaper proxy (total edge length from -Tplain? crossing count?).
3. Direction: does the cross-axis violation counter + chain order actually produce
   deterministic, execution-ordered placement for LR/RL? MEASURE.
4. LCA-scope emission into the deferred cluster builder: any case where the LCA is
   computed wrong (siblings with partial module-path overlap, atomic-module suffix
   handling per rendering.py:3590) -> chain emitted at wrong scope -> distortion that
   verify then rejects (correct but wasteful) OR ghost?
5. Independence guard in unrolled mode: any residual reachability error (multi-output
   container ops, shared params, buffer source edges) that over-includes (rank conflict,
   caught by verify but wasteful) or over-excludes (silent no-op)?
6. The verify TOL (10%): adversarially pick a model that distorts just under 10% and
   ships a visibly worse graph; or one that improves but trips 10% and is wrongly
   reverted. Tune/justify TOL.
7. Determinism/idempotence: same model + same args -> byte-identical decision every time?
   (step_index ties, dict iteration order, dot nondeterminism across runs.)
8. Any interaction with existing `ordering=out` (keep it) or the existing module-cluster
   construction that the post-pass could corrupt?

## Validation plan (programmatic)

- 0 cross-axis violations after, on: toy 5-branch fan-out + googlenet (all 9 inception
  splits), via `dot -Tplain` coordinate measurement.
- Anti-distortion: residual toy, the 1/3/1 adversarial, resnet50, a small transformer ->
  the FINAL chosen layout has bbox-area within TOL of baseline AND violations not
  increased AND no ghost nodes (rendered node count unchanged vs baseline) AND dot exits
  0 with no new warnings.
- Determinism: render twice, identical chosen-layout decision + identical injected DOT.
- Perf: post-pass time << dot time on googlenet/resnet50; shared-cone perf test.
- Existing `pytest -m smoke` + tier-2 visualization tests green.

## Reference harness / POCs

`/tmp/poc_fix.py`, `/tmp/corank_verify.py` (ranks from -Tplain), `/tmp/adv_offrank.py`
(Claude's distortion construction), `/tmp/googlenet_diag.py`, `/tmp/residual_guard.py`.
Repo `/home/jtaylor/projects/torchlens`, env py311, graphviz + torchvision installed.
