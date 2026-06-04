# Design v3: deterministic sibling ordering in TorchLens Graphviz output

Supersedes v2 (`/tmp/ordering_design_v2.md`). v2 was critiqued by Claude + Codex (round 2,
findings in `/tmp/ordering_claude_findings_v2.md`, `/tmp/ordering_codex_findings_v2.md`).
BOTH returned NOT BULLETPROOF and CONVERGED on the same root cause + fix. v3 implements
that convergence. ATTACK IT AGAIN.

## The round-2 convergence (why v2 failed, what both labs agreed on)

- **Root cause:** v2's `accept = violations down AND global bbox-area within TOL` is a
  GLOBAL SCALAR gate, and a global scalar cannot adjudicate LOCAL per-chain distortion.
  - Codex: a bad chain gets ACCEPTED because good chains dominate the global counters.
  - Claude: good chains get REVERTED because one bad chain blows global area (measured
    5-11 fixable chains lost over one distorter; distortion is on the cross-flow WIDTH
    axis so it does NOT dilute).
- **bbox-area is the wrong metric** (both): false-rejects layouts that improve on
  crossings/edge-length (Claude: area 1.16 reject but crossings 1->0, edge-len 1.007);
  misses local distortion area can't see (Codex: branch dragged 32-64 ranks, area shrinks).
- **The fix (both):** per-chain (or per-LCA-cluster) verify on an EDGE metric, not a
  global bbox gate. Codex showed the discriminating metric is LOCAL real-edge stretch
  (GoogLeNet worst ~3.9x; adversarial distorter ~32x -> a cap ~6-8x separates cleanly).
- **Verified clean by both:** ghost-node remap is implementation-gated not a hole;
  quadratic guard resolved by memo/cone-bound; multi-output `chunk`, buffer-source,
  clean diamonds, the assert backstop, determinism, `ordering=out` interaction;
  GoogLeNet accept still fires (9/9->0/9, byte-identical across runs). NO public RL
  direction exists (drop that worry); LR places members deterministically with axis=y,
  first-exec topmost.

## What changed from v2

1. **Independence/ancestor-drag guard replaced by the sole-parent rule** (empirically
   validated): keep a sibling for ordering ONLY IF its sole RENDERED parent is the fanout
   source. Uses DEFAULT-ON parent lists, O(1), no traversal, no distance fields.
   - Proven: GoogLeNet inception entries each have exactly 1 parent (the split) -> KEEP.
     Residual `add` has 2 parents (branch-out + source) -> SKIP. Codex-#3 ancestor-drag
     child has an extra parent -> SKIP. (This is what Claude B5 said the old guard
     "structurally cannot catch" -- the sole-parent rule catches it for free.)
   - It does NOT catch unequal-depth reconvergence among sole-parent siblings (e.g. the
     1/3/1 toy distorts, GoogLeNet does not, both sole-parent). That residue is exactly
     what the per-chain verify is for.
2. **Per-chain local edge-stretch verify replaces the global bbox-area gate.**
3. **LCA emission keyed by the recursion path, not `module_addr:call_index`** (Claude B3:
   sibling clusters `b1/b2/b3` all render as `cluster_parent_pass1` -> collision).
4. **No mandatory new capture-time state.** Guards use default-on parent lists. The
   distortion metric is computed at draw time from the `-Tplain` passes we already run.
   Numeric `*_distance_*` fields are OFF by default (verified None on a default trace);
   if ever needed they are computed on-demand at draw, never forced into capture.

## Distance fields: ON by default (sprint change) + static pre-filter

SPRINT CHANGE (JMT-requested, also a conformance fix): set the capture-config default
`compute_input_output_distances` from `False` (options.py:631) to `True`, so the numeric
`min/max_distance_from_input` / `min/max_distance_to_output` fields are populated by
default. Rationale: one O(V+E) flood-fill is negligible vs the forward pass; the glossary
`tl.trace` signature ALREADY documents `=True`, so the code is currently violating the
locked spec (code-is-the-bug). Escape hatch (`=False`) stays for pathological graphs.
To handle: update tests asserting None-by-default; confirm `invariants.py:2112` distance
checks pass when on; serialized traces grow 4 ints/node (negligible).

USE IN THE ALGO (efficiency, optional but valuable): a STATIC PRE-FILTER. A fanout whose
kept siblings ALL share equal `max_distance_from_input` AND equal `max_distance_to_output`
is structurally symmetric -> very likely co-rank-safe -> order it WITHOUT the dot-verify
round-trips. Only ASYMMETRIC fanouts (unequal sibling depth, the genuine distortion risk)
need the per-chain stretch verify. For a fully-symmetric model this eliminates the extra
dot passes entirely. NOTE: the guards (sole-parent) and the verify (dot layout) remain the
load-bearing correctness mechanisms; distances are a cheap accelerator, not a substitute.
Round-3 must confirm "equal from-input AND to-output distance => safe to order without
verify" is actually sound, not merely plausible.

## Algorithm v3

Scope: FORWARD graph, UNROLLED mode only. Skip rolled/collapsed (no-op or out-of-order
already handled), backward, bundle-diff, and conditional branch-point fanouts (detected
via `conditional_*_children`).

```
# 0. Cheap gate (O(V)): any fanout (node with >= 2 forward children)? If not -> no-op,
#    ZERO extra cost, render normally.

# 1. Build candidate chains (no dot yet):
for each fanout source S with >= 2 forward children:
    # map children to RENDERED node names by CAPTURING the names emitted by
    # _add_edges_for_node as it writes into module_cluster_dict[key]["edges"]
    # (both labs confirmed this is where real rendered names live). Assert membership.
    targets = distinct rendered targets of S's children
    # SOLE-PARENT GUARD (default-on parent lists): keep only targets whose sole rendered
    # parent is S. Drops residual/merge children and ancestor-drag children.
    kept = [t for t in targets if rendered_parents(t) == {S}]
    if len(kept) >= 2:
        order kept by (step_index, rendered_name)        # deterministic
        record chain c = (S, kept, lca_recursion_key(kept))

if no chains -> render normally.

# 2. Baseline layout:
L0 = dot -Tplain on baseline DOT      # node coords + real-edge geometry

# 3. Inject ALL candidate chains and lay out ONCE:
#    each chain emitted as { rank=same; t1 -> t2 -> ... [style=invis, tl_layout_edge] }
#    at its LCA recursion-path cluster scope.
L1 = dot -Tplain on injected DOT

# 4. PER-CHAIN verify on LOCAL edge stretch (the convergent fix):
for each chain c at fanout S:
    # local real edges = forward edges incident to S and to each kept child
    stretch(c) = max over those real edges of
                 flow_axis_span_in_L1(edge) / max(eps, flow_axis_span_in_L0(edge))
    c.bad = stretch(c) > STRETCH_CAP            # CAP ~ 6-8 (3.9x good vs 32x bad)
survivors = [c for c in chains if not c.bad]

# 5. If survivors == all chains: L1 is final layout -> render image from L1's DOT.
#    Else: rebuild DOT with ONLY survivor chains, dot -Tplain again (L2), confirm no
#    survivor now exceeds cap (dropping bad chains only relaxes), render image from L2.
#    (>= 1 candidate distorter -> at most one extra pass. Bounded, independent of #chains.)

# 6. Backstops: assert no emitted same-rank pair has a real forward edge between them;
#    assert rendered count unchanged vs baseline (no ghosts).
```

Direction comparator for "in execution order": TB/BT -> increasing x; LR -> decreasing y
(first-exec topmost). Used both to define a "violation" and is irrelevant to the stretch
metric (stretch is on the flow axis).

## Efficiency

- No fanouts -> zero overhead.
- Fanouts present -> baseline `-Tplain` + injected `-Tplain` + final render; +1 more
  `-Tplain` ONLY if some chain was dropped. So 2-3 dot layout passes, INDEPENDENT of the
  number of chains (per-chain verify is measured on ONE injected layout, not N layouts).
- Measured dot layout cost: googlenet 0.069s, resnet50 0.068s, tiny_transformer 0.057s
  -> ~0.07-0.21s added on the interactive draw path only.
- Guards are O(1) per child on default-on parent lists. No capture-time cost.
- CAP the whole feature above a node threshold (~2000) where a dot pass approaches ~1s.

## How each round-2 blocker is resolved

- **Codex B1 / Claude B2 (bbox-area wrong metric):** replaced by per-chain LOCAL
  edge-stretch. Codex's own numbers separate good (3.9x) from bad (32x).
- **Codex B2 / Claude B1 + B4 (global granularity, brittle TOL):** per-chain accept/reject
  -- each chain kept or dropped on its OWN local stretch; one distorter cannot revert good
  chains (Claude B1) nor ride in on good ones (Codex B2). No global TOL knife-edge (B4).
- **Codex B3 / Claude B5 (independence misses ancestor-drag):** sole-parent rule (default
  parent lists) drops any child with an extra parent. Proven.
- **Codex Major 4 / Claude B3 (LCA emitted-key contract + cluster name collision):** key
  rank_groups by the RECURSION PATH used in `_setup_subgraphs_recurse`, not raw
  `module_addr:call_index`; capture rendered names at emission; assert queued == emitted.
  (Per-chain verify also catches any residual wrong-scope distortion -> drop that chain.)
- **Codex Major 5 (LR/RL comparator):** TB/BT x-asc, LR y-desc; no public RL.
- **Codex Major 6 (multi-output/buffer/param):** sole-parent rule + build candidates from
  the EXACT rendered forward edges after buffer-visibility/skip filtering; skip targets
  with `in_multi_output` unless a proven need; dedup buffer-source vs input-source.

## Open questions for round-3 critique (final pass -- break these)

1. Does per-chain local edge-stretch ACTUALLY keep GoogLeNet (all 9) AND drop an
   F1-style distorter, in a MIXED graph? (Codex's isolated numbers say yes; verify the
   mixed case end-to-end with the real harness.)
2. Is "local real edges incident to S and its children" the right stretch neighborhood,
   or can a chain distort an edge OUTSIDE that neighborhood (e.g. two ranks away) that
   the per-chain metric misses? Construct it.
3. STRETCH_CAP calibration: a real model whose legitimate fix exceeds 8x, or a distorter
   under 6x. Does a fixed cap hold across model families, or does it need normalization?
4. The "drop bad chains -> re-layout -> survivors still pass" assumption: can dropping a
   bad chain make a previously-good chain newly bad (non-monotonic)? If so, is one extra
   pass enough or can it oscillate? Bound it.
5. Recursion-path LCA keying: any case where the kept siblings' LCA recursion key is
   ambiguous or not visited by `_setup_subgraphs_recurse`?
6. Sole-parent rule: any rendered fanout where a TRUE parallel sibling legitimately has a
   second rendered parent (so it's wrongly skipped -> silent no-op), and is that
   acceptable? (Conservative: yes, but confirm it doesn't gut common architectures.)
7. Determinism of the per-chain decision across dot runs (stretch ratios near the cap).
8. DISTANCE-DEFAULT change: any hidden downside to `compute_input_output_distances=True`
   by default beyond the obvious (large-graph BFS cost, invariants.py:2112 checks,
   serialization size)? MEASURE the flood-fill cost on a big model. Any save/load or
   determinism impact?
9. STATIC PRE-FILTER soundness: is "kept siblings share equal `max_distance_from_input`
   AND equal `max_distance_to_output` => safe to order without dot-verify" actually
   sound? CONSTRUCT a counterexample where siblings have equal both-distances yet
   `rank=same` distorts (so the pre-filter would wrongly skip verification and ship a bad
   layout). If found, the pre-filter must be demoted to "skip only when ALSO co-rank in a
   cheap structural sense," or dropped.

## Validation plan (programmatic)

- Mixed model with clean fanouts (googlenet-like) + an unequal-depth distorter: assert
  survivors fix all clean chains (0 violations) AND drop the distorter (its real edges
  within baseline stretch) AND final bbox not materially worse than baseline.
- googlenet (all 9), resnet50, a small transformer, residual toy, 1/3/1 toy, multi-output
  `chunk`, buffer model: 0 ghosts, dot exit 0 no new warnings, deterministic decision
  across 2 renders, post-pass time << dot time.
- Existing `pytest -m smoke` + tier-2 visualization tests green.

## Harnesses

`/tmp/v2_harness.py` (Claude's faithful v2 loop), `/tmp/poc_fix.py`, `/tmp/soleparent.py`
(sole-parent rule), `/tmp/corank_verify.py`, `/tmp/verify_googlenet.py`. Codex's mixed
googlenet+distorter fixture in `/tmp/ordering_codex_findings_v2.md`. Repo
`/home/jtaylor/projects/torchlens`, env py311, graphviz + torchvision installed.
