# OpenAI Codex round-3 critique: ordering design v3

Reviewed:

- `/tmp/ordering_design_v3.md`
- `/tmp/ordering_codex_findings_v2.md`
- `/tmp/ordering_claude_findings_v2.md`
- `torchlens/visualization/rendering.py`
- `torchlens/data_classes/op.py`
- `torchlens/options.py:631`
- `torchlens/postprocess/graph_traversal.py`

Environment:

- Repo: `/home/jtaylor/projects/torchlens`
- Python: active `py311`
- Graphviz: `dot` on PATH
- Scope: design review only; no tracked-file edits. `git status --short` was clean.

## Executive result

v3 closes the round-2 blockers in the important mixed case. Replacing the global
bbox-area gate with per-chain local real-edge stretch prevents one bad chain from either
shipping inside a mostly-good graph or reverting all good chains. The sole-parent rule
also closes the ancestor-drag class that the v2 independence guard could not catch.

I still do **not** consider the design bulletproof because the proposed fixed
`STRETCH_CAP ~ 6-8` ships a small but real unequal-depth distorter. This is not the old
global-gate pathology; it is a calibration/semantic hole in the new machinery. If a
5x local edge stretch and four-rank displacement is acceptable by definition, then the
cap is an explicit quality tradeoff, not a proof of "never worse than today's layout."

## 1. Mixed GoogLeNet + unequal-depth distorter: v3 succeeds

I built the requested mixed graph from a real TorchLens GoogLeNet DOT plus the same
unequal-depth distorter family used in round 2:

```text
BP -> BS
BS -> BA
BS -> BB
BA -> BZA
BB -> BZB
BP -> BQ0 -> ... -> BQ31 -> BB
```

Then I simulated the v3 loop:

1. Baseline `dot -Tplain`.
2. Inject all 9 real GoogLeNet chains plus the bad chain.
3. One injected `dot -Tplain`.
4. Compute per-chain local flow-axis edge stretch over real edges incident to `S` and
   its kept children.
5. Drop chains over cap and relayout survivors.

Measured:

```text
chains: 10 total = 9 GoogLeNet + 1 distorter
baseline violations: 9
all-injected violations: 0

top stretches:
BAD_DISTORTER       26.271x on BP -> BS
cat_6_132:1          3.939x
maxpool2d_2_11:1     3.931x
maxpool2d_5_52:1     3.931x
cat_1_31:1           3.931x
cat_4_92:1           3.931x
cat_3_72:1           3.931x
cat_8_173:1          3.925x
maxpool2d_11_153:1   3.925x
cat_5_112:1          3.921x

cap=8.0 survivors: 9
dropped: BAD_DISTORTER
final survivor violations: 0
final survivor max stretch: 3.939x
```

This directly resolves Codex B1/B2 and Claude B1/B2/B4 from round 2 for the mixed case.
There was no global-gate pathology: all 9 GoogLeNet chains survived and the bad chain was
dropped.

## 2. Stretch neighborhood: no counterexample found, but keep the metric explicit

I attempted to construct a graph where an injected chain distorts a real edge outside
the local neighborhood while all incident edges for `S`/children stay under cap. The
directed construction I tried was the plausible evasion shape:

```text
R -> P -> S -> A/B
```

with side chains intended to make `P`, `S`, `A`, and `B` translate together while
stretching `R -> P` two hops away. I did not find a hit. In the harmful unequal-depth
families, the damage appears on an incident edge first, typically the incoming edge into
the fanout source.

This supports the proposed neighborhood for the known bad class. It is not a formal
proof over Graphviz's network simplex, so implementation should keep the post-drop
survivor stretch assertion and report/drop rather than silently render if the assertion
fails.

## 3. Blocking: `STRETCH_CAP ~ 6-8` accepts a real distorter

The same unequal-depth family can be tuned below the proposed cap while still visibly
moving a branch several ranks.

Measured isolated fixture:

```text
P -> S
S -> A
S -> B
A -> ZA
B -> ZB
P -> Q0 -> ... -> Q4 -> B
{ rank=same; A -> B [style=invis] }
```

For depth `L=5`:

```text
max local stretch: 5.00x on P -> S
S displacement:    4.03 ranks
A displacement:    4.00 ranks
P -> S span:       1.00 -> 5.00
```

For depth `L=6`:

```text
max local stretch: 6.00x
S displacement:    5.03 ranks
A displacement:    5.00 ranks
```

Because the design says `bad = stretch > STRETCH_CAP`, a cap of 6 keeps the 5x case, and
a cap of 8 keeps both 5x and 6x cases. That contradicts the v3 framing that cap 6-8
cleanly separates legitimate GoogLeNet fixes (~3.94x) from distorters. It separates the
large `L=32` adversary, but not the same adversary at moderate depth.

Required hardening:

- Either define this bounded distortion as acceptable and document the cap as a quality
  tolerance, not a safety proof.
- Or lower/tune the cap. The measured GoogLeNet max is ~3.94x, so a cap near 4.5-5.0 is
  the first plausible range, but that needs calibration on more real models.
- Prefer a normalized metric that accounts for absolute rank displacement as well as
  ratio. A 5x stretch on a one-rank edge is still a four-rank layout change.

This is the only blocking issue I found.

## 4. Drop-then-relayout monotonicity: mixed case passes; one-pass proof still absent

In the mixed GoogLeNet + `L=32` distorter case, dropping the bad chain and relaying out
survivors kept all survivor stretches under cap:

```text
survivor max final stretch: 3.939x
final survivor violations: 0
```

I did not construct an oscillation or a case where dropping one bad chain makes a
previously-good survivor newly bad. However, Graphviz layout is not monotone in the
mathematical sense needed by the design statement "dropping bad chains only relaxes."
The safe bounded algorithm is:

```text
repeat: layout current survivors -> drop over-cap chains
until no drops
```

Since chains are only removed, this terminates in at most `N + 1` layout passes for `N`
candidate chains. The v3 claim of "at most one extra pass" is empirically true for the
mixed case I measured, but not proven by the design.

## 5. Sole-parent rule: conservative and acceptable for common architectures

Corrected parent-label measurement:

```text
GoogLeNet:
  ops=313, fanouts=9, kept_groups=9, skipped_groups=0, children_kept=36/36

ResNet-50:
  ops=283, fanouts=16, kept_groups=4, skipped_groups=12, children_kept=20/32
  skipped examples are residual merges such as iadd with parents
  [batchnorm_N, relu_N].

TransformerEncoderLayer:
  ops=38, fanouts=3, kept_groups=1, skipped_groups=2, children_kept=5/7
  skipped examples are residual add merges.
```

This confirms the rule does not gut the target Inception/GoogLeNet case. It does skip
true rendered children that have a legitimate second rendered parent, especially residual
merge nodes. That is an acceptable conservative no-op: those are exactly the nodes where
forcing same-rank is most likely to drag an ancestor or merge edge.

Implementation note: compare raw labels to raw labels. My first count compared
pass-qualified source labels to raw child parent labels and incorrectly reported
0 GoogLeNet groups kept.

## 6. Distance default: cost and invariants look fine; loaded invariant issue is preexisting

Current code still has:

```text
CaptureOptions.compute_input_output_distances = False
```

I measured ResNet-50 traces with the flag off/on:

```text
flag=False trace_s=1.196 ops=283 none_max_from_input=283
flag=True  trace_s=0.894 ops=283 none_max_from_input=106 invariants=True
flag=False trace_s=0.733 ops=283 none_max_from_input=283
flag=True  trace_s=0.796 ops=283 none_max_from_input=106 invariants=True
```

The on/off timing was within noise after warmup. `check_metadata_invariants` passed with
distance fields populated.

Nuances:

- Even with distance marking on, not every op gets numeric distances. In ResNet-50,
  106/283 ops still had `max_distance_from_input is None`, apparently for nodes without
  an input ancestor under TorchLens's current metadata semantics. Tests should not assert
  every op is numeric.
- Portable save/load persisted distance fields correctly in a small model.
- `check_metadata_invariants(loaded_trace)` failed after portable load with
  `AttributeError: 'NoneType' object has no attribute 'keys'` at
  `out_versions_by_child`, but the same failure occurs with distances disabled. This is
  not a new downside of the default change.
- `user_funcs.trace` documentation still calls distance computation "expensive"; if the
  default changes, that docstring should be updated.

## 7. Static pre-filter: no empirical counterexample found, but do not make it load-bearing

I attempted a random DAG search constrained so `A` and `B` have equal
`max_distance_from_input` and equal `max_distance_to_output`, with `A`/`B` sole-parent
children of `S`, then injected `{ rank=same; A -> B }`. I did not find a concrete bad
layout in the time budget.

Still, equal max-distance fields are scalar graph summaries, not a Graphviz co-rank
proof. They ignore:

- min distances,
- module cluster placement,
- rendered-edge filtering/collapse,
- cross-axis effects from the invisible ordering edge,
- node sizes and labels,
- Graphviz slack choices.

Recommendation: keep the pre-filter as an optimization only when it is paired with a
cheap rendered-layout check, e.g. "siblings are already co-rank in baseline `L0`" or
"equal distances plus no baseline rank spread." If the pre-filter skips all dot-verify
round-trips solely on equal max distances, it remains an unproven heuristic.

Because I did not construct the requested equal-distance bad case, I am not counting this
as a blocking issue.

## 8. Recursion-path LCA keying and determinism

The code confirms why v3's recursion-path keying is the right implementation contract:
`_setup_subgraphs_recurse` emits subgraph contents only for keys reached by its
`parent_graph_list` traversal, and cluster names can collide if keyed only by rendered
cluster labels. Keying queued rank groups by the same recursion path used by
`_setup_subgraphs_recurse`, then asserting every queued group is emitted, closes the
round-2 LCA/keying objection on paper.

Determinism check:

```text
5 repeated dot -Tplain runs on the same injected graph:
unique sha256 hashes = 1
```

The per-chain decision should be deterministic if implementation preserves ordered
iteration over `entries_to_plot`/rendered edges and uses `(step_index, rendered_name)` as
the chain order key rather than iterating through sets.

## Round-2 blocker status

- Global bbox-area gate ships/reverts wrong chains: resolved by per-chain edge stretch.
- Global accept/reject granularity: resolved in the mixed case; per-chain survivors work.
- Bbox-area as wrong metric: resolved; no bbox gate remains.
- Ancestor-drag missed by independence guard: resolved by sole-parent guard for the
  direct multi-parent class; unequal-depth sole-parent residue is handled by stretch.
- LCA cluster key collision: resolved in design by recursion-path keying plus queued-vs-
  emitted assert.
- Ghost rendered names: still implementation-gated, but v3's "capture emitted names at
  edge emission" is the correct contract.

## Concerns to carry into implementation

- The cap value is now the load-bearing policy. Do not present `6-8` as proven safe.
- If final survivor verification fails after a drop, the implementation needs a defined
  response: iterate drops to a fixed point or fall back all remaining chains. An assert
  without recovery is not a user-facing algorithm.
- Static pre-filter should not skip verification solely from equal max distances unless a
  stronger co-rank/layout condition is added.
- Distance-default tests must allow `None` distances for nodes without input/output reach
  under current semantics.

VERDICT: NOT BULLETPROOF -- 1 blocking issue
