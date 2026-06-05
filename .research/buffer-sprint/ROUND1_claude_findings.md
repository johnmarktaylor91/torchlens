# Buffer-Refactor SPEC — Adversarial Findings (Claude / Anthropic)

Design review of `.research/buffer-sprint/SPEC.md`. All probes run under env `py311`,
torch 2.8.0, against current `main`. No tracked files modified; scratch in `/tmp`.

The spec's load-bearing claim is: "sprint is largely ADDITIVE … the write-detection,
one-node passthrough, and main-graph placement EXIST … does NOT build capture from
scratch." I empirically tested that premise. **It is false for the cases that motivate the
spec.** Current capture produces exactly ONE buffer node = version 1 = the INITIAL value
for every realistic overwrite mechanism; the multi-version / dual-label / value-history
model the spec sells has essentially no input data to compute from, and the one case that
DOES produce multiple version-nodes contradicts the spec's structural claims.

---

## BLOCKING

### B1. The flagship case (BatchNorm train: read + overwrite running stats) captures ZERO overwrites — one node, the pre-update value, num_overwrites=0
SEVERITY: blocking.
Construction: `nn.BatchNorm1d(4).train()`, `tl.trace(bn, randn(8,4))` (`/tmp/probe2_bn.py`).
Measured:
```
buffers:  addr=running_mean pass=1 is_overwritten=False num_overwrites=0 out=[0,0,0]
          addr=running_var  pass=1 is_overwritten=False num_overwrites=0 out=[1,1,1]
buffer node count per address: {'running_mean': 1, 'running_var': 1}
```
The running-stat UPDATE happens inside the fused `batch_norm` ATen kernel (the project's
own anti-pattern: "fused kernels won't expose hidden internal tensors"). The post-update
`running_mean` is NEVER re-read as a buffer, so no version-2 node, no `value_after`, no
overwrite. The spec's headline validation test ("BatchNorm in train — read + overwrite
running stats — versions, value_at/after") **cannot pass against current capture**: there is
nothing to version. The spec assumes a capability TorchLens does not have for its #1 example.
Fix: either (a) descope BatchNorm/fused buffers entirely and say so loudly, OR (b) add real
capture work (hook the module to snapshot pre/post buffer state) — which contradicts the
spec's "does NOT build capture from scratch" scoping. Pick one in the spec; right now the
spec promises (a)'s simplicity with (b)'s capability.

### B2. In-place buffer mutation (`running_mean.mul_(0.9)`, `copy_`) is NOT captured as read-vN + write-v(N+1). There is no v(N+1) node; the mutated value becomes a plain forward tensor with zero buffer linkage
SEVERITY: blocking. This is the spec's explicitly flagged risk #1, and it fails.
Construction: `self.decay.mul_(0.9); return x*self.decay` (`/tmp/probe1b_inplace.py`).
Measured edges:
```
buffer_1   parents=[]            children=['mul_1']    out=[1,1,1,1]   # version 1, ORIGINAL value
mul_1      parents=['buffer_1']  children=['mul_2']    out=[0.9,...]   # plain compute op, is_buffer=False
mul_2      parents=['input_1','mul_1'] children=['output_1']          # reads mul_1's tensor, NOT a buffer node
```
Only ONE buffer node exists (v1 = pre-mutation 1.0). The `mul_` mutator (`/tmp/probe14`):
`is_buffer=False, buffer_source=None, address=None, is_inplace=False`, no attribute links it
to the buffer it mutated. The spec's model — "the COMPUTE op (`mul_`) reading version N
(in-edge) and writing version N+1 (out-edge to the new buffer node); `is_in_place` flags that
mutator" — has NO capture support: no v(N+1) node is created, and the mutator op carries no
buffer identity. `copy_` across a recurrent cell is identical (`/tmp/probe5`): 1 node, v1
zeros, `num_overwrites=0`. The spec's whole in-place story and the `is_in_place`/`value_after`
fields are unbuildable additively. (Bonus: `is_in_place` near-collides with the EXISTING
`is_inplace` field on Op — a naming trap.)
Fix: the spec must either drop the in-place-versioning claim or budget genuine capture work to
intercept in-place buffer writes (new, not additive). The "VERIFY this matches capture" note
in the spec is doing load-bearing work it should not — it IS the blocker, verified negative.

### B3. The dual-label worked example (two loops -> two buffer-Layers, op-label resets, global version continuous) is fictional: all recurrent overwrites of one address collapse into ONE Layer, and op-label == address-version in every case
SEVERITY: blocking (the dual-label is the spec's central NEW affordance).
Construction: buffer reassigned in TWO separate `for` loops, 3 passes each (`/tmp/probe8`).
Spec's worked example wants `buffer_1:1,2,3` (Layer 1) then `buffer_2:1,2,3` (Layer 2), global
`running_mean:1..6`. Measured:
```
buffer_1..buffer_6  ALL layer_label=buffer_1   buffer_pass=1..6
Layers containing buffer nodes: {buffer_1: num_passes=6}   # ONE Layer
```
There is ONE buffer-Layer with 6 passes; op-label-pass (1..6) is IDENTICAL to the global
address-version (1..6). The two-Layer split the dual-label exists to disambiguate **never
happens** for buffers. Contrast a real module across two loops (`/tmp/probe9`): `relu` DOES
split into TWO Layers (`relu_1_2:1,2,3` + `relu_2_4:1,2,3`). So the "ModuleCall-parallel"
claim the spec leans on is FALSE for buffers — ops/modules split per loop, buffers do not,
because buffer `equivalence_class = "buffer_<address>"` (sources.py:258) keys purely on
address and ignores loop position. The mechanism that would produce divergent op-/address-
labels (loop-position-sensitive grouping) is exactly the one buffers lack.
Net: in 100% of constructed cases op-label and address-version coincide, so the dual-label
carries zero information and its motivating example cannot be produced by current loop
detection. Either the spec must change buffer equivalence to be loop-position-sensitive (a
loop-detection change with cycle/regression risk, NOT additive and NOT in scope), or admit
the dual-label collapses to a single label and drop it.

### B4. `log.buffers` accessor sees only 1 of N version-nodes and reports num_overwrites=0 / is_overwritten=False even when N real version nodes exist — `versions`/`num_versions`/`is_static` are NOT computable from current machinery
SEVERITY: blocking (the entity's core lifecycle API has no data source).
Construction: recurrent reassign cell, 3 calls (`/tmp/probe11_accessor_bug.py`).
Measured:
```
len(log.buffers): 1
  accessor entry: addr=cell.s buffer_pass=1 is_overwritten=False num_overwrites=0
ACTUAL version nodes in graph: 3 -> [buffer_1, buffer_2, buffer_3]
```
The accessor is keyed by address and retains only the first node; the version-2/3 nodes
(which carry `buffer_source`, the real overwrite signal) are invisible to it. `is_overwritten`
and `num_overwrites` (buffer.py:132-146) walk `trace.buffers` and therefore see one record and
return 0 — WRONG by construction. So the spec's `Buffer.versions` (flat execution-ordered
version-node Ops), `num_versions`, `is_static`, `value_at`/`value_after`, and the
`address:N == versions[N-1]` lookup all need a new entity->version-node index built from the
raw graph, NOT from the existing accessor. This is a non-trivial new postprocess pass, not
"additive over what exists." The spec's "What's already built" section materially
overstates reuse.

---

## MAJOR

### M1. One-node-per-version + "save all includes buffers" => linear node and tensor-copy blowup for deep recurrent buffers, with no rolling
SEVERITY: major.
Construction: N-step recurrent buffer overwrite (`/tmp/probe13_blowup.py`).
Measured: n=50 -> 50 version nodes; n=200 -> 200 version nodes (linear). A 1000-step RNN with
a stateful buffer => 1000 version nodes, and under the spec's `layers_to_save="all" INCLUDES
buffers` rule each saves a full tensor copy => 1000x the buffer's memory. The spec only
"NAMEs the rolled-view multi-axis case" and defers the answer. For buffers specifically the
default save path is now a memory footgun (regular ops in a 1000-step loop are already N
nodes, but buffers ADD a parallel N-node chain). Spec should (a) make buffer version-node
saving opt-in OR rolled by default, and (b) state the node-count contract explicitly, not
defer it.

### M2. Value-based dedup in `_fix_buffer_layers` conflicts with "each overwrite = a new version"
SEVERITY: major.
Pointer: `control_flow.py:678-693` merges buffer nodes whose hash
(`str(modules)+str(buffer_source)+address`) matches AND `torch.equal(out)` holds. For the
static-read case this is correct (one node, many children — spec-desired). But the spec
declares "Each overwrite = a new version-node" as an invariant. A genuine overwrite that
happens to restore a previously-seen value with a matching source hash WOULD be merged,
silently dropping a version and breaking `versions[N-1] == trace["addr:N"]` (off-by-k after a
merge). In my probes distinct `buffer_source` per pass kept hashes distinct so no collapse was
observed (`/tmp/probe12`), but the dedup is value-sensitive and the spec's version-count
invariant is not. Spec must specify: does versioning happen BEFORE or AFTER dedup, and is
dedup disabled for overwritten (sourced) buffers? Undefined today.

### M3. `value_at` / `value_before` / `value_after` are underspecified AND mostly uncomputable
SEVERITY: major.
The spec defines `value_at(op_label)` = "value as SEEN by that op (before it ran)" and
`value_after` for in-place. But (a) for the dominant single-node case there is exactly one
value, so "before/after/at" all collapse and the API is decorative; (b) for the in-place case
that would make them meaningful, B2 shows there is no v(N+1) node and no mutator->buffer link,
so `value_after` has nothing to read; (c) for BatchNorm (B1) the post-value is never captured.
So `value_after` is well-defined only in a case (in-place versioning) the system cannot
capture. Spec hand-waves the exact tie-break ("collapses to the same for pure reads") without
acknowledging that the non-collapsing case is precisely the uncapturable one.

### M4. Removing `Buffer(Op)` touches validation invariants and replay in non-obvious places — migration is wider than "swap a base class"
SEVERITY: major.
Pointers: `validation/invariants.py` checks `is_buffer` across Checks B, K, P, and ordering
(`buffer_layers` <-> `is_buffer` flag agreement at line 440; step_index/source-node carve-outs
at 641-663 treat buffers as "genuine functionless graph source, exactly like a buffer");
`validation/core.py:335` exempts `is_buffer and buffer_source is None` from replay; `core.py:472`
treats buffers as replay sources. The `func=identity` passthrough version nodes (the SOURCED
ones, e.g. `buffer_2` with `src=add_1_4`) are replayable today and validation PASSES
(`/tmp/probe10`). But the spec moves "Buffer" from an Op subclass to a non-Op entity while the
GRAPH NODES become plain `Op(is_buffer=True)`. Every one of these checks keys on `is_buffer`,
which survives — fine — but `_check_buffer_xrefs` (1750) iterates `ml.buffers` expecting Buffer
records with `.address`; under the new model `ml.buffers` returns ENTITIES (not nodes), so this
invariant and `buffer_layers` semantics must be re-pointed at version-nodes vs entities
carefully. Per the LOCKED tripwire principle, any of these going green via a loosened check is
forbidden; the spec must enumerate which invariant now reads the entity vs the node. It
currently says only "buffer nodes must not break … invariants" — no mapping.

---

## MINOR

### m1. `Module.buffers` (entities) vs `Module.buffer_layers` (deferred field) — shared/aliased buffers underspecified
The spec resolves the long-deferred `Module.buffer_layers` by making `Module.buffers` return
entities, but a buffer shared at multiple addresses (`all_addresses`,
`all_module_addresses`) has no stated rule for which module "owns" it or whether it appears in
multiple `Module.buffers`. `_check_buffer_xrefs` already tolerates buffers on never-entered
modules and non-module containers (invariants.py:1774-1791) — the entity model must preserve
that tolerance or it trips the tripwire on real models (anchor_generator, rope).

### m2. `is_in_place` name collides with existing `Op.is_inplace`
`Op` already has `is_inplace` (probe14). The spec adds `is_in_place`. Two near-identical
booleans with different meanings on the same class is a documentation/UX trap; pick one name
or namespace it (`is_buffer_mutator`).

### m3. `initial_value`/`final_value`/`current_value` capture cost unscoped
Three value snapshots per buffer entity, on top of per-version-node `out`. For
`final_value`/`current_value` of an attached buffer the live ref is cheap, but `initial_value`
+ every version `out` double-stores the v1 tensor. Spec lists the fields without a capture-cost
or dedup-with-version-node-`out` statement.

---

## Overall factoring judgment

The `Buffer`-entity / plain-`Op`-version-node split is the RIGHT shape for the read-side
(static buffer read N times: one node, N children — `/tmp/probe4` confirms this already
works cleanly). The entity-as-noun + `.buffer` backref is a genuine improvement over
`Buffer(Op)` mixing two namespaces.

The factoring BREAKS on the write/version axis, because the spec designs a rich
version/history/dual-label model on top of capture that does not exist (B1, B2) and loop
grouping that does not behave as claimed (B3). The cleaner factoring given current capture:

1. Ship the **entity** (noun) + `is_buffer` flag on Op + read-side (one node, N children).
   This is genuinely additive and most of the user value ("everything about running_mean in
   one hop").
2. Define versions as **exactly the captured buffer-source nodes** (the `buffer_2..N`
   identity-passthrough chain that DOES exist for re-read reassigned buffers), and state
   plainly that BatchNorm/fused/in-place overwrites yield ONE version because TorchLens does
   not capture them. `is_static` = "only one captured version" — honest and computable.
3. DROP the dual-label entirely (B3: op-label == address-version always). Keep ONE label:
   `<address>:<version>`. Re-introduce a second label only if/when buffer equivalence is made
   loop-position-sensitive (separate, scoped loop-detection change).
4. Make in-place/`value_after`/BatchNorm-versioning an EXPLICIT, separately-budgeted capture
   sub-project, not a field on the entity that silently returns nothing.

This preserves the spec's real win (the entity) while removing the four blocking claims that
ride on uncaptured state.

---

VERDICT: NOT BULLETPROOF -- 4 blocking issues
