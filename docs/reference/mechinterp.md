# The mech-interp kit (`torchlens.mechinterp`)

TransformerLens's daily-driver analysis tools rebuilt on torchlens: residual
decomposition, direct logit contributions (DLA), per-head results, the
component edge graph, patching grids, head-taxonomy scores, prompt
utilities, an alias layer, and retention plans -- derived from the captured
graph (no architecture registry), with verification IN the API, not only in CI. Every spelling is
DOCUMENTED-UNSTABLE pending the naming sprint; the facade name is a
placeholder.

Axis disambiguation: `torchlens.attribution` attributes model OUTPUTS to
INPUTS; this kit attributes logits to internal WRITERS. No kit name contains
the word "attribution".

## The engine and its gate

`residual_decomposition(trace)` walks BACKWARD from the captured tensor the
final norm consumes: each qualifying addition is a spine step, the other
operand is the writer; hops across value-preserving ops are grammar-proposed
and payload-verified; every step carries the op's pass-qualified label and
live site key. The built-in oracle is an EQUALITY assertion, not a
tolerance: accumulating the literal captured writers in execution order
replays the forward's own fp32 add sequence, so all partial sums equal the
captured spine states under `torch.equal` (fp32/CPU; the per-dtype degrade
clause is disclosed in the identity receipt off that envelope). A failing
strict walk refuses OPEN; `strict=False` appends an `unresolved_remainder`
row, stamps the stack diagnostic-only, and is refused by DLA, gallery, and
launch claims.

Stack grading (D7): `complete` (every row a named writer) /
`closed_constant` (non-writer rows are VERIFIED constants) /
`closed_unresolved` (diagnostic). OPEN never constructs.

## DLA

`direct_logit_contributions(trace, answer=..., vs=...)`: directions-first
(peak memory scales with requested tokens, never vocabulary), frozen
validated norm scale (bit-exact against TLens's own `ln_final.hook_scale` on
real gpt2), ONE minimal constant row (final-norm beta + unembedding bias --
writer-side biases are already inside the literal writers; double-counting
costs a measured 6.35 logits), and the identity `sum(rows) + constant ==
the captured native logit` runs on EVERY call. Convention, in one sentence:
DLA linearly decomposes the ACTUAL logits at the actual operating point; it
is not a set of counterfactuals.

The identity gate cannot be bitwise (the kit re-associates the model's
matmul), so it runs against a PER-ELEMENT budget
(`identity_receipt["budget_model"] == "per_element_cancellation_aware_v2"`):
each element's own accumulated |addend| basis -- the rows, constant and
native logit projected on the UN-differenced answer and `vs` directions, so
a logit DIFF inherits the rounding of its two ~100-logit operands instead of
being budgeted at its small value -- times the pairwise-summation depth term
`1 + log2(n_rows + 2)`, times 4x ULP headroom, at the machine epsilon of the
coarsest floating dtype in the chain (float32 floor). A global scalar taken
at the largest-magnitude element (the former model) handed low-magnitude
elements ~300x their observed residual. The receipt discloses
`max_abs_residual`, `tolerance` (the budget AT the worst-fraction element),
`max_budget_fraction` (<= 1 when verified; measured 0.002-0.07 on gpt2
layer- and head-grain, full and partial stacks), `budget_eps` and
`n_addends`; a failing element is named in the `mi_dla_identity_failed`
refusal (`worst_element`).

## Coverage and honesty surfaces

- `attention_head_contributions`: per-QUERY-head rows from the validated
  `result` facet + a remainder labelled `output_bias` only when PROVEN.
- `head_weight_views`: `[H, E, D]`/`[H, D, E]` views with orientation decided
  by module class (Conv1D `[in, out]` vs Linear `[out, in]`), fused-QKV
  slice evidence from the captured graph, GQA counts, shared-storage
  disclosure; payload-verified before serving.
- `resolve_alias` / `alias_report`: TLens spellings (both 2.x and 3.x-bridge
  generations, `translation.json`) resolve to native coordinates with an
  honest status: `real` / `reconstructed_read_only` / `needs_capture` /
  `structurally_absent` / `ambiguous`, each with a remedy.
- `retention_plan`: printable, priced, `save=`-composable plans; every kit
  function refuses with the COMPLETE missing-site set when payloads are
  absent -- never a partial table, never a silent widening.
- `grid` + presets: receipted patching cells over the live-hook rerun route
  (a hook that did not fire is an exception, never a number), pre-run cost
  disclosure, hard `budget=`, `engine="replay"` reserved typed.
- `patch_query_heads_grid`: under grouped-query attention a `k`/`v`-site
  grid patches a whole key/value group, shared by several query heads. This
  grid patches one query head's own copy instead, at the op that expands the
  group to the query heads (Hugging Face `repeat_kv`, or the registered
  `"torchlens"` attention backend), selected by function, module and capture
  order and receipted like `grid` (one fire per cell, or a refusal). Head
  coordinates are query heads; `receipts["effect_site"]` names the
  interceptor. Torch SDPA with `enable_gqa=True` expands inside the kernel
  and refuses `mi_kv_expansion_unavailable` (load with
  `attn_implementation="eager"` or the `"torchlens"` backend); equal head
  counts refuse `mi_kv_heads_not_grouped`.
- `lowered_counterfactual`: by-head and by-pattern patching on DEFAULT-loaded
  fused-SDPA models with no eager reload -- the delta lowers EXACTLY to an
  add at the real post-projection op, with a receipt naming the virtual
  site, the real site, the formula version, and the invalidated facets;
  reading an invalidated facet on the patched rerun refuses typed.
- `head_scores` / `inspect_prompt` / `test_prompt`: taxonomy masks with
  TLens-compatible conventions (credited), structured prompt records with
  explicit BOS disclosure and teacher-forced multi-token scoring.
  `head_scores(trace, kind="induction", period=n)` scores the fixed repeat
  of a sequence with period `n` instead of the token-equality mask:
  destination `i` on the one source `i - n + 1`, counted when its token
  equals the token at `i - n`, so a token that repeats inside the sequence
  adds nothing (measured on GPT-2 small in the circuit section below).

Fused attention (torch SDPA) needs
`capture=CaptureOptions(save_arg_values=True)` for reconstructed
pattern/z/result reads; the facets stay READ-only -- edits are the lowered
counterfactual's job, or an eager recapture.

## The component graph

`component_graph(trace)` returns the circuit-discovery graph of a pre-norm
additive-residual language model, in the vocabulary of the MIB circuit track
and EAP-IG. Nodes are the residual writers: `input` (the embedding sum),
`a{L}.h{H}` (one head's output, without the attention output bias), `b{L}`
(layer L's attention output bias, a node of its own so the decomposition
never drops it) and `m{L}` (one MLP's output, bias included). Receivers are
the residual readers: `a{L}.h{H}<q>`, `<k>` and `<v>`, `m{L}` and `logits`.
An edge `sender->receiver` exists when the sender wrote into the residual
state the receiver reads, and its `contribution` is the sender's output per
batch row and position.

A receiver's input is the tensor entering its pre-norm, never the norm's
output: the norm is not linearised, and on an additive residual stream that
tensor is exactly the sum of the receiver's incoming contributions. The
builder checks that identity for every receiver before it returns
(`graph.identity_receipt`). Which receivers read which writers comes from
the captured dataflow, so on a parallel-block model (GPT-NeoX, Pythia) MLP L
reads the same state as attention L and receives no edge from layer L's
heads. The vocabulary edge count excludes the bias nodes: count the edges
whose `sender.kind` is not `"bias"`.

Capture with every op saved and eager attention (the per-head `result`
view). A stream that is not additive under this decomposition (a
renormalized or rescaled residual, a writer that is not an embedding, head
or MLP) refuses `mi_edge_graph_nonadditive`; grouped-query attention refuses
`mi_edge_graph_gqa_unsupported`.

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

config = GPT2Config(
    n_layer=2, n_head=2, n_embd=8, vocab_size=32, n_positions=16, attn_implementation="eager"
)
model = GPT2LMHeadModel(config).eval()
ids = torch.randint(0, 32, (1, 5))
trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
graph = tl.mechinterp.component_graph(trace)

receiver = graph.receivers["a1.h1<q>"]
print([node.name for node in receiver.senders])  # input, a0.h0, a0.h1, b0, m0
total = sum(node.value for node in receiver.senders)
assert torch.allclose(total, receiver.value, atol=1e-5)
edge = graph.edges["m0->a1.h1<q>"]
assert edge.contribution is graph.nodes["m0"].value

# A circuit runs as one patch_edges call: every edge outside it reads its
# sender's value on the corrupted input.
ready = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)
clean = tl.trace(model, ids, capture=ready)
corrupt = tl.trace(model, torch.randint(0, 32, (1, 5)), capture=ready)
corrupt_graph = tl.mechinterp.component_graph(corrupt)
kept = {"input->a0.h0<q>", "a0.h0->a1.h1<v>", "a1.h1->logits"}
outside = [e for name, e in tl.mechinterp.component_graph(clean).edges.items() if name not in kept]
circuit = tl.mechinterp.patch_edges(clean, outside, corrupt=corrupt_graph)
print(circuit.modules["lm_head"].out[0, -1, :3])
for t in (trace, clean, corrupt):
    t.cleanup()
```

Edges are intervention selections. `graph.edges["m0->logits"]` passed to
`trace.do` addresses the receiver's pre-norm input edge (an EDGE selection,
edge substitution on the replay engine); an edge into `a{L}.h{H}<q>` (or
`<k>`, `<v>`) addresses head H's slice of that layer's projection output (an
ACT selection), because the heads and roles of one layer share their norm.
`patch_edges(trace, edges, *, corrupt=corrupt_graph)` builds the matching
values and returns a fork in which every patched receiver reads `input -
sum(s) + sum(s_corrupt)` over its patched edges. With edges into one
receiver, `input` and `s` are the clean values, and every receiver that is
not downstream of it reads exactly what it read before. With edges into
several receivers, `input` and `s` are the values of the patched forward
itself (the MIB circuit track's convention): a receiver sees its patched
senders' corrupted values while every other sender carries what it computes
in that forward, including the effect of patches upstream of it. A circuit
(a set of kept edges, everything else patched) is one call, as in the
example above: pass every edge outside the circuit.

Receivers are patched in topological order, a stage at a time (a layer's
attention receivers, then its MLP, then `logits`), and the component graph
is rebuilt on the partly patched fork at each later stage, so a circuit over
all 445 GPT-2 small receivers costs one replay per patched receiver (two for
`q`, `k`, `v`) plus 24 graph builds. No edges returns an unpatched fork; an
edge passed twice refuses `mi_edge_patch_edge_repeated`. Patching needs a
trace captured with `intervention_ready=True`.

## Edge attribution patching and circuit faithfulness

`edge_scores(model, inputs, *, corrupted, metric)` scores every edge of the
component graph by edge attribution patching, the first-order estimate of
patching that edge alone: `score(s->t) = (s_corrupt - s_clean) . dM/d(input
of t)`, summed over positions and averaged over examples
(`example_reduction="sum"` adds them instead). `metric` maps the
unembedding output to one value per example. `method="eap"` takes the
gradient once, at the clean input. `method="eap_ig"` (the default, the best
edge method of the MIB circuit track) averages it over `steps` points (5 by
default) on the straight line between the corrupted and the clean `input`
node, `e_corrupt + (k/m) (e_clean - e_corrupt)` for `k = 0..m-1`, a left
Riemann sum.

Each evaluation point costs one capture and one backward for all edges at
once. A receiver's gradient is its own, never the total gradient at the
shared residual state: the backward stops at the receiver's own read (its
pre-norm's output for `m{L}` and `logits`; head H's q, k or v slice of the
projection output for `a{L}.h{H}<q|k|v>`), and a local pull-back through the
norm (and the projection slice) carries it to the receiver's input, reusing
the captured autograd graph. The result is a one-backward `ReadTable` of
`EDGE` rows addressed `(edge name,)` in the graph vocabulary; bias-sender
edges score exactly zero, since a bias is the same on both inputs. The table
is scalar, so `table.save(path)` persists it.

`circuit_performance_ratio(faithfulness)` (CPR) and
`circuit_model_difference(faithfulness)` (CMD) summarize a circuit sweep the
way the MIB circuit track does. Given the faithfulness `f = (M_circuit -
M_empty) / (M_full - M_empty)` at each kept edge fraction (the ten MIB
fractions 0.001 to 1 by default, `fractions=` otherwise, or a `{fraction:
f}` mapping), CPR is the trapezoidal area under `f` (higher is better) and
CMD the area under `|1 - f|` (lower is better); `log_scale=True` integrates
over the log fraction.

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

config = GPT2Config(
    n_layer=2, n_head=2, n_embd=8, vocab_size=32, n_positions=16, attn_implementation="eager"
)
model = GPT2LMHeadModel(config).eval()
clean = torch.randint(0, 32, (2, 5))
corrupted = torch.randint(0, 32, (2, 5))


def logit_difference(logits):
    return logits[:, -1, 3] - logits[:, -1, 4]


table = tl.mechinterp.edge_scores(
    model, clean, corrupted=corrupted, metric=logit_difference, steps=2
)
scores = {address[0]: row.score for (_target, kind, address), row in table.items()}
print(max(scores, key=lambda name: abs(scores[name])))
assert scores["b0->m0"] == 0.0

curve = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.05, 1.0, 0.98, 1.0]
cpr = tl.mechinterp.circuit_performance_ratio(curve)
cmd = tl.mechinterp.circuit_model_difference(curve)
assert 0 < cmd < cpr < 1
```

Measured on GPT-2 small with prompts from the `mib-bench/ioi` dataset (CPU):
EAP, and EAP-IG at 1 and 5 steps, equal an independent plain-PyTorch
implementation on every one of the 32,491 edges in float64 (largest
relative difference 5.0e-10). On a closed-form nonlinear fixture, EAP-IG
equals the left Riemann sum at every step count from 1 to 128 within 3e-16.
Against brute-force patching of all 32,491 edges one at a time (16 prompts,
float32), EAP-IG at 5 steps has Pearson correlation 0.939 with the patching
effect and shares 75 of the 100 largest edges; EAP has 0.806 and 87. CPR
and CMD equal the MIB circuit track's loop exactly.

Circuit test cases on GPT-2 small (CPU):

- Induction heads: `head_scores(trace, kind="induction", period=50)` on
  `[bos] + R + R` (`R` 50 random tokens in `[1000, 50000)`, generator seed
  0, batch 8) equals an independent Hugging Face eager-attention oracle
  within 5.0e-11 for all 144 heads in float64 (2.7e-7 in float32), and its
  top 5 heads are the oracle's: 5.5, 6.9, 7.10, 5.1, 7.2.
- Recovering Wang et al.'s IOI circuit: EAP-IG at 5 steps over 1000 rows
  of the `mib-bench/ioi` train split, corrupted with its
  `s2_io_flip_counterfactual` column (not the ABC corruption the circuit
  was found with), ranks heads by the summed |score| of their outgoing
  edges. Against the 26 circuit heads the ROC AUC is 0.926 and 20 of the
  top 26 heads are circuit heads (recall 0.77). Over the 29,089 edges among
  the input, the heads and the logits, 72 of the circuit's 182 head-level
  edges are in the top 182 (recall 0.40); over all 32,491 edges the ROC AUC
  against ACDC's encoding of the circuit is 0.870. The run took 57 minutes
  on 4 cores (3.4 s per row).
- Circuits run as one `patch_edges` call; on GPT-2 small a call that patches
  edges into all 445 receivers equals an independent forward-hook
  implementation of the same convention within 1e-10 in float64.

ACDC-style discovery is a recipe, not a TorchLens loop:
[docs/recipes/acdc_ioi.md](../recipes/acdc_ioi.md) walks the receivers from
the output, patches each incoming edge with `patch_edges`, prunes it when
the metric moves by less than a threshold, and records every decision as a
ledger verdict.

Bad settings refuse `mi_edge_scores_setting_invalid`, a metric that does
not return one value per example `mi_edge_scores_metric_invalid`, and
corrupted inputs of another shape `mi_edge_scores_inputs_mismatched`; a
model whose attention and MLP share one norm refuses
`mi_edge_scores_shared_norm`. Malformed faithfulness curves refuse
`mi_faithfulness_curve_invalid`.

## Refusal vocabulary

Every kit refusal carries a stable `fields["code"]` (`mi_*` rows in
`docs/reference/error_refusal_contract.md`) and a remedy; callers branch on
codes, never message text. The any-architecture claim stays bounded: any
executed architecture whose target has a certifiable additive residual
topology works; everything else refuses typed with the frontier named.
