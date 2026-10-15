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
trace.cleanup()
```

Edges are intervention selections. `graph.edges["m0->logits"]` passed to
`trace.do` addresses the receiver's pre-norm input edge (an EDGE selection,
edge substitution on the replay engine); an edge into `a{L}.h{H}<q>` (or
`<k>`, `<v>`) addresses head H's slice of that layer's projection output (an
ACT selection), because the heads and roles of one layer share their norm.
`patch_edges(trace, edges, corrupt=corrupt_graph)` builds the matching value
and returns a fork in which the edges' one receiver reads `clean input -
sum(s_clean) + sum(s_corrupt)`; every receiver that is not downstream of it
reads exactly what it read before. Patching needs a trace captured with
`intervention_ready=True`; edges into several receivers refuse
`mi_edge_patch_receivers_mixed`.

## Refusal vocabulary

Every kit refusal carries a stable `fields["code"]` (`mi_*` rows in
`docs/reference/error_refusal_contract.md`) and a remedy; callers branch on
codes, never message text. The any-architecture claim stays bounded: any
executed architecture whose target has a certifiable additive residual
topology works; everything else refuses typed with the frontier named.
