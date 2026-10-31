## Common Patterns

Cost first: to steer or patch a model over many forwards or through HF `generate()`, use the
capture-free `tl.when(site, action).bind(model)` (about 1x a plain forward hook) instead of a
`tl.trace` per step; use `tl.record(..., intervene=..., return_output=True)` when each step also
needs recorded activations. Which path costs what, and what each reuse path refuses:
[Performance: choosing a fast path](../guides/fast_paths.md).

```python
import torch
import torchlens as tl

log = tl.trace(model, x, save=tl.func("relu"))
activation = log["relu_1_2"].out
# A default trace is a pure observer: buffers, modes, random state and plain module
# attributes are put back afterwards. A value the forward stashes on a module
# (self.last_weights = w) is read from the trace; carry_state=True keeps it on the module.
stashed = log.modules["self"].attribute_writes  # {name: AttributeWrite(kind, value)}
# Module and stage names resolve to that module's output at every selector door,
# including timm stage names (FeatureGetterNet's "blocks.9" for "model.blocks.9").
backbone = torch.nn.Sequential()
backbone.add_module("layer1", torch.nn.Linear(4, 4))
backbone.add_module("layer2", torch.nn.Linear(4, 2))
stages = tl.extract(backbone, torch.randn(1, 4), ["layer1", "layer2"])
stage_log = tl.trace(backbone, torch.randn(1, 4), save=["layer1", "layer2"])  # list form
stage_log["layer2"].out
# Appliances hang off one door: trace.using(name), registered with
# torchlens.accessors.register_appliance(name, factory) (spellings DOCUMENTED-UNSTABLE).
print(log.summary())
print(tl.report.explain(log))
log.draw(order_siblings=True)  # default: verified sibling ordering for dot/unrolled graphs
log.draw(collapse="auto", show_containers=False)  # readability-targeted module overview
print(log.module_collapse_order[:10])
tl.release_model(model)  # restore whole-model pickle / torch.save serializability

# Steered rerun: a module-targeted staged spec reruns through the guarded fast engine
# (native forward + hooks, close to plain-forward cost); any typed refusal falls back
# to the capture engine and says why.
direction = torch.randn(4)               # one entry per channel of encoder's output
steered = tl.trace(model, x, save=tl.module("encoder"),
                   intervene=tl.when(tl.module("encoder"),
                                     tl.steer(direction, magnitude=4, feature_axis=1)))
steered.run(model, torch.randn(1, 3, 20, 20))  # larger input, same graph structure
steered.last_run["engine"]              # "guarded_fast", or "rerun" after a fallback
steered.last_run["fast_refused"]        # None, or "<code>:<stage>" naming the guard
steered.last_run["shape_varied"]        # True when the input size differed from capture

# Influence geometry is lazy: the first property access solves the captured DAG.
op = log["relu_1_2"]
rf = op.receptive_field
box = rf.at((10, 10))
unit = rf.center_unit(batch_index=0)
check = rf.check(unit)
projective = op.projective_field.at((10, 10))
layer_to_layer = op.receptive_field.at((10, 10), source=log.input_ops[0])
table = log.receptive_fields(level="layer")
outgoing = log.projective_fields(level="layer")
# Arming the gradient tripwire needs requires_grad inputs, backward_ready=True,
# and save_mode="reference"; verify().verdict is PASS / FAIL / INDETERMINATE.
armed = tl.trace(model, x.requires_grad_(True),
                 capture=tl.options.CaptureOptions(backward_ready=True),
                 save_mode="reference")
armed_op = armed["relu_1_2"]
armed_unit = armed_op.receptive_field.center_unit(batch_index=0)
gradient = armed_op.receptive_field.gradient(armed_unit, retain_graph=True)
validated = tl.receptive_field.verify(armed, units="center")
# show(gradient=True) recomputes the gradient WITHOUT retain_graph and frees the
# autograd graph -- call it last (or re-capture) if later backward passes are needed.
overlay = armed_op.receptive_field.show(armed_unit, gradient=True)
# tl.validate(model, x, scope="receptive_field") captures an armed trace itself.
# tl.validate(gpu_model, x, scope="forward", output_device="cpu", save_budget=None) keeps
# the validator capture's saved activations in host memory; output_device / save_budget
# take CaptureOptions' values and defaults and apply to the forward/saved/intervention scopes.
# Saved function arguments other than parameters go to host memory too and replay moves each
# back to its op's device, so the GPU keeps only the parameters and one forward's transients.
```

Use the unified predicate surface for selective capture, windowed saves, interventions, and
storage:

```python
conv_before_relu = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
log = tl.trace(
    model,
    x,
    save=conv_before_relu,
    lookback=4,
    lookback_payload_policy="detached_raw",
)

ablated = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)
# Reruns re-arm the capture predicate and apply the edit exactly once; the
# staged spec never grows. tl.save keeps the ablated values but not the recipe:
# a loaded copy refuses run(model, x) with
# run_intervention_spec_not_persisted, so save the recipe with save_intervention.
ablated.run(model, x)

disk_log = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
full_structure = recording.to_trace()

# Selection algebra (L6): compose regions, resolve explicitly, edit with do().
log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
u1, u2 = log["relu_1_2"], log["conv2d_2_3"]
inter = u1.receptive_field.at((3, 3)) & u2.receptive_field.at((5, 5))  # a Selection
resolved = inter.resolve(log)              # frozen, trace-bound, session-only
fork = log.fork()
fork.do(inter, tl.zero_ablate())           # edit-then-scatter: only masked elements
fork2 = log.fork()
fork2.do(tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(fork2), tl.patch_from(log))
fork3 = log.fork()
fork3.do(tl.params("head.weight"), tl.scale(0.5))  # "as if" the weight changed, replay-only:
# every consumption is substituted; the live nn.Parameter is NEVER written.
print(fork.intervention_audit[-1])         # query repr + resolve digest + relations
```

Use `backend=` only when the backend is intentionally part of the test or example:

```python
torch_trace = tl.trace(model, x, backend="torch")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
assert torch_trace.backend == "torch"
```

Before debugging wrapper-specific failures, run:

```python
print(tl.compat.report(model, x).to_markdown())
```

### Steering many forwards / generation

To apply one intervention to many forwards (greedy decoding, `generate()`, a sweep over
prompts), pick the path by what you need back. Build the spec once; every path takes it.

- Steering only: `spec.bind(model)`. A capture-free executor that runs at about the cost of
  a plain forward hook. Call it like the model, or call `.generate(...)` on it (KV cache on or
  off); `.last_report` counts the firings. `torchlens.intervention.steer_generate` is one-call
  sugar for `spec.bind(model).generate(...)`. A binding selects ops structurally
  (`tl.func("relu") & tl.in_module("1")`, `tl.module("1")`); it refuses final or short labels such
  as `tl.label("relu_1_2")` when bound, because labels are assigned after a capture.
- Steering plus activations as evidence: `tl.record(model, x, save=..., intervene=spec,
  return_output=True)` once per forward. It returns the model output and a sparse `Recording`
  and is the cheapest path that also captures.
- Activations across a whole generation, without steering:
  `tl.record(model, ids, save=..., generate_kwargs={"max_new_tokens": 16, "do_sample": False})`
  (DOCUMENTED-UNSTABLE). One real cached `generate()` runs and every forward it issues is one
  Recorder pass, so each decode step records only its new position and the cost is linear in
  generated tokens. `generation.steps[k].positions` names the absolute positions step `k`
  computed (step 0 is the prefill) and `generation.records_at(k)` returns its records. Beams,
  batched prompts, more than one return sequence and `intervene=` refuse with
  `record_generation_unsupported`. Do not loop a fresh `tl.record` over the growing prefix: it
  recomputes and re-saves every earlier position at every step.
- A correctness oracle: one full `tl.trace(model, x, intervene=spec)`, run once and compared
  against the fast path. It is several times slower per op than `tl.record`, so never run it
  per generation step.

Trace-then-rerun is a fast path when the staged spec targets modules: `trace.run(model, x)`
runs the native forward with the staged hooks (the guarded fast engine, about 1.5x a plain hook,
exact), admits a longer input when the model takes the same sequence of calls, and falls back to
the capture engine with `last_run["fast_refused"]` naming the guard that refused;
`trace.run(inputs=..., fast=True)` applies the staged spec the same way on a live trace. Value
replacements and non-module targets still take the capture engine. Glossary: "Guarded fast
rerun".

```python
import torch
import transformers

torch.manual_seed(0)
lm = transformers.LlamaForCausalLM(  # a tiny random decoder stands in for your model
    transformers.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2)
).eval()
ids = torch.tensor([[1, 5, 9, 2]])
direction = torch.randn(32)
site = tl.module("model.layers.1.mlp")
spec = tl.when(site, tl.steer(direction, magnitude=4.0, feature_axis=-1))

# Steering only: bind once, then call or generate as often as needed.
steered = spec.bind(lm)
logits = steered(ids).logits
tokens = steered.generate(ids, max_new_tokens=5, do_sample=False, use_cache=True)
assert steered.last_report.fire_count == 5  # one firing per generated token

# Steering plus evidence: one tl.record per forward returns the output and the recording.
out, rec = tl.record(lm, ids, save=site, intervene=spec, return_output=True)
steered_mlp = rec.to_trace().find_sites(site).first().out  # the steered site output
assert torch.equal(out.logits, logits)

# Correctness oracle, once: a full trace with the same spec agrees exactly.
oracle = tl.trace(lm, ids, save=site | tl.module("lm_head"), intervene=spec)
assert torch.equal(oracle.find_sites(tl.module("lm_head")).first().out, logits)
assert torch.equal(oracle.find_sites(site).first().out, steered_mlp)
```

### Per-head attention from a Hugging Face model without a trace

`tl.bridge.hf.register_attention(model)` makes TorchLens the model's attention backend
(the `"torchlens"` attention function plus its matching mask function, so causal and padding
masks always arrive); `tl.bridge.hf.record_attention(model)` then records every call's per-head
probabilities by module name. It materializes the full `(batch, heads, q, kv)` matrix per call
(32 GiB per 4096-token sequence for a Llama-2-7B shape in bfloat16). Glossary:
"tl.bridge.hf.register_attention".

```python
import torch
import transformers

torch.manual_seed(0)
lm = transformers.LlamaForCausalLM(
    transformers.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2)
).eval()
batch = {"input_ids": torch.tensor([[1, 5, 9, 2], [7, 3, 0, 0]]),
         "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])}
tl.bridge.hf.register_attention(lm)
with torch.no_grad(), tl.bridge.hf.record_attention(lm, check_sdpa=True) as attn:
    lm(**batch)
probs = attn["model.layers.0.self_attn"]  # (batch, heads, q, kv), the tensor the model used
assert probs.shape == (2, 4, 4, 4)
assert torch.count_nonzero(probs[1, :, :2, 2:]) == 0  # padded keys get exactly zero
```

### Newer namespace exports

Exports added to existing namespaces during 2.x; spellings DOCUMENTED-UNSTABLE, each defined
in the glossary under its name. The fence builds edits and results without a transformer; the
comments name the exports that need one.

```python
# trace.using(name) appliances, with a one-line summary each.
appliances = tl.accessors.registered_appliances()  # {name: summary}, sorted by name
# Subspace and position edits are plain tl.Edit objects; apply them with do() or fork.do().
basis = torch.eye(4)[:1]  # [k, d]: one direction over a 4-wide feature axis
ablate = tl.intervention.subspace_ablate(basis, feature_axis=1)  # x - P x on the span
assert isinstance(ablate, tl.Edit)
# tl.intervention.subspace_interchange(basis, source_trace) writes the source trace's
# in-span component over the target's instead.
first_token = tl.intervention.at_positions(tl.scale(0.0), [0])  # token axis 1 by default
assert isinstance(first_token, tl.Edit)
# Namespaces are imported, not root attributes: import torchlens.mechinterp first.
import torchlens.mechinterp

# torchlens.mechinterp.FactoredMatrix(A, B) keeps A @ B factored.
factored = torchlens.mechinterp.FactoredMatrix(torch.randn(4, 2), torch.randn(2, 4))
assert factored.ndim == 2 and factored.AB.shape == (4, 4)
# On a transformer trace, torchlens.mechinterp.qk_circuit(trace, attention) and
# torchlens.mechinterp.ov_circuit(trace, attention) return one FactoredMatrix per query head, and
# torchlens.mechinterp.mlp_neuron_contributions(trace) returns per-neuron MLP rows.
disagreement = tl.repgeom.metric_disagreement(
    {"cka": {"layer1": 0.9, "layer2": 0.4}, "rsa": {"layer1": 0.3, "layer2": 0.8}}
)
assert isinstance(disagreement, tl.repgeom.MetricDisagreement)
pair = disagreement.pairs[0]  # a tl.repgeom.DisagreeingPair
assert pair.higher == {"cka": "layer1", "rsa": "layer2"}
score = tl.stats.cka(torch.randn(8, 3), torch.randn(8, 5))
assert isinstance(score, tl.stats.SimilarityScore)  # a float that names its measure
measure = score.similarity_repository_id
# An output_style no detector resolves refuses before the forward.
try:
    tl.trace(model, x, capture=tl.options.CaptureOptions(output_style="no_such_style"))
except tl.errors.OutputStyleUnknownError as error:
    assert error.fields["code"] == "output_style_unknown"
else:
    raise AssertionError("output_style_unknown did not refuse")
# Also exported: tl.debug.DebugReport (base class of report-shaped tl.debug results),
# tl.report.PrunedParamFacts (prune state in the FactCore parameter facts),
# torchlens.trackers.MLflowSink and torchlens.trackers.CometSink (forward-watch sinks over a run or
# experiment the user owns) and torchlens.tviz.render_logit_lens_heatmap(result, path)
# (the layers-by-positions logit-lens figure).
```

### Attribution graphs

`tl.attribution.build_graph(trace, transcoders, embedding_module=..., frozen=...)` builds an
attribution graph from one prompt's capture: `transcoders=` maps each MLP module name to a
transcoder (`encode`, `W_enc`, `b_enc`, `W_dec`, `b_dec`; mappings for `W_dec` and `b_dec`
give a cross-layer transcoder), and `frozen=` names the attention-pattern and norm-scale
sites held at their captured values. The result is a `tl.attribution.AttributionGraph`:
one `tl.attribution.GraphNode` per feature, error, token and logit node, a dense
`adjacency_matrix` with targets as rows, and `node_values` that equal each feature or logit
row's incoming edges plus its named `constants`. Spellings are DOCUMENTED-UNSTABLE; the
runnable example is in `docs/reference/attribution.md`, section Attribution graphs.

A transcoder that reads a point inside a module (Gemma Scope reads the pre-feedforward
norm's value before its `(1 + w)` gain) is keyed by that op's label instead of the MLP
module name, with `W_dec` and `b_dec` given as mappings naming the MLP outputs it writes;
the key must run before them. `frozen=` also takes facet selectors:
`frozen=[tl.facet("pattern"), tl.facet("inv_rms")]` holds every Hugging Face Gemma-2
attention pattern and every `Gemma2RMSNorm` scale, where `inv_rms` is the norm's
`rsqrt` output, the factor that multiplies its input. DOCUMENTED-UNSTABLE.

`graph.to_neuronpedia(slug="my-graph", model_id="gemma-2-2b", tokenizer=tokenizer, path="g.json")`
returns the graph, pruned by circuit-tracer's rule (`node_influence_coverage=0.8`,
`edge_influence_coverage=0.98`), in Neuronpedia's published attribution-graph schema, and writes it
to `path`; the file opens in Neuronpedia's graph viewer. DOCUMENTED-UNSTABLE.

`tl.attribution.overlay(trace, graph)` paints the graph onto its trace for
`trace.draw(color_by=...)`: each site holding feature or error nodes gets the total influence
of its nodes (`graph_value="activation"` paints feature activations instead).
`tl.attribution.graph.upload_to_neuronpedia("g.json", api_key=os.environ["NEURONPEDIA_API_KEY"])` publishes the exported
file on neuronpedia.org under the key's account and returns its URL;
`direction.upload_to_neuronpedia(api_key=os.environ["NEURONPEDIA_API_KEY"], model_id="gemma-2-2b", name="honesty")`
publishes a `Direction` the same way. Both check every field before the first request and
never write the key into a message. DOCUMENTED-UNSTABLE.
