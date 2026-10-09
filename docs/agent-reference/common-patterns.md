## Common Patterns

```python
import torch
import torchlens as tl

log = tl.trace(model, x, save=tl.func("relu"))
activation = log["relu_1_2"].out
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
# Saved function arguments stay on the model device, so the GPU footprint roughly halves and
# the GPU save_budget can still be exceeded (then pass save_budget=None or a larger value).
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
  sugar for `spec.bind(model).generate(...)`.
- Steering plus activations as evidence: `tl.record(model, x, save=..., intervene=spec,
  return_output=True)` once per forward. It returns the model output and a sparse `Recording`
  and is the cheapest path that also captures.
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
