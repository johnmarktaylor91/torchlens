# Intervention API Reference

This page documents the public v2 intervention API that shipped in the current
branch. It avoids proposed naming changes from separate design workstreams.

To apply one intervention to many forwards or a `generate()` loop, use `spec.bind(model)`
(capture-free, about the cost of a plain hook) or `tl.record(..., intervene=spec,
return_output=True)` when you need activations too, or rerun a steered trace with
`trace.run(model, x)` (the guarded fast engine when the spec targets modules; see the glossary
entry "Guarded fast rerun"). Recipe: "Steering many forwards / generation" in
[agent-reference/common-patterns.md](agent-reference/common-patterns.md).

## Selectors

Selectors resolve against completed `Trace.layers` records.

| Selector | Signature | Use |
| --- | --- | --- |
| `tl.label` | `label(name: str)` | Exact final, raw, short, or pass-qualified label. |
| `tl.func` | `func(name: str)` | Match captured function name OR normalized layer type such as `"relu"` or `"add"`. |
| `tl.module` | `module(address: str)` | Match a module output boundary. |
| `tl.label_contains` | `contains(substring: str)` | Case-insensitive label substring search. |
| `tl.label_matches` | `regex(pattern: str)` | Case-sensitive `re.search` over labels. |
| `tl.where` | `where(predicate, *, name_hint=None)` | Predicate over layer pass records; non-portable. |
| `tl.in_module` | `in_module(address: str)` | Match sites contained in a module address. |
| `tl.grad_fn` | `grad_fn(name=None, *, label=None, is_custom=None)` | Backward grad_fn by autograd class name, label substring or custom flag (backward sites only). |

One interpreter evaluates every selector in every lifecycle (capture-time
`save=`, post-hoc `find_sites`, live hooks); `contains` is case-insensitive and
`regex` case-sensitive everywhere. Post-hoc `contains`/`regex` search the
final `layer_label` only; exact `tl.label` additionally matches raw, short,
and pass-qualified spellings. Capture-only selectors (`tl.followed_by`,
`tl.preceded_by`) and mutator-only selectors (`tl.facet`, `tl.head`) refuse
unsupported lifecycles with the typed `SelectorCapabilityError` (a
`SiteResolutionError` subclass) instead of a generic message — upfront, before
any per-site evaluation, for post-hoc resolution and live hook attachment
alike.

Composites (`&`, `|`, `~`) SHORT-CIRCUIT per site in every lifecycle: a
`tl.where` predicate is only invoked for sites its siblings have not already
decided, so predicates must not rely on side effects from seeing every site.
`&`/`|` build nested binary composites, and deserialized target specs may
carry flat n-ary child tuples; both shapes evaluate identically. Conjunctions
are association-insensitive for the temporal sugar — `a & tl.followed_by(x) & b`
behaves exactly like `a & b & tl.followed_by(x)` and the flat three-child
spec — and degenerate arities keep identity semantics in evaluation and spec
round-trips alike: an empty `and` matches everything, an empty `or` matches
nothing, and a unary composite matches like its child.
`tl.followed_by`/`tl.preceded_by` target specs serialize a selector inner
structurally at every save level; an opaque-callable inner remains audit-only
and refuses typed at rebuild rather than being reconstructed lossily.

Selectors compose with `&` and `|` for in-memory discovery:

```python
candidate = tl.in_module("block") & tl.func("relu")
sites = log.find_sites(candidate, max_fanout=8)
exact = tl.label(sites.labels()[0])
```

The same selector objects can drive capture-time save and intervention predicates:

```python
saved = tl.trace(model, x, save=tl.func("relu"))
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.scale(0.5)),
)
conditional = tl.trace(
    model,
    x,
    intervene=tl.when(tl.func("relu") & tl.preceded_by(tl.in_module("block")), tl.zero_ablate()),
    lookback=4,
)
```

For saved or replayed specs, prefer exact `tl.label(...)` or another simple
portable selector after discovery.

## Forward Helpers

Forward helpers return `Edit` objects that can be passed to `set`,
`attach_hooks`, `do`, live capture `hooks=...`, replay, or rerun.

| Helper | Signature | Portability |
| --- | --- | --- |
| `tl.zero_ablate` | `zero_ablate(*, allow_shape_change=False)` | Portable built-in; append-compatible unless shape changes are forced. |
| `tl.mean_ablate` | `mean_ablate(source=None, *, dim="self", allow_shape_change=False)` | Portable for supported tensor/self sources; batch-dependent policies can block append. |
| `scramble_elements` | `torchlens.intervention.scramble_elements(source=None, *, from_=None, seed=None, allow_shape_change=False)` (the honest rename of the removed `resample_ablate` spelling; elementwise iid scramble, NOT coherent resampling ablation -- see MIGRATIONS.md) | Built-in, stochastic; seeded runs are reproducible, append-incompatible. |
| `tl.steer` | `steer(direction, strength=1.0, *, coef=None, dim=None, allow_shape_change=False)` | Portable when `direction` is serializable tensor data. |
| `tl.scale` | `scale(factor, *, allow_shape_change=False)` | Portable built-in and append-compatible. |
| `tl.clamp` | `clamp(*, low=None, high=None, allow_shape_change=False)` | Portable built-in and append-compatible. |
| `tl.add_noise` | `add_noise(std, *, seed=None, allow_shape_change=False)` | Built-in stochastic helper; seeded runs avoid global RNG consumption. |
| `tl.project_onto` | `project_onto(direction, *, dim=None, allow_shape_change=False)` | Portable when `direction` is tensor data. |
| `tl.project_out` | `project_out(direction, *, dim=None, allow_shape_change=False)` | Portable when `direction` is tensor data. |
| `tl.splice_module` | `splice_module(module, *, input_route="in", output_route="out", allow_shape_change=False)` | Executable in the same environment; not portable and not append-compatible. |

The packaged SAE splice experiment, `torchlens.bridge.sae.splice(...)`
(documented-unstable), composes `splice_module` with fork + push: it swaps a
duck-typed SAE's reconstruction (`encode`/`decode` pair; no SAE package
required) back into the model at one site, replays downstream, and reports
reconstruction fidelity plus output-level causal effect. The optional
`latents_edit=` callable transforms the encoded latents before decoding --
the feature-level causal-testing knob. See
`notebooks/sae_splice_tutorial.ipynb` for the end-to-end experiment.

## Backward Helpers

Backward helpers are Tier-1 live/rerun-only helpers.

| Helper | Signature | Portability |
| --- | --- | --- |
| `tl.hook_grad` | `hook_grad(fn)` | Live/rerun-only; not portable. |
| `tl.zero_grad` | `zero_grad(*, allow_shape_change=False)` | Live/rerun-only; not portable. |
| `tl.scale_grad` | `scale_grad(factor, *, allow_shape_change=False)` | Live/rerun-only; not portable. |

Hook callables receive one positional tensor and a required keyword-only
`hook` context. For forward hooks the positional tensor is the activation; for
backward hooks it is the gradient. The positional parameter is matched by
position, so `def hook(activation, *, hook): ...` and
`lambda g, *, hook: ...` are both valid. Hooks must return the replacement
tensor.

## Attribution Patching

Attribution-patching examples in this guide use the `(corrupted_graph - clean) * patched`
sign convention so positive scores mean the patched activation moves the run
toward the corrupt direction under that local linearization. The more common
TransformerLens presentation is `grad * (clean - corrupted_graph)`. Both are valid as
long as the sign convention is named and used consistently.

## Raw PyTorch Forward Hooks

The TorchLens intervention API is the preferred path when you want selectors,
fire records, replay, rerun, or saved intervention specs. Raw
`nn.Module.register_forward_hook` also works during `tl.trace()` /
`tl.trace()` when you need module-local PyTorch behavior:

```python
def halve_relu(module, args, output):
    return output * 0.5

handle = model.relu.register_forward_hook(halve_relu)
try:
    log = tl.trace(model, x)
finally:
    handle.remove()
```

If a raw hook returns a new tensor, TorchLens instruments that replacement, so
downstream op-level graph structure remains available: torch ops the hook runs
are traced as ordinary ops, and a tensor with no traceable provenance is
recorded as an `interventionreplacement_*` op marked `intervention_replaced=True`.

A capture-time module edit (`intervene=tl.when(tl.module("block"), ...)`) is
always recorded as its own `interventionreplacement_*` op, with
`intervention_replaced=True` and its fire records on `.interventions`, placed
between the module's last op and that op's consumers. The module's last op keeps
the value it computed, before the edit. The edited value is the replacement
op's `.out`, and it is what `tl.module("block")` reads as the module's output.
Every later op is renumbered (a label's last number is its position), so compare
traces through `Bundle` site reads, which pair ops by structure, not by label.

## Capture-Time Recording

`tl.record(..., save=...)` is the sparse capture sibling of `tl.trace(..., save=...)`.
It returns a `Recording`, not a `Trace`; call `Recording.to_trace()` when you need the full
postprocessed graph structure later.

```python
recording = tl.record(model, x, save=tl.func("relu"))
trace = recording.to_trace()
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
```

`record(save=...)` is the only predicate spelling; the old `keep_op=` /
`keep_module=` alias kwargs are removed and raise `TypeError`. Module-boundary
event recording is gated by `default_module=`, which records ALL module
enter/exit events uniformly — predicate-gated module-event selection has no
public spelling (see `docs/reference/deprecations.md` for the honest
capability statement).

Forward exceptions keep the historical behavior unless you opt in. With
`on_forward_error="attach_partial"`, TorchLens attaches `exc.partial_recording` and re-raises
the original exception. With `on_forward_error="return_partial"`, `tl.record(...)` returns a
failed partial `Recording`; `return_output=True` returns `(None, partial)` because there is no
valid model output. Failed partials carry `status="partial_error"`, `failed=True`,
string-only error metadata, `num_committed_ops`, and best-effort `last_event_*` fields. user-op
failures exclude the failing call; TL-side capture failures may include a skipped/partial
current-call event. Failed partials are not topology-complete, so `Recording.to_trace()` and
`Recording.backward()` reject them.

For full `tl.trace(...)` failures, inspect `exc.partial_log` directly or call
`tl._partial.from_failed_capture(exc)`.

## Trace Mutators

| Method | Use |
| --- | --- |
| `log.set(site, value)` | Record a one-shot tensor or callable replacement and mark the recipe stale. |
| `log.attach_hooks(site, hook)` | Add sticky helper/callable hooks to the recipe. |
| `log.do(...)` | Apply an intervention and dispatch to `push`, `run`, or `set_only`. |
| `log.fork(name=None)` | Create an isolated branch for experiments. |
| `log.push(replay=ReplayOptions(...))` | Propagate over the saved DAG without calling `model.forward` (the former `replay()` alias is removed). |
| `log.run(model, x, replay=ReplayOptions(append=False))` | Re-execute the model under the active spec (the former `rerun()` alias is removed). |
| `log.save_intervention(path, level=...)` | Write a `.tlspec/` intervention recipe. |

There are two distinct intervention paths, and they do not mix implicitly:
(1) edit a SAVED value and push the effect downstream on the captured DAG
(`do()` on a resolved selection, `push_from`, direct writes), and (2)
intervene on a FRESH execution (`do(..., model=..., inputs=..., intervention=InterventionOptions(engine="rerun"))`
or a new capture with `intervene=...`). A new-input `run(inputs=...)` on a
trace carrying path-1 value-edits is a fresh execution — the edits do NOT
apply to it, and TorchLens discloses that at the run door with
`PendingValueEditsWarning` (documented-unstable spelling) rather than
silently returning an un-edited verified run. The run itself proceeds:
this is a disclosure, never a refusal.

`log.do([(selection, edit), ...])` applies a list of selection and edit pairs
as one transaction with one intervention event. Activation (ACT) pairs attach
together and push once. A batch may also carry parameter selections
(`tl.param(...)`) and dataflow edge selections (from `log.graph.edges`); on the
replay engine the pairs then run in order exactly as one `do()` per pair would
(each run of consecutive activation pairs attaches and pushes together; each
parameter or edge pair goes through its own substitution path), and the result
equals those calls made one at a time. The batch is atomic: if any pair is
refused, every value, hook, audit row and history record the batch wrote is
put back, and the trace keeps only the one error event for the failed call.
Parameter and edge edits apply on the replay engine only, so a mixed batch on
another engine refuses with `selection_batch_cross_kind` before anything runs;
pass `intervention=tl.options.InterventionOptions(engine="replay")`. Leaf sites
(inputs and buffers) still cannot join a batch (`selection_batch_leaf_unsupported`).

The two engines differ sharply in cost when you sweep many edits over one
captured input. On the 144-head ablation sweep of a 12-layer, 12-head GPT-2
configuration (585 captured ops; `benchmarks/head_sweep_engine_cost.py`), the
push engine (`engine="replay"`) took about 2.1 minutes for all 144 heads
(129 s and 125 s in two runs; median 0.60-0.62 s per head) and the rerun
engine (`engine="rerun"`) about 34 minutes (2052 s and 2008 s; median
13.5-13.7 s per head), on CPU with four torch threads (torch 2.13) on a shared
32-core x86-64 Linux host. Both engines produce bitwise-identical ablated
traces on all 144 heads, so the difference is cost only. A rerun re-executes
every captured op, and from an intervention-ready source it also runs under
the host RNG monitor, so its cost is the same for every site; push cost grows
with the edited site's downstream cone. Prefer push for many edits on one
captured input, and use rerun when the edit must see a fresh execution.
A rerun from an intervention-ready source also snapshots the model state
before its forward pass, as every intervention-ready capture does; each state
entry whose bytes are unchanged since the source capture shares the source's
snapshot instead of being cloned again, so on this sweep a rerun copies about
54 MB of tensors instead of about 706 MB, and an entry that changed (an
in-place weight edit, say) is cloned afresh. Every edit on either engine
starts with `trace.fork()`, whose cost is fixed by the size of the trace, not
by the edit: on this 585-op trace a fork takes about 0.08-0.09 s and makes
about 419,000 Python calls (about 580,000 before its label tables were copied
directly), which is the floor under the fastest push edits. A rerun that
refreshes its fork in place releases the capture it discarded before it
returns, so a fork made right after a rerun costs the same as any other
(about 420,000 calls; about 1.25 million before, when that capture's cleanup
ran inside whatever came next, typically the next fork).

`Trace.draw(vis_intervention_mode=...)` visualizes the planned intervention
recipe stored on an intervention-ready trace, such as sites registered with
`log.set(...)` or `log.do(...)`. Each edited node gets a purple ring and a
last row naming the edit (`Δ zero_ablate`), and the header names the edits;
`vis_show_cone=True` (off by default) adds a translucent purple underlay
under the arrows and node outlines the edit changed downstream. Capture-time `intervene=...` calls record
fire metadata on matched layers, but they are not replay plans and are not
drawn as planned sites.

## Bundle

Construct with `tl.Bundle(...)`:

```python
bundle = tl.Bundle({"clean": clean_log, "patched": patched_log}, baseline="clean")
```

Common operations:

| Operation | Use |
| --- | --- |
| `bundle.names` | Member names in order. |
| `bundle["clean"]` | Access one `Trace`. |
| `bundle.node(site)` | Return a `SuperOp` across members after relationship checks. |
| `bundle.compare_at(site)` | Pairwise comparison matrix at a shared site. |
| `bundle.joint_metric(fn)` | Apply a metric to the whole bundle. |
| `bundle.do(...)`, `bundle.attach_hooks(...)`, `bundle.push()`, `bundle.run(model, x)` | Apply mutator/propagation calls to each member (the former `replay`/`rerun` aliases are removed). |
| `bundle.fork(name=None)` | Fork all members into a new bundle. |

Site reads (`most_changed`, `diff_pair`) pair ops across members through the
bundle's structural alignment, not by literal label, so an op renumbered by an
inserted edit op still meets its counterpart. A site that only some members have
(such as the replacement op a module edit inserts) is not scored.

Relationship gates are intentional, and they check two predicates separately:
a model-axis floor (`node` needs `same_param_shapes`; comparison reads need
`shared_graph`) and, for comparison reads, VALUE-LEVEL input identity derived
from retained input payloads. An identity relationship is never accepted as
proof of input equality, unprovable input identity refuses fail-closed
(`bundle_gate_input_identity_unproven`), and members joined by an ordering
relation (`successor_of`/`forked_from`/`escalates`) are not comparison
operands regardless of rank (`bundle_gate_ordering_topology`). Pair-scoped
reads (`diff_pair(a, b)`, `most_changed` baseline pairs) gate only the
operands they actually read.

## Cohort Migration

| TransformerLens pattern | TorchLens pattern |
| --- | --- |
| `act_patch` attribution patching | Attach a `tl.hook_grad(...)` gradient observer at the site, then `log.run(model, x)` under the active spec and score from the captured gradients/outs. |
