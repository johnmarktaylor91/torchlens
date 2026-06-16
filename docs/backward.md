# Backward Capture

TorchLens records backward execution with two clocks. The forward event stream remains the
source for operations, activations, modules, and graph position. Backward execution is written
to a runtime-only sidecar diary, then projected into `BackwardPass`, `GradFn`, `GradFnCall`, and
per-op gradient records. Saved `.tlspec` files persist the projections and tensor blobs, not the
live diary.

Backend status: torch is the only true-backward capture backend in this checkout. MLX and
TensorFlow declare backward capture unsupported. JAX, tinygrad, and Paddle expose derived-gradient previews:
leaf gradients from a second backend AD execution over declared leaves, not
`GradFn`/`GradFnCall` records or live backward hooks. Real non-torch backward graphs remain
research.

## Capturing Gradients

Use `save_grads=` to choose which operation gradients are retained:

```python
trace = tl.trace(model, x, save_grads=True)
trace.log_backward(trace[trace.output_layers[0]].out.sum())

relu_trace = tl.trace(model, x, save_grads=tl.func("relu"))
relu_trace.log_backward(relu_trace[relu_trace.output_layers[0]].out.sum())
```

`save_grads=True` saves all observed op gradients. `False` or `None` observes backward structure
without retaining gradient tensors. Predicate expressions use the same forward selector language,
plus backward selectors such as `tl.grad_fn(...)`, `tl.grad_fn_label(...)`,
`tl.intervening()`, `tl.grad_input()`, `tl.grad_output()`, and `tl.in_backward_pass(k)`.

`storage=tl.to_disk(path)` streams retained gradient payloads into a `.tlspec` bundle. The
standing trace policy can be widened or narrowed per trigger:

```python
trace.log_backward(loss, save_grads=tl.func("attn") & tl.in_backward_pass(2))
```

Plain `loss.backward()`, `torch.autograd.backward`, `torch.autograd.grad`,
`trace.log_backward(...)`, and recording backward calls are all triggers when the autograd graph
belongs to a live TorchLens trace. If an engine path bypasses those wrappers, tensor hooks open an
implicit pass. Consecutive bypassed backwards may merge because hook events alone do not reveal a
hard engine boundary; use TorchLens triggers for exact pass boundaries.

## Backprop To The Stimulus

For activation maximization, adversarial examples, and GAN inversion, the optimized stimulus can
be either a plain leaf tensor or an `nn.Parameter` owned by an outer optimizer:

```python
z = torch.nn.Parameter(torch.randn(1, latent_dim))
trace = tl.trace(generator, z, save_grads=True)

loss = score(trace[trace.output_layers[0]].out)
trace.log_backward(loss)

optimizer.step()  # reads z.grad
stimulus_grad = trace[trace.input_layers[0]].grad
```

TorchLens clones forward inputs before capture so it can label and move tensors without mutating
the caller's object. The clone is not detached, so gradients still propagate back to the original
leaf tensor or `nn.Parameter`; `x.grad` / `z.grad` is the optimizer-facing gradient. TorchLens also
records the same input gradient on `trace[trace.input_layers[0]].grad` when gradient saving covers
the input. If the original input was an `nn.Parameter`, the input op records
`input_was_parameter=True`; it is still represented as an input node in the graph, not as a model
parameter.

## Records

`trace.backward_passes` is indexed positionally with Python's 0-based rules, and supports named
1-based lookup with `trace.backward_passes.for_pass(k)`. `trace.last_backward_pass` returns the
latest projected pass.

`BackwardPass.backward_call_context` stores the Python source location where
`trace.log_backward(loss)` or `trace.backward(loss)` was invoked. It is a `FuncCallLocation`, so it
exposes fields such as `file`, `line_number`, and `func_name`. Implicit passes inferred from
orphan tensor-hook emissions have no explicit TorchLens call site and leave
`backward_call_context` as `None`.

`GradFn` records represent distinct autograd node objects. Labels use backward-native names such
as `addmm_back_1_4`; the forward correspondence is metadata, not part of the label. A fired node
has local dense call labels like `addmm_back_1_4:2`. Positional access remains 0-based, while
colon suffixes and pass numbers are 1-based:

```python
grad_fn = trace.grad_fns["addmm_back_1_4"]
first_call = grad_fn.calls[0]
second_pass_call = grad_fn.calls.for_pass(2)
```

`op.grads` is a local-dense accessor over saved per-pass gradient records. `op.grad` returns a
tensor only when exactly one gradient was saved. With zero or several saved gradients it raises a
specific actionable error; use `op.grads[...]` or `op.grad_for(bwd=k)` for multi-pass traces.

`param.grad` remains the live accumulated PyTorch tensor. Per-pass parameter increments are only
available when autograd fires `AccumulateGrad`, so `torch.autograd.grad` calls that do not
accumulate into `.grad` do not populate parameter increment records.

## Multiple Passes And Accumulation

Each backward engine invocation becomes one `BackwardPass`. Loops are not backward passes and do
not affect backward naming. Retained op gradients default to all observed passes when gradient
saving is enabled, so repeated backwards over a retained graph produce multiple entries in
`op.grads`.

`BackwardPass.save_grads_policy` records the policy snapshot active for that pass. This makes
mixed policies auditable when one trace has several backward triggers.

## Higher Order

When `create_graph=True`, TorchLens re-walks newly-created autograd nodes at bracket end.
Creator-attributed nodes get `order = creator.order + 1`; forward-created nodes are order 1.
Mixed-order passes are normal because earlier-order nodes may refire while higher-order nodes are
created. When attribution cannot be resolved deterministically, `order` is `None` and
`BackwardPass.order_attribution_coverage` reports the resolved fraction.

## Intervention And Replay

Backward interventions use `intervene=tl.when(<backward selector>, <grad helper>)` on `tl.trace`.
Helpers include `tl.grad_zero`, `tl.grad_scale`, `tl.grad_clip`, `tl.grad_noise`, `tl.grad_clamp`,
and custom hooks.

Differentiable replay creates a new trace:

```python
patched = source_trace.replay(..., differentiable=True)
patched.log_backward(patched[patched.output_layers[0]].out.sum())
```

Saved frontier tensors enter replay as fresh leaves, so gradients flow within the replayed
computation and do not leak into the source run's buffers.

Backward-only replay from a saved trace is intentionally deferred. A saved TorchLens backward
projection does not contain PyTorch's live autograd graph or saved-tensor closures; reconstructing
that would require a TorchLens-side autograd engine. Live multi-pass capture, live backward
intervention, and differentiable replay cover the useful in-session workflows.

## Module Containment

Paired backward nodes inherit the paired forward op's module containment. `AccumulateGrad` nodes
use the parameter's module home. Intervening in-forward nodes are inferred from neighboring
op-anchored nodes when possible. Post-forward loss-construction nodes before the first op-anchored
node carry no module membership. `module_membership_source` is `"paired"`, `"inferred"`, or
`None`.

## Validation

Backward validation checks parameter-gradient parity, module-output gradient parity through a
separate stock run with minimal tensor hooks, flow consistency for non-intervened quantities, and
structural invariants. Invariants include dense pass indices, local-dense call ordinals, one
record per retained autograd object, resolved order chains, label grammar, and containment flags.

## Dataframes And Serialization

`trace.to_pandas()` never reads loud gradient properties. Ambiguous op gradients appear as
`None` in the `grad` column with `num_saved_grads` set. Backward tables are available through
`trace.grad_fns.to_pandas()`, `trace.grad_fn_calls.to_pandas()`, and
`trace.backward_passes.to_pandas()`.

Unified `.tlspec` manifests include `backward_summary`, gradient blob kinds in `body_index`, and
the public policy for old bundles: legacy class remaps are accepted at load, while removed
backward gradient configuration fields are dropped during state normalization.

## Limitations And Costs

Torch keeps strong references to discovered autograd wrapper objects until `trace.cleanup()` or
trace deletion, preventing id reuse from corrupting registry lookups. This can grow memory for
large or higher-order graphs; PyTorch also warns about reference cycles with `create_graph=True`.

Live traces register their known autograd roots so wrapped engine calls can find owning traces.
If you keep a trace only for analysis while continuing training, call `trace.disarm_triggers()` to
detach it from engine interception and tensor-hook emission.

MLX currently declares backward capture unsupported and raises tiered backend errors. Torch-only
features such as autograd node records and live backward intervention remain out of reach unless
another backend exposes hookable backward graphs. JAX exposes `trace.derived_grads`, populated by
`tl.backends.jax.GradOptions` through a second pure `jax.value_and_grad` run over
`fn(params, *inputs)`, and can optionally expose exact saved-op records through
`trace.intermediate_derived_grads` plus read-only `op.derived_grad` with
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)`. The JAX intermediate producer
runs as a separate zero-tap AD replay and only exposes oracle-confirmed `status == "exact"`
records. MLX exposes `trace.derived_grads`, populated by
`tl.backends.mlx.GradOptions` through a second pure `mx.value_and_grad` run that rebinds
`mlx.nn.Module` params with `model.update(params)` and refuses records unless the AD-rerun raw
output matches the captured output. MLX can optionally expose exact saved-op records through
`trace.intermediate_derived_grads` plus read-only `op.derived_grad` with
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)`. The MLX intermediate producer
uses custom-VJP identity taps in that auxiliary AD replay, attaches by unambiguous grouped
signatures, skips duplicates, and only exposes oracle-confirmed `status == "exact"` records.
tinygrad exposes `trace.derived_grads`, populated by `tl.backends.tinygrad.GradOptions` through a
bracketed `DEV=PYTHON` leaf-gradient run, and can optionally expose exact unambiguous per-op
records through `trace.intermediate_derived_grads` plus read-only `op.derived_grad`. These are not
Paddle exposes `trace.derived_grads`, populated by `tl.backends.paddle.GradOptions` through a
second guarded Paddle AD pass over module params and selected inputs, and can optionally expose
exact saved-op records through `trace.intermediate_derived_grads` plus read-only
`op.derived_grad`. Paddle traces keep `has_backward_pass=False`; this is not true backward
capture. TensorFlow currently defers true backward capture and T1/intermediate derived gradients;
TF traces are forward eager-capture records only. `trace.log_backward(...)`,
`trace.backward_passes`, `trace.saved_grad_ops`, and `op.grads` raise on JAX, MLX, tinygrad,
Paddle, and TensorFlow traces.

Future follow-ups are filed for real per-fire timing via prehooks and better implicit-boundary
detection. Current `GradFnCall` timing is a single hook timestamp, and implicit passes are closed
at synchronization points rather than at an engine boundary that TorchLens did not observe.
