# JAX backend

The JAX backend captures raw functions through their jaxpr, with parameter records
derived from explicit pytree leaves. Equinox and Flax NNX module roots also carry
module attribution. Static-label payload selection, portable array payloads, keyword
inputs, source lines, capture-time edits and derived gradients (including module roots
and closed-over constants) are supported. Transformed roots and unsupported effects
refuse; supported control-flow regions disclose partial replay verification.

## Tutorial

Install the runtime first:

```bash
pip install "torchlens[jax]"
```

```python
# tutorial: executed by tests/test_jax_docs_page.py under the backend_jax marker
import jax.numpy as jnp
import torchlens as tl

def fn(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

params = {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))}
x = (params, jnp.ones((4, 3)))
model = fn
trace = tl.trace(model, x, backend="jax")
print(trace.summary())
status = trace.validation_replay_status
print(status.state)
```

## Can and cannot

| Capability | Status | Note |
|---|---|---|
| backward_capture | supported | `trace.backward(loss_fn)` records one backward pass per call from JAX's own derivative rules; custom-VJP calls and whole control-flow regions refuse. |
| has_validation_replay | supported | Replay distinguishes verified operations and partial regions. |
| fastlog | supported | `tl.record(..., backend="jax")` records per-equation values with the same raw labels as `tl.trace`; see Recording. |
| can_intervene | supported | `intervene=` edits equation outputs at capture and, with `grad_options=`, the gradient at `tl.grad(...)` sites; derived gradients follow the edited program; `stop_after=` halts at the first matching equation. See Edits. |
| rng_replay | unsupported | RNG replay is not declared. |
| payload_materialization | supported | Portable forward and derived array payloads are materialized. |
| has_streaming | unsupported | Streaming capture refuses. |
| structure_only_capture | unsupported | Structure-only capture is not declared. |
| intermediate_derived_grads | supported | Exact intermediate gradients use a separate AD replay. |

### Keyword inputs

`input_kwargs=` works. A keyword whose value is an array or a pytree of arrays traces as
input leaves (input source ops with saved payloads, like positional inputs). Any other
keyword value, such as a Python flag or a string, is bound into the call statically.

### Derived gradients

`grad_options=` differentiates raw function roots, Equinox and Flax NNX module roots,
and functions that close over arrays. On a module root the parameters come from the
module itself (`GradOptions(params=model)` or `params=None`), and the gradients land on
`trace.params` under each parameter's module address. A closed-over array is recorded
as a `constant` source op and is held fixed: it is not differentiated.

### Edits

`intervene=` edits equation outputs while the jaxpr is interpreted, so every later
equation computes from the edited value. A predicate sees the label the op will carry,
the primitive name (`tl.func`), the `named_scope` module stack (`tl.in_module`) and the
concrete output: its shape and dtype, and for a scalar boolean its value, so
value-dependent predicates work. The builtin helpers `zero_ablate`, `scale`, `add`,
`clamp`, `mean_ablate` (the self form) and `replace_with` have JAX-native versions, and a
callable hook `fn(out, *, ctx)` receives the `jax.Array`. An edited trace passes replay
validation: the validator replays each edited equation, applies the same edit, and
compares the result with the saved output. A selector that matches nothing follows
`on_zero_fire` like torch (by default a `trace_zero_fire` warning).

These shapes refuse with a typed error:

- a non-callable `intervene=` (`trace_option_unsupported`);
- a gradient site without `grad_options=` (`backward_capture_unsupported`; see Gradient
  sites);
- `tl.module(...)`, a module's output boundary, which the jaxpr does not carry
  (`trace_option_unsupported`);
- `followed_by` and `preceded_by` selectors (`followed_by_unsupported`);
- `isolate`, `keep_grad`, other helpers, `mean_ablate` with a source, a torch tensor
  passed to a helper, and edits of multi-output equations (`trace_option_unsupported`);
- a hook result that is not an array or changes the shape or dtype
  (`intervention_result_invalid`);
- a predicate that returns something other than a decision, helper, callable or
  `None` (`predicate_return_invalid`).

`stop_after=` stops interpreting at the first matching equation: the trace keeps the
captures up to that frontier, unwinds every nested frame before any later primitive binds,
and reports `is_stopped` and `stop_reason` as on PyTorch.

#### Edits with derived gradients

`intervene=` combined with `grad_options=` differentiates the edited program: the
gradients of parameters and inputs are those of the forward the trace recorded, with each
edit applied at its equation, never of a rerun of the unedited model. This holds on raw
function roots, Equinox and Flax NNX module roots and functions that close over arrays,
and for an edit inside a nested `jax.jit` call. An edit that has no single position in
the program refuses with `intervention_with_derived_grads_unsupported`:

- an edit on an equation inside a `fori_loop`, `scan` or `while_loop` body, or in a
  jitted helper called more than once (the equation runs more than once);
- an edit inside a `lax.cond` branch (the gradient program binds the `cond` whole);
- `GradOptions(intermediate_grads=True)`, whose per-boundary replay is not edit-aware
  yet.

#### Gradient sites

`tl.when(tl.grad(site), edit)` with `grad_options=` edits the gradient that flows into
each equation output `site` matches: the derived gradients of parameters and inputs are
those of `jax.vjp` with the edit applied to that output's cotangent. The forward values
are unchanged, so no op is marked edited. `tl.grad(a) | tl.grad(b)` edits both sites, and
`tl.in_backward_pass(1)` names the one backward pass derived gradients run. The helpers
and callables of Edits apply to the cotangent; an edit must keep its shape and dtype.
These refuse with `backward_capture_unsupported`: a gradient site without
`grad_options=` (JAX computes gradients only as derived gradients), a clause that mixes
`tl.grad(...)` with activation sites, autograd-node selectors, and
`tl.in_backward_pass(n)` with `n > 1`. A gradient site inside a loop body or `cond`
branch refuses like an edit there (`intervention_with_derived_grads_unsupported`).
`trace.do` with a `tl.grad(...)` site on a finished JAX trace refuses with
`run_capability_unavailable`.

### Finished-trace edits

`trace.fork().do(site, edit)` edits a finished JAX trace: the edit is applied at the
site and each downstream equation is replayed once through the JAX replay adapter, so
the fork equals a live `intervene=` capture with the same edit and passes the backend
replay check. The six helpers above and a callable `fn(out, *, ctx)` work; a donor
value from another JAX capture patches in through a callable. These refuse with a
typed error before anything is committed:

- an op downstream of the site that the adapter cannot replay, named in the message:
  the two ops of a `custom_vjp` region and the stateful `rng_uniform` primitive
  (`backend_replay_op_unreplayable`);
- a gradient edit (`replay_gradient_edit_unsupported`) and a differentiable replay
  (`backend_replay_not_differentiable`);
- `tl.patch_from` with a JAX donor array (`backend_replay_edit_helper_unsupported`; the
  callable form `fork.do(site, lambda out, *, ctx: donor)` works).

### Recording

`tl.record(..., backend="jax")` records per-equation values with the same raw labels as
`tl.trace`; only selected values are kept, and `stop_after` stops interpreting at the
frontier.

### Modules inside transforms

Inside an Equinox or NNX module, submodules called in `jax.jit`, a taken `lax.cond`
branch, `lax.scan` or `lax.while_loop` are attributed; each loop iteration is its own module
call, and their returns are unrecorded (`CaptureGapWarning`). `checkpoint`/`remat`,
`custom_vjp` and `shard_map` inside a module refuse typed.

### Source lines

Each op's `source.frames` holds the user frames from the equation's traceback, plus the
call site, in the same order as torch: the line that called `tl.trace`, then the model entry (the function,
or a module root's `__call__`), down to the line that ran the primitive. Frames from
`jax`, `jaxlib` and TorchLens are dropped; model-library frames such as Flax or Equinox
layers stay. `save_source_context=True` snapshots the source text at capture.

### Known gaps

- Loaded traces report replay validation `unavailable`.
- The ecosystem roster is parked until Flax checkpoints have a dependency path:
  transformers 5 removed its Flax models.

## Framework requirement

- Supported range: `jax>=0.4.30`.
- Tested through: JAX `0.10.2`.
- Pinned CI version: JAX `0.10.2`, equinox `0.13.8`, flax `0.12.8`.
