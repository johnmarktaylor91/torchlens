# MLX backend

TorchLens captures [MLX](https://github.com/ml-explore/mlx) forwards with eager wrappers
over `mlx.core`, `mlx.core.fast`, the `mx.array` operators and methods, and the `mlx.nn`
layer classes. Its scorecard ships in the wheel and shows in `tl.report.doctor()` under
`backend_scorecard:mlx`.

| Requirement | Value |
|---|---|
| Distribution | `mlx` |
| Supported range | `[0.26, 0.27)`, excluding `0.26.5` (its manylinux wheel ships without `libmlx.so`) |
| Pinned CI version | `0.26.3` |
| Install extra | `pip install "torchlens[mlx]"` |
| Materialization | lazy (realize-once: one `mx.eval` after the forward, before any payload is read) |

## Tutorial

```python
import mlx.core as mx
import mlx.nn as nn
import torchlens as tl

mx.random.seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
x = mx.array([[0.5, -1.0, 2.0, 0.25], [1.0, 1.0, -1.0, 0.0]])

trace = tl.trace(model, x, backend="mlx")
print([op.func_name for op in trace])  # ['input', 'linear', 'gelu', 'linear']
print(trace.output_ops[0].shape)  # (2, 2)

status = tl.backends.spec("mlx").validate_trace(trace)
print(status.state)  # 'passed': every op replayed from its parents' saved payloads
```

Operators and methods are captured as the `mlx.core` function they stand for: `a + b` logs
`add`, `a @ w` logs `matmul`, `v.sum()` logs `sum`, `v[0]` logs `getitem`, `v > 0` logs
`greater`. Modules keep their MLX parameter-tree addresses (`layers.0`); the list or dict
holding sub-modules gets its own module record (`layers`, class `list`), as a torch
`ModuleList` does. An array the forward uses that is neither an input, a parameter nor the output
of a recorded op (a closure array, a module-level array) is recorded as a `constant` source
op with its payload, and the op that used it lists the constant as a parent. `mx.ones`,
`mx.zeros`, `mx.arange` and the other literal producers stay ops with no parents and a saved
payload.

When the forward still reaches an operation no wrapper saw (a function outside the wrapped
surface such as `mx.linalg.norm`, or an `mx.array(...)` built inside the forward), the trace
is marked `provenance.verified=False` with reason `unwrapped_op_in_capture` and a warning says
how many arrays were affected. Callers who want a hard stop instead switch the backend-local
policy:

```python
from torchlens.backends.mlx import unwrapped_policy

with unwrapped_policy("refuse"):
    tl.trace(model, x, backend="mlx")  # raises BackendUnsupportedError, code unwrapped_op_in_capture
```

## Source lines

Each op records the user source lines that ran it, as on PyTorch. `op.source.frames` runs
outermost to innermost: the first frame is the line that called `tl.trace`, the last is the
line in your model (or in the `mlx.nn` layer it called) that ran the op. MLX modules are called
through `__call__` and a model may be a plain function, so the model boundary is the call into
the model rather than a `forward` method. `CaptureOptions(save_source_context=True)` snapshots
the source text at capture time and `source_context_lines` sets how many lines each frame
keeps; both behave as on PyTorch and survive `save` / `load`.

## Value-dependent conditions

`intervene=tl.when(condition, action)` and `stop_after=` accept a static selector (`tl.func`,
`tl.label`, `tl.module`, `tl.in_module` and their `&`, `|`, `~` composites), a `tl.where`
predicate, or a callable taking the op context:

```python
def wide_relu(ctx):
    return ctx.layer_type == "relu" and ctx.shape[-1] >= 8

trace = tl.trace(model, x, backend="mlx", intervene=tl.when(tl.where(wide_relu), tl.zero_ablate()))
partial = tl.trace(model, x, backend="mlx", stop_after=lambda ctx: ctx.layer_type == "gelu")
```

The context is the one PyTorch's live capture gives a condition: the op's type, function
name, label, shape, dtype and module fields are filled, and `ctx.is_scalar_bool` and
`ctx.bool_value` are `None` during the forward on both backends. Nothing is evaluated at the
op, so a run with a value-dependent condition keeps MLX graph fusion and stays realize-once.
`ctx.tensor_requires_grad` has no MLX value (arrays carry no gradient flag; gradients come
from function transforms), so reading it raises `MLXValueUnavailableError`. An exploratory
callable `stop_after` that never fires warns with `stop_after_never_fired_callable` and
records the site in `trace.annotations`; a selector that never fires refuses with
`stop_after_never_fired`, as on PyTorch.

## Edits and gradients

Ten built-in helpers have an MLX application: `tl.zero_ablate`, `tl.scale`, `tl.add`,
`tl.clamp`, `tl.mean_ablate` (the self form, `dim=None`), `tl.replace_with`, `tl.steer`,
`tl.project_onto`, `tl.project_out` and `tl.clip_norm`. Each is written against `mlx.core` with
the PyTorch helper's math, and the same application runs in live capture (`intervene=`), on a
finished trace and in the gradient rerun, so an edit means the same thing on every path. Any
other helper refuses; an edit written as a callable on `mx.array` values covers the rest.

A finished trace takes edits through `trace.fork().do(site, edit)`, with a helper or a
callable. The fork applies the edit at the site, reruns each downstream op once from its
updated parents through the MLX replay adapter, and the edited trace passes the same replay
check as a capture.

`grad_options=` combines with `intervene=` and `stop_after=`. The derived gradients are then
the gradients of the program that actually ran: TorchLens reruns the model under
`mx.value_and_grad` with the same edits applied, checks the rerun reproduces the capture's
outputs, and refuses if it does not. With `stop_after=` the loss is taken on the output of the
op where the capture stopped, and parameters past that op get zero gradients.

`tl.grad(site)` edits the gradient that flows back through a forward op. The forward values
are unchanged; the op's gradient and every gradient upstream of it use the edited value, as
on PyTorch. A gradient site needs `grad_options=`:

```python
from torchlens.backends.mlx import GradOptions

halved = tl.trace(
    model,
    x,
    backend="mlx",
    capture=tl.options.CaptureOptions(
        grad_options=GradOptions(loss_fn=lambda out: mx.sum(out * out), input_grad_argnums=(0,))
    ),
    intervene=tl.when(tl.grad(tl.func("gelu")), tl.scale(0.5)),
)
```

Gradient influence probes (`probe_grad`, `.of(method="gradient")`, both directions) replay the
captured operations between the two units and differentiate them with `mx.vjp` / `mx.jvp`.

A trace saved with `level="executable_with_callables"` also stores one replay recipe per op, so
after `tl.load` it validates by real replay, also in a fresh process without the model.

## Selective save and recording

- `save=` with a static selector is a payload filter at capture time: selected ops keep their
  payloads, and every other value is spilled to a temporary directory for validation and
  removed when the trace is collected, so peak memory falls. Selectors that cannot be decided
  during the forward keep every value in memory and filter after capture.
- `capture=CaptureOptions(structure_only=True)` keeps the full graph with no payloads; value
  validation refuses.
- `tl.record(..., backend="mlx")` emits lean per-op events and keeps values only for the
  selected ops, with `stop_after=`; `storage=` writes them to disk with bounded held bytes.
  `tl.trace(storage=...)` streaming is not available on MLX.

## Can and cannot

The rows are the backend's capability table in the scorecard's vocabulary: `supported` (the
flag is True and consumed by the suite), `unsupported` (False, and the request refuses with the
named code), `not yet proven` (False, refusal not yet pinned by a passing suite cell).

| Capability | State | Refusal code when requested |
|---|---|---|
| Forward capture (`tl.trace`) | supported | |
| Replay validation (`has_validation_replay`; `validate_trace`, status object with per-op reasons) | supported | |
| Payload materialization (`payload_materialization`; `save` / `load` with array payloads) | supported | |
| Interventions and stop sites (`can_intervene`; `intervene=tl.when(...)`, `stop_after=`), static or value-dependent | supported | |
| Source lines (`op.source.frames`, `save_source_context`) | supported | |
| Intermediate derived gradients (`intermediate_derived_grads`; `grad_options=`) | supported | |
| Backward capture (`backward_capture`; `save_grads`, `backward_ready`) | unsupported | `backward_capture_unsupported` |
| Fastlog (`fastlog`) | supported | |
| RNG replay (`rng_replay`; `save_rng`) | unsupported | `rng_replay_unsupported` |
| Streaming (`has_streaming`) | unsupported | `streaming_unsupported` |
| Structure-only capture (`structure_only_capture`) | supported | |
| Explicit `random_seed=` | unsupported | `explicit_seed_unsupported` |
| Trace options outside `module_identity_mode`, `grad_options` | unsupported | `trace_option_unsupported` |
| `mx.compile` / `mx.grad` / `mx.vmap` wrappers as the traced entry | unsupported | `traced_transform_entry_unsupported` |
| `grad_options=` together with `intervene=` or `stop_after=` at op sites (forward edits, `tl.grad` sites, early stop) | supported | |
| `grad_options=` together with a module edit (`tl.module(...)` site) | unsupported | `intervention_with_derived_grads_unsupported` |
| A `tl.grad` site edit without `grad_options=` | unsupported | `intervention_with_derived_grads_unsupported` |
| `tl.grad` of a module site, and one selector mixing forward and gradient sites | unsupported | `backward_capture_unsupported` |
| An edit that reads forward state (`tl.patch_from`) at a `tl.grad` site | unsupported | `edit_signal_mismatch` |
| A built-in helper outside the MLX helper table, live | unsupported | `interventions_unsupported` |
| A built-in helper outside the MLX helper table, on a fork | unsupported | `backend_replay_edit_helper_unsupported` |
| Helper arguments that do not fit the matched output (zero-norm projection direction, direction of the wrong length) | unsupported | `intervention_helper_arguments_invalid` |
| An intervention that returns a non-`mx.array` or changes shape or dtype | unsupported | `intervention_result_invalid` |
| `intervene=` as a raw decision callable (not `tl.when`) | unsupported | `backend_unsupported` |
| A module selector combined with a non-module condition | unsupported | `interventions_unsupported` |
| Replay validation of a selective `save=` capture (torch gap) | passes where torch does not | MLX keeps every unselected value as a replay witness and returns `passed` after a full replay; torch reports the same options as unverified (`parent_payload_missing`) or raises `PayloadUnavailableError` |
| `ctx.tensor_requires_grad` in a condition | no MLX value | raises `MLXValueUnavailableError` |
| Ops outside the wrapped surface | flagged (`unwrapped_op_in_capture`) or refused under `unwrapped_policy("refuse")` | `unwrapped_op_in_capture` |
| Arrays constructed inside the forward (`mx.array(...)`) | flagged as unwrapped (build constants outside the forward) | `unwrapped_op_in_capture` |

Backward capture is a framework limit. MLX builds its backward pass inside C++, so TorchLens
cannot list the backward operations. Gradient values for every recorded operation are
available, and gradient edits work (`tl.grad` sites, below).

## Validation decisions

`validate_trace` never returns a bare bool. The status carries one decision per op with a
reason from the shared closed list: `replay_mismatch`, `parent_payload_missing`,
`no_replayable_function`, `perturbation_insensitive`, `user_intervention_replacement`,
`constant_producer_leaf` (a constant leaf or a literal producer: replayed when a function
exists, exempted from perturbation because it has no tensor parent) and
`constant_parent_payload` (a consumer whose dependency was proven by perturbing a constant
parent), `declared_parents_inconsistent` (an op's declared parents disagree with the arrays its
saved call reads) and `replay_coverage_incomplete` (the replay did not cover every captured
op). A failed status names the first offending op in its message.
