# MLX backend (preview)

TorchLens captures [MLX](https://github.com/ml-explore/mlx) forwards with eager wrappers
over `mlx.core`, `mlx.core.fast`, the `mx.array` operators and methods, and the `mlx.nn`
layer classes. The backend is a **preview**: its scorecard ships in the wheel and shows in
`tl.report.doctor()` under `backend_scorecard:mlx`; only a maintainer decision flips the
label to `stable`.

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
is marked `capture_verified=False` with reason `unwrapped_op_in_capture` and a warning says
how many arrays were affected. Callers who want a hard stop instead switch the backend-local
policy:

```python
from torchlens.backends.mlx import unwrapped_policy

with unwrapped_policy("refuse"):
    tl.trace(model, x, backend="mlx")  # raises BackendUnsupportedError, code unwrapped_op_in_capture
```

## Can and cannot

The rows are the backend's capability table in the scorecard's vocabulary: `supported` (the
flag is True and consumed by the suite), `unsupported` (False, and the request refuses with the
named code), `not yet proven` (False, refusal not yet pinned by a passing suite cell).

| Capability | State | Refusal code when requested |
|---|---|---|
| Forward capture (`tl.trace`) | supported | |
| Replay validation (`has_validation_replay`; `validate_trace`, status object with per-op reasons) | supported | |
| Payload materialization (`payload_materialization`; `save` / `load` with array payloads) | supported | |
| Static-label interventions (`can_intervene`; `intervene=`, `stop_after=`) | supported | |
| Intermediate derived gradients (`intermediate_derived_grads`; `grad_options=`) | supported | |
| Backward capture (`backward_capture`; `save_grads`, `backward_ready`) | unsupported | `backward_capture_unsupported` |
| Fastlog (`fastlog`) | unsupported | `fastlog_unsupported` |
| RNG replay (`rng_replay`; `save_rng`) | unsupported | `rng_replay_unsupported` |
| Streaming (`has_streaming`) | unsupported | `streaming_unsupported` |
| Structure-only capture (`structure_only_capture`) | unsupported | `structure_only_capture_unsupported` |
| Explicit `random_seed=` | unsupported | `explicit_seed_unsupported` |
| Trace options outside `module_identity_mode`, `grad_options` | unsupported | `trace_option_unsupported` |
| `mx.compile` / `mx.grad` / `mx.vmap` wrappers as the traced entry | unsupported | `traced_transform_entry_unsupported` |
| `grad_options=` together with `intervene=` or `halt=` | unsupported (capture them in separate calls) | `intervention_with_derived_grads_unsupported` |
| An intervention that returns a non-`mx.array` or changes shape or dtype | unsupported | `intervention_result_invalid` |
| Ops outside the wrapped surface | flagged (`unwrapped_op_in_capture`) or refused under `unwrapped_policy("refuse")` | `unwrapped_op_in_capture` |
| Arrays constructed inside the forward (`mx.array(...)`) | flagged as unwrapped (build constants outside the forward) | `unwrapped_op_in_capture` |

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
