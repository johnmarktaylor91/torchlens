# tinygrad backend

`backend="tinygrad"` captures a lazy tinygrad forward from its UOp graph. It fills the
shared `BackendProtocol`, runs the shared conformance suite, and ships its scorecard.

| Requirement | Value |
|---|---|
| Distribution | `tinygrad` |
| Supported range | `>=0.13.0, <0.14` |
| Pinned CI version | `0.13.0` |
| Install extra | `torchlens[tinygrad]` |
| Materialization | `lazy` (realize-once rule applies) |
| Device | `DEV=PYTHON` for live payloads, replay validation and derived gradients |

## Tutorial

Trace a pure tinygrad callable, read the recorded ops, and validate the trace by replay.
The capture evaluates the forward's lazy graph once, after the forward returns, and every
payload comes from that one evaluation. Run this in a fresh process with `DEV=PYTHON`.

```python
import os

os.environ["DEV"] = "PYTHON"

from tinygrad import Tensor
import torchlens as tl

weight = Tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [1.0, -1.1, 1.2]])
bias = Tensor([0.0, 0.1, -0.1])

def mlp(x):
    return (x.matmul(weight) + bias).relu()

x = Tensor([[-1.0, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]])
trace = tl.trace(mlp, x, backend="tinygrad")

print(len(trace), "ops recorded")
print(trace.output_ops[0].out.numpy())          # equals mlp(x).numpy()
assert trace.validate_forward_pass([mlp(x)]) is True
```

The protocol object is the surface the conformance suite drives. Its `validate` returns the
replay status object with per-op decisions instead of a boolean:

```python
protocol = tl.backends.spec("tinygrad").protocol()
status = protocol.validate(trace, [mlp(x)])
print(status.state)          # "passed"
print(status.message)        # names the first offending op when it fails
```

Inputs may be positional or keyword tensors (`input_kwargs=` or plain keyword arguments to
`tl.trace`). Each op records the user source lines it ran from in `op.source.frames`,
outermost first, as the torch backend does:

```python
frames = trace.output_ops[0].source.frames
print(frames[-1].func_name, frames[-1].line_number)   # the line in mlp that made the op
```

A static-selector edit replaces a matched op's value with a lazy expression over it, so
the capture still realizes once and the edit's own ops are not recorded. The edited op keeps
its label and parents, downstream ops see the edited value, and the trace validates:

```python
edited = tl.trace(mlp, x, backend="tinygrad", intervene=tl.when(tl.func("where"), tl.scale(0.5)))
relu = next(op for op in edited.ops if op.is_edited)
print(relu.out.numpy())                          # half of mlp(x).numpy()
assert edited.validate_forward_pass([mlp(x) * 0.5]) is True
```

Refusals are typed. A request the backend cannot honour raises `BackendUnsupportedError`
with a `code` from the shared registry before any framework side effect:

```python
from torchlens.errors import BackendUnsupportedError

try:
    tl.trace(
        mlp,
        x,
        backend="tinygrad",
        capture=tl.options.CaptureOptions(module_identity_mode="nope"),
    )
except BackendUnsupportedError as error:
    print(error.code)        # "module_identity_mode_unsupported"
```

Layers with random default init (`tinygrad.nn.Linear`, `Conv2d`) capture as they are: their
weights are realized in the capture's single evaluation, so the trace and the random-number
state afterwards match one eager forward. Random numbers drawn inside the traced callable
(`Tensor.rand`, dropout) refuse `rng_state_in_graph`, since replay would draw them again;
draw them before capture and pass them in. When comparing a capture with an eager forward,
read the eager output before reading tinygrad's RNG counters: in tinygrad itself, realizing
the counters first changes the values a still-lazy random weight takes.

## Can and cannot

The table equals the backend's capability summary; `untested` means the flag is `False`
and its refusal cell has not yet passed on the pinned CI leg (not yet proven, not broken).

| Capability | State | Notes |
|---|---|---|
| `has_validation_replay` | supported | UOp replay plus parent perturbation on `DEV=PYTHON` payloads |
| `payload_materialization` | supported | `array_payloads` via `.tlspec`; one realization per capture (realize-once) |
| `intermediate_derived_grads` | supported | opt-in through `GradOptions(intermediate_grads=True)` |
| `backward_capture` | unsupported | refusal `backward_capture_unsupported` |
| `fastlog` | unsupported | refusal `fastlog_unsupported` |
| `can_intervene` | supported | static selectors (`tl.func`, `tl.label`, `tl.label_contains`, `tl.in_module` and their `&`, `|`, `~` combinations) with `tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`, `tl.replace_with` or a callable transform; see Static-selector edits |
| `rng_replay` | unsupported | refusal `rng_replay_unsupported` |
| `has_streaming` | unsupported | refusal `streaming_unsupported` |
| `structure_only_capture` | unsupported | refusal `structure_only_capture_unsupported` |
| Closure constant with a readable buffer | supported | recorded as a parentless `buffer` op with payload (`tinygrad_constant_kind="closure"`) |
| Literal producer (`Tensor.ones`, `Tensor.full`, scalar `const`) | supported | parentless op with payload (`tinygrad_constant_kind="literal"`) |
| Closure constant built from a NumPy array (`Tensor(ndarray)`, not yet realized) | supported | realized in the capture's single evaluation; a value tinygrad cannot realize refuses `constant_unrealizable` naming the op |
| Random-init layers (`tinygrad.nn` default init) | supported | weights realized in the capture's single evaluation; values and RNG state equal one eager forward |
| In-forward random numbers (`Tensor.rand`, dropout inside the traced callable) | unsupported | refusal `rng_state_in_graph` |
| `random_seed=` | unsupported | refusal `explicit_seed_unsupported` |
| Trace options outside `module_identity_mode`, `grad_options` | unsupported | refusal `trace_option_unsupported` |
| Mid-capture `Tensor.realize()` or TinyJit execution | unsupported | refusal `mid_capture_execution_unsupported` |
| Input mutation (`Tensor.assign()`, `Tensor.replace()`, setitem on an input) | unsupported | refusal `input_mutation_unsupported` |
| A callable that returns no tinygrad Tensor | unsupported | refusal `tensor_output_required` |
| Keyword tensor inputs (`input_kwargs=` or keyword arguments) | supported | each keyword leaf is a trace input |
| User source lines (`op.source.frames`, `save_source_context=`) | supported | user frames from the `tl.trace` call site to the op's line; TorchLens and tinygrad frames are skipped |
| No tinygrad Tensor input (positional or keyword) | unsupported | refusal `tinygrad_input_required` |
| Replay validation or derived gradients on an audit-only trace | unsupported | refusal `live_payloads_required` |

## Static-selector edits

`trace(intervene=tl.when(selector, edit))` runs the forward twice. The first run is a plain
capture: the selector is matched against its finished ops, as `save=` matches, and each
matched op is mapped to the Tensor calls that returned it. The second run replaces those
calls' results with the edit applied lazily, so it still realizes once. The edited op's
payload is the edited value, and replay validation replays the original op, applies the
edit and compares. tinygrad interns identical expressions, so an edit at an op applies at
every call that returned that op.

These stay refused, each with a typed code:

| Request | Refusal |
|---|---|
| Value-dependent predicates (`tl.where(...)`, a lambda over values) | `interventions_unsupported`: a lazy forward has no values before its one realization |
| `stop_after=` | `interventions_unsupported` |
| Module-output selectors (`tl.module(...)`) and ops no Tensor call returned | `interventions_unsupported` |
| Gradient sites (`tl.grad(...)`) | `backward_capture_unsupported` |
| `grad_options=` together with `intervene=` | `intervention_with_derived_grads_unsupported` |
| An edit that changes shape or dtype | `interventions_unsupported` |
| A forward reading lazy random values (unrealized default-init weights) | `rng_state_in_graph`: the two runs would draw them twice; realize the parameters first |

A selector that matches nothing warns, as on torch, and returns the plain capture.

## Scorecard

`torchlens/backends/tinygrad/scorecard.json` is written by the pinned CI leg
(`python -m tests.conformance.scorecard --backend tinygrad --write`) and surfaces as the
`backend_scorecard:tinygrad` row of `tl.report.doctor()`. The roster behind it is
`tests/conformance/_rosters/tinygrad.py`: ten fixed-weight `config_built` rows, three
`random_init` rows (default-init Linear, Conv2d and LayerNorm), and two `ecosystem` rows that run only
under `TORCHLENS_CONFORMANCE_ECOSYSTEM=1`.
