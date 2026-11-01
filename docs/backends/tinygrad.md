# tinygrad backend

`backend="tinygrad"` captures a lazy tinygrad forward from its UOp graph. The backend is
a **preview** (`maturity="preview"`): it fills the shared `BackendProtocol`, runs the shared
conformance suite, and ships its scorecard, but its spellings may still move before the
label flips to `stable`.

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
Every weight the callable closes over is a Python literal here, so the forward consumes no
random state (see the random-init row in the table below).

```python
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

Refusals are typed. A request the preview cannot honour raises `BackendUnsupportedError`
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

Closure weights built from a NumPy array (`Tensor(ndarray)`) stay host-side buffers until
something realizes them; capturing a forward through one refuses `constant_unrealizable`
naming the op. Realize the constant first (`weight.realize()`) or pass it as an input.

## Can and cannot

The table equals the backend's capability summary; `untested` means the flag is `False`
and its refusal cell has not yet passed on the pinned CI leg (not yet proven, not broken).

| Capability | State | Notes |
|---|---|---|
| `has_validation_replay` | supported | UOp replay plus parent perturbation on `DEV=PYTHON` payloads |
| `payload_materialization` | supported | `array_payloads` via `.tlspec`; realized copies per UOp today |
| `intermediate_derived_grads` | supported | opt-in through `GradOptions(intermediate_grads=True)` |
| `backward_capture` | unsupported | refusal `backward_capture_unsupported` |
| `fastlog` | unsupported | refusal `fastlog_unsupported` |
| `can_intervene` | unsupported | refusal `interventions_unsupported` |
| `rng_replay` | unsupported | refusal `rng_replay_unsupported` |
| `has_streaming` | unsupported | refusal `streaming_unsupported` |
| `structure_only_capture` | unsupported | refusal `structure_only_capture_unsupported` |
| Closure constant with a readable buffer | supported | recorded as a parentless `buffer` op with payload (`tinygrad_constant_kind="closure"`) |
| Literal producer (`Tensor.ones`, `Tensor.full`, scalar `const`) | supported | parentless op with payload (`tinygrad_constant_kind="literal"`) |
| Closure constant with no readable buffer (`Tensor(ndarray)` not realized) | unsupported | refusal `constant_unrealizable` naming the op |
| Random-init layers (`tinygrad.nn` default init) or in-forward `Tensor.rand` | unsupported | refusal `rng_state_in_graph`; random-init layers must be realized before capture until realize-once capture lands |
| `random_seed=` | unsupported | refusal `explicit_seed_unsupported` |
| Trace options outside `module_identity_mode`, `grad_options` | unsupported | refusal `trace_option_unsupported` |
| Mid-capture `Tensor.realize()`, `assign()`, TinyJit, input mutation | unsupported | typed `BackendUnsupportedError` |

## Scorecard

`torchlens/backends/tinygrad/scorecard.json` is written by the pinned CI leg
(`python -m torchlens.conformance.scorecard --backend tinygrad --write`) and surfaces as the
`backend_scorecard:tinygrad` row of `tl.report.doctor()`. The roster behind it is
`torchlens/conformance/_rosters/tinygrad.py`: ten fixed-weight `config_built` rows, three
`random_init` rows (refused until realize-once), and two `ecosystem` rows that run only
under `TORCHLENS_CONFORMANCE_ECOSYSTEM=1`.
