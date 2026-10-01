# TorchLens for AI coding agents

This page is a compact map for agents writing or reviewing TorchLens code. Prefer the current
v2 spelling: `tl.trace(..., backend=None)`, predicate `save=...`, `intervene=...`,
`storage=...`, and grouped `capture=tl.options.CaptureOptions(save_grads=...)` (the flat
flat `save_grads=` kwarg is removed).

Backend note: `backend=None` preserves the torch eager default, and EVERY preview backend
auto-routes genuine framework models (MLX, JAX, tinygrad, Paddle, TensorFlow).
`tl.record()`/fastlog and true backward capture are torch-only in backend v1. Backend-neutral
metadata lives on `Trace.backend`, `Trace.module_identity_mode`, `Trace.param_source`,
`Trace.derived_grads`, `Trace.intermediate_derived_grads`, `Trace.payload_load_status`,
`Trace.validation_replay_status`, `dtype_ref`, `device_ref`, `backend_address`, and
`resolver_status`.
JAX leaf gradients are requested with `tl.backends.jax.GradOptions`; they are derived by a
second functional AD run and never populate backward-pass or op-gradient surfaces.
JAX intermediate derived gradients are requested with
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)`; they run a separate zero-tap AD
replay plus per-boundary VJP and finite-difference oracle. Only `status == "exact"` records reach
`Trace.intermediate_derived_grads` and `Op.derived_grad`; producer drift degrades to leaf grads plus
an empty intermediate accessor.
tinygrad leaf gradients use `tl.backends.tinygrad.GradOptions`; they are bracketed
`DEV=PYTHON` leaf-gradient runs. tinygrad op-level intermediate derived gradients are requested
with `GradOptions(intermediate_grads=True)` and are exposed only through
`Trace.intermediate_derived_grads` and `Op.derived_grad`; they never populate true backward
surfaces.
MLX leaf gradients use `tl.backends.mlx.GradOptions`; they run a second
`mx.value_and_grad` pass with module param rebinding and expose `Trace.derived_grads` after
the raw-output honesty guard passes. MLX intermediate derived gradients are requested with
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)`; the same AD replay installs
custom-VJP taps through the MLX wrappers, then exposes only exact grouped-signature matches whose
replacement-gradient and perturbation oracle passes.
Paddle leaf gradients use `tl.backends.paddle.GradOptions`; they run a second guarded Paddle AD
pass and expose `Trace.derived_grads` without setting `has_backward_pass`. Paddle intermediate
derived gradients are requested with
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)` and expose only exact replay
matches through `Trace.intermediate_derived_grads` and `Op.derived_grad`. Paddle capture is
dygraph/eager only; in-place mutation, RNG, tensor-derived Python scalar escapes, and active
stochastic/training composites are denied in the preview.
TensorFlow uses `backend="tf"` / `backend="tensorflow"` for the Keras-3 / TF>=2.16 preview when
`keras.backend.backend() == "tensorflow"`. Eager `op_callbacks` capture is the primary shipped
mechanism and records real values, real taken-branch control flow, op-level records, and
Keras/`tf.Module` module stacks. Graph-only FuncGraph fallback is the static-mode design for
compiled/SavedModel-style entries. SHIPPED for eager entries: static-label `intervene=`
(two-level writable layer, fail-closed site reachability) and leaf + exact T1 intermediate
derived gradients via `tl.backends.tf.GradOptions`. Still deferred: `halt=`/`recipes=`, true
backward capture, and value-dependent predicates (graph-only captures also refuse
`grad_options` typed).
JAX `array_payloads` saves round-trip typed PRNG keys and fully addressable single-host sharded
arrays by value. `jax_named_sharding` metadata is a reconstructible JSON-primitive contract,
but default load stays value-only; explicit re-sharding goes through `PayloadLoadHints` /
`JaxPayloadLoadHint`, not `map_location`. Multi-host/unaddressable sharded arrays fail closed.
The retained-memory baseline for Op `__slots__` lives at
`benchmarks/perf/slots_baseline.md` and records roughly 10-15% lower trace-level retained memory
on the measured fixtures.

## Public surface map

`torchlens.__all__` currently exposes 114 names. The most-used ones, grouped by job (this
table is a selection, not the full list — read `torchlens.__all__` for that):

| Job | Names |
| --- | --- |
| Capture and sparse recording | `trace`, `fastlog`, `span`, `tap` (the former `record_span` alias is removed) |
| Persistence and bundles | `load`, `save`, `bundle`, `Bundle`; schema-v2 manifests add `backend`, `backend_runtime`, and `payload_policy` |
| Replay and edits | `do`, `push`, `push_from`, `run` (the former `replay`/`replay_from`/`rerun` aliases are removed) |
| Data objects | `Trace`, `Layer`, `Op`, `Quantity`, `Bytes`, `Duration`, `Flops`, `Macs` |
| Site discovery | `label`, `func`, `func_transform`, `module`, `contains`, `where`, `in_module`, `head`, `output`, `grad_fn`, `facet` |
| Predicate composition | `followed_by`, `preceded_by`, `without_op`, `when` (the former `intervening` alias is removed) |
| Activation helpers | `zero_ablate`, `mean_ablate`, `resample_ablate`, `replace_with`, `swap_with`, `steer`, `scale`, `clamp`, `noise`, `project_onto`, `project_off`, `splice_module` |
| Backward helpers | `bwd_hook`, `grad_zero`, `grad_scale`, `grad_clamp`, `grad_noise`, `grad_clip` |
| Extraction and validation | `pluck`, `extract`, `extract_dataset`, `validate` (the former `peek` and `batched_extract` aliases are removed); disk-mode `extract_dataset` writes a self-describing manifest, resumes with `resume=True`, and loads back via `torchlens.dataset_extraction.load_extraction` (DOCUMENTED-UNSTABLE spellings) |
| Subpackages | `facets`, `fastlog` |

Submodules such as `tl.report`, `tl.stats`, `tl.viz`, and `tl.compat` are available as attributes
but are deliberately not listed in `__all__`.

## Predicate language

Predicates describe sites, not final labels guessed from a previous run. Compose them directly:

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(18, 3)).eval()
x = torch.randn(1, 1, 5, 5)

conv_before_relu = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
trace = tl.trace(
    model,
    x,
    save=conv_before_relu,
    lookback=4,
    lookback_payload_policy="detached_raw",
)

assert trace.find_sites(tl.func("conv2d")).first().out.shape == (1, 2, 3, 3)
```

Use exact `tl.label(...)` only after discovery when reproducibility matters. For broad capture,
prefer `tl.func(...)` or `tl.in_module(...)`.

## Common recipes

Pull one activation:

```python
import torch
from pathlib import Path
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
x = torch.randn(2, 4)

trace = tl.trace(model, x, save=tl.func("relu"))
relu_out = trace.find_sites(tl.func("relu")).first().out

assert relu_out.shape == (2, 4)
```

Run a capture-time intervention:

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

trace = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)

assert torch.count_nonzero(trace.find_sites(tl.func("relu")).first().out) == 0
```

Capture gradients:

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.options import CaptureOptions


model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1)).eval()
x = torch.randn(2, 3, requires_grad=True)

trace = tl.trace(model, x, capture=CaptureOptions(save_grads=True, backward_ready=True))
loss = trace[trace.output_layers[0]].out.sum()
loss.backward()

relu_site = trace.find_sites(tl.func("relu")).first()
assert relu_site.grad is not None
```

For forward-only analysis where backward will never be run, use
`tl.trace(..., inference_only=True)` or `CaptureOptions(inference_only=True)` to capture under
`torch.no_grad()`. Do not combine it with `backward_ready=True`, `save_grads=...`, or
`intervention_ready=True`.

Draw after capture:

```python
import torch
from pathlib import Path
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(2, 2), nn.Tanh()).eval()
x = torch.randn(1, 2)

trace = tl.trace(model, x, save=tl.func("tanh"))
graph = trace.draw(
    view="unrolled",
    vis_outpath=str(Path(DOCS_TMPDIR) / "agent-graph"),
    vis_save_only=True,
    vis_fileformat="dot",
    return_graph=True,
)

assert graph is not None
```

## Machine-readable trace dump and budgeted reports

Two agent-facing spellings (both DOCUMENTED-UNSTABLE pending the naming ratification
sprint) describe the same surface the rest of this page drives:

- `trace.to_agent_json()` returns a JSON-serializable, self-describing dump under the
  `torchlens.agent_trace.v1` schema: capture outcome/verification honesty facts, counts,
  execution-ordered pass-qualified op rows with graph edges, the module hierarchy, and an
  embedded `guide` block that maps every record back to the live spelling to call next.
  Tensor payloads are never inlined; read them as `trace[layer_label].out`. Pass
  `max_ops=N` to cap op rows — any omission is disclosed in the `truncation` block, and
  the `counts` block stays full-capture truth.
- `tl.report.explain(trace, max_tokens=N)` budget-prunes the text report by whole
  sections (low-value first), disclosing every drop in a trailing `Truncation` section.
  The `Capture status` honesty facts and partial-capture failure evidence are never
  dropped: a budget below that floor returns the floor plus a disclosure instead of a
  misleading fragment. `max_tokens` refuses with `format="json"` (that schema is
  fixed-shape and already minimal).

```python
import json

import torch
from torch import nn

import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)
trace = tl.trace(model, x, save=tl.func("relu"))

dump = trace.to_agent_json()
assert dump["schema"] == "torchlens.agent_trace.v1"
assert json.loads(json.dumps(dump)) == dump
relu_row = next(row for row in dump["ops"] if row["layer_label"] == "relu_1_2")
assert relu_row["saved"] is True
assert trace[relu_row["layer_label"]].out.shape == (2, 4)

budgeted = tl.report.explain(trace, max_tokens=120)
assert "Capture status" in budgeted
assert "Truncation" in budgeted  # drops are disclosed, never silent
```

## MCP server (`torchlens.bridge.mcp`)

For hosts that speak the Model Context Protocol, `python -m torchlens.bridge.mcp` runs a
local stdio server (extra: `pip install torchlens[mcp]`, requires `mcp>=2.0`;
DOCUMENTED-UNSTABLE). It exposes read-only tools over SAVED `.tlspec` artifacts and the
runtime environment — the same public surface as the rest of this page, never a parallel
API, with no user-code execution and no mutation:

- `torchlens_doctor` — environment health check (`tl.utils.doctor()` rows).
- `torchlens_api_map` — machine-readable index of `torchlens.__all__` (name, kind, first
  docstring line) plus the deliberately-unlisted submodules.
- `torchlens_load_overview` — `tl.load(path)` + `trace.summary()` + capture honesty facts.
- `torchlens_agent_dump` — `trace.to_agent_json(max_ops=...)` over a saved artifact.
- `torchlens_explain` — `tl.report.explain(trace, max_tokens=..., audience=...)`.

Live capture stays a Python-process concern: run `tl.trace(...)` in code, `tl.save(...)`
the result, and point the tools at the artifact. The tool registry is importable without
the `mcp` package for direct in-process use:

```python
import torchlens.bridge.mcp as tlmcp

api_map = tlmcp.call_tool("torchlens_api_map")
assert api_map["schema"] == "torchlens.api_map.v1"
assert {row["name"] for row in api_map["names"]} == set(__import__("torchlens").__all__)
```

## Anti-patterns

- Do not trace `torch.compile`, `torch.jit`, or `torch.export` artifacts. Trace the original
  Python `nn.Module`.
- Do not expect per-element eager ops inside `torch.func` / functorch transforms; TorchLens records
  transform boundaries conservatively.
- Do not call TorchLens capture from multiple threads or worker processes. Capture is single-process
  and single-threaded because it uses global toggle state.
- Do not use deprecated `layers_to_save`, `vis_mode`, or `hooks` spellings in new code
  (`keep_op=`/`keep_module=` are fully removed and raise TypeError)
  unless you are intentionally testing compatibility.
- Do not assume unsaved payloads can be read later. Re-trace with a wider `save=` predicate or use
  torch `tl.record(...).to_trace()` with the records you need. JAX/tinygrad/Paddle/TF `.tlspec` saves
  materialize array payloads, but loaded traces cannot replay-validate stripped runtime captures;
  check `trace.validation_replay_status` (`ValidationReplayStatus`) and
  `trace.payload_load_status`.
  A live importer-owned region can make replay status `unverified`: the trace is available and
  replayable checks passed, but per-op replay is partial and `bool(status)` raises.
