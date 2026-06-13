# TorchLens Agent Guide

TorchLens logs backend-resolved execution into a `Trace`. The stable default is PyTorch
eager capture: run a normal forward pass, record operation metadata and activations, then
inspect the result. Torch function wrapping is lazy in 2.x: `import torchlens` keeps torch
clean, and the first torch capture calls `wrap_torch()` through model preparation. The
wrappers then stay installed until an explicit `torchlens.decoration.unwrap_torch()`.

## Install

```bash
pip install torchlens
pip install -e ".[test]"  # local development with test extras
```

Graphviz rendering needs Graphviz (`apt install graphviz` on Debian/Ubuntu). Optional
extras gate appliance and bridge namespaces; see `pyproject.toml` for the current list.

## Common Patterns

```python
import torchlens as tl

log = tl.trace(model, x, save=tl.func("relu"))
activation = log["linear_1_1"].out
print(log.summary())
print(tl.report.explain(log))
log.draw(order_siblings=True)  # default: verified sibling ordering for dot/unrolled graphs
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

disk_log = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
full_structure = recording.to_trace()
```

Use `backend=` only when the backend is intentionally part of the test or example:

```python
torch_trace = tl.trace(model, x, backend="torch")
assert torch_trace.backend == "torch"
```

Before debugging wrapper-specific failures, run:

```python
print(tl.compat.report(model, x).to_markdown())
```

## Current 2.x Surface

- Top-level `torchlens.__all__` has 60 names: capture, save/load, intervention,
  selectors, helper transforms, observers, validation, and the three main log classes.
- `tl.record(..., save=...)` is the sparse predicate recorder; it returns `Recording`.
  `Recording.to_trace()` cooks the event stream into a full-structure `Trace`, with unsaved
  payload reads rejected explicitly. `tl.record()`/fastlog is torch-only in the backend-v1
  registry. `keep_op=` and `keep_module=` are deprecated aliases.
- `tl.trace(..., backend=None)` routes through `BackendSpec`; explicit backend mismatches,
  unknown names, unsupported capabilities, and audit-only payload reads raise typed backend
  errors. Public backend-neutral metadata lives on `Trace.backend`, `Trace.module_identity_mode`,
  `Trace.param_source`, and record fields such as `dtype_ref`, `device_ref`,
  `backend_address`, and `resolver_status`.
- `Trace.draw(order_siblings=True)` is the default Graphviz sibling-ordering pass for
  forward unrolled graphs; set it to `False` to render the raw dot layout.
- `torchlens._io` and `torchlens.io` own portable `.tlspec` save/load helpers. Manifest
  schema v2 is backend-aware; non-torch preview bundles may be audit-only or metadata-only.
- `torchlens.debug` owns power-user diagnostics such as `bisect_nan` and `hot_path`;
  the submodule is imported as `tl.debug` and is deliberately not in `__all__`.
- `torchlens.bridge` contains optional adapters for Captum, HF, SHAP, SAE Lens, LIT,
  profiler, and related tools.
- Appliance packages `viewer`, `paper`, `notebook`, `llm`, and `neuro` reserve extras
  boundaries; most are stubs except import gating in `notebook` and `neuro`.

## Anti-Patterns

- Do not log `torch.compile`, TorchScript, or `torch.export` artifacts; log the eager
  source module.
- Do not expect `torch.func` / functorch transforms to expose per-element internal ops;
  TorchLens captures transform calls as boundary nodes with provenance edges.
- Do not run captures concurrently across Python threads or worker processes.
- Do not expect fused kernels to expose hidden internal tensors.
- Do not put opaque callables in portable artifacts unless audit-only behavior is
  acceptable.
- Do not add new top-level API names casually; use submodules and deprecation shims.

## Validation Integrity (LOCKED PRINCIPLE — never violate)

The `validation/` pipeline (forward replay, backward checks, metadata invariants) is a
**TRIPWIRE, not a formality.** Its entire purpose is to CATCH capture bugs — ops that
weren't traced, wrong replay inputs, broken metadata, silent corruption.

**NEVER weaken, loosen, exempt, broaden a tolerance, or skip a validation check / invariant
to make a test pass.** A validation failure is the system *working*: ROOT-CAUSE it and fix
the actual bug. Silencing a failing check defeats the entire point and lets exactly the kind
of silent breakage validation exists to prevent ship undetected.

The ONLY legitimate exemption is behavior that is **correct by design and provably outside the
check's contract** (e.g. a user-injected intervention tensor genuinely has no traceable
function to replay). Even then the carve-out must be NARROW (only the intended case) and must
NOT mask the unintended case — e.g. an auto-synthesized placeholder op appearing during PLAIN
capture is a capture bug, and validation must STILL fail on it.

**Incident (2026-06-02):** `test_mistral` / `test_audio_vits` emitted functionless
`interventionreplacement` placeholder ops during plain tracing — a real capture gap (ops
TorchLens failed to wrap). An exemption was added to the metadata invariant to pass them. That
was backwards: it disarmed the tripwire. The correct fix is to make capture actually trace those
ops so no placeholder is synthesized during plain capture; any replacement-op exemption must be
scoped to GENUINE user interventions only.

## Keep the glossary + docs in lockstep with code (LOCKED)

The glossary is the **canonical** API spec (vault `brain/projects/torchlens/reports/<date>-glossary-vN/torchlens_glossary.md`); code conforms to it (spec-drives-code). A rename is not *done* until the docs match too:

- **Rename / add / remove any PUBLIC name** (dataclass field, `@property`, method, top-level `tl.*` name, kwarg) → in the SAME change, update: (1) the **glossary** entry (canonical), (2) this `CLAUDE.md` + `AGENTS.md` examples, (3) the audit notebooks (`notebooks/audit/`) and `examples/` that use it.
- A change that touches code but leaves the glossary/docs stale is **INCOMPLETE.** This is exactly how the v7 `memory → activation_memory` gap and the stale `log_forward_pass`/`vis_opt` examples slipped through.
- After a rename/conformance sprint: re-file the updated glossary to the vault (it supersedes the prior dated version), and confirm a `grep` of every old name is clean across `torchlens/`, `tests/`, `examples/`, `notebooks/`, AND the glossary itself.

## Testing Tiers

```bash
ruff check . --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short
pytest tests/ -m "not slow" -x --tb=short  # for public API or boundary changes
```

Use `pytest.importorskip()` for optional migration dependencies. Keep tests
deterministic and run documentation examples when they are meant to be executable.
