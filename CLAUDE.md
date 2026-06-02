# TorchLens Agent Guide

TorchLens logs PyTorch eager execution: run a normal forward pass, record operation
metadata and activations, then inspect the resulting `Trace`. Torch function
wrapping is lazy in 2.x: `import torchlens` keeps torch clean, and the first capture
calls `wrap_torch()` through model preparation. The wrappers then stay installed until
an explicit `torchlens.decoration.unwrap_torch()`.

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

log = tl.trace(model, x)
activation = log["linear_1_1"].out
print(log.summary())
print(tl.report.explain(log))
```

Use selectors for intervention discovery:

```python
log = tl.trace(model, x, intervention_ready=True)
sites = log.find_sites(tl.func("relu"))
edited = log.fork("zero_relu")
edited.attach_hooks(tl.func("relu"), tl.zero_ablate())
edited.rerun(model, x)
```

Before debugging wrapper-specific failures, run:

```python
print(tl.compat.report(model, x).to_markdown())
```

## Current 2.x Surface

- Top-level `torchlens.__all__` has 40 names: capture, save/load, intervention,
  selectors, helper transforms, observers, validation, and the three main log classes.
- `torchlens.fastlog` is the sparse predicate recorder; it returns `Recording`, not a
  faithful `Trace`.
- `torchlens._io` and `torchlens.io` own portable `.tlspec` save/load helpers.
- `torchlens.bridge` contains optional adapters for Captum, HF, SHAP, SAE Lens, LIT,
  profiler, and related tools.
- Appliance packages `viewer`, `paper`, `notebook`, `llm`, and `neuro` reserve extras
  boundaries; most are stubs except import gating in `notebook` and `neuro`.

## Anti-Patterns

- Do not log `torch.compile`, TorchScript, or `torch.export` artifacts; log the eager
  source module.
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
