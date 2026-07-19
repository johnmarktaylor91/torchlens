# TorchLens Agent Guide

TorchLens logs backend-resolved execution into a `Trace`. The stable default is PyTorch
eager capture: run a normal forward pass, record operation metadata and activations, then
inspect the result. Torch function wrapping is lazy in 2.x: `import torchlens` keeps torch
clean, and the first torch capture calls `wrap_torch()` through model preparation. The
wrappers then stay installed until an explicit `torchlens.backends.torch.unwrap_torch()`.

## Install

```bash
pip install torchlens
pip install -e ".[test]"  # local development with test extras
```

Graphviz rendering needs Graphviz (`apt install graphviz` on Debian/Ubuntu). Optional
extras gate appliance and bridge namespaces; see `pyproject.toml` for the current list.

## Torch Version Compatibility

TorchLens supports torch 2.1 -> 2.12+ for eager torch capture. The declared floor stays
`torch>=2.1`; torch 2.0 may work best-effort through guarded fallbacks, but it is not a
declared support floor.

Every fragile torch-private-API probe or cross-version torch signature must route through
`torchlens/utils/_torch_compat.py`. Feature-detect the runtime capability; do not parse
`torch.__version__` for behavioral branching. Every graceful degradation must flip a named
`HAS_*` capability flag and be visible through the torch capability snapshot in
`torchlens.utils.doctor()` / `torchlens.compat.report()`.

## Model Menagerie (`menagerie/`)

`menagerie/` is a browsable atlas of 10,000+ neural-net architecture families captured with TorchLens:
a queryable catalog (`python -m menagerie.catalog stats|query|recipe`), ~300+ hand-built historical
"classics" with no prior PyTorch implementation (`menagerie/classics/`, each trace-verified), and a
disk-safe graph renderer (`python -m menagerie.generate_menagerie`).

**To DISCOVER new families** — periodically, after each conference cycle, or **whenever a more
capable model becomes available** (a smarter auditor finds more) — use the canonical durable prompt at
**`menagerie/DISCOVER_MODELS.md`**. It is the reusable, adversarial "hunt exhaustively for architecture
families we missed" sweep: hostile framing, every-axis + non-English + newly-published coverage,
strict family-not-variant discipline, and exact instructions for folding finds into the catalog or
`classics/`. Dispatch cross-lab adversarial sub-hunters with it; seed candidates with the starter
`python -m menagerie.discover_crawler` (recent-arXiv harvester, meant to be extended).

### To ADD / BUILD found models into the roster (LOCKED — READ THE METHODOLOGY, DO NOT REINVENT)

**BEFORE adding ANY model, READ and FOLLOW `menagerie/METHODOLOGY.md` + `menagerie/UPDATE_RECIPE.md` +
`menagerie/HARVEST_SOURCES.md`.** The catalog's 8,400+ rows were built by ONE established process; do not
re-derive it. The build-bridge is: harvest the model's **REAL constructor** into a 9-column source row
(`name, zoo, constructor_call, input_shape, input_dtype, family, domain, era, notes`), run it through
`python -m menagerie.tools.tsv_to_jsonl` → typed JSONL record in `menagerie/data/master_catalog.jsonl`
(or `deferred.jsonl`), then `python -m menagerie.catalog build` and `python -m menagerie.validate_menagerie`
(renders/validates random-init in **grouped/fat envs** — the renderer amortizes dependency installs; use a
few fat pixi env-islands via `menagerie/envs.py`, NOT one env per model).

**IF SOURCE CODE EXISTS FOR A MODEL, USE THE REAL SOURCE — never write a from-scratch "approximation".**
That is SLOP and is forbidden (2026-07-01 incident: ~1029 such reimpls deleted, huge token/$ waste). The ladder
per candidate: (1) real class from an installed base lib IF the arch is unmodified; (2) the real repo code, run
it in a (fat) env / vendor its actual model file; (3) **faithful PORT** transcribed from the real repo code, only
if it genuinely can't be made to run; (4) **faithful REIMPLEMENT from a DETAILED description** (paper/thesis/etc.)
only when NO usable code exists at all — the triage's REIMPLEMENT class, still faithful, not a gist; (5) skip +
document ONLY if not even a detailed description exists (triage UNAVAILABLE) or it is not a real trainable NN.
`classics/` is ONLY for no-prior-code models (faithful ports + rung-4 reimpls). The triage's
SOURCE_AVAILABLE / ENV_SETUP / REIMPLEMENT / UNAVAILABLE / NOT_TRACEABLE class IS the signal for which rung — honor it.

## Common Patterns

```python
import torchlens as tl

log = tl.trace(model, x, save=tl.func("relu"))
activation = log["linear_1_1"].out
print(log.summary())
print(tl.report.explain(log))
log.draw(order_siblings=True)  # default: verified sibling ordering for dot/unrolled graphs
log.draw(collapse="auto", show_containers=False)  # readability-targeted module overview
print(log.module_collapse_order[:10])
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
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
assert torch_trace.backend == "torch"
```

Before debugging wrapper-specific failures, run:

```python
print(tl.compat.report(model, x).to_markdown())
```

## Current 2.x Surface

- Top-level `torchlens.__all__` has 90 names: capture, save/load, intervention,
  selectors, helper transforms, observers, validation, and the three main log classes.
- `tl.record(..., save=...)` is the sparse predicate recorder; it returns `Recording`.
  `Recording.to_trace()` cooks the event stream into a full-structure `Trace`, with unsaved
  payload reads rejected explicitly. `tl.record()`/fastlog is torch-only in the backend-v1
  registry. `keep_op=` and `keep_module=` are deprecated aliases. Failed forwards default to
  the historical `on_forward_error="raise"` behavior; opt into
  `on_forward_error="attach_partial"` to attach `exc.partial_recording` and re-raise, or
  `on_forward_error="return_partial"` to return a failed partial `Recording`. Failed partials
  set `status="partial_error"`, `failed=True`, string-only error metadata, `n_ops_completed`,
  and best-effort `last_event_*` fields. user-op failures exclude the failing call; TL-side
  capture failures may include a skipped/partial current-call event. Failed partials cannot be
  converted with `Recording.to_trace()` or used with `Recording.log_backward()`.
  Trace-side failed captures expose `exc.partial_log`, recoverable with
  `tl.partial.from_failed_capture(exc)`.
- `tl.trace(..., backend=None)` routes through `BackendSpec`; explicit backend mismatches,
  unknown names, unsupported capabilities, and audit-only payload reads raise typed backend
  errors. Public backend-neutral metadata lives on `Trace.backend`, `Trace.module_identity_mode`,
  `Trace.param_source`, and record fields such as `dtype_ref`, `device_ref`,
  `backend_address`, and `resolver_status`.
- TensorFlow is available as `backend="tf"` / `backend="tensorflow"` for the Keras-3 / TF>=2.16
  preview when `keras.backend.backend() == "tensorflow"`. The shipped path is eager live capture
  with `op_callbacks` as the primary mechanism: real values, real taken-branch control flow,
  op-level records, and Keras/`tf.Module` module stacks. Graph-only FuncGraph fallback is the
  static-mode design; interventions, true backward capture, and T1 derived gradients remain
  deferred like sibling preview gaps.
- `Trace.draw(order_siblings=True)` is the default Graphviz sibling-ordering pass for
  forward unrolled graphs; set it to `False` to render the raw dot layout.
- `Trace.draw(collapse="none"|"auto"|"max"|t, fold_repeats=None|True|False)` controls v2 smart
  collapse for rolled and unrolled graphs, where float `t` in `[0.0, 1.0]` follows the public
  monotone schedule (`0.0 == "none"`, `1.0 == "max"`). `auto` is the first schedule point whose
  visible count enters the readable band, but its implementation remains frozen for compatibility.
  `None` preserves defaults (`"none"` has no run folding; `"auto"`/`"max"` use band-pressure
  folding), `True` folds eligible repeated runs even with `collapse="none"`, and `False` disables
  run folding. `collapse="max"` may emit segment boxes; `(xN)`, ellipsis, and segment labels must
  stay honest about hidden calls or ranges. `Trace.collapse_plan(mode=...)` returns the diagnostic
  plan, and `Trace.collapse_schedule()` returns the float schedule metadata.
- Smart-collapse metadata is computed at access time: `Module.collapse_score`,
  `Trace.module_collapse_order`, and `Trace.collapse_order(weights=..., mode=...)`. These are
  not portable fields and must not be added to `*_FIELD_ORDER` without an explicit schema change.
- `torchlens._io` and `torchlens.io` own portable `.tlspec` save/load helpers. Manifest
  schema v2 is backend-aware; non-torch preview bundles may be audit-only or metadata-only.
- `torchlens.debug` owns power-user diagnostics such as `bisect_nan` and `hot_path`;
  the submodule is imported as `tl.debug` and is deliberately not in `__all__`.
- `torchlens.bridge` contains optional adapters for Captum, HF, SHAP, SAE Lens, LIT,
  profiler, and related tools.
- Appliance packages `notebook` and `neuro` reserve extras boundaries and enforce
  import gating for their optional dependencies.

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

### Trusted custom callable imports

Intervention-spec loads tolerate foreign `custom` callable keys for safe structural and metadata
analysis without importing their modules. Resolution for execution denies those foreign imports by
default because module imports execute top-level code. Trusted execution may opt in with
`trust_custom_callables=True`; prefer the narrower
`allowed_custom_callable_modules={"my_trusted_module"}`, which remains enforced even alongside broad
trust. TorchLens-owned `torchlens.*` custom callables and the fixed `torch`, `torch.Tensor`,
`torch.nn.functional`, and `operator` namespaces always resolve.

### Sparse runnable state binding

Loaded sparse runnable traces accept `trace.load_state_dict(sd)` to strictly validate and atomically
stage canonically named parameter and persistent-buffer tensors. The method never executes the DAG
or writes tensor payloads into the sparse descriptor. Run preflight selects explicitly staged user
state, then optional embedded capture state, then the versioned
`torchlens_role_init_v2` fallback (degenerate-total: empty slots consume zero RNG); random reports
name every initialized slot.

`tl.save(trace, path, level="runnable", include_weights=True)` bundles the full capture-time
`state_dict` (all named parameters plus persistent buffers) as the separate, schema-versioned
`state_dict_v1` blob family. The default is `include_weights=False`, so the sparse core stays
tensor-value-free. Load validates embedded state through the same strict binder; run reports
`embedded_capture_state`, never a reconstructed model, and a later `load_state_dict()` overrides it.
Used NON-persistent buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family
(declared state, not gated on either include flag; the save discloses it). State ALIAS topology is
declared (r37): repeated live object identity (tied weights, double-registered buffers) becomes a
shared alias group staged as ONE allocation; distinct-object overlapping or unprovable state
topology refuses at save with `state_alias_topology_unsupported`. Payload blobs keep their
`map_location` transport placement; readiness capability-checks recorded slot devices without
allocating, and one atomic run-preparation staging pass moves all state families to their recorded
devices (a CUDA artifact on a CPU-only host loads for analysis and refuses `.run()` typed).

`tl.save(trace, path, level="runnable", include_activations=True)` independently archives exactly
the activations already retained by the capture-time `save=` decision, including retained raw and
transformed outputs, as `selected_activation_v2` with physical `InputAttestationFingerprint`
eligibility records. Load exposes them through
`trace.archived_activations` for inspection. They never seed the sparse DAG. On original-input runs
with embedded or capture-equivalent staged state, recomputed saved raw slots are compared by exact
bytes and report `attested`; the first mismatch raises `numeric_attestation_failed` and rolls back.
Changed-input (logical or physical), random-state, non-equivalent-state, and
nondeterministic-capture-context runs report `not_applicable`; `attested` always implies
`verified`.

### Sparse runnable execution

`trace.run(inputs=x, seed=...)` is the provider-neutral execution spelling. A live Trace delegates
on a fork to the existing `save_new_outs` fast capture path; a loaded sparse Trace binds cloned
input leaves plus staged/random state and executes its resolved taken-path DAG under
`pause_logging()`. Both return `RunResult(output, trace, report)` and leave the source Trace
unchanged. Analysis-only loads raise typed `run_capability_unavailable`. Stage 5 populates
`report.path_faithfulness`; Stage 6 enforces the three-layer honesty transaction. Divergence raises
and rolls back by default; `return_diverged` is the sole opt-in and returns a monotonic poisoned
Trace refused by validation, export, faithful comparison, and path-assuming intervention chaining.
Incomplete witness coverage is `unverifiable`, never `verified`; numeric attestation is
`not_applicable` for sparse-only or ineligible activation-payload runs. Model outputs with ZERO
tensor leaves (all-literal trees, literal roots, empty containers) and namedtuple/mapping/
registered-container outputs carrying extra per-instance state refuse at save
(`missing_output_container_contract`; one per-kind capability table governs capture proof, save
refusal, and load-time recompute). Host nondeterminism beyond the two replayable global engines
-- RNG instances (incl. outside-held NumPy Generators, witnessed by a thread-independent
process-wide state inventory; bare `_random.Random`; unseeded-construction entropy via the
`randbits` alias), `SystemRandom`/`secrets`, OS entropy, `uuid4`, the `default_rng` factory, and
the full clock family (`time.*` counters, `localtime`/`strftime`/`datetime.now`/`date.today`,
`os.times`/`getrusage`) -- ceilings every replay permanently (`unverifiable` + `not_applicable`);
monitor uncertainty (install/chain/restore/inventory failure) downgrades completeness, never reads
as no-consumption. A realistic pre-existing-thread draw from a persistent numpy generator is
witnessed thread-independently by the state inventory; only an adversarial persistent-instance
draw+state-restore on a pre-existing thread is a documented residual (a benign background thread
never ceilings a capture). The loaded-sparse and live-refresh
providers settle through ONE finalizer (identical verdict class): a live opaque-container output
is `unverifiable` + poisoned (never a wrongly-blessed bare tensor), a parse-refused descriptor
degrades EVERY payload family to analysis-only with its typed diagnostic intact, and an
inexecutable divergent input raises `PathDivergenceError` (not `RuntimeSignatureDrift`). Structseq
reconstruction trust keys on the RESOLUTION AUTHORITY (`spec.type_module == "torch.return_types"` +
identity re-resolution), never the spoofable `__module__` attribute; a namedtuple TYPE that can
carry instance state refuses at save even with an empty instance. Persisted execution-context values validate at parse
time against closed vocabularies (`context_field_invalid`); the recorded default device is entered
as a scoped `with torch.device(...)` context, never via `set_default_device`.

Runnable descriptors are `sparse_recorded_taken_path_v2`: per-call `CallExecutionContext` and the
capture-scoped `AmbientExecutionContext` are REQUIRED and EXPLICIT, restored at replay or refused
typed; a legacy v1 artifact loads analysis-only with a typed readiness refusal (absent context is
never defaulted).

The declared state model is the capture-time `state_dict` (named parameters plus persistent buffers)
PLUS the capture-time values of used non-persistent buffers (the required
`runnable_nonpersistent_buffer_v1` family), and the taken-path DAG. `verified` is faithfulness
against a *fresh* live-model run from that state on
the given inputs (oracle 1) — NOT reproduction of a specific already-run instance's later, differently
branched forwards. Hidden non-`state_dict` Python state mutated *across* forwards (an arbitrary
attribute, or a retained activation-derived handle — a kept `numpy()`/`untyped_storage()` view or a
detached tensor) is out of scope and stays `verified`; the "divergence" exists only against re-running
the same mutated instance, never oracle 1. In scope and witnessed identically for activations,
parameters, and buffers: a host write *within* the captured forward into captured storage — caught by
whole-storage byte comparison + per-consumption sampling, with the raw `data_ptr()` surface fail-closed
to `unverifiable`, and a read-only exposure staying `verified`. Full boundary in
`docs/reference/runnable_tlspec_contract.md` section 11.

The frozen runnable enums live in `torchlens.runnable`: `ReadinessStatus`, `RunProvider`,
`StateSource`, `PathFaithfulness`, `DivergencePolicy`, `NumericAttestationStatus`, and
`RunnableErrorCode`. Public code branches on these values or the structured report, not exception
text. The exhaustive error vocabulary and release threshold are maintained in
`docs/reference/runnable_tlspec_contract.md`.

## Internal notes stay PRIVATE (LOCKED — this repo is PUBLIC)

`johnmarktaylor91/torchlens` is a **public** GitHub repo. Internal planning, riffing, sprint
specs, adversarial reviews, STATE/SUMMARY files, and the working task tracker are **JMT's eyes
only** and must NEVER be committed.

- **Private (gitignored, never commit):** all of `.research/`, and `.project-context/` EXCEPT the
  two whitelisted curated docs. The agent task tracker `.project-context/todos.md` and the
  agent-facing `.project-context/torchlens_glossary.md` (canonical lives in the vault) are private.
- **Public (the only tracked `.project-context/` files):** `architecture.md`,
  `state_of_torchlens.md`. The user-facing glossary, when it ships, is `docs/reference/glossary.md`
  — a separate, curated artifact, NOT the agent copy.
- **Enforcement:** `.gitignore` excludes them and a `no-internal-notes` pre-commit hook
  (`.pre-commit-config.yaml`) HARD-FAILS any commit that stages a private path. Never `git add -f`
  to bypass it; never `git rm` the local files (they are your working notes). Long-form
  human-readable reports go to the Obsidian vault, not the repo.

## Testing Tiers

```bash
ruff check . --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short                            # ~28s: true sub-minute per-step gate
pytest tests/ -m "not rare and not slow and not heavy" -x --tb=short  # ~8min mid backstop (heavy = 5-20s tests)
pytest tests/ -m "not slow" -x --tb=short  # full fast suite (~15min); for public API or boundary changes
```

Tiers by cost: `smoke` (~28s) is the real per-step gate; `heavy` carries the 5-20s tests and
`slow` the >20s ones, so the `not rare and not slow and not heavy` tier (~8min) is a mid
backstop and full `not slow` (~15min) is the phase-boundary backstop. NOTE: `pytest -n auto`
(xdist) is NOT faster here — torch's intra-op threads oversubscribe the 20 workers and the
fast tier rises to ~13min; the per-test bottleneck is torch import/fixture setup, not CPU.

Use `pytest.importorskip()` for optional migration dependencies. Keep tests
deterministic and run documentation examples when they are meant to be executable.
