# Torchlens project instructions

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

TorchLens logs backend-resolved execution into `Trace`, `Layer`, and `Op` objects. The stable default is PyTorch
eager capture: run a normal forward pass, record operation metadata and activations, then
inspect the result. Torch function wrapping is lazy in 2.x: `import torchlens` keeps torch
clean, and the first torch capture calls `wrap_torch()` through model preparation. The
wrappers then stay installed until an explicit
`torchlens.backends.torch.wrappers.unwrap_torch()`.


## Required reference reading

The referenced sections below are part of these instructions and retain their obligations. Before any code change, read conventions, critical invariants, known gotchas, and the glossary/docs contract. Before changing an API, feature, or module boundary, also read the corresponding surface, architecture, and feature reference. Paths in reference text are relative to the repository root.

## Install

```bash
pip install torchlens
pip install -e ".[dev,test,tabular,viz]"  # local development: the extras CI installs
```

Graphviz rendering needs Graphviz (`apt install graphviz` on Debian/Ubuntu; on Debian forky
and Ubuntu 26.04 or later also `libgvplugin-neato-layout8`, the neato engine rank layout needs). Optional
extras gate appliance and bridge namespaces; see `pyproject.toml` for the current list.

## Torch Version Compatibility

TorchLens supports torch 2.1 -> 2.15 for eager torch capture (the newest gating CI row; newer releases run in the non-gating nightly canary). The declared floor stays
`torch>=2.1`; torch 2.0 may work best-effort through guarded fallbacks, but it is not a
declared support floor.

Every fragile torch-private-API probe or cross-version torch signature must route through
`torchlens/_utils/_torch_compat.py`. Feature-detect the runtime capability; do not parse
`torch.__version__` for behavioral branching. Every graceful degradation must flip a named
`HAS_*` capability flag and be visible through the torch capability snapshot in
`torchlens.report.doctor()`.

## Model Menagerie and the classics corpus

The model menagerie (catalog, crawler, and the full classics battery) is its own project and no
longer lives in this repo; its public gallery is https://modelmenagerie.ai. TorchLens keeps a
coverage-chosen sample of those classics as a test corpus in `tests/classics_corpus/` (see its
`README.md`): a comprehensive `slow` tier plus a small `smoke` subset.

## Common Patterns

Required reading for this area: [Common Patterns](docs/internal/agent-reference/common-patterns.md).
Before steering, patching or capturing in a loop, pick the path in [Performance: choosing a fast path](docs/guides/fast_paths.md).

## Current 2.x Surface

Required reading for this area: [Current 2.x Surface](docs/internal/agent-reference/current-2-x-surface.md).

## Anti-Patterns

- Do not log `torch.compile`, TorchScript, or `torch.export` artifacts; log the eager
  source module.
- Do not expect `torch.func` / functorch transforms to expose per-element internal ops;
  TorchLens captures transform calls as boundary nodes with provenance edges.
- Do not run captures concurrently across Python threads or worker processes.
- Do not expect fused kernels to expose hidden internal tensors.
- Do not put opaque callables in portable artifacts unless audit-only behavior is
  acceptable.
- Do not add new top-level API names casually; use submodules. Interim-phase policy is remove-and-rename, never deprecation shims (scripts/checks/deprecation_inventory.py pins the package deprecation-free).
- Major-version bumps happen only on the maintainer's explicit command; commit messages never carry `!` types or BREAKING CHANGE footers; `.github/MAJOR_RELEASE_APPROVED` is created only on that command.

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

Replacement-op exemptions cover genuine user interventions only; a placeholder op during plain
capture is a capture gap to fix, never to exempt.

## Keep the glossary + docs in lockstep with code (LOCKED)

The glossary is the canonical API spec (maintained privately; `docs/reference/glossary.md` is
the public copy); code conforms to it. Renaming, adding or removing any
PUBLIC name (dataclass field, `@property`, method, top-level `tl.*` name, kwarg) updates, in the
SAME change, the glossary entry, the examples in `docs/internal/agent-reference/`, and the audit notebooks
(`notebooks/audit/`) and `examples/` that use it. A change that leaves them stale is INCOMPLETE.
Required reading for the full rule (canonical re-file, old-name grep) and the runnable-state
contracts filed with it: [lockstep reference](docs/internal/agent-reference/keep-the-glossary-docs-in-lockstep-with-code-locked.md).

## Internal notes stay PRIVATE (LOCKED — this repo is PUBLIC)

`johnmarktaylor91/torchlens` is a **public** GitHub repo. Internal planning, riffing, sprint
specs, adversarial reviews, STATE/SUMMARY files, and the working task tracker are
**maintainer-only** and must NEVER be committed.

- **Private (gitignored, never commit):** all of `.research/`, and `.project-context` EXCEPT the
  two whitelisted curated docs. The agent task tracker `todos.md` in `.project-context` and the
  agent-facing glossary copy `torchlens_glossary.md` beside it are private (the canonical glossary
  is maintained privately).
- **Public (the only tracked `.project-context` files):** `architecture.md`,
  `state_of_torchlens.md`. The user-facing glossary is `docs/reference/glossary.md` (shipped)
  — a separate, curated artifact, NOT the agent copy.
- **Enforcement:** `.gitignore` excludes them and a `no-internal-notes` pre-commit hook
  (`.pre-commit-config.yaml`) HARD-FAILS any commit that stages a private path. Never `git add -f`
  to bypass it; never `git rm` the local files (they are your working notes). Long-form
  reports go to the maintainer's private research location, not the repo.
- **Agent knowledge (private, hub only):** the `knowledge` folder in `.project-context` holds
  `torchlens_ui_api_sprint_inputs.md` (read first when the UI/API sprint starts),
  `torchlens_hardening_protocol.md` (the TorchLens hardening run record; general rules live in the
  maintainer's private hardening protocol), `torchlens_memory_lifetime.md` and `torchlens_test_isolation.md`.

## Testing Tiers

```bash
ruff check . --fix
mypy torchlens/
pytest tests/<files for the code you touched> -x --tb=short     # per-step gate: targeted suites (seconds-minutes)
python scripts/smoke_ci_parity.py -n 4                          # commit-level gate: smoke in CI's pinned enforcing environment
OMP_NUM_THREADS=1 pytest tests/ -m smoke -n 4 -x --tb=short     # quick smoke in your own venv (not the gate)
pytest tests/ -m "not rare and not slow and not heavy" -x --tb=short  # mid backstop (heavy = 5-20s tests)
pytest tests/ -m "not rare and not slow" -x --tb=short  # phase-boundary backstop; public API/boundaries
```

Byte goldens enforce only in the pinned environment (python 3.10, torch 2.13.0) and fail closed
outside it and outside CI, so only the script's answer equals CI's (`tests/AGENTS.md`, "The smoke tier").

Tiers by cost (every tier's count: `tests/tiers/collection.json`): `smoke` selects
~0.4k tests (384/18,644 collect-only, measured 2026-11-03),
a coverage-chosen subset of the former ~1.5k-test tier that holds every per-package coverage
floor (see `tests/AGENTS.md`, "The smoke tier"). The smoke wall measurement (measured 2026-11-03, one
4-core Linux worker, `-n 4`, one torch thread per worker) took 80s (~1 min) after about 50s
of collection. The size ceiling is `SMOKE_TIER_SIZE_CEILING`
(400) in `tests/conftest.py`. Smoke is NOT
sub-minute and NOT a per-step gate — per-step verification is the targeted test files for
the code touched; smoke is the commit-level gate, `not rare and not slow and not heavy`
the mid backstop, and `not slow` the phase-boundary backstop. Partition: `smoke` tests
must each run <5s measured, `heavy` carries the 5-20s tests, `slow` the >20s ones.
`tests/_tiers_plugin.py` refuses a session at collection when `smoke` combines with
`heavy`/`slow`/`serial`/`rare` (markers are additive — the test would still run under
`-m smoke`). Duration is a repository check, not a runtime tripwire: `tests/tiers/durations.csv`
records every test file's measured test count, summed seconds, slowest test and loosest tier
marker, and `scripts/tier_durations.py check` (the `tier-durations` row of
`scripts/checks/run_all.py`) holds smoke/unmarked files to budget 5s and heavy 20s per test
(the slowest measured test per file in `tests/tiers/durations.csv`, allowed up to 4x under a
parallel or coverage-instrumented measurement, plus a 2s grace). Regenerate the file from a
`--junitxml` run with `python scripts/tier_durations.py regenerate --source-commit SHA --junit
FILE --workers N`; `python scripts/tier_durations.py shards --count N --index I` prints one
duration-balanced shard of test files for a parallel gate. `pytest-xdist` ships in the
`dev` extra: run `OMP_NUM_THREADS=1 pytest tests/ -m smoke -n N` with N equal to the cores you
declared (never `-n auto`). Measured 2026-10-02 on a 32-core Linux worker: smoke took 5.5 min at
`-n 8` against 28 min serial. Every worker collects first, so the gain shrinks below 4 workers.
Smoke runs collect only the smoke file list: under `-m smoke` the tiers plugin skips every test
module off it before import, and `python scripts/tier_durations.py files --tier smoke` recomputes
the list from the tree on every call, so nothing needs regenerating. Measured 2026-11-02 on a
4-core Linux worker: smoke collection took about 40-45 s, serial or `-n 4`, against about 60-65 s
for the whole tree; passing the listed files as arguments took 130-140 s, because pytest re-walks
the directory for every file argument.

Use `pytest.importorskip()` for optional migration dependencies. Keep tests
deterministic and run documentation examples when they are meant to be executable.

## Architecture

Required reading for this area: [Architecture](docs/internal/agent-reference/architecture.md).

## Brainpipe (F20; spellings DOCUMENTED-UNSTABLE)

Required reading for this area: [Brainpipe (F20; spellings DOCUMENTED-UNSTABLE)](docs/internal/agent-reference/brainpipe-f20-spellings-documented-unstable.md).

## Conventions

Required reading for this area: [Conventions](docs/internal/agent-reference/conventions.md).

## Quality Gates

Every task must pass before completion unless the task explicitly narrows verification:

```bash
ruff format .
ruff check . --fix
mypy torchlens/
python scripts/smoke_ci_parity.py -n 4
```

Byte goldens enforce only in the pinned environment (python 3.10, torch 2.13.0); the script runs
smoke there, so its answer is CI's.

(CI lint runs `ruff format --check` plus `ruff check` over `torchlens tests scripts tools
benchmarks examples notebooks`; run `ruff format` locally or the format-check leg fails.)

For changes touching module boundaries or public API, also run:

```bash
pytest tests/ -m "not rare and not slow" -x --tb=short
```

## Critical Invariants

Required reading for this area: [Critical Invariants](docs/internal/agent-reference/critical-invariants.md).

## Known Gotchas

Required reading for this area: [Known Gotchas](docs/internal/agent-reference/known-gotchas.md).

## Build & Test

```bash
pip install -e ".[dev,test,tabular,viz]"
pip install build && python -m build
python scripts/smoke_ci_parity.py -n 4  # commit-level gate
pytest tests/ -m "not rare and not slow"
pytest tests/
ruff format && ruff check --fix
```
