# tests/ - Test Suite

## Overview
The test suite is broad and 2.x-heavy: core capture, postprocess, validation, visualization,
portable I/O, intervention, fastlog, bridges, examples, and train-mode behavior. Default
pytest config excludes `rare` tests via `addopts = -m 'not rare'`.

## Main Areas

| Area | Representative files |
|------|----------------------|
| Core capture and metadata | `test_toy_models.py`, `test_metadata.py`, `test_layer_log.py`, `test_module_log.py`, `test_param_log.py` |
| Decoration and wrappers | `test_decoration.py`, `test_arg_positions.py`, `test_two_pass_inplace_fix.py` |
| Postprocess conditionals | `test_conditional_*.py`, `test_ast_branches.py`, `test_field_lifecycle_matrix.py` |
| Visualization | `test_large_graphs.py`, `test_node_spec_api.py`, `test_node_modes.py`, `test_themes.py`, `test_overlays.py`, `test_bundle_diff_renderer.py` |
| Validation/backward | `test_validation.py`, `test_validate_consolidated.py`, `test_backward.py`, `test_backward_streaming.py` |
| Portable I/O | `test_io_*.py`, `test_tlspec_*.py`, `fixtures/tlspec_v2_16/` |
| Intervention | `test_intervention_phase*.py`, `test_sites.py`, `test_selector_unification_phase4.py`, `test_bundle_*.py` |
| Fastlog | `test_fastlog/` |
| Bridges/compat/export | `test_bridges_*.py`, `test_compat_report.py`, `test_exports.py`, `test_extractor_compat.py` |
| Examples/audit | `test_examples.py`, `test_examples_load.py`, `test_not_mvp_audit.py` |
| Train mode | `test_train_mode/` |
| Backend contracts | `backend_conformance/`, `backend_parity/`, `backends/` |
| Capture and surface oracles | `capture_oracle/`, `godobject_oracle/`, `surface_oracle/` |
| Producer and semantic parity | `producer_parity/`, `semantic/`; the producer ledger is a committed gate artifact |
| Classics corpus and benchmarks | `classics_corpus/`, `bench/` |
| Validation and visualization goldens | `validation_goldens/`, `visualization/`, `golden/`, `snapshots/` |
| Shared data and helpers | `fixtures/`, `support/` |

## Running Tests

```bash
pytest tests/test_toy_models.py            # single file — targeted suites ARE the per-step gate
pytest tests/test_toy_models.py::test_name # single test
python scripts/smoke_ci_parity.py -n 4     # commit-level gate: smoke in CI's pinned enforcing environment
pytest tests/ -m smoke -n 4                # quick smoke in your own venv (~1.4k tests; not the gate)
pytest tests/ -m "not rare and not slow and not heavy" -x --tb=short  # mid backstop
pytest tests/ -m "not rare and not slow"   # phase-boundary backstop (keeps rare excluded)
pytest tests/                              # default suite excluding rare
pytest tests/ -k "loop"                    # keyword filter
```

Run memory-heavy real-world tests sequentially. Optional dependency tests should use
`pytest.importorskip()` or extras-aware skips.

## Markers

| Marker | Meaning |
| --- | --- |
| `smoke` | Critical-path checks, <5s each (measured); the commit-level gate, not per-step. A coverage-chosen set of about 1,400 tests, capped at `SMOKE_TIER_SIZE_CEILING` (1,500) in `tests/conftest.py`. |
| `smoke_cells` | `smoke_cells("test_x[a]", ...)` on a function, class or module applies `smoke` to exactly the named items, so a parametrized family keeps one or two representative smoke cells. |
| `heavy` | Mid-cost (5-20s) tests, excluded from smoke and the mid backstop. |
| `slow` | Long-running (>20s) real-world tests. |
| `serial` | Load-sensitive tests that should run away from parallel worker load. NOT a budget exemption: serial items resolve their heavy/smoke/unmarked duration budget normally. |
| (none) | Unmarked tests — a large majority (16,890/20,445, 83%, measured 2026-10-02) — run only in the backstops, never the commit gate, but are budgeted at the same <5s tier as smoke. |
| `rare` | Always excluded by default unless explicitly selected. |
| `optional` | Requires an optional dependency or runtime. |
| `requires_assertions` | Needs Python `assert`; skipped under `python -O`. |
| `backend_parity` | Active backend-substrate parity gate. |
| `backend_jax` | Requires the optional JAX runtime. |
| `backend_mlx` | Requires the optional MLX runtime. |
| `backend_tinygrad` | Requires the optional tinygrad runtime. |
| `backend_paddle` | Requires the optional Paddle runtime. |
| `tf_backend` | Requires the optional TensorFlow runtime. |

Markers are additive: a test carrying `smoke` together with `heavy`/`slow`/`serial`/`rare`
still runs under `-m smoke`, so those combinations are forbidden — drop `smoke` instead
(a per-parametrize-cell `slow` refinement of a `heavy` family is the one sanctioned combo).
`tests/test_marker_lint.py` enforces the partition and the runtime tripwire: smoke/unmarked
tests budget 5s and heavy 20s (load-scaled 1x-4x plus a 2s boundary-noise grace, charged on min(wall, cpu)), checked at
the end of every session — literally: `tests/conftest.py::pytest_sessionfinish` flips a green
session to failing on any recorded offender, so targeted runs that never collect
`test_marker_lint.py` are enforced too (r7 R41). Known bounds of the tripwire: it charges only
tests that actually RAN in the session (a permanently deselected test is never bounded), the
budget is load-scaled so the same family can pass loaded and fail quiet (the boundary is
compute cost, not wall time), and a test that mostly sleeps is uncatchable by design
(min(wall, cpu) charging). Subprocess-per-cell parametrized families whose AGGREGATE compute
is heavy-class belong in `heavy` even when each cell is under 5s (the lazy-module
import-pattern families are the precedent).
The load scale is MEASURED, not read from loadavg: a fixed CPU-bound probe's thread CPU time
against its reference-host time (`SLOWDOWN_PROBE_REFERENCE_SECONDS`), measured at session start
and once more for any test about to be flagged, clamped 1x-4x. It sees slower cores (a slow
runner, a busy hyperthread sibling) and ignores run-queue length, so it is correct under
core-pinned job runners and on CI runners. Test processes run torch with ONE intra-op thread
(`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` and `torch.set_num_threads(1)`);
set `TORCHLENS_TEST_THREADS=N` to override. Tests in `tests/test_duration_tripwire_calibration.py`.

## The smoke tier

The smoke tier is a coverage-chosen subset (re-selected 2026-10-02 from about 9,900 tests):
fastest tests that keep line coverage of `torchlens/` close to the old tier, every package
file and public `tl.*` name smoke reaches, the Critical Invariant and validation-integrity
tests, the classics smoke entries, enough tests to hold every package the nightly smoke-coverage
job floors (`scripts/check_package_coverage_floors.py`) well above its floor, and every
env-fingerprinted golden family (those enforce
only on the canonical smoke CI row). Everything else that used to be smoke is unmarked or
`heavy` and still runs in the backstops and the nightly and weekly CI tiers. A new test stays
unmarked unless it covers a critical path nothing in smoke reaches; the size ceiling keeps the
tier from regrowing.

Under `-m smoke` (or `-m "smoke and ..."`) `tests/conftest.py` skips, before import, every
test module whose source never spells `mark.smoke` or `mark.smoke_cells`; whole-tree
collection otherwise costs 3-4 minutes. Spell smoke marks in the test file itself, never
through an imported alias: `tests/test_marker_lint.py::test_smoke_prefilter_never_hides_a_smoke_test`
fails any non-prefiltered session in which a smoke item sits in a file the pre-filter would
skip. `TORCHLENS_SMOKE_PREFILTER=0` turns the pre-filter off.

Byte goldens enforce only in the pinned environment (python 3.10, torch 2.13.0), so the
commit-level gate is `python scripts/smoke_ci_parity.py [-n N]`, not a plain
`pytest tests/ -m smoke`. The env-fingerprinted golden families enforce only on the environment
their goldens were recorded under and fail closed on any other non-CI interpreter by design,
so a smoke run in your own venv reports those few golden tests as failures. The script builds a
venv pinned to CI's enforcing smoke row (once, in `.venv-ci-smoke/`) and runs the workflow's
smoke-step commands there; `tests/test_ci_packaging_gates.py` keeps its pins in lockstep with
`tests.yml` and these docs. A smoke run off that environment ends with a `torchlens CI parity:`
line naming the differences. The `test` extra includes `tabular`, because CI installs it on
every row and the tabular surfaces are smoke-tier paths.

## Fixtures
`tests/conftest.py` owns deterministic seeding and common inputs such as image tensors,
small inputs, vector/2D/complex inputs, and output directories.
`tests/backends/conftest.py` supplies backend test isolation, and
`tests/test_train_mode/conftest.py` supplies train-mode fixtures. Model fixtures/classes live
primarily in `tests/example_models.py`.

## Shared package-source corpus
A lint/census test that walks `torchlens/` source must consume
`tests/_source_corpus.py` (`package_files()` / `package_source(path)` /
`package_ast(path)`; `module_ast()` / `module_source()` for helpers that may
also receive non-package paths) instead of running its own
`rglob` + `ast.parse` sweep — each private sweep costs ~5s CPU and ~270 MB of
AST churn, duplicated per file. Import `_source_corpus` at MODULE level: that
import is what triggers the one prewarm in
`conftest.pytest_collection_finish`, BEFORE the import-time `gc.freeze()`, so
the corpus lands in the frozen generation and gen-2 collections never scan
it. Returned trees and sources are SHARED — never mutate them (a scanner that
annotates parent pointers keeps its own private parse instead).

## Output Directories
All generated outputs go under pytest's private basetemp at
`<basetemp>/torchlens-generated/` (assigned in `tests/conftest.py::pytest_configure` and
exported as `TORCHLENS_TEST_OUTPUTS_DIR`):
- `reports/` for coverage, aesthetics, profiling.
- `visualizations/` for rendered graph artifacts.

## Adding Tests
- New model class: add to `tests/example_models.py` unless the test needs a one-off local class.
- New fields: test metadata, FIELD_ORDER consistency, pandas/export behavior when user-facing.
- Visualization changes: run targeted render tests and inspect generated artifacts when needed.
- Portable I/O changes: include save/load, lazy, corruption/security, and backcompat coverage.
- Intervention changes: test selector resolution, hook behavior, save-level behavior, and bundle
  comparisons where relevant.
