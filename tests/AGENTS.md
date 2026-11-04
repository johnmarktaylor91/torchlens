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
| Postprocess conditionals | `test_conditional_*.py`, `test_ast_branches.py` |
| Visualization | `test_large_graphs.py`, `test_node_modes.py`, `test_themes.py`, `test_overlays.py`, `test_bundle_diff_renderer.py` |
| Validation/backward | `test_validation.py` (siblings `test_validation_perturbation_units.py`, `test_validation_bool_exemption.py`, `test_validation_key_forgery.py`), `test_validate_consolidated.py`, `test_backward.py`, `test_backward_streaming.py` |
| Portable I/O | `test_io_*.py`, `test_tlspec_*.py`, `fixtures/tlspec_v2_16/` |
| Intervention | `test_intervention_phase*.py`, `test_sites.py`, `test_selector_unification_phase4.py`, `test_bundle_*.py` |
| Fastlog | `test_fastlog/` |
| Bridges/compat/export | `test_bridges_*.py`, `test_compat_report.py`, `test_exports.py`, `test_extractor_compat.py` |
| Examples/audit | `test_examples.py`, `test_examples_load.py`, `test_not_mvp_audit.py` |
| Train mode | `test_train_mode/` |
| Backend contracts | `backend_conformance/`, `backend_parity/`, `backends/`, `conformance/` (the C0-C2 conformance harness and scorecard builder; not a package of `torchlens`) |
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
pytest tests/ -m smoke -n 4                # quick smoke in your own venv (~400 tests; not the gate)
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
| `smoke` | Critical-path checks, <5s each (measured); the commit-level gate, not per-step. A coverage-chosen set of 383 tests, capped at `SMOKE_TIER_SIZE_CEILING` (400) in `tests/conftest.py`. |
| `smoke_cells` | `smoke_cells("test_x[a]", ...)` on a function, class or module applies `smoke` to exactly the named items, so a parametrized family keeps one or two representative smoke cells. |
| `heavy` | Mid-cost (5-20s) tests, excluded from smoke and the mid backstop. |
| `slow` | Long-running (>20s) real-world tests. |
| `serial` | Load-sensitive tests that should run away from parallel worker load. NOT a budget exemption: a file's duration tier is its loosest tier marker, and `serial` is not one. |
| (none) | Unmarked tests — a majority (15,036/18,941, 79%, measured 2026-11-03) — run only in the backstops, never the commit gate, but are budgeted at the same <5s tier as smoke. |
| `rare` | Always excluded by default unless explicitly selected. |
| `optional` | Requires an optional dependency or runtime. |
| `backend_parity` | Active backend-substrate parity gate. |
| `backend_jax` | Requires the optional JAX runtime. |
| `backend_mlx` | Requires the optional MLX runtime. |
| `backend_tinygrad` | Requires the optional tinygrad runtime. |
| `backend_paddle` | Requires the optional Paddle runtime. |
| `tf_backend` | Requires the optional TensorFlow runtime. |
| `cuda` | Needs a CUDA device. Skipped without one (`tests/conftest.py`); the GPU leg selects them with `-m cuda`. Every `skipif(not torch.cuda.is_available(), ...)` site carries it (`tests/test_marker_lint.py`). |

Markers are additive: a test carrying `smoke` together with `heavy`/`slow`/`serial`/`rare`
still runs under `-m smoke`, so those combinations are forbidden — drop `smoke` instead
(a per-parametrize-cell `slow` refinement of a `heavy` family is the one sanctioned combo).
`tests/_tiers_plugin.py` enforces the partition at collection: a session that selects a
forbidden combination (after `-m` deselection) fails before its first test, in targeted runs too. Duration is a
repository check over a tracked measurement, not a runtime tripwire. `tests/tiers/durations.csv`
holds one row per test file (`file,tests,seconds,max_seconds,tier`: measured test count,
summed wall seconds, slowest test, loosest tier marker the file spells, `missing` once the
file is gone) and `tests/tiers/durations.meta.json` records the measurement's commit,
inputs, worker count and whether coverage tracing was on. `scripts/tier_durations.py check`
(the `tier-durations` row of `scripts/checks/run_all.py`) holds smoke/unmarked files to
budget 5s and heavy 20s per test (the slowest measured test per file in
`tests/tiers/durations.csv`, allowed up to 4x under a parallel or coverage-instrumented
measurement, plus a 2s grace); `slow` and `rare` files are unbounded. Known bounds: the tier
is resolved per FILE (a file with one `heavy` test gets the heavy budget for all its tests),
only measured tests are bounded (a test deselected in the measurement is never bounded),
and the measurement is wall time, so a loaded or instrumented run is judged at the 4x line.
Subprocess-per-cell parametrized families whose AGGREGATE compute is heavy-class belong in
`heavy` even when each cell is under 5s (the lazy-module import-pattern families are the
precedent). Regenerate after a measured run:
`python scripts/tier_durations.py regenerate --source-commit SHA --junit run.junit.xml --workers N`
(also accepts `--per-test-csv test_id,seconds` files and `pytest-split` `--pytest-split
.test_durations` files; add `--instrumented` when coverage tracing was on).
`python scripts/tier_durations.py shards --count N` balances the test files on disk over N
shards by measured seconds (`--index I` prints one shard's files for `pytest`); unmeasured
files take the median file time.
Test processes run torch with ONE intra-op thread
(`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` and `torch.set_num_threads(1)`);
set `TORCHLENS_TEST_THREADS=N` to override. Tests in `tests/test_thread_pinning.py`.

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
tier from regrowing. The tier holds 383 tests (`tests/tiers/collection.json`, collected
2026-11-03; per-tier counts, wall estimates and the collect commands are there).

Smoke runs collect only the smoke file list, the test files whose source spells `mark.smoke`
or `mark.smoke_cells`: under `-m smoke` `tests/_tiers_plugin.py` skips every other test module
before import (`TORCHLENS_SMOKE_PREFILTER=0` turns the skip off). `python
scripts/tier_durations.py files --tier smoke` prints the list, recomputed from the tree on every
call, so nothing needs regenerating. Measured 2026-11-02 on a 4-core Linux worker: smoke
collection took about 40-45 s against about 60-65 s for the whole tree. Spell smoke marks in the
test file itself, never through an imported alias or a hook: the tiers plugin fails any session
that collects a smoke item from a file off the list, and the `tier-durations` check of
`scripts/checks/run_all.py` fails a helper module under `tests/` that applies the mark.

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

## Conformance harness (`tests/conformance/`)
The conformance suite is test-side: it is imported as `tests.conformance` and is not shipped in the
wheel. It runs the same case ids over every backend (C0 protocol, C1 capture, C2 durable) from
per-backend rosters in `_rosters/` and folds the result into one scorecard per backend. The
shipped record stays in the product at `torchlens/backends/<name>/scorecard.json` (read by
`tl.report.doctor()`); regenerate or check it from a checkout with
`python -m tests.conformance.scorecard --backend <name> --write` or `--check` (the pinned CI legs
run `--check`) and print the six advisory graduation items with `--graduation`.
`TORCHLENS_CONFORMANCE_ECOSYSTEM=1` enables the roster rows that download pretrained weights.

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
