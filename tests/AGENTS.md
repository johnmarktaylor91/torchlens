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

## Running Tests

```bash
pytest tests/                              # default suite excluding rare
pytest tests/ -m smoke                     # critical path
pytest tests/ -m "not slow"                # skip slow real-world tests
pytest tests/test_toy_models.py            # single file
pytest tests/test_toy_models.py::test_name # single test
pytest tests/ -k "loop"                    # keyword filter
```

Run memory-heavy real-world tests sequentially. Optional dependency tests should use
`pytest.importorskip()` or extras-aware skips.

## Markers
- `slow` - long-running real-world or heavy tests.
- `smoke` - fast critical-path checks.
- `rare` - excluded by default unless explicitly selected.

## Fixtures
`tests/conftest.py` owns deterministic seeding and common inputs such as image tensors,
small inputs, vector/2D/complex inputs, and output directories. Model fixtures/classes live
primarily in `tests/example_models.py`.

## Output Directories
All generated outputs go under `tests/test_outputs/` (gitignored):
- `reports/` for coverage, aesthetics, profiling.
- `visualizations/` for rendered graph artifacts.

## Adding Tests
- New model class: add to `tests/example_models.py` unless the test needs a one-off local class.
- New fields: test metadata, FIELD_ORDER consistency, pandas/export behavior when user-facing.
- Visualization changes: run targeted render tests and inspect generated artifacts when needed.
- Portable I/O changes: include save/load, lazy, corruption/security, and backcompat coverage.
- Intervention changes: test selector resolution, hook behavior, save-level behavior, and bundle
  comparisons where relevant.
