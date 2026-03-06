# tests/ — Test Suite

## Overview
~690 tests across 12 test files. Uses pytest with deterministic torch seeding.

## Test Files

| File | Tests | What It Covers |
|------|-------|----------------|
| `conftest.py` | — | Fixtures, deterministic seeding, output directory setup, coverage reporting |
| `example_models.py` | — | 170+ toy model class definitions for controlled testing |
| `test_toy_models.py` | ~165 | Validation + visualization for toy models (14 sections by category) |
| `test_real_world_models.py` | ~87 | Real-world architectures (20 fast, 67 `@pytest.mark.slow`) |
| `test_metadata.py` | ~102 | Field-level coverage for ModelLog and LayerPassLog |
| `test_module_log.py` | ~44 | ModuleLog/ModulePassLog/ModuleAccessor |
| `test_param_log.py` | ~68 | ParamLog/ParamAccessor |
| `test_decoration.py` | ~61 | Permanent decoration architecture (toggle, crawl, JIT, signals) |
| `test_validation.py` | ~50 | Validation subpackage (registries, perturbation, invariants A-R) |
| `test_layer_log.py` | — | LayerLog aggregate class |
| `test_internals.py` | — | Internal implementation details |
| `test_save_new_activations.py` | — | `save_new_activations()` re-logging |
| `test_profiling.py` | — | Performance profiling |
| `test_output_aesthetics.py` | ~9 | Aesthetic report + vis PDFs for human review |

## Running Tests

```bash
pytest tests/                              # all tests
pytest tests/ -m smoke                     # smoke tests (~6s, 18 tests across 9 files)
pytest tests/ -m "not slow"                # skip slow real-world tests
pytest tests/test_toy_models.py            # single file
pytest tests/test_toy_models.py::test_name # single test
pytest tests/ -k "loop"                    # keyword filter
```

**Important**: Use `timeout: 600000` or `run_in_background: true` for long-running tests.
Run memory-heavy real-world tests SEQUENTIALLY.

## Markers
- `@pytest.mark.slow` — Tests taking >5 min (mostly real-world models)
- `@pytest.mark.smoke` — 18 critical-path tests for fast validation during dev

## Fixtures (conftest.py)

| Fixture | Shape | Use |
|---------|-------|-----|
| `default_input1-4` | `(6,3,224,224)` | Standard image input |
| `zeros_input` | `(6,3,224,224)` | All-zeros edge case |
| `ones_input` | `(6,3,224,224)` | All-ones edge case |
| `vector_input` | `(5,)` | 1D models |
| `input_2d` | `(5,5)` | 2D models (recurrent, LSTM) |
| `input_complex` | `(3,3)` complex | Complex tensor edge case |
| `small_input` | `(2,3,32,32)` | Fast metadata tests |

## Test Patterns

### Toy model test (test_toy_models.py)
```python
def test_model_descriptive_name(default_input1):
    model = example_models.MyNewModel()
    assert validate_saved_activations(model, default_input1)
    show_model_graph(
        model, default_input1, save_only=True, vis_opt="unrolled",
        vis_outpath=opj(VIS_OUTPUT_DIR, "toy-networks", "my_new_model"),
    )
```
Every toy test validates activations AND generates a visualization.

### Real-world model test (test_real_world_models.py)
```python
def test_architecture_name(default_input1):
    lib = pytest.importorskip("some_library")
    model = lib.SomeModel()
    assert validate_saved_activations(model, default_input1)
```
Optional deps use `pytest.importorskip()` inside the test function.

### Metadata test (test_metadata.py)
```python
class TestSomeFields:
    def test_field(self, small_input):
        model = example_models.SomeModel()
        mh = log_forward_pass(model, small_input)
        assert isinstance(mh.some_field, expected_type)
```
Uses `log_forward_pass()` directly, not `validate_saved_activations`.

## Output Directories
All test outputs go to `tests/test_outputs/` (gitignored):
- `reports/` — coverage reports (text + HTML), aesthetic report, profiling report
- `visualizations/` — vis PDFs organized by model family (19 subdirs)

## Adding Tests
1. Model class → `example_models.py`
2. Test function → appropriate `test_*.py` file
3. New fields → add test in `test_metadata.py`, update constants.py FIELD_ORDER
4. After changing vis/repr → run `test_output_aesthetics.py` and inspect outputs

## Aesthetic Testing
After changing user-facing features (reprs, accessors, error messages, visualization):
```bash
pytest tests/test_output_aesthetics.py -v
```
Then inspect `tests/test_outputs/reports/aesthetic_report.pdf` visually.
