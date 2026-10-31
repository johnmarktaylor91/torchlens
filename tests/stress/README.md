# Stress matrix

One generated matrix. Axes are lists of rows; a cell is one row from each axis:
(model, input, dtype, device, options). Six oracles judge every cell, and each
oracle is one parametrized test in `test_stress.py`. A finding is added as a row
or recorded against a cell id, never as a new test file.

## Layout

| File | Holds |
|---|---|
| `generator.py` | row dataclasses, `generate(axes)`, cell ids, subset selection, the `subset.json` tool |
| `oracles.py` | `run_cell` (one execution per cell) and the six oracle functions |
| `axes/control_flow.py` | ten synthetic models with data- or shape-dependent control flow |
| `axes/dtypes_cpu.py` | the dtype cast rows, the models carrying the integer, bool and complex rows, and the classics smoke tier as corpus model rows |
| `axes/inputs.py` | input-shape rows and the small models that accept them |
| `axes/__init__.py` | `default_axes()`, plus the device and options rows |
| `subset.json` | the pre-merge cell ids, with the matrix fingerprint they were selected from |
| `test_stress.py` | the six oracle tests and two structural tests |

## Adding an axis value

Append one row to the list in the axis module. Every row has an id (lowercase
`[a-z0-9_]`, never `-`) and a one-line `reason`.

- A model: a `ModelSpec` with `build`, `example_input(model)`, the dtype rows it
  may be cast to (`dtypes`), a `cost` rank (1 cheapest to 5) and a
  `memory_budget_mib`.
- An input shape: an `InputSpec` with `make(model) -> (args, kwargs)` and the
  model ids it fits (`models`); `None` fits every model.
- A dtype: a `DtypeSpec` whose `cast(model, inputs)` returns the cast pair; list
  the new id in the `dtypes` of the models that support it.
- A device or options row: a `DeviceSpec` or `OptionsSpec` in `axes/__init__.py`.

Then rewrite the subset so it covers the new value:

```bash
python -m tests.stress.generator --write-subset   # rewrites subset.json
python -m tests.stress.generator --list           # prints every cell id
```

`test_subset_matches_the_axes` fails until `subset.json` matches the axes.

## The oracles

Each oracle returns `pass`, `fail` (a finding, reported with the cell id) or
`not_applicable` (reported as a skip with the reason, for example an eager
forward that itself rejects the dtype).

| Oracle | Judges |
|---|---|
| `replay_equals_eager` | default forward validation: every captured op replays to its eager output |
| `invariants_pass` | the metadata invariants of that same default validation (never loosened) |
| `save_load_equals_live` | a portable save and load returns the same op labels and byte-identical saved outputs |
| `typed_refusal_names_condition` | every error TorchLens raised is a `TorchLensError` with a stable `fields["code"]` and a message |
| `torch_clean_after_cell` | no held capture slots, global hooks or torch-function modes left; wrappers never uninstalled or swapped; trace, save and load leave the global RNG streams as they found them (`tl.validate` documents that its seed reseeds without restoring, so its RNG effect is recorded, not judged) |
| `peak_memory_within_budget` | the `tracemalloc` peak and the resident high-water growth over the cell stay under the row's budget |

A cell whose eager forward itself raises (a dtype the model's own ops reject,
for example) ran no TorchLens stage: every oracle reports it as not
applicable, with the eager error as the reason.

The torch-clean reads are the same process-global reads the suite's autouse
leak fixtures make in `tests/conftest.py`; those fixtures still run and reset
state between tests. The memory oracle under-reports resident growth when the
process already peaked higher earlier in the session, and `tracemalloc` sees
Python allocations only; both are budgets, not measurements.

## Running

The pre-merge subset (about 200 cells, chosen to cover every axis value at
least once, then by cheapest):

```bash
pytest tests/stress -n 4 --dist loadgroup
```

The full matrix, for the nightly or weekly worker leg:

```bash
TORCHLENS_STRESS_FULL=1 pytest tests/stress -n 4 --dist loadgroup
```

`--dist loadgroup` keeps a cell's six oracle tests on one worker so the cell
runs once. The oracle tests carry `slow`; the two structural tests are
unmarked. Nothing here is `smoke`.
