# The behaviour net

One parametrized harness replaces hand-written per-feature test files: every
model in the net (the classics corpus plus the toy models) runs under pairs of
the 21 interaction-matrix features, and every cell asserts the same six
oracles through the public surface:

| oracle | what it checks |
| --- | --- |
| `trace` | `tl.trace` finishes with ops and the outcome the features declare (COMPLETE, or HALTED under `halt`) |
| `validate` | `tl.validate(model, inputs, scope="forward", validate_metadata=True)` is true, with the default tolerances |
| `save` | `tl.save(trace, path, level="portable")` succeeds |
| `load` | `tl.load(path)` returns a trace |
| `equality` | the loaded trace equals the live one: op labels, structural hash, per-op shape, dtype, parents, children, pass index, module addresses and saved payloads |
| `forward_match` | the trace's reconstructed output equals a plain `model(*inputs)` exactly; under a capture edit it is the replay (`trace.run`) that must reproduce the captured output; under `halt` the halt feature checks the op prefix instead |

A seventh test per cell, `test_feature_checks`, asserts each active feature's
own promise (for example `save` keeps exactly the selected ops, `fork_do`
halves the edited site, `record` reproduces the op sequence).

An oracle outcome is `pass`, `refused` (a typed TorchLens error, recorded with
its class and code), `not_applicable` (the case cannot take the feature by
design, for example `multipass` on a model with no repeated layer) or `fail`.
Only `fail` fails the test, and a failure is a capture, validation or
persistence finding: root-cause it. Never loosen a tolerance, exempt a check
or mark the cell to make it pass (the validation-integrity rule in
`AGENTS.md`).

## Layout

| file | holds |
| --- | --- |
| `cases.py` | the case registry: corpus entries (from `tests/classics_corpus/manifest.json`) and the toy rows |
| `features.py` | the 21 feature rows: name, cost class, oracles changed, tags required, `prepare` and `after` hooks |
| `harness.py` | `run_cell(case, features, tmp) -> CellResult` and the public-surface helpers |
| `test_matrix.py` | the cell universe, one test per oracle, and the pins that tie the feature axis to `tests/interaction_matrix.json` |
| `conftest.py` | the per-cell result cache and the xdist grouping |

## The cell universe

* The cheap corpus cases (the manifest's `smoke` tier, the 13 cheapest
  measured entries) times `CORE_PAIRS`: unmarked, the mid tier.
* Every pair of the 21 features once, spread round-robin over the cheap
  cases: `slow`.
* Every other case (the rest of the corpus and the toys) times `CORE_PAIRS`:
  `slow`.

Run it with the cell grouping so one cell's seven tests share one worker
(measured 2026-10-31: under this repository's root `conftest.py` the
`xdist_group` marks are not yet honoured, so each worker recomputes the cells
it draws; the cheap tier still finishes in under 3 minutes at `-n 4`):

```bash
OMP_NUM_THREADS=1 TORCHLENS_SMOKE_PREFILTER=0 pytest tests/behaviour -m "not slow" -n 4 --dist loadgroup
pytest tests/behaviour -n 8 --dist loadgroup          # everything, including slow
pytest tests/behaviour -k "Neural_Gas-save-index"       # one cell
```

Nothing here carries `smoke`: the smoke tier sits near its size ceiling
(`SMOKE_TIER_SIZE_CEILING` in `tests/conftest.py`), and smoke admission is a
separate decision recorded in `tests/AGENTS.md`.

## Adding a case

Add a row, never a file.

* A corpus model: add the entry to `tests/classics_corpus/manifest.json`
  following `tests/classics_corpus/README.md` (vendored byte-identical, pinned
  by sha256). Tags are derived from its census features.
* A toy model: add a `_toy(name, ModelClass, inputs_factory, *tags)` row to
  `TOY_CASES` in `cases.py`. Tags the features consult: `child_module`,
  `float_output`, `recurrent`, `stochastic` (the forward draws random
  numbers, so second-run comparisons are not applicable), `containers` (the
  inputs are not one tensor), `inplace`, `views`, `branching`, `dtype`.

## Adding a feature

Add a `Feature(...)` row to `FEATURES` in `features.py`: the matrix key, a
cost class (`cheap`, `medium`, `heavy`), the oracles it changes
(`forward_match`, `outcome`, `payloads`), the tags it requires, a `prepare`
hook that sets `cell.trace_kwargs` or `cell.capture_fields`, and an `after`
hook that checks its promise with `cell.expect(...)`. Raise `NotApplicable`
when a case cannot take the feature by design. The feature axis must stay
equal to `tests/interaction_matrix.json`'s features (pinned by a test here),
so a new feature is first a matrix row.

## Admission rule

A new test in this repository is admitted when the change says which kind it
is: a new public behaviour as a row in the area's table (here, a feature or a
case row); a bug that reached a user or the corpus as a regression row that
fails before the fix; a new model pattern as a corpus entry; an internal
algorithm as a property test. It is refused when it imports a private module
outside `tests/internals/`, pins a count, reads the package source or
repository files, patches a private symbol, asserts only `is not None`,
asserts undocumented message text, or duplicates an existing row. Anything
over 5 s carries `heavy` or `slow` with the reason.
