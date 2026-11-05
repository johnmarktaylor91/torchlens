# The behaviour net

One parametrized harness replaces hand-written per-feature test files: every
model in the net (the classics corpus plus the toy models) runs under pairs of
the interaction-matrix features, and every cell asserts the same six
oracles through the public surface:

| oracle | what it checks |
| --- | --- |
| `trace` | `tl.trace` finishes with ops and the outcome the features declare (COMPLETE, or STOPPED under `halt`) |
| `validate` | `tl.validate(model, inputs, scope="forward", validate_metadata=True)` is true, with the default tolerances |
| `save` | `trace.save(path, level="portable")` succeeds |
| `load` | `tl.load(path)` returns a trace |
| `equality` | the loaded trace equals the live one: op labels, structural hash, per-op shape, dtype, parents, children, pass index, module addresses and saved payloads |
| `forward_match` | the trace's reconstructed output equals a plain `model(*inputs)` exactly; under a capture edit it is the replay (`trace.run`) that must reproduce the captured output; under `halt` the halt feature checks the op prefix instead |

After the six, each active feature's own promise is checked as the oracle
`feature:<name>` (for example `save` keeps exactly the selected ops, `fork_do`
halves the edited site, `record` reproduces the op sequence).

Each cell is ONE test, `test_cell[<case>-<featureA>-<featureB>]`, that judges
every oracle and never stops at the first unsatisfied one. A failing cell
fails with one message that names each failing oracle with its status and
reason, one line per oracle:

```text
Neural_Gas-save-index: 2 oracle(s) failed
save: fail: ...
equality: fail: ...
```

An oracle outcome is `pass`, `refused` (a typed TorchLens error, recorded with
its class and code), `not_applicable` (the case cannot take the feature by
design, for example `multipass` on a model with no repeated layer) or `fail`.
`pass`, `refused` and `not_applicable` satisfy an oracle; `fail` (or
`not_run`, a base oracle the cell never reached) fails the cell's test, and a failure is a capture, validation or
persistence finding: root-cause it. Never loosen a tolerance, exempt a check
or mark the cell to make it pass (the validation-integrity rule in
`AGENTS.md`).

## Layout

| file | holds |
| --- | --- |
| `cases.py` | the case registry: corpus entries (from `tests/classics_corpus/manifest.json`) and the toy rows |
| `features.py` | the feature rows (one per interaction-matrix feature): name, cost class, oracles changed, tags required, `prepare` and `after` hooks |
| `harness.py` | `run_cell(case, features, tmp) -> CellResult` and the public-surface helpers |
| `test_matrix.py` | the cell universe, one test per cell (`test_cell`), the smoke hook (`SMOKE_CELLS`), and the four structural pins (the feature axis equals `tests/interaction_matrix.json`'s, the core pairs are admitted by it, the six base oracles, the stochastic tag), plus a pin that a cell's verdict names every unsatisfied oracle |
| `conftest.py` | the `cell_result` fixture: runs one cell in its own temp directory |
| `test_cell_tmp.py` | pins that a cell removes its temp directory when it ends |
| `test_f7_intervention_persistence.py` | the saved intervention recipe (io.1), one cell per promise: a built-in edit recipe reruns to the live activations, loads equal to the live recipe, a user callable saves as a disclosed placeholder, every rerun door refuses it, and both saves and loads run their validators |

## The cell universe

* The cheap corpus cases (the manifest's `smoke` tier, the 13 cheapest
  measured entries) times `CORE_PAIRS`: `heavy` (each cell takes 1 to 7 s, too
  many for an unmarked family's aggregate budget).
* Every pair of the interaction-matrix features once, spread round-robin over the cheap
  cases: `slow`.
* Every other case (the rest of the corpus and the toys) times `CORE_PAIRS`:
  `slow`.

Each cell runs once, inside its own test, so no worker ever recomputes a
cell another worker ran. Cells keep their `xdist_group` marks; run with
`--dist loadgroup`:

```bash
OMP_NUM_THREADS=1 TORCHLENS_SMOKE_PREFILTER=0 pytest tests/behaviour -m "not slow" -n 4 --dist loadgroup
pytest tests/behaviour -n 8 --dist loadgroup          # everything, including slow
pytest "tests/behaviour/test_matrix.py::test_cell[Neural_Gas-save-index]"   # one cell
TORCHLENS_BEHAVIOUR_UNIVERSE=sweep pytest tests/behaviour -n 4   # every pair once (slow)
```

`TORCHLENS_BEHAVIOUR_UNIVERSE` is `cheap`, `sweep`, `core` or `all` (the default)
and narrows which part of the universe is collected.

Each cell writes its artifacts (the saved bundle, disk-streamed payloads,
rendered graphs) into its own temp directory and removes it as soon as the
cell has run, pass or fail, so a long run does not fill the disk. To inspect
a failing cell's files, set `TORCHLENS_BEHAVIOUR_KEEP_TMP=1`: the directory of
a cell with a failing oracle (or one that crashed) is then kept under pytest's
basetemp, named after the cell id. Passing cells are removed either way.

Nothing here carries `smoke` yet: the smoke tier sits near its size ceiling
(`SMOKE_TIER_SIZE_CEILING` in `tests/conftest.py`), and smoke admission is a
separate decision recorded in `tests/AGENTS.md`. The hook for it is
`SMOKE_CELLS` in `test_matrix.py`: cheap-cell ids listed there run in the
smoke tier instead of `heavy` once `_SMOKE_TIER` is set to the smoke marker
spelled in that file (the smoke file list is a text scan of each test file's
own source, so the marker is spelled there, never through a conftest alias).
An id that is not a cheap cell raises at import.

## Adding a case

Add a row, never a file.

* A corpus model: add the entry to `tests/classics_corpus/manifest.json`
  following `tests/classics_corpus/README.md` (vendored byte-identical, pinned
  by sha256). Tags are derived from its census features.
* A toy model: add a `_toy(name, ModelClass, inputs_factory, *tags)` row to
  `TOY_CASES` in `cases.py`. Tags the features consult: `child_module`,
  `float_output`, `recurrent`, `stochastic` (the forward draws random
  numbers, so second-run comparisons are not applicable; a corpus case gets it
  when a `func:` feature is on the hand list or names an ATen sampler that
  always draws: torch tags it `nondeterministic_seeded`, it takes a `Generator?`
  argument and no `train`/`training` gate, so `exponential_` counts and
  eval-mode `lstm` or `dropout_` do not), `containers` (the
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
