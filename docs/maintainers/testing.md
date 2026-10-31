# Testing

This page states what a test in this repository is for, when a new one is admitted, which
tiers run where, and how to add a row to the behaviour net or the stress battery. The
commands and markers are the ones in `tests/AGENTS.md` and `test-tiers.json`; when those
two and this page disagree, fix this page.

## Philosophy

A test exists to catch one of three things: a wrong capture, a wrong public result, or a
public contract that changes. Three kinds of test are admitted.

1. **Behaviour tests** through the public surface. One documented seam per subsystem may be
   tested directly, in `tests/internals/`, and those tests stay at most about a tenth of the
   suite.
2. **Property and differential tests**: replay equals the plain forward, every invariant holds
   on every corpus model, save followed by load is the identity.
3. **Corpus tests**: the classics corpus, exploit payloads and example models, each run as one
   parametrized function over a case list.

Refusals are asserted by typed error class and code. Message text is asserted only where a
document promises the wording. Goldens exist only for formats people read or files people
keep; each stays under 200 KB and names its regenerate command. Repository policy (file
caps, marker rules, lint) is checked by scripts and standard linters, never by pytest.

## Admission rule

A new test is admitted when the change says which kind it is:

- a new public behaviour, as a row in the area's existing table;
- a bug that reached a user or the corpus, as a regression row that fails before the fix;
- a new model pattern, as a corpus entry;
- an internal algorithm, as a property test.

It is refused when it imports a private module outside `tests/internals/`, pins a count,
reads the package source or repository files, patches a private symbol, asserts only
`is not None`, asserts undocumented message text, or duplicates an existing row. Anything
over five seconds carries `heavy` or `slow` with the reason in the marker.

## Tiers

The tiers are declared once in `test-tiers.json` at the repository root; CI runs exactly
those tiers and never regenerates a tracked artifact. Every pytest run for this repository
on a shared machine goes through the project's test runner, never a bare `pytest` on a
coordinating host.

| Tier | Selection | Where it runs | Budget |
| --- | --- | --- | --- |
| `inner` | the files mapped to the changed path, or one named test | your worktree | 300 s, one CPU |
| `worker_lane` | one targeted file with `-x` | a worker | 300 s, one CPU |
| smoke (commit gate) | `-m smoke`, about 1,400 coverage-chosen tests under the `SMOKE_TIER_SIZE_CEILING` of 1,500 | `python scripts/smoke_ci_parity.py -n 4`, in CI's pinned environment | about 3 minutes serial |
| mid backstop | `-m "not rare and not slow and not heavy"` | a worker | minutes |
| `pre_merge` (phase boundary) | `-m "not rare and not slow"`, 8 shards by least duration, one seed | a worker, 8 shards | 12,000 s |
| full | `pytest tests/` (`rare` stays excluded by `addopts`) | nightly and weekly CI | per workflow budget |

Byte goldens enforce only in the pinned environment (python 3.10, torch 2.13.0) and fail
closed outside it and outside CI. That is why the commit gate is the parity script and not a
plain `pytest -m smoke` in your own virtual environment.

Markers partition tests by measured cost: `smoke` under 5 s, `heavy` 5 to 20 s, `slow`
over 20 s, `serial` away from parallel workers, `rare` excluded unless selected, `optional`
for optional dependencies. Markers are additive, so `smoke` never combines with `heavy`,
`slow`, `serial` or `rare`. A runtime tripwire in `tests/conftest.py` charges each test's
cost against its tier budget (load-scaled, on the smaller of wall and CPU time) and fails the
session that ran an offender. Run `-n N` with `N` equal to the cores you declared, with one
torch thread per worker; every worker collects the whole suite first, so the gain shrinks
below four workers.

```bash
pytest tests/test_toy_models.py -x --tb=short                 # per-step: the files for the code you touched
python scripts/smoke_ci_parity.py -n 4                        # commit gate
pytest tests/ -m "not rare and not slow and not heavy" -x     # mid backstop
pytest tests/ -m "not rare and not slow" -x --tb=short        # phase boundary
```

## Adding a behaviour-net row

The behaviour net is the corpus-by-feature matrix under `tests/behaviour/`: corpus models
(the classics sample in `tests/classics_corpus/` plus toy models) crossed with features in
pairs and chosen triples, each cell asserting trace, validate, save, load, equality and a
plain-forward match. Its `README.md` in that directory is the authority once the harness
lands; until then:

1. Add the model or input to the corpus list with one line saying what makes it special.
2. Add the feature, or the feature pair, to the feature table; a cell is a row in that
   table, never a new test function.
3. Mark the cell for the tier it belongs to. The cheapest corpus entries crossed with the
   core pairs run in smoke; everything else runs in `slow`; a nightly subset of the model
   gallery runs through the same harness.
4. Run the cell's file through the `inner` tier before the commit gate.

## Adding a stress row

The stress battery under `tests/stress/` holds one axis file per stress family
(`tests/stress/axes/*.py`: models and inputs, dtypes on CPU, control flow, backward and
user hooks, checkpointing, concurrency misuse). Its `README.md` in that directory is the
authority once the harness lands; until then:

1. Pick the axis file whose family the case belongs to, or add an axis file with a
   one-paragraph docstring saying what it varies.
2. Add the case as a row: the model or input constructor, the capture options, and the
   expected outcome (a result, or a typed refusal by error class and code).
3. A failure that is a real bug becomes a finding with the row id; the row stays red until
   the fix lands. Never weaken an assertion or a validation tolerance to turn a row green.

## Shrinking the suite

Deletions follow measured coverage, never judgment alone: a test whose covered lines and
killed mutants are a subset of the kept tests' is a deletion candidate, and the coverage
diff after a batch drops no covered line or branch in `torchlens/` without a named reason.
Sprint-named files are converted before deletion: the model and input that made each one
special becomes a corpus or table row with one line of reason, then the file goes. When a
private module moves, the tests that imported it are deleted unless they hold an edge case
the net lacks. Throughout, the full classics corpus, `tests/test_validation.py` and
`tests/test_toy_models.py` run unchanged around every batch.
