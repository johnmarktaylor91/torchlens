# Repository policy checks

`python scripts/checks/run_all.py` runs every check below and prints one line
each; it exits 1 when any check fails. Each check is a ratchet set at the
numbers measured when it was added: it fails only when a number grows, so
today's tree passes and every later change can only hold or tighten.

| Check | Script | What it measures | Recorded in |
|-------|--------|------------------|-------------|
| `layering` | `layering.py` | Size of the largest import cycle (strongly connected component) in `torchlens`, and the number of in-function internal import statements | `baselines.json` (`layering`) |
| `import-contracts` | `layering.py` | import-linter's `forbidden` contracts between the layers of `torchlens/_architecture.py` | `.importlinter` at the repository root |
| `file-cap` | `file_cap.py` | Package files over 800 lines and functions over 60 lines | `file_cap_allowlist.txt` |
| `id-lint` | `id_lint.py` | Private planning ids (walk rulings, tracker items, review rounds, findings, sprint features, memo items) in package comments and strings and in `docs/` | `baselines.json` (`id_lint`) |

## The import graph

`tools/layering/graph.py` builds the graph the `layering` check measures. It
counts module-top imports and imports inside function bodies alike, because a
deferred import is still a runtime dependency; imports under
`if TYPE_CHECKING:` never run and are counted separately. Run
`python tools/layering/graph.py torchlens --members` for the full summary,
including the members of the largest cycle and the in-function import count
per module.

## The baseline rule

Every recorded number only shrinks.

- When a change lowers a number, lower its baseline in the same change (the
  check prints `lower the baseline` when a number is below its record).
- Raising a baseline, adding an allowlist entry, raising an allowlisted line
  count, removing or loosening an import-linter contract, or narrowing an
  `id_lint.py` family regex needs a one-line reason in the commit message.
- `python scripts/checks/file_cap.py --update` rewrites the allowlist from the
  tree: it drops entries that are now under the cap and lowers counts, and it
  never adds an entry or raises a count.

## Running in CI

A CI leg installs import-linter and runs:

```bash
pip install import-linter
TORCHLENS_CHECKS_STRICT=1 python scripts/checks/run_all.py
```

Without import-linter the `import-contracts` check reports `SKIP`;
`TORCHLENS_CHECKS_STRICT=1` turns a skip into a failure. The checks need only
the standard library and the source tree; nothing imports torch.

## Adding a check

Write `scripts/checks/<name>.py` with a `check(repo: Path) -> Outcome`
function (see `_common.py`; `ratchet` compares measured numbers with a
`baselines.json` section) and append `("<name>", <module>.check)` to `CHECKS`
in `run_all.py`. Add a fixture case to `tests/test_checks_scripts.py`.

## Checks moved from the test suite

These checks used to be test files under `tests/`. They test the repository
(its configuration, documentation, generated files and ledgers), not the
library, so they run here instead of under the test runner. Each module keeps
the old test file's assertions, inputs, thresholds and allowlists, and its
self-tests (planted violations that the check must catch). `governance.py`
lists them in `MOVED`; `run_all.py` runs each through `governance.as_check`.
`python scripts/checks/<name>.py` runs one alone.

| Check | Script | What it checks |
|-------|--------|----------------|
| `internal-id-lint` | `internal_id_lint.py` | No walk or tracker id in `torchlens/` comments and strings |
| `mypy-ratchet` | `mypy_ratchet.py` | The `[tool.mypy]` strictness flags only grow |
| `not-mvp-audit` | `not_mvp_audit.py` | Deferred features stay out of code, docs and tests |
| `filename-convention-lint` | `filename_convention_lint.py` | No new sprint-token test file names |
| `deprecation-inventory` | `deprecation_inventory.py` | The package ships no deprecation shims |
| `file-size-ratchet` | `file_size_ratchet.py` | Large-file line ceilings, two-way stale-checked |
| `claims-lint` | `claims_lint.py` | `docs/migration/_claims.json` claims hold |
| `arch-spine-layer-lint` | `arch_spine_layer_lint.py` | Eager upward imports never increase (`arch_spine_layer_baseline.tsv`) |
| `factcore-import-lint` | `factcore_import_lint.py` | One compute-field reader, no health truthiness |
| `arch-tail-vocab-lint` | `arch_tail_vocab_lint.py` | Vocabulary rule and closure census |
| `arch-tail-seam-census` | `arch_tail_seam_census.py` | Seam dispositions |
| `arch-tail-surface-audit` | `arch_tail_surface_audit.py` | The public surface table and its reachable walk |
| `state-access-ratchet` | `state_access_ratchet.py` | Raw `_state` access sites only shrink |
| `error-contract-lockstep` | `error_contract_lockstep.py` | Error-refusal contract document matches the code |
| `docs-lockstep-names` | `docs_lockstep_names.py` | Public documentation names, paths, links and tier claims match the code |
| `terminology-lint` | `terminology_lint.py` | Glossary terminology |
| `removed-spelling-lint` | `removed_spelling_lint.py` | Removed spellings are not taught again |
| `ruff-ratchet` | `ruff_ratchet.py` | Ruff `select` only grows; deferred-rule site counts never grow |
| `known-failures-ledger` | `known_failures_ledger.py` | `tests/known_failures.py` is the single, accurate xfail source |
| `skip-audit` | `skip_audit.py` | Every skip under `tests/` is ledgered |
| `schema-lockstep` | `schema_lockstep.py` | Field catalogs and their ledgers; generated files are current, byte for byte |
| `golden-governance-lint` | `golden_governance_lint.py` | Every golden file is governed or ledgered; golden generators carry the wrap-state guard |

Unlike the checks above them, these import `torchlens` (and so torch), and
`known-failures-ledger` runs a collect-only pass of the test suite, so their
CI leg needs the `dev` and `test` extras installed and a full git checkout.
They are written as `assert` statements, so they refuse to run under
`python -O`. A moved check reports `SKIP` only where its old test skipped
(for example `skip-audit` without the full `test` extra), so run the strict
leg in an environment with the `test` extra.
