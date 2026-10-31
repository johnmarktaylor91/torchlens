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
