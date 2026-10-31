# .github/ — CI/CD Configuration

The workflow FILES are the authority; this inventory summarizes them. If they
disagree, the YAML wins — update this doc in the same change. `test-tiers.json`
(`ci.workflows`, `ci.budget_minutes_per_job`) lists the same four files.

## Workflows (four)

| File | Trigger | What It Does |
|------|---------|-------------|
| `workflows/pr.yml` | PR to main, push to `ci/**`, dispatch, `workflow_call` | Each tier once: `lint` (ruff format/check, `pre-commit run --all-files`, full-tree gitleaks, the ruff hook SHA check, actionlint), `type-check` (mypy), `smoke` (the one pinned environment, python 3.10 + torch 2.13.0, where the byte goldens enforce; R0 PASSED floor; render and surface byte oracles), `checks` (`scripts/checks/run_all.py`), `packaging` (wheel and sdist manifest tripwires). |
| `workflows/nightly.yml` | Cron + dispatch | `smoke-matrix` (every other supported torch/python row: the torch floors, canonical 2.8, newest admitted on 3.11-3.13, the randomized-order row), `tier` (the `not rare and not slow` tier in the 8 single-process pytest-split shards `test-tiers.json` declares), `latest-canary`, coverage floors, capture byte oracle, preview backends, metadata gates, wheel double-build reproducibility and PEP 561 smoke, platform canary, HF band legs, LIT, bridges, peer recipes, Brain-Score, TransformerLens oracle, netron and Model Explorer vendor contracts, pip-audit, then `attest`. |
| `workflows/weekly.yml` | Weekly cron, monthly cron, dispatch | Slow and rare tiers, mutation margins, the rotating per-arm mutation campaign, hook-pin staleness, optimized-verdict identity, netron latest drift; the large-graph suite only from the monthly cron or dispatch with `large_graphs`; then `attest`. |
| `workflows/release.yml` | PR to main, push of a `v*` tag, dispatch | Builds the reproducible wheel and sdist on every PR as a dry run. On a tag only: re-runs `pr.yml`, refuses a tag that disagrees with `pyproject.toml` and `torchlens/__init__.py` or is not on main, publishes to PyPI by trusted publishing and creates the GitHub Release. |

Executed-test floors: every leg that runs tests writes its junit and a
`legs.floor` line (`<junit> <floor> [skip fraction] [--min-passed N]
[--passed-ids FILE]`) into `$RUNNER_TEMP/junit`. Each workflow checks all of
them with one `scripts/check_ci_executed_tests.py --legs DIR` call (pr: at the
end of the smoke job; nightly and weekly: the `attest` job).

## CI must NEVER auto-commit (locked)

No workflow commits, pushes, tags or bumps a version. Lint is a pure gate: it
checks and fails; it never rewrites files.

## Release Pipeline

- A release is the maintainer's act: set the version in `pyproject.toml` and
  `torchlens/__init__.py`, merge to main, push the `vX.Y.Z` tag on that commit.
  Nothing in CI computes or bumps a version.
- The builder (`build` and the setuptools backend) installs hash-verified,
  wheels-only from `.github/workflows/release-requirements.txt`
  (`--require-hashes --only-binary :all:`).
- Artifacts are bit-reproducible (`SOURCE_DATE_EPOCH` from the tag commit;
  `scripts/normalize_sdist.py`); the nightly double-build gate attests both.
- PyPI: OIDC trusted publishing (no API tokens). The publish job holds only
  `contents: write` and `id-token: write`.

## Pinning & Permissions

- Every job that apt-installs Graphviz runs on `ubuntu-24.04` (Graphviz 2.42), never
  `ubuntu-latest`: the committed galleries and render goldens are byte-checked against one
  Graphviz release, and 26.04's Graphviz 14.1 also leaves neato out of the `graphviz`
  package. `tests/test_packaging_diet.py` enforces it; each install step probes neato.
- Third-party actions are SHA-pinned, including `pypa/gh-action-pypi-publish`.
- Pre-commit hook repos are SHA-pinned, including ruff-pre-commit (the one
  hook with `--fix` write access); the ruff version is lockstep-parsed by
  `tests/test_packaging_diet.py`.
- Every workflow declares `permissions: contents: read`; every checkout uses
  `persist-credentials: false`.

## Conventions

- Conventional commits: `fix(scope):`, `feat(scope):`, `chore(scope):`,
  `docs:`, `ci:`, `test:`. Commit types do not drive versions.
