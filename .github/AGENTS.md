# .github/ — CI/CD Configuration

The workflow FILES are the authority; this inventory summarizes them. If they
disagree, the YAML wins — update this doc in the same change.

## Workflows (all eight)

| File | Trigger | What It Does |
|------|---------|-------------|
| `workflows/lint.yml` | PR to main + `workflow_call` | ruff `format --check` + `check` (CHECK-ONLY — see below), full-tree `pre-commit run --all-files` parity job, full-tree gitleaks scan, actionlint + cross-file pin-lockstep gates (torch, pydot, graphviz, jsonschema, pip-audit). |
| `workflows/tests.yml` | PR to main + `workflow_call` | Smoke matrix over exact CPU torch pins (floor 2.1.2 / canonical 2.8.0 / newest-admitted, py 3.10–3.13; the 3.13 row rides the newest-admitted torch) with executed-floor attestations and the render-byte-oracle row; plus var-gated crawler round21 release-proof jobs (`MENAGERIE_RELEASE_RUNNERS`). |
| `workflows/quality.yml` | PR to main + `workflow_call` | mypy on py3.11 + newest-admitted torch; PR-blocking wheel/sdist manifest tripwires; pip-audit with NO suppressions. |
| `workflows/release.yml` | Push to main | Calls lint/tests/quality via `workflow_call`, then python-semantic-release (pinned) versions, builds reproducible artifacts, publishes to PyPI via OIDC trusted publishing and to GitHub Releases with a minimal-scope App token. |
| `workflows/nightly.yml` | Cron + `workflow_dispatch` | Fifteen jobs: perf regression gate, full fast tier, coverage floor (incl. the per-package floor script), capture byte oracle, preview-backend matrix (tf/jax/tinygrad/paddle/mlx), metadata-gates, wheel+sdist double-build reproducibility gate + PEP 561 consumer smoke + forward-load gate (previous PyPI release loads the new goldens), platform-canary, shuffle-stress, HF band legs (hf_floor / hf_4_current / hf_5_candidate constraint legs; the two 4.x legs deselect the R0 rows keyed to the 5.x band until per-band expectations land; the candidate leg runs every R0 row plus the offline R1 + RG gallery against the exact passed-ID floor), LIT bridge on real lit-nlp (+ [lit,hf] resolve on every interpreter, [lit,shap] asserted unsatisfiable), Brain-Score seam on real brainscore-vision, TransformerLens oracle canary ([tlens-oracle] extra), netron vendor parser (pinned 9.2.2 + node), Model Explorer vendor contract (pinned 0.1.32 + node, exact executed count, PyPI/npm drift summary). |
| `workflows/weekly.yml` | Cron + `workflow_dispatch` | Five jobs: slow/rare extended tiers with executed-floor attestations, mutation-margin, hook-pin-staleness, optimized-verdict-identity, netron unpinned-latest drift canary (continue-on-error; the nightly pin is the floor). |
| `workflows/latest-canary.yml` | Cron + `workflow_dispatch` | Smoke tier against latest released torch/torchvision (ecosystem-drift isolation). |
| `workflows/mutation.yml` | Cron (Sun) + `workflow_dispatch` | Rotating per-arm mutation-campaign shard (1-of-4 arm shards weekly; any family on dispatch) in the canonical pinned CPU env; fails on any SURVIVOR/ERROR/TIMEOUT; archives per-mutant verdicts. |

## CI must NEVER auto-commit (locked)

Lint is a pure gate: it checks and fails; it never rewrites files, commits, or
pushes. An earlier design ran `ruff format` + `ruff check --fix` in CI and
pushed the fixes to main — that push raced the Release workflow (both trigger
on push to main) and cascaded release runs (the runaway-release incident
class). Do not reintroduce any workflow that commits or pushes, except the
release job's own semantic-release commit, which is guarded by `[skip ci]`
AND a `chore(release):` head-commit condition.

## Release Pipeline Details

- python-semantic-release and its FULL transitive stack (plus the `build`
  backend) install hash-verified, wheels-only from
  `.github/workflows/release-requirements.txt` (`--require-hashes
  --only-binary :all:`) — no package code executes at install while the
  repo-write App token is on disk; conventional commits.
- Three never-ship-a-major defense layers: commit-msg hook, pre-push hook,
  custom parser that refuses `LevelBump.MAJOR` (see pyproject
  `[tool.semantic_release]` and `scripts/`). Major bumps require explicit
  owner authorization.
- PyPI: OIDC trusted publishing (no API tokens). GitHub auth: App token minted
  with `permission-contents: write` only, installed before checkout persists
  credentials.
- Artifacts are bit-reproducible (SOURCE_DATE_EPOCH wheel;
  `scripts/normalize_sdist.py` sdist); the nightly double-build gate attests
  both. Reproduce-from-tag recipe: pyproject build_command comment.
- Release-notes body uses the capped `templates/.release_notes.md.j2`
  (hard 100K-character budget; the uncapped builtin caused the 422-blocks-PyPI
  incident).

## Pinning & Permissions

- Third-party actions are SHA-pinned. ONE documented exception:
  `pypa/gh-action-pypi-publish@release/v1` (PyPA's own guidance; rationale in
  release.yml).
- Pre-commit hook repos are SHA-pinned — INCLUDING ruff-pre-commit (its old
  stay-on-tag exception was reversed: it is the one hook with `--fix` write
  access, so a mutable tag was the wrong exception; the ruff version is
  lockstep-parsed by `tests/test_packaging_diet.py`).
- Every workflow declares `permissions: contents: read`; checkout uses
  `persist-credentials: false` outside the release job.

## Conventions

- Conventional commits required: `fix(scope):`, `feat(scope):`, `chore(scope):`
- `fix:` → patch bump, `feat:` → minor bump; major-bump markers (`feat!:`,
  `BREAKING CHANGE:`) are BLOCKED by the three defense layers above
- `chore:`, `docs:`, `ci:`, `test:` → no release
