# .github/ — CI/CD Configuration

## Workflows

| File | Trigger | What It Does |
|------|---------|-------------|
| `workflows/lint.yml` | Push/PR | Auto-linting with ruff (`ruff format` + `ruff check --fix`), auto-commits fixes via GitHub App |
| `workflows/quality.yml` | Push/PR | Two jobs: (1) mypy type-checking on Python 3.11, (2) pip-audit dependency audit. Both use CPU torch. |
| `workflows/release.yml` | Push to main | Semantic-release v9 (conventional commits). Publishes to PyPI via trusted OIDC + GitHub Releases. |

## Release Pipeline Details
- Semantic-release v9 (pinned `>=9,<10`), `major_on_zero = true`
- `fetch-tags: true` in checkout step for proper version calculation
- PyPI trusted publishing via OIDC (no API tokens)
- GitHub App (`torchlens-release`) for auth
- Branch protection via rulesets

## Conventions
- Conventional commits required: `fix(scope):`, `feat(scope):`, `chore(scope):`
- `fix:` → patch bump, `feat:` → minor bump, `feat!:` → major bump
- `chore:`, `docs:`, `ci:`, `test:` → no release
