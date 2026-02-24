# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

TorchLens is a Python package for extracting activations from PyTorch models. It provides functionality for extracting model activations, visualizing computational graphs, and extracting exhaustive metadata about models.

## Commit Convention

This project uses **conventional commits** for semantic-release. Every commit message must follow this format:

```
<type>(<scope>): <description> (#<issue>)
```

### Types and their version effects:
- `fix:` — patch bump (0.1.36 → 0.1.37)
- `feat:` — minor bump (0.1.36 → 0.2.0)
- `feat!:` or footer `BREAKING CHANGE:` — major bump (0.1.36 → 1.0.0)
- `chore:`, `docs:`, `ci:`, `refactor:`, `test:`, `style:` — no release

### Issue references:
Always reference the GitHub issue being addressed at the end of the first line:

```
fix(logging): handle duplicate tensor entries (#55)
feat(vis): add dark mode to graph visualization (#72)
chore(ci): update release workflow (#80)
```

If there is no issue, omit the issue reference — but prefer having an issue for trackability.

## Build & Packaging

- Build system: setuptools via `pyproject.toml`
- Build: `pip install build && python -m build`
- Install (dev): `pip install -e ".[dev]"`
- Install (test): `pip install -e ".[test]"`

## Testing

- Run tests: `pytest tests/`
- Linting: `black --check .`

## Project Structure

- `torchlens/` — main package source
- `tests/` — test suite
- `images/` — documentation images
- `local_jmt/` — local development scripts (not packaged)
