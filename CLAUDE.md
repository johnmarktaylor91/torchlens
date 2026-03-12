# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

TorchLens is a Python package for extracting activations from PyTorch models. It permanently
wraps all PyTorch functions at import time with toggle-gated wrappers, runs forward passes
with the toggle enabled, and logs every operation into ModelLog/LayerLog/LayerPassLog objects.
~20,800 lines core code (36 modules across 7 subpackages), ~1,004 tests across 16 test files.

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

- Run all tests: `pytest tests/`
- Smoke tests (~6s): `pytest tests/ -m smoke`
- Skip slow tests: `pytest tests/ -m "not slow"`
- Linting: `ruff format` + `ruff check --fix`
- Dagua bridge tests: `pytest tests/test_dagua_theme.py`
- TorchLens Dagua gallery builder:
  - `python scripts/build_torchlens_theme_gallery.py`

## Project Structure

- `torchlens/` — main package source ([see subpackage docs](torchlens/CLAUDE.md))
- `tests/` — test suite ([see test docs](tests/CLAUDE.md))
- `scripts/` — development utilities ([see scripts docs](scripts/CLAUDE.md))
- `.github/` — CI/CD workflows ([see CI docs](.github/CLAUDE.md))
- `images/` — documentation images

## Dagua Integration

TorchLens now has an optional Dagua-based visualization path. Graphviz remains the
default renderer.

Key files:
- `torchlens/visualization/dagua_bridge.py`
  - `ModelLog -> DaguaGraph` semantic mapping
  - field audit helpers
  - Dagua rendering entrypoint
- `torchlens/visualization/rendering.py`
  - dispatches between Graphviz and Dagua
- `torchlens/data_classes/model_log.py`
  - exposes `to_dagua_graph`, `render_dagua_graph`, and `visualization_field_audit`
- `scripts/build_torchlens_theme_gallery.py`
  - builds the TorchLens Dagua reference gallery
- `tests/test_dagua_theme.py`
  - smoke coverage for the bridge and renderer

Current policy:
- `vis_renderer="graphviz"` is still the default
- Dagua is opt-in
- visual semantics are still under iteration; do not switch defaults casually

Useful workflow:
- generate a quick Dagua render through `show_model_graph(..., vis_renderer="dagua")`
- inspect the `ModelLog -> DaguaGraph` mapping in `dagua_bridge.py`
- use `tests/test_dagua_theme.py` as the minimum regression surface before refactors

## Architecture Quick Reference

```
import torchlens  ->  decorate_all_once() wraps ~2000 torch functions (ONE TIME)

log_forward_pass(model, input)
  1. Prepare model (decoration/model_prep.py)
  2. Run forward pass with logging (capture/)
  3. 18-step postprocess pipeline (postprocess/)
  4. Return ModelLog with all data
```

Key packages: `capture/` (7 files), `data_classes/` (10 files), `decoration/` (2 files),
`postprocess/` (6 files), `validation/` (3 files), `visualization/` (2 files: rendering.py + elk_layout.py),
`utils/` (7 files).

## Visualization Notes

- Graphviz is still the stable/default end-user visualization path.
- The Dagua path is valuable as infrastructure:
  - semantic graph extraction from `ModelLog`
  - theme iteration
  - alternative rendering experiments
  - future large-graph / cinematic workflows
- When working on visualization, keep the distinction clear between:
  - semantic mapping quality (`ModelLog -> graph structure`)
  - layout quality
  - rendering/theme quality
  They regress independently and should be judged separately.
