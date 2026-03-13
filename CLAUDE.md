# TorchLens — Claude Code Architect Briefing

## Project Overview
TorchLens is a Python package for extracting activations and computational graph metadata
from PyTorch models. Published in Scientific Reports and used in Neuromatch Academy. It
permanently wraps all PyTorch functions at import time with toggle-gated wrappers, runs
forward passes with the toggle enabled, and logs every operation into ModelLog/LayerLog/
LayerPassLog objects. ~20,800 lines core code (36 modules across 7 subpackages), ~1,004
tests across 16 test files.

## Architecture
See `.project-context/architecture.md` for the full map.

Key entry points:
- Main API: `torchlens/user_funcs.py` — `log_forward_pass()`, `show_model_graph()`, `validate_forward_pass()`
- Tests: `pytest tests/ -m "not slow" -x --tb=short`
- Build: `pip install build && python -m build`

## How to Read This Codebase
- Start with `torchlens/__init__.py` (public API + import-time decoration trigger), then `user_funcs.py` (entry points), then `capture/trace.py` (forward-pass orchestration)
- Complexity lives in `postprocess/loop_detection.py` (isomorphic subgraph expansion), `capture/output_tensors.py` (exhaustive/fast-path split), and `visualization/elk_layout.py` (ELK subprocess + large graph layout)
- Stable: `_state.py`, `utils/`, `decoration/`. In-flux: `visualization/` (dagua integration ongoing), `capture/arg_positions.py` (coverage expanding)

## Testing Tiers
```
# Tier 1 — Fast (run on every change)
pytest tests/ -m smoke -x --tb=short -q

# Tier 2 — Medium (run when module boundaries change)
pytest tests/ -m "not slow" -x --tb=short

# Tier 3 — Full (run during downtime / before release)
ruff check . && mypy torchlens/ && pytest tests/ -v
```

## Build & Packaging
- Build system: setuptools via `pyproject.toml`
- Build: `pip install build && python -m build`
- Install (dev): `pip install -e ".[dev]"`
- Install (test): `pip install -e ".[test]"`
- Current version: see `pyproject.toml` and `torchlens/__init__.py`

## Commit Convention
Conventional commits for semantic-release:
```
<type>(<scope>): <description> (#<issue>)
```
- `fix:` → patch bump, `feat:` → minor bump, `feat!:` → major bump
- `chore:`, `docs:`, `ci:`, `refactor:`, `test:`, `style:` → no release
- Always reference the GitHub issue at end of first line when one exists

## Linting
- `ruff format` — auto-formatting
- `ruff check --fix` — linting with auto-fixes
- Line length: 100 (configured in pyproject.toml)
- Ignored rules: E721 (type comparison), F401 (unused imports)

## Dispatch Configuration

### Branch Strategy
Codex works on feature branches: `codex/<task-id>`
Default: one branch at a time besides main. Don't spawn extras unless asked.

### Task ID Convention
Descriptive kebab-case: `fix-trace-module`, `add-sweep-dataclass`

### Quality Gates (every Codex task must pass)
```
ruff check . --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short
```

## PR Workflow
```bash
# Create
gh pr create --title "<title>" --body "<description>"

# After merge (user says "merged" or "clean up")
git checkout main && git pull origin main
git branch -d <branch> && git remote prune origin
```

## What NOT to Dispatch
- API design decisions (discuss with user first)
- Changes to public interfaces without approval
- CI/CD, deployment, release configs
- CLAUDE.md, AGENTS.md, .project-context/*.md modifications

## Project Structure
```
torchlens/           # main package (see torchlens/CLAUDE.md)
  _state.py          # global toggle, session state, context managers
  user_funcs.py      # public API entry points
  constants.py       # FIELD_ORDER tuples, function discovery sets
  capture/           # real-time tensor operation logging (7 files)
  data_classes/      # ModelLog, LayerLog, LayerPassLog, etc. (10 files)
  decoration/        # one-time torch wrapping + model prep (2 files)
  postprocess/       # 18-step pipeline (6 files)
  validation/        # forward replay + invariant checks (3 files)
  visualization/     # graphviz + ELK rendering + dagua bridge (3 files)
  utils/             # shared utilities (7 files)
tests/               # test suite (see tests/CLAUDE.md)
scripts/             # dev utilities (see scripts/CLAUDE.md)
.github/             # CI/CD workflows (see .github/CLAUDE.md)
.project-context/    # architecture docs, conventions, task state
```

## Architecture Quick Reference
```
import torchlens  →  decorate_all_once() wraps ~2000 torch functions (ONE TIME)

log_forward_pass(model, input)
  1. Prepare model (decoration/model_prep.py)
  2. Run forward pass with logging (capture/)
  3. 18-step postprocess pipeline (postprocess/)
  4. Return ModelLog with all data
```

Two-pass strategy: when `layers_to_save` is a specific list, Pass 1 runs exhaustive
(metadata only), Pass 2 runs fast (saves only requested layers).

## Dagua Integration
Optional dagua-based visualization path. Graphviz remains the default.

Key files:
- `torchlens/visualization/dagua_bridge.py` — ModelLog → DaguaGraph semantic mapping
- `torchlens/visualization/rendering.py` — dispatches between Graphviz and dagua
- `torchlens/data_classes/model_log.py` — exposes `to_dagua_graph`, `render_dagua_graph`
- `scripts/build_torchlens_theme_gallery.py` — reference gallery builder
- `tests/test_dagua_theme.py` — smoke coverage for the bridge

Policy: `vis_renderer="graphviz"` is still the default. Dagua is opt-in. Visual semantics
still under iteration — do not switch defaults casually.

## Visualization Notes
- Graphviz is the stable/default end-user visualization path
- The dagua path is valuable as infrastructure: semantic graph extraction, theme iteration,
  alternative rendering experiments, future large-graph workflows
- When working on visualization, keep the distinction clear between semantic mapping quality
  (ModelLog → graph structure), layout quality, and rendering/theme quality — they regress
  independently
