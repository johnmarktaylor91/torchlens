# TorchLens Rename Impl Summary

Updated: 2026-05-03

## Branch

- `naming-sprint-impl`

## Phases Completed

- 6/6 phases completed.
- Phase commits:
  - `30179a8 refactor(rename): top-level vocab + class renames (phase 1)`
  - `55c22d1 refactor(rename): field renames per cluster locks (phase 2)`
  - `dc73744 refactor(rename): new accessor classes + new fields (phase 3)`
  - `3541292 refactor(rename): method renames (phase 4)`
  - `cb425ec refactor(rename): update notebooks (phase 5)`
  - `c1ac957 chore(rename): green smoke tests after rename (phase 6)`

## Files Modified

- 376 files changed relative to `main`.
- Shortstat: 17,482 insertions, 17,370 deletions.

## Smoke Test Result

- PASS: `pytest tests/ -m smoke -x --tb=short`
- Result: 170 passed, 2082 deselected, 104 warnings.
- Also passed: `ruff check . --fix`, `mypy torchlens/`, and `python -c "import torchlens"`.

## Notebook Status

- Notebook and example references were mechanically updated for the locked renames.
- Executed and passed:
  - `notebooks/total_audit/00_install_and_smoke.ipynb`
  - `notebooks/total_audit/01_basic_capture.ipynb`
  - `notebooks/total_audit/02_layer_indexing.ipynb`
  - `notebooks/total_audit/03_save_load_basics.ipynb`
  - `notebooks/total_audit/05_visualization_basics.ipynb`
- Remaining notebooks are rename-only, execution not verified in this dispatch.

## Deferred Items

- `Site` / `find_sites` / `resolve_sites` / `attach_hooks` and intervention API parameter cascade.
- Conditional flow fields such as `conditional_then_edges`.
- `fastlog` namespace, including `preview_fastlog`.
- Streaming helpers `replace_run_state_from` and `append_run_state_from`.
- `vis_*` visualization parameter prefix cascade.
- Bundle cluster G dynamic helpers via `__getattr__`.
- Larger storage refactors for ParamLog dedup and LayerLog/OpLog layout.

## Known Issues

- No blocking package issues found in phase 6 verification.
- Smoke emits existing deprecation warnings for APIs outside the locked rename scope and for deferred clusters.
- Details are in `.research/torchlens-rename-impl_KNOWN_ISSUES.md`.
