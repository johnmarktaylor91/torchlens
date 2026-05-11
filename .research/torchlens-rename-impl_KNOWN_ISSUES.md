# TorchLens Rename Impl Known Issues

Updated: 2026-05-03T01:02:23-04:00

## Verification Status

- `pytest tests/ -m smoke -x --tb=short`: PASS
- `ruff check . --fix`: PASS
- `mypy torchlens/`: PASS
- `python -c "import torchlens"`: PASS

## Known Issues

- No blocking package issues found in phase 6 verification.
- Smoke emits pre-existing deprecation warnings for APIs outside the locked rename scope, including intervention/site helpers and option aliases. Deferred rename clusters were intentionally left alone.
- Notebook execution verification was limited to the first five `notebooks/total_audit/*.ipynb` notebooks. The remaining notebooks were mechanically renamed but not executed in this dispatch.

## Deferred Per Audit

- `Site` / `find_sites` / `resolve_sites` / `attach_hooks` and intervention API parameter cascade.
- Conditional flow fields such as `conditional_then_edges`.
- `fastlog` namespace, including `preview_fastlog`.
- Streaming helpers `replace_run_state_from` and `append_run_state_from`.
- `vis_*` visualization parameter prefix cascade.
- Bundle cluster G dynamic helpers via `__getattr__`.
- Larger storage refactors for ParamLog dedup and LayerLog/OpLog layout.
