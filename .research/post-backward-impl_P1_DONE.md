# Post-backward Megasprint P1 DONE

Date: 2026-05-12
Branch: `codex/post-backward-megasprint`
Code SHA before this note: `c37ad47`
Status: Ready for P2

## Summary

Implemented P1 multi-output module handling:

- Parameterized multi-output ops include output identity in eq-class keys.
- Module exits preserve structured output order for LSTM, GRU, MHA, dict, and nested outputs.
- `ModuleCallLog.outputs` and `ModuleLog.outputs` return `OpLog` objects.
- `ModuleCallLog.output_structure` and `ModuleLog.output_structure` preserve `ContainerSpec`.
- `OpLog.multi_output_role` is populated from semantic role hints or container paths.
- `tl.output(...)` and `tl.func(..., output=...)` selectors resolve output index/role.
- Save/load reattaches module output references, including legacy bundle tolerance.
- Ambiguous `.out` access raises `MultiOutputModuleError`.

## Commits

- `b98f659` fix(loop-detection): include output index in multi-output eq-class
- `3b9c0d3` feat(data): add module role hints and output structure capture
- `e449190` feat(intervention.types): rebuild containers from output specs
- `d3fd652` feat(module-log): expose structured multi-output module outputs
- `e69932f` feat(selectors): add output selector disambiguation
- `20790d4` fix(save-load): reattach module output references
- `f6cfe75` feat(errors): add multi-output module error
- `dfdc1a5` test(module): cover multi-output module traces
- `4273d0d` chore(changelog): note multi-output module semantics
- `c37ad47` fix(save-load): tolerate legacy module logs without calls

## Tests

- `pytest tests/test_multi_output_modules.py -x --tb=short`: 10 passed.
- `pytest tests/test_tlspec_backcompat.py::test_v2_16_tlspec_fixture_loads_and_matches_in_memory_counterpart -x --tb=short`: 6 passed.
- `mypy torchlens/`: success, 200 source files.
- `pytest tests/ -m smoke -x --tb=short`: 196 passed, 2214 deselected.
- `ruff check torchlens/ tests/test_multi_output_modules.py --fix`: passed.
- `ruff check . --fix`: blocked by unrelated dirty notebook syntax in `notebooks/torchlens_in_10_minutes.ipynb` cell 27: `from torchvision import `.

## Audit

P1z audit command:

```bash
grep -rn 'module\.outputs\b' tests/ torchlens/ docs/ examples/ notebooks/ 2>/dev/null || true
```

Result: no callers expecting label-string `module.outputs`.

## Files

Primary implementation files:

- `torchlens/backends/torch/tensor_tracking.py`
- `torchlens/backends/torch/ops.py`
- `torchlens/backends/torch/model_prep.py`
- `torchlens/backends/torch/sources.py`
- `torchlens/data_classes/_module_role_hints.py`
- `torchlens/data_classes/module_log.py`
- `torchlens/data_classes/op_log.py`
- `torchlens/data_classes/layer_log.py`
- `torchlens/data_classes/model_log.py`
- `torchlens/postprocess/finalization.py`
- `torchlens/intervention/types.py`
- `torchlens/intervention/selectors.py`
- `torchlens/intervention/resolver.py`
- `torchlens/intervention/hooks.py`
- `torchlens/intervention/errors.py`
- `torchlens/errors/__init__.py`
- `torchlens/_io/accessor_rebuild.py`
- `tests/test_multi_output_modules.py`
- `CHANGELOG.md`

## Notes

Visualization-specific code did not require a separate renderer change in P1:
the root cause was the layer/module output collapse and LSTM `num_calls`
misclassification. Smoke coverage includes existing visualization smoke tests.

No dead code was identified as newly unreachable.
