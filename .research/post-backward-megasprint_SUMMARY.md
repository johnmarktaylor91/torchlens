# Post-Backward Megasprint Summary

date: 2026-05-12
branch: codex/post-backward-megasprint
state: DONE

## Scope

The post-backward megasprint integrated the P1-P6 work across multi-output
module capture, oracle/replacement state preservation, stack/backward handling,
early-abort behavior, MLX coverage, and finalize hardening.

## Final T2 Sweep

Initial comprehensive T2 found 12 non-pre-existing failures:

- module-pass CSV/JSON round-trip recursion in pandas JSON export
- stale module containment snapshots for multi-output module outputs
- recurrent and sequence-model metadata invariant failures for multi-output
  parameterized ops
- stale report namespace budget assertion

All were fixed or refreshed with targeted verification, then the full T2 sweep
passed:

```text
pytest tests/ -m "not slow" --tb=no -q
2228 passed, 24 skipped, 211 deselected, 2 xfailed, 942 warnings in 1549.81s
```

## Fix Commits

- `68f1280` fix(io): serialize module pass tabular outputs
- `8b85905` fix(validation): allow multi-output parameterized ops
- `dc4a2da` test(module): refresh multi-output containment snapshots
- `54ff9f2` test(report): align top-level api budget

## Quality Gates

```text
ruff check torchlens/ --fix
All checks passed!

mypy torchlens/
Success: no issues found in 203 source files

pytest tests/ -m smoke -x --tb=short
199 passed, 2266 deselected, 119 warnings in 37.56s
```

The known whole-tree ruff blocker remains pre-existing and outside this sprint:
`notebooks/torchlens_in_10_minutes.ipynb` cell 27 contains
`from torchvision import ` with an empty import list.

## Ready State

The branch is ready for review after the finalize documentation commits. No
new dead code was identified during the T2 regression fixes.
