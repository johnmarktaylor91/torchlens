# P7C Done

## Commits

- `e31592e` `test(api): update backward parity surface invariants`
- `c0584b5` `fix(alpha3): materialize failed partial traces`
- `b7b3627` `fix(alpha3): restore streaming strict capture writes`
- `b665119` `fix(alpha3): keep auxiliary T2 paths compatible`

## Final T2 Result

Command:

```bash
pytest tests/ -m "not slow" -x --tb=short
```

Result:

```text
2165 passed, 24 skipped, 209 deselected, 2 xfailed, 934 warnings in 580.73s (0:09:40)
```

The MLX smoke test passed in this environment.

## Other Checks

```text
ruff check . --fix
```

Blocked by pre-existing dirty notebook syntax in `notebooks/torchlens_in_10_minutes.ipynb`
cell 27: `from torchvision import `. The notebook was outside P7C scope and was not edited.

```text
ruff check <P7C touched files> --fix
Success: all checks passed.

mypy torchlens/
Success: no issues found in 199 source files.

pytest tests/ -m smoke -x --tb=short
194 passed, 2206 deselected, 117 warnings in 19.48s.
```
