# scripts/ — Development Utilities

## Files

| File | Purpose |
|------|---------|
| `check_flops_coverage.py` | Reports FLOPs module coverage against all decorated torch functions |

## check_flops_coverage.py

Analyzes how many of the ~2000 decorated torch functions have FLOPs computation support
in `capture/flops.py`. Reports:
- Total covered ops (ELEMENTWISE_FLOPS + SPECIALTY_HANDLERS + ZERO_FLOPS_OPS)
- Uncovered ops categorized as private (`_*`), dunder (`__*__`), or public
- Coverage percentage

Run with:
```bash
python scripts/check_flops_coverage.py
```
