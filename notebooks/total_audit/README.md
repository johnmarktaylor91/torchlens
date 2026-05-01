# TorchLens Total Audit

Total Audit is the maintainer sandbox for checking every public TorchLens name
with small, refreshable notebooks.

## Start Order

Run the notebooks in numeric order. The fastest orientation path is:

1. `00_install_and_smoke.ipynb`
2. `01_basic_capture.ipynb`
3. `02_layer_indexing.ipynb`
4. `06_modellog_anatomy.ipynb`
5. `17_intervention_helpers.ipynb`
6. `25_compat_truth_table.ipynb`

Every notebook is self-contained and creates its own temporary directory.

## Refresh Procedure

1. Regenerate the manifest:
   ```bash
   python scripts/generate_audit_coverage_manifest.py
   ```
2. Check strict coverage:
   ```bash
   python scripts/check_audit_coverage.py --strict
   ```
3. Execute changed notebooks with papermill:
   ```bash
   papermill notebooks/total_audit/00_install_and_smoke.ipynb notebooks/total_audit/00_install_and_smoke.ipynb
   ```
4. For a full rebuild, follow `REGEN_PROMPT.md`. It is intentionally
   self-contained and does not depend on planning documents outside the repo.

## Coverage Commands

```bash
python scripts/generate_audit_coverage_manifest.py
python scripts/check_audit_coverage.py --strict
python scripts/check_coverage_delta.py
```

`_coverage_manifest.json` is the public-name inventory. `_coverage_matrix.md`
is the generated notebook-by-item report. Compatibility aliases are ignored by
strict notebook coverage and protected by the delta check.

## If A Notebook Breaks

First rerun just that notebook with papermill so the failing cell is visible.
If the failure is an API rename, regenerate the manifest and move the affected
coverage marker to the most relevant notebook. If the failure is an optional
dependency, keep the notebook on the local smoke path and turn the dependency
case into a small expected-failure cell. Keep outputs bounded: no cell output
over 50 KB, no image output over 200 KB, and no notebook over 1 MB.
