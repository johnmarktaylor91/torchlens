# scripts/ - Development Utilities

## Files

| File | Purpose |
|------|---------|
| `backend_ci_changes.py` | Decides whether a preview backend's per-change `backend-protocol` leg runs for a diff (`pr.yml`; rules pinned by `tests/test_backend_framework_pins.py`) |
| `check_ci_executed_tests.py` | Executed-floor attestation over a junit XML (wired into CI legs) |
| `check_flops_coverage.py` | Report FLOPs handler coverage (`torchlens/_capture/flops.py`) against decorated torch functions |
| `check_no_breaking_markers.py` | Reject major-bump markers (`!` types, breaking-change footers) in commit messages and pushed commits (commit-msg / pre-push hooks; behavioral tests: `tests/test_no_breaking_markers_arming.py`) |
| `check_release_major.py` | Release-time hard stop: refuse a major version unless `.github/MAJOR_RELEASE_APPROVED` names it (`release.yml` build job, every pull request and tag) |
| `check_package_coverage_floors.py` | Per-package coverage-floor gate over a coverage JSON (live nightly gate, `nightly.yml`; floors pinned by `tests/test_coverage_floor_governance.py`) |
| `normalize_sdist.py` | Deterministic sdist/wheel normalization for reproducible releases (invoked by `build_command` + the nightly double-build gate) |
| `render_collapse_reference.py` | Regenerate the collapse reference images |
| `render_large_graph.py` | Render synthetic large graphs with layout backends |
| `run_notebooks.py` | Execute every user notebook with nbclient (nightly `notebooks` leg; needs the `test` extra); the `HEAVY_NOTEBOOKS` run alone after the `-j` pool; `--write-outputs` regenerates the stored outputs of the ten-minute tutorial, the learning path, the README-linked notebooks and the Mamba-130m Jacobian lens Colab |

## Release Safety
A major version ships only on the maintainer's explicit command. Three stops, each failing
closed, with the tripwire `tests/test_major_version_guard.py` (run by name in the `pr.yml` smoke job) pinning their wiring:
- `release.yml` runs `check_release_major.py` before every build (pull requests and tags); `publish`
  needs that job, so an unapproved major tag or version never uploads,
- the commit-msg hook rejects major-bump markers,
- the pre-push hook checks outbound commits.
`.github/MAJOR_RELEASE_APPROVED`, containing exactly the approved version, is the only override and is
created only on the maintainer's explicit command.

Do not edit release scripts as part of docs or feature work unless explicitly requested.

## FLOPs Coverage
`check_flops_coverage.py` compares `_capture/flops.py` rule coverage with
`constants.ORIG_TORCH_FUNCS` and groups uncovered functions. Use it after changing FLOPs
handlers or torch function discovery.

## Large Graph Rendering

```bash
python scripts/render_large_graph.py 250000
python scripts/render_large_graph.py 1000000 --format png --seed 123
```

Use this for stress-testing rank layout behavior, not for normal unit tests.
