# scripts/ - Development Utilities

## Files

| File | Purpose |
|------|---------|
| `bench_log_forward_pass.py` | Local benchmark harness for capture overhead |
| `build_torchlens_theme_gallery.py` | Generate visualization theme gallery artifacts |
| `check_audit_coverage.py` | Validate total-audit notebook coverage manifest |
| `check_coverage_delta.py` | Coverage delta helper for CI/local checks |
| `check_flops_coverage.py` | Report FLOPs handler coverage against decorated torch functions |
| `check_no_breaking_markers.py` | Reject semantic-release major-bump markers in text inputs |
| `generate_audit_coverage_manifest.py` | Regenerate `notebooks/total_audit/_coverage_manifest.json` |
| `no_major_parser.py` | Semantic-release parser layer that blocks major versions |
| `render_large_graph.py` | Render synthetic large graphs with layout backends |

## Release Safety
The release pipeline has three defenses against accidental major bumps:
- commit-msg hook checks for major-bump markers,
- pre-push hook checks outbound commits,
- `scripts/no_major_parser.py` is configured in `[tool.semantic_release]`.

Do not edit release scripts as part of docs or feature work unless explicitly requested.

## Audit Notebook Tools
`generate_audit_coverage_manifest.py` and `check_audit_coverage.py` gate the Total Audit
Notebook System in `notebooks/total_audit/`. Update the manifest when audit notebooks are
added, removed, or retitled.

## FLOPs Coverage
`check_flops_coverage.py` compares `capture/flops.py` rule coverage with
`constants.ORIG_TORCH_FUNCS` and groups uncovered functions. Use it after changing FLOPs
handlers or torch function discovery.

## Large Graph Rendering

```bash
python scripts/render_large_graph.py 250000
python scripts/render_large_graph.py 1000000 --format png --seed 123
```

Use this for stress-testing ELK/topological layout behavior, not for normal unit tests.
