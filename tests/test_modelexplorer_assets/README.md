# Pinned Google Model Explorer vendor contract assets (F15 harness)

These files pin the EXECUTED vendor contract for the TorchLens Model Explorer
exporter (`torchlens/export/_model_explorer/`). Three of the four data-loss
bugs the 2026-08-22 tri-lab panel found are invisible to a dataclass parse and
visible only to Model Explorer's real graph processor, so the harness runs the
real pinned worker under Node instead of trusting any schema reading.

| File | Provenance |
|---|---|
| `worker.js` | `dist/worker.js` from npm `ai-edge-model-explorer-visualizer` 0.1.2 (Apache-2.0, Google Inc.); sha256 pinned in `checksums.sha256` |
| `visualizer_custom_element.d.ts` | `src/custom_element/index.d.ts` from the same npm package (embed-config interface snapshot for the future tsc/Playwright leg) |
| `run_worker.cjs` | TorchLens-authored headless Node runner: loads `worker.js` in a `vm` sandbox with worker-global shims, feeds it graph payloads, prints per-graph stats JSON |
| `vendor_loader.py` | Test-support loader for the pinned pip package `ai-edge-model-explorer` 0.1.32 schema modules without executing the package `__init__` (whose server imports need flask etc.) |
| `checksums.sha256` | sha256 pins for the two vendor files |

The pip-side schema of record is `model_explorer.graph_builder` /
`model_explorer.node_data_builder` from `ai-edge-model-explorer==0.1.32`
(install with `--no-deps` for the schema-only environment). The npm
TypeScript-interface compile leg needs a `tsc` toolchain and is recorded as an
open remainder in the F15 lane report; the interface snapshot is pinned here so
that leg can be added without re-fetching the package.

Update procedure: bump BOTH package pins deliberately, re-copy the files,
regenerate `checksums.sha256`, and treat every harness assertion change as a
reviewed vendor-contract-profile update (megaplan B11 canary discipline) --
never a silent rebaseline.
