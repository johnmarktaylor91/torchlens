# TorchLens Migration Policy

## `.tlspec` Compatibility

TorchLens 2.16.0 wrote two public `.tlspec` directory formats:

- Intervention specs with `spec.json` containing `format_version`.
- Portable `ModelLog` bundles with `manifest.json` containing `io_format_version`.

These 2.16.0 formats are permanently readable. TorchLens does not auto-migrate
them in place; readers dispatch by detected format and preserve support for the
legacy schemas.

New writers introduced during the Phase 11 schema graduation emit the unified
manifest format. The polymorphic loader detects the on-disk format and dispatches
to the appropriate reader. During the Phase 11.0 transition, intervention specs
also include `kind: "intervention"` in `manifest.json` while preserving the
2.16.0 fields.

An optional future utility, `torchlens.io.migrate_tlspec(path, dest)`, may write
an upgraded copy in the unified format. It will not be required for loading
existing 2.16.0 files.

## Visualization Collapse Engine

`Trace.draw(collapse=...)` is one v2 smart-collapse surface: `"none"`, `"auto"`, `"max"`,
or a float `t` in `[0.0, 1.0]` on the public monotone schedule (`collapse="max"` is the v2
mode that may emit segment boxes). There is no engine-selection environment variable.

Two S5 grain metric rows are intentionally rebaselined: `deeplabv3_resnet50` and
`convnext_tiny` are superseded by the flip-gate visual ruling. For DeepLabV3,
v1's lower variance came from a degenerate opaque ASPP cut; v2 exposes the ASPP
branches. For ConvNeXt Tiny, v2's extra downsample detail is treated as a
legitimate overview detail rather than a blocking regression.

## run() Declared-State Restore (default behavior change)

The default live `trace.run(inputs=...)` now snapshot-restores the model's declared
state (named parameters plus registered buffers, alias topology preserved) around the
run, so repeated `run()` calls leave the model bit-identical. Previously, forward-pass
mutations of declared state (BatchNorm running statistics, buffer counters, `no_grad`
in-forward parameter updates) persisted on the live model across `run()` calls. Pass
`carry_state=True` (provisional spelling, documented-unstable) for the old persistence;
the report discloses the choice as `report.state_carried`.

## `resample_ablate` -> `scramble_elements` (honest rename, same bytes)

The elementwise-scramble helper is renamed: `resample_ablate` implied the
field's "resampling ablation" (replacing a site's value with another run's
COHERENT donor value), but the helper has always been an elementwise iid
scramble from a flattened source -- a structure-destroying noise baseline.
The canonical constructor is now `torchlens.intervention.scramble_elements`
(same bytes, same seeds, same draws; specs constructed through either
spelling carry helper name `"scramble_elements"`).

Which "resample" do you mean?

| You want | Use |
|---|---|
| Elementwise iid scramble (noise baseline) | `scramble_elements(source, seed=)` |
| Coherent donor patch at the same site ("resampling ablation") | `tl.patch_from(other_trace)` |
| Batch-row permutation | planned stochastic-edit verb (not this helper) |
| Per-row donor resampling | planned stochastic-edit verb (not this helper) |

Compatibility: saved intervention specs and pickles that persist the helper
name `"resample_ablate"` keep loading (they reconstruct through the renamed
constructor). No runtime deprecation shim is added; the old top-level
spelling is retired with the facade export flip.
