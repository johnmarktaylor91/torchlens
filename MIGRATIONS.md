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

`Trace.draw(collapse="auto")` now uses the v2 collapse engine by default. The
`TORCHLENS_COLLAPSE_ENGINE` environment variable still overrides the default, and
`collapse="max"` remains on the v1 engine until the R4 max-mode work lands.

Two S5 grain metric rows are intentionally rebaselined: `deeplabv3_resnet50` and
`convnext_tiny` are superseded by the flip-gate visual ruling. For DeepLabV3,
v1's lower variance came from a degenerate opaque ASPP cut; v2 exposes the ASPP
branches. For ConvNeXt Tiny, v2's extra downsample detail is treated as a
legitimate overview detail rather than a blocking regression.
