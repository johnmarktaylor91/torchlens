# TorchLens Roadmap

## Substrate Hardening: next 6 months

- Bug fixes for known compatibility rows in `tl.report.doctor`.
- Performance work for capture, streaming, and summary generation.
- Schema migrations for `.tlspec/` and bundle metadata as formats settle.
- Quantized models in the summary: a bits-per-weight footer. It waits on honest quantized
  capture: today quantized activation metadata is best-effort (the `quantized_tensor` row in
  `LIMITATIONS.md`), so a footer would report numbers the capture cannot yet back. The
  "coming from flax `tabulate`" page (`docs/migration/from_flax_tabulate.md`) covers
  floating-point summaries now and states this limit.

## Appliance Subfolders: next 6-12 months

- Explore a richer local inspection appliance.
- Mature the shipped `torchlens.notebook` and `torchlens.neuro` package boundaries around
  concrete workflows.
- Keep these as subfolder APIs rather than expanding the top-level namespace.

## Format Stability: 12+ months

- Public commitment for the `.tlspec` body format.
- Alternate storage backends after the current manifest/body split is stable.
- Clearer migration tooling for older saved artifacts.

## Major releases

3.0 shipped on 6 November 2026; 3.0.1 is the first 3.0 release on PyPI (`CHANGELOG.md` lists the
changes, `MIGRATIONS.md` the upgrade path). Further breaking changes wait until enough incompatible
cleanup has accumulated to justify another major release.
