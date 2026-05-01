# TorchLens Roadmap

## Substrate Hardening: next 6 months

- Bug fixes for known compatibility rows in `tl.compat.report`.
- Performance work for capture, streaming, and summary generation.
- Schema migrations for `.tlspec/` and bundle metadata as formats settle.

## Appliance Subfolders: next 6-12 months

- Mature `torchlens.viewer` for richer local inspection.
- Mature `torchlens.paper`, `torchlens.notebook`, `torchlens.llm`, and `torchlens.neuro` around
  concrete workflows.
- Keep these as subfolder APIs rather than expanding the top-level namespace.

## Format Stability: 12+ months

- Public commitment for the `.tlspec` body format.
- Alternate storage backends after the current manifest/body split is stable.
- Clearer migration tooling for older saved artifacts.

## 3.0

Breaking changes are deferred until enough incompatible cleanup has accumulated to justify a major
release. Scope and timing are TBD.
