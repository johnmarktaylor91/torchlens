# io/ - Implementation Guide

Public I/O and administrative helper namespace. Single module (`__init__.py`) that
DELEGATES: the portable save/load implementation lives in `torchlens/_io/` (scrub to
metadata + safetensors blobs, directory bundles); log administration delegates to
`user_funcs`. Keep this package thin — implementation goes in `_io/`, not here.

## Public surface

- `load` / `save` — re-exported from `_io.bundle` (also top-level `tl.load` /
  `tl.save`).
- `detect_tlspec_format(path)` / `inspect_tlspec(path)` — artifact inspection without
  full rehydration; metadata paths reject symlinks
  (`_reject_symlinked_metadata_path`).
- `load_intervention_spec(...)` — intervention-spec loads; foreign `custom` callable
  keys are tolerated for structural analysis but execution resolution denies foreign
  imports unless explicitly trusted (`trust_custom_callables=True`, or better the
  narrow `allowed_custom_callable_modules={...}`).
- `list_logs()` / `reset_naming_counter()` / `log_model_metadata()` — admin
  helpers (`get_model_metadata` was removed; it was an alias of
  `log_model_metadata`).
- `save_intervention(...)` — write an intervention recipe (`Trace.save_intervention`
  routes here).
- `TraceState` — the runtime state enum (`torchlens._trace_state`).
- `suppress_mutate_warnings()` — context manager silencing mutation warnings.
- `PayloadLoadHints` / `JaxPayloadLoadHint`, `TorchLensIOError`, `rehydrate_nested`,
  `cleanup_tmp` — re-exports from `_io`.

## Gotchas

- Rehydration floor: artifacts older than tlspec_version 6 refuse with
  `ArtifactVersionBelowFloorError` (drop-not-resurrect); the floor covers Trace
  rehydration only — legacy 2.16 intervention specs still load.
- The `io`/`_io` split is a recorded dual-home (see
  `.project-context/architecture.md`); do not fold one into the other casually.
