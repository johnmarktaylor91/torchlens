# ir/ - Implementation Guide

Internal backend-neutral IR for TorchLens capture unification (per the package docstring).
Almost everything here is internal — with ONE public exception:
`ir/container.py::register_container` is imported by the package
`__init__.py` and listed in the top-level `__all__` (97 names), so a rename
or signature change there is a PUBLIC API break governed by the deprecation
rules, not an internal refactor.

## capture_events.py
- `CaptureEvents` is the ONE logical journal per run: its `append*` methods are the single
  writers and stamp every event of every kind with one run-monotonic `seq` (`next_seq()`).
  Never append to a lane list directly.
- Lane appenders: `append()` (op lane), `append_module_prep()`/`append_module_enter()`/
  `append_module_exit()`, `append_pre_hook()`, `append_buffer_write()`,
  `append_intervention()`, `append_output_version()`, `append_backward()`.
- The op lane is append-only. Post-commit knowledge rides `append_amendment()`
  (`OpAmendment`); amended-state reads fold through `amended_op_records()` /
  `amended_op_record()`.
- Sealed journals refuse writes typed: `SealedJournalWriteError` and its
  `SealedJournalAppendError` / `SealedJournalAmendmentError` subclasses.
- grad-fn handles live only in the journal side index (`grad_fn_handles_by_label_raw`),
  never on op records. `register_live_event()` is the live-capture ingress helper.

## op_record.py / op_record_scatter.py / op_record_manifest.py
- `OpRecord` decomposes as `OpCore` plus typed facets (`GraphFacet`, `ModulesFacet`,
  `AncestryFacet`, `AutogradFacet`, `TransformFacet`, `ControlFacet`, `ParamsFacet`,
  `AnnotationsFacet`, `PolicyFacet`, `RecordingFacet`, `InterventionFacet`);
  `validate_amendment()` guards the exact-set amendment families.
- `scatter_record_to_cells()` maps a record into op-store cells at ingest.
- `op_record_manifest.py` is GENERATED (`CELL_SOURCE_MANIFEST`, regenerate with
  `python -m tools.generate_op_record_manifest`); never hand-edit, never mutate v2 in place.

## events.py
- Compat event dataclasses still emitted by preview backends: `OpEvent`, module events
  (`ModulePrepEvent`, `ModuleEnterEvent`, `ModuleExitEvent`), `BufferWriteEvent`,
  `OutputVersionEvent`, `PreHookProvenanceEvent`, and the backward family
  (`BackwardPassStart`/`BackwardPassEnd`, `GradFnDiscovered`, `GradFnFired`,
  `OpGradObserved`, `ParamGradObserved`, `BackwardCoverageGap`).

## container.py / container_registry.py
- `register_container(...)` is the PUBLIC user entry point for declaring
  custom output-container types (top-level `tl.types.register_container`); treat
  its name and signature as frozen public API.
- `ContainerSpec` + `rebuild_container_from_spec()` describe and rebuild output containers;
  `reconstruction_is_lossy()` gates what may round-trip.
- `ContainerRegistry`, `walk_container()`, `ContainerRecord`/`ContainerSnapshot` record
  container occurrences by `Role`/`Phase` and site (`FuncSite`/`ModuleSite`/`ModelSite`).
- `_container_errors.py` defines `ContainerReconstructionError` (public via `tl.errors`);
  `container.py` re-exports it. It sits below `container.py` and `distribution_container.py`
  so the two import it without importing each other.

## Selection and predicates
- `predicate.py`: `RecordContext` is the predicate-visible view during capture; deferred
  values via `is_deferred_value()` / `coerce_deferred_value()`.
  It also holds what a predicate returns and what a capture retains (`CaptureSpec`,
  `StorageIntent`, `ActivationRecord`, `GradRecordContext`, `GradRecord`, `PredicateFailure`),
  published from `torchlens.recording.types` (their `__module__` says so); torch types are
  annotation-only here.
- `selector_eval.py`: selector walking and capability gating (`walk_selector()`,
  `ensure_supported()`, `contains_followed_by()`) and the compile-once evaluator:
  `evaluate()` compiles a selector once per lane (`capture`, `site`, `live`, and
  the backward `grad_fn` lane) into a plan of matcher nodes cached on the selector,
  so per-subject evaluation never branches on selector shape; refusals compile to
  nodes that raise when reached, keeping short-circuit behaviour. Unsupported
  shapes raise `SelectorCapabilityError` (defined in `intervention/errors`). The
  `facet` kind refuses in every lifecycle here; its working doors sit above the
  evaluator (`resolver.resolve_sites` for a bare facet selector, the save-only
  predicate for `trace(save=)`), and the refusal message names them.
  `selector_from_spec()` is a builder table keyed by spec kind.
- `_selector_nodes.py` / `_selector_universe.py`: modules that import only leaf
  package modules (the live-trace guard, the backend name, the record-type reader),
  holding the duck-typed plan nodes (label, substring, regex, func, func-transform,
  grad-kind and backward-pass matchers) and the per-lifecycle subject adapters
  (label universes, module-address and module-output candidates, output roles,
  module-scope aliases, GradFn type and call readers, the GradFn pass pairing). The `tl.grad` nodes live in
  `selector_eval.py`: on a forward lane `grad` matches what its inner names, and
  on the `grad_fn` lane it is the only bridge from a forward selector to the GradFn
  behind it (a bare forward kind never matches a GradFn).
  Nodes that need package types (container paths, `RecordContext`, GradFn pairing,
  episode sessions) stay in `selector_eval.py` so the split adds nothing to the
  import cycle measured by `scripts/checks/layering.py`.

## Small modules
- `refs.py`: `DtypeRef`, `DeviceRef`, `TensorRef`, `ParamRef`, `DeferredRef`, `ReservedLabel`.
- `live_index.py`: `LiveIndex` windowed lookback index (`LiveIndexWindowError`).
- `intervention.py`: `FunctionEventInput`, `FireResult`, `InterventionTemplateRef`.
- `semantics.py`: `BackendSemantics`, `CapturePolicy` per-backend declarations.
- `workspaces.py`: the three per-phase transient workspaces (`RawGraphWorkspace`,
  `ModuleCaptureWorkspace`, `WrapperRuntimeWorkspace`) that replaced `TraceBuildState`.

## Local invariants
- Cross-kind and forward/backward ordering is an exact recorded fact ONLY because `seq` is
  stamped inside the journal appenders; bypassing `CaptureEvents` breaks that guarantee.
- The generated cell-source manifest is version-frozen: edit the spec in
  `op_record_scatter.py`, regenerate, and bump the version rather than mutating it.
