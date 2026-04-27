# CHANGELOG


## v2.3.0 (2026-04-27)

### Features

- **fastlog**: High-throughput activation recording for dynamic models
  ([#155](https://github.com/johnmarktaylor91/torchlens/pull/155),
  [`1b19663`](https://github.com/johnmarktaylor91/torchlens/commit/1b19663d22b2ed4d7e8db5472d7aa8f1198da2b8))

* fix(perf): defer col_offset extraction until after frame filtering

Improves log_forward_pass throughput substantially on models with deep Python call stacks.
  _get_col_offset and _get_code_qualname now run only on frames that survive Phase 1 filtering
  instead of every raw frame.

* feat(fastlog): scaffolding and types contract

* feat(fastlog): predicate evaluator and RecordContext builder

* feat(fastlog): capture-layer predicate dispatchers

* feat(fastlog): orchestrator with root events

* fix(fastlog): freeze Recording dataclass

* feat(fastlog): RAM and sync-disk storage backends + recover()

* feat(fastlog): public record / Recorder / dry_run API

* feat(fastlog): preview_fastlog visualization

Per execution plan Step 6. Adds model_log.preview_fastlog() and tl.preview_fastlog() top-level
  alias. Generates node_spec_fn at call time; does not touch MODE_REGISTRY. Uses
  _build_record_context to synthesize RecordContext from postprocessed LayerLog so a predicate that
  accesses a postprocess-only field fails the same way as in dry_run. preview_fastlog(...,
  vis_renderer='dagua') raises NotImplementedError for v1.

* feat(fastlog): dry_run live visualization

Per execution plan Step 7. Adds print_tree, to_pandas, show_graph (graphviz with module rails, no
  containment boxes), summary, timeline_html, repredicate. RecordingTrace stores decisions without
  tensor payloads.

* feat(fastlog): opt-in incremental enrichments

Per execution plan Step 8. Adds Recording.enrich() with module_path_strings preset and all-feasible
  alias. param_addresses preset declared but raises pending capture-time field plumbing (scoped for
  follow-up; documented in status report).

* test(fastlog): comprehensive sprint test suite

* docs(fastlog): tutorial notebook + module CLAUDE.md


## v2.2.1 (2026-04-25)

### Bug Fixes

- **validation**: Tolerate small replay drift
  ([`8126fe2`](https://github.com/johnmarktaylor91/torchlens/commit/8126fe20808fa9cae5f61543c39d4b3301f688f3))

Validation replay used a fixed absolute float tolerance, which was too tight for deep convolution
  replays in timm models. Compare tolerated replay outputs with a small absolute floor plus relative
  tolerance so numerically equivalent conv outputs validate while real mismatches still fail.


## v2.2.0 (2026-04-25)

### Features

- **visualization**: Side-by-side code panel
  ([`6d3101e`](https://github.com/johnmarktaylor91/torchlens/commit/6d3101e72c1b66c4c02ca90a8b9085d2f43f1e26))


## v2.1.0 (2026-04-25)

### Features

- **visualization**: Node-mode presets (default, profiling, vision, attention)
  ([`ddfdd34`](https://github.com/johnmarktaylor91/torchlens/commit/ddfdd34fad8e608bc4fce0b56ae4d2c42d1a074b))

- **visualization**: Single-module focus (module= arg + ModuleLog.show_graph)
  ([`155be75`](https://github.com/johnmarktaylor91/torchlens/commit/155be7585b4a4143b49cf80485d9b2836b252e43))


## v2.0.0 (2026-04-25)

### Features

- **visualization**: Nodespec callback + collapse_fn + skip_fn + default important args
  ([`0697787`](https://github.com/johnmarktaylor91/torchlens/commit/06977876c670717737d8952ffce651e833a3d45e))

BREAKING CHANGE: vis_node_overrides and vis_nested_node_overrides are removed.

Use the new node_spec_fn / collapsed_node_spec_fn callbacks instead.

See PR body for migration notes.

### Breaking Changes

- **visualization**: Vis_node_overrides and vis_nested_node_overrides are removed.


## v1.7.0 (2026-04-24)

### Features

- **robustness**: Pr4 opaque-wrapper guards + LIMITATIONS.md + CI matrix
  ([`9db5b2e`](https://github.com/johnmarktaylor91/torchlens/commit/9db5b2e11b30b0f72d2a19b1e19b9514d0e897e6))

Final wave of the robustness sprint. Adds the remaining must-fail-loudly guards for wrappers whose
  forward pass is not ordinary Python, adds a user-facing limitations doc, wires in a lean
  torch-version CI matrix, and links the doc from the README.

### Opaque-wrapper guards (new) torchlens/user_funcs.py now ships _reject_opaque_wrappers(model),
  called immediately before _unwrap_data_parallel at every log entry point (log_forward_pass,
  get_model_metadata, show_model_graph, validate_forward_pass). It raises RuntimeError with a
  pointer to docs/LIMITATIONS.md for:

- torch.compile'd models (OptimizedModule) — dynamo replaces the forward with a compiled graph; our
  Python-level wrappers are optimized away or bypassed depending on the backend. -
  torch.jit.ScriptModule (script + trace) — the forward runs on the TorchScript interpreter, not
  Python; wrappers never fire. - torch.export.ExportedProgram — a serialised IR, not a callable
  nn.Module. Detected defensively since ExportedProgram isn't an nn.Module in the first place.

For every case the message tells the user to call log_forward_pass on the original un-wrapped model.

### docs/LIMITATIONS.md (new) Canonical one-page matrix + per-context explanation of every
  limitation surfaced by the robustness sprint. Sections:

- At-a-glance matrix (what TorchLens does / workaround) - Per-context detail: compile / jit / export
  / FSDP / meta / sparse / SymInt / quantized / vmap / multiprocess / nested / partial-support -
  "Reporting a new failure" instructions

### README pointer (new) A "Known limitations / unsupported contexts" section right before "Planned
  Features" links to docs/LIMITATIONS.md with a one-line summary so users can find it without
  reading the source.

### CI torch-version matrix (new) .github/workflows/tests.yml runs smoke tests on two points in the
  support window: python 3.10 + torch 2.4.* (declared floor) and python 3.11 + torch 2.7.*.
  fail-fast is off so both rows report independently. Kept lean (smoke-only, two entries) so CI
  minutes stay under control.

### Tests (tests/test_robustness_pr4.py, 11 cases) - torch.compile'd model raises with
  "torch.compile" in the message - Logging the un-compiled original still works after compilation -
  torch.jit.script raises with "ScriptModule" - torch.jit.trace raises with "ScriptModule" - Logging
  the un-scripted original still works after scripting - torch.export.ExportedProgram fails loudly
  at entry - _reject_opaque_wrappers is a no-op on a bare nn.Module - _reject_opaque_wrappers raises
  on ScriptModule at the helper level - docs/LIMITATIONS.md exists and is long enough to be
  meaningful - README links to docs/LIMITATIONS.md - docs/LIMITATIONS.md mentions every context with
  a runtime guard (staleness regression guard)

### Verification - pytest tests/ -m smoke: 35/35 ✅ (up from 32) - pytest targeted regression + PR4:
  470/470 ✅ (3m47s) - ruff check . ✅ / ruff format ✅ - mypy: 67 source files clean ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.6.0 (2026-04-24)

### Features

- **robustness**: Pr3 parallel-wrapper unwrap + framework I/O guards
  ([`c80c21d`](https://github.com/johnmarktaylor91/torchlens/commit/c80c21d9a4399194bb59fe6497d75540beff2411))

Wave 3 of the robustness sprint. Extends parallel-wrapper handling and adds regression coverage for
  HuggingFace-style dict-subclass inputs and dataclass outputs.

### DistributedDataParallel unwrap Previously _unwrap_data_parallel only handled nn.DataParallel.
  DDP and DataParallel share the same .module attribute but the isinstance check excluded DDP, so
  users of the much more common DDP got silently wrong layer addressing (parameter barcodes pointing
  at the wrapper's module hierarchy instead of the real model tree).

Fix: isinstance check extended to DistributedDataParallel. Guarded by ImportError so torch installs
  without torch.distributed still work.

### FullyShardedDataParallel clear-error FSDP parameters are sharded across ranks — there is no
  single unsharded module to log. Silently unwrapping via .module yields a model whose parameters
  are FlatParameter shards, which produces misleading layer metadata and breaks loop detection's
  param-sharing heuristic.

Fix: raise RuntimeError at unwrap time with a message explaining the sharding issue and pointing to
  "run log_forward_pass on a rank-local copy of the underlying module (before FSDP wrapping)".

### Framework I/O regression tests No code change here — these tests lock in the current behavior so
  future refactors don't silently break HuggingFace use cases:

- Dict-subclass inputs (BatchEncoding-shaped UserDict) flow through _move_tensors_to_device
  correctly and logging completes. - Dataclass-style outputs (ModelOutput-shaped dataclass) don't
  crash the output-extraction BFS crawl; at minimum the inner ops are logged.

### Tests (tests/test_robustness_pr3.py, 7 cases) - DataParallel unwrap still works (regression
  guard for pre-existing behavior) - DistributedDataParallel isinstance -> unwrap via .module - FSDP
  isinstance -> raise with clear message (both at the helper level and at log_forward_pass entry) -
  UserDict-based BatchEncoding-like input logs (smoke) - ModelOutput-like dataclass output doesn't
  crash and Linear is logged - Standard model still logs (golden-path regression)

### Verification - pytest tests/ -m smoke: 32/32 ✅ - pytest targeted regression + PR3: 466/466 ✅
  (3m47s) - ruff check . ✅ / ruff format ✅ - mypy: 66 source files clean ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.5.0 (2026-04-24)

### Features

- **robustness**: Pr2 tensor-variant pre-flight guard + channels_last fix
  ([`0f1367d`](https://github.com/johnmarktaylor91/torchlens/commit/0f1367d2f678881873dcceb6e2ced31c056f55de))

Wave 2 of the robustness sprint. Centralises detection of tensor variants TorchLens cannot handle
  and adds a channels_last preservation fix to safe_copy.

### Pre-flight variant guard New module torchlens/_robustness.py: - UnsupportedTensorVariantError
  (subclass of RuntimeError) with a clear bullet-listed message + pointer to docs/LIMITATIONS.md. -
  check_model_and_input_variants(model, input_args, input_kwargs) walks model params/buffers and the
  input tensor tree and: - RAISES for meta tensors (no backing storage — activation capture can't
  produce usable values), sparse tensors (safe_copy/print paths assume dense strided layouts),
  symbolic-shaped tensors (torch.SymInt dims break counter alignment and flops). - WARNS for
  quantized modules (logging works, FLOPs are wrong).

Wired into four entry points in user_funcs.py (log_forward_pass, get_model_metadata / summary,
  show_model_graph, validate_forward_pass), in each case right after _unwrap_data_parallel. Failures
  happen up front with a clear message instead of partway through the 18-step postprocess pipeline.

### channels_last preservation in safe_copy torchlens/utils/tensor_utils.py: - New
  _safe_get_memory_format(t) probes channels_last / channels_last_3d via
  is_contiguous(memory_format=...), falling back to preserve_format on exotic layouts. - safe_copy
  now calls clone(memory_format=...) on both the attached and detached paths. If a tensor variant
  rejects the memory_format kwarg (some sparse/subclass paths), we fall back to a plain clone. -
  Previous behavior silently converted channels_last activations to channels_first on copy, which
  could produce silent wrong results for layout-sensitive downstream ops.

### Tests tests/test_robustness_pr2.py (16 cases): - Meta tensor inputs + meta params raise
  UnsupportedTensorVariantError - Sparse COO and CSR inputs raise with a layout-mentioning message -
  Quantized models emit exactly one UserWarning and logging keeps going - safe_copy preserves
  channels_last / channels_last_3d; detach path also preserves memory format - Standard forward pass
  still logs (golden-path regression) - CUDA channels_last + CUDA forward pass smoke tests (skipped
  if no GPU) - Internal helpers (_is_meta_tensor, _is_sparse_tensor) covered

### Verification - pytest tests/ -m smoke: 32/32 ✅ (up from 30; 2 new smoke) - pytest targeted
  regression + PR2: 475/475 ✅ (3m44s) - ruff check . ✅ / ruff format ✅ - mypy: 67 source files clean
  ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.4.0 (2026-04-24)

### Features

- **robustness**: Pr1 correctness guards + torch floor pin
  ([`f6efc9b`](https://github.com/johnmarktaylor91/torchlens/commit/f6efc9ba84142fe209323782041f52e72d637a40))

Wave 1 of the robustness sprint. Correctness-critical hardening only; no behavior change for
  non-pathological forward passes.

- _state.active_logging(): raise RuntimeError on nested entry instead of silently overwriting
  _active_model_log and clearing it on inner exit (which corrupted the outer ModelLog mid-pass). The
  docstring already declared the context non-nestable; now the runtime enforces it. The realistic
  trigger is a user forward-hook or activation pipeline that calls log_forward_pass on a sub-model.
  - torch_func_decorator: when a functorch/vmap/grad transform is active, TorchLens continues to
  skip logging inside the transform (internal ops like safe_copy lack vmap batching rules), but now
  emits a one-shot UserWarning per forward pass so users know the resulting ModelLog omits whatever
  ran inside the transform. Flag lives in _state and resets at the top of every active_logging()
  session. - pyproject: pin torch>=2.4. TorchLens already calls
  torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) (rng.py), both of
  which require a device argument only from torch 2.4+. The bare 'torch' dep was permitting installs
  that would fail at runtime. - pyproject: advertise Python 3.13 classifier (stable since 2024-10,
  supported by torch 2.5+).

New tests (tests/test_robustness_pr1.py, 8 cases): - direct active_logging nesting raises with the
  non-re-entrant message - a forward-hook that re-enters log_forward_pass raises and the outer
  session state is cleaned up cleanly afterward - a vmap-containing forward pass emits exactly one
  functorch warning per session, and the flag resets between sessions - non-vmap passes emit no
  functorch warning - pyproject torch dep pins >=2.4 and Python 3.13 is a classifier

All 30 smoke tests and 209 targeted regression tests pass. ruff, mypy clean.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.3.0 (2026-04-24)

### Documentation

- **ux-sprint**: Clarify defaults in docstrings + add regression guard
  ([`46a8555`](https://github.com/johnmarktaylor91/torchlens/commit/46a85556643ace08b66b236a64a8a187a10b111e))

Wave 4c of the defaults audit. No defaults changed (all rows in defaults-table.md marked change: no,
  because the public wrappers already put fast+informative defaults in the right place and expensive
  paths are opt-in).

Documentation-only improvements: - log_forward_pass / show_model_graph docstrings for detect_loops /
  detect_recurrent_patterns now include the \">1M operations\" speedup guidance users want to find
  when postprocessing gets slow. - save_source_context docstring now explicitly separates the
  always-captured identity fields (file, line, func_name, etc.) from the opt-in rich source text, so
  users understand branch attribution works regardless of this flag. - keep_unsaved_layers docstring
  now shows the typical memory-conscious usage pattern (layers_to_save=[...],
  keep_unsaved_layers=False).

New regression guard: - tests/test_defaults.py — 7 tests locking in the current resolved defaults on
  log_forward_pass, show_model_graph, log_model_metadata, torchlens.summary, and
  validate_forward_pass. Tests the wrapper behavior (which routes through MISSING sentinels + option
  groups) rather than raw signature defaults.

Follow-up noted in wave4c-issues.md: ModelLog.__init__ still defaults
  mark_input_output_distances=True while the public wrapper resolves to False. Not user-visible in
  this sprint; would need a separate constructor parity pass.

CHANGELOG.md: new v1.3.0 (unreleased) documentation stub covering the whole UX sprint.

Gates: - ruff check: clean - mypy torchlens/: no issues (66 source files) - pytest smoke (28):
  passed - pytest test_defaults.py (7 new): passed - pytest not-slow full (1144 passed, 21 skipped):
  passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

### Features

- **ux-sprint**: Add ModelLog.summary, compact __repr__, torchlens.summary
  ([`6a64cf1`](https://github.com/johnmarktaylor91/torchlens/commit/6a64cf1b4da2380c1f61fd44a8e3f771909474ff))

Implements the Wave 4a deliverables from summary-api.md:

- ModelLog.summary(level=..., fields=..., show_ops=..., ...) renders a preset-driven textual
  summary. Presets: overview (default, Keras-style), graph (hybrid module+op), memory (tensor-size
  accounting), control_flow (conditionals + recurrence), cost (params + FLOPs + MACs + time). -
  ModelLog.__repr__ replaced with a concise 1-2 line identity card. - torchlens.summary(model,
  input_args, input_kwargs=None, **kw) top-level convenience: runs a metadata-only forward pass,
  renders the summary, cleans up, returns the string. - New package torchlens/_summary/
  (underscore-prefixed to avoid collision with the public torchlens.summary function) holds the
  builder + preset logic in 1169 LOC. - tests/test_summary.py: 11 new tests covering each preset,
  __repr__ brevity, truncation markers, pre-postprocess ModelLog, fidelity against claims we cannot
  keep (no peak-memory, no per-op device column).

Gates: - ruff check: clean - mypy torchlens/: no issues (63 source files) - pytest smoke (28 tests):
  passed - pytest test_summary.py (11 new): passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **ux-sprint**: Api renames + VisualizationOptions/StreamingOptions groups
  ([`1d737b7`](https://github.com/johnmarktaylor91/torchlens/commit/1d737b739279747d4d074a5467952101cb355310))

Implements Wave 4b deliverables (rename-table.md + option-groups.md):

Renames (additive, old names remain working with DeprecationWarning): - top-level get_model_metadata
  -> log_model_metadata - kwarg num_context_lines -> source_context_lines - kwarg
  mark_input_output_distances -> compute_input_output_distances - kwarg detect_loops ->
  detect_recurrent_patterns - ModelLog.validate_saved_activations -> ModelLog.validate_forward_pass

Option groups (new dataclasses, flat kwargs remain as deprecated aliases): -
  torchlens.VisualizationOptions — 16 fields, replaces vis_* sprawl. Per-function defaults:
  log_forward_pass uses mode=none, show_model_graph uses mode=unrolled. Mixing flat+grouped same
  field raises TypeError; mixing flat+grouped DIFFERENT fields merges. - torchlens.StreamingOptions
  — 3 fields for bundle_path + retain_in_memory + activation_callback.

Shared plumbing: - torchlens/_deprecations.py — warn_deprecated_alias dedup + MISSING sentinel +
  kwarg-resolution helper. - torchlens/_literals.py — Literal type aliases for autocomplete:
  OutputDeviceLiteral, VisModeLiteral, VisDirectionLiteral, VisNodePlacementLiteral,
  VisRendererLiteral. - torchlens/options.py — 496 LOC, option-group dataclasses +
  resolve_visualization_options / resolve_streaming_options / render kwarg translation.

Tests: - tests/test_api_renames.py — 61 new tests covering each alias, DeprecationWarning firing,
  flat+grouped mixing rules, per-function VisualizationOptions defaults, StreamingOptions migration.

Gates: - ruff check: clean - mypy torchlens/: no issues - pytest smoke (28): passed - pytest
  test_api_renames.py (61): passed - pytest not-slow full (1137 passed, 21 skipped): passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

### Refactoring

- **ux-sprint**: Inline ModelLog cross-file method bindings
  ([`b82f176`](https://github.com/johnmarktaylor91/torchlens/commit/b82f176f4f130be493e6f38e5f01fcba0a999bec))

Step 1 of the refactor-list.md plan: convert all 19 class-body attribute rebindings at
  model_log.py:706-724 into explicit `def X(self, ...)` methods on ModelLog. Implementation helpers
  remain in their feature modules (visualization/, validation/, capture/, postprocess/); the class
  body now has real method definitions that delegate via local imports where needed to avoid
  circular-import regressions.

Step 2: extract `_give_user_feedback_about_lookup_key` and `_get_lookup_help_str` from
  `data_classes/interface.py` into a new `data_classes/_lookup_keys.py` so `capture/trace.py` no
  longer needs to import from the former helper module. This unblocks any future collapse of
  interface.py / cleanup.py.

Step 3 intentionally SKIPPED this sprint (design marked it optional). interface.py and cleanup.py
  remain as legitimate helper modules with non-method-pack utilities.

Also added `tests/test_cross_file_refactor.py` with an AST-based guard that fails if anyone
  reintroduces `X = imported_X` class-body rebindings on ModelLog.

Gates: - ruff check: clean - mypy torchlens/: no issues - pytest smoke (28 tests): passed - pytest
  not-slow (1065 passed, 21 skipped): passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.2.0 (2026-04-23)

### Bug Fixes

- **io**: Io-s9 path-traversal hardening + two-pass streaming rejection + docs security warning +
  preflight/reader cleanup
  ([`42c047f`](https://github.com/johnmarktaylor91/torchlens/commit/42c047fef95c126caca9069b40140df384f2ca4d))

### Chores

- **io-sprint**: Add plan and research docs
  ([`9bd8272`](https://github.com/johnmarktaylor91/torchlens/commit/9bd827241a80ee7a0a621081cd61128846346e9d))

Plan v6 (Round 8, GREEN) at .project-context/plans/io-sprint/plan.md. Audits and review rounds 1-8
  archived alongside.

### Documentation

- **io**: Io-s8 docstrings + README section + architecture doc
  ([`ddb8981`](https://github.com/johnmarktaylor91/torchlens/commit/ddb898126bb6590e999260fbfeb7732115b482b0))

- **io-sprint**: Phase 5 dual-review records
  ([`806868a`](https://github.com/johnmarktaylor91/torchlens/commit/806868a154d14cd9b93dadb429ada0dcd1e93e13))

### Features

- **io**: Io-s1 pickle hardening + PORTABLE_STATE_SPEC + rehydrate loader
  ([`b49c748`](https://github.com/johnmarktaylor91/torchlens/commit/b49c748439837b19e65daba8c09417facf8e937f))

- **io**: Io-s2 Param/Buffer/ModulePass DataFrames + summary fields
  ([`4512a64`](https://github.com/johnmarktaylor91/torchlens/commit/4512a64f7161b838c4e1a618cfc306fae6859fd0))

- **io**: Io-s3 export wrappers (csv/parquet/json) on all DataFrame surfaces
  ([`7686ec4`](https://github.com/johnmarktaylor91/torchlens/commit/7686ec4b61b419863532bf423d0ee955bb289bd6))

Add to_csv / to_parquet / to_json to ModelLog, LayerAccessor, ModuleAccessor, ModuleLog. Normalize
  wrapper signatures on the four S2 surfaces (ParamAccessor, BufferAccessor, ModulePassLog) to match
  the shared spec.

- **io**: Io-s4 bundle save/load with manifest + integrity + tensor policy
  ([`380d3f5`](https://github.com/johnmarktaylor91/torchlens/commit/380d3f5b530ec4eb88b0be7c97bffdc12af93f92))

Adds torchlens.save, torchlens.load, torchlens.cleanup_tmp. Adds ModelLog.save / ModelLog.load
  sugar. Adds safetensors as required dep. Enforces Fork F version policy, Fork G exception
  wrapping, Fork J symlink rejection, Fork L replay-not-supported-on-portable guard.

- **io**: Io-s5 streaming-save during forward pass + postprocess steps 19/20
  ([`7a172a9`](https://github.com/johnmarktaylor91/torchlens/commit/7a172a9ef326ab41dd3f03dc40976fa6eb7f2e64))

- **io**: Io-s6 lazy activation refs + drift detection + rehydrate_nested
  ([`1c870ce`](https://github.com/johnmarktaylor91/torchlens/commit/1c870cee22ea35ccaa5158a7d7058d344fb5192c))

### Testing

- **io**: Io-s7 integration + corruption + plain-pickle regression suite
  ([`3fba0c0`](https://github.com/johnmarktaylor91/torchlens/commit/3fba0c0430d61fe0dfb71e57d43cf41aad7849db))

Add the IO-S7 integration, corruption, and plain-pickle regression coverage.

While adding the corruption battery, wrap safetensors corruption read failures in TorchLensIOError
  so truncated blobs stay inside the portable I/O exception contract.


## v1.1.0 (2026-04-22)

### Bug Fixes

- **conditionals**: Make MultilinePredicateModel predicate genuinely multi-line
  ([`597e075`](https://github.com/johnmarktaylor91/torchlens/commit/597e0757d5d794871ac6eff0d68d1080ad2dea5a))

test_multiline_predicate_model_tracks_multiline_if_event asserts that the conditional's test_span
  covers multiple lines, but the model's predicate was single-line (and ruff-format would collapse
  any naive re-wrap back to one line since it fits under 100 chars). Use `# fmt: off` / `# fmt: on`
  to preserve the genuinely multi-line predicate structure.

Final tier-2: 966 passed, 12 skipped, 0 failed.

### Documentation

- **conditionals**: Update user_funcs + AGENTS + architecture + limitations (Phase 9/10)
  ([`6d1814e`](https://github.com/johnmarktaylor91/torchlens/commit/6d1814e5c7f716fb3f54e0ed2e0f00c02402c8ff))

### Features

- **conditionals**: 13 consistency invariants for conditional metadata (Phase 6/10)
  ([`4236c24`](https://github.com/johnmarktaylor91/torchlens/commit/4236c2419570fae084b38c450c9880181db58d8d))

- **conditionals**: Ast_branches module for bool classification + op attribution (Phase 2/10)
  ([`21eabe8`](https://github.com/johnmarktaylor91/torchlens/commit/21eabe88e53bc8ae833a31ace8251f5744c1c662))

New module torchlens/postprocess/ast_branches.py with public API: - get_file_index(filename) ->
  Optional[FileIndex] - classify_bool(filename, line, col=None) -> BoolClassification -
  attribute_op(func_call_stack) -> List[Tuple[ConditionalKey, str]] -
  invalidate_cache(filename=None)

Internal structures: FileIndex, ScopeEntry, ConditionalRecord (if_chain + ifexp), BoolConsumer
  index, branch interval trees, mtime-keyed file cache. Elif flattening, IfExp first-class support,
  (line, col) point lookup, D14 scope resolution with fail-closed ambiguity handling, D18
  col-degraded mode fail-closed.

Tests: 19 passing unit tests on synthetic source snippets covering all 9 bool-context kinds, wrapper
  bool_cast nesting, multi-line if-tests, nested ifs, ternary attribution, ternary same-line
  col=None fail-closed, elif flattening, scope resolution, ambiguous scope fail-closed, cache
  invalidation.

Quality gates: pytest ast_branches (19/19), smoke (22/22), mypy clean, ruff clean.

No changes to postprocess/control_flow.py (Phase 4) or any other torchlens module.

- **conditionals**: Foundation for if/elif/else attribution (Phase 1/10)
  ([`ced5a06`](https://github.com/johnmarktaylor91/torchlens/commit/ced5a0675612ee3a443535a706dd3730938b394d))

Add Phase 1 conditional foundation fields across LayerPassLog, LayerLog, ModelLog, and
  FuncCallLocation. Ungate func_call_stack capture while keeping source text lazily disabled via
  source_loading_enabled. Add metadata coverage for save_source_context=False, including
  disabled-source accessors and pickle round-trip.

Assumption: legacy direct FuncCallLocation construction defaults code_firstlineno to line_number
  when the new field is omitted.

- **conditionals**: Full integration test matrix (Phase 8/10)
  ([`f97f19d`](https://github.com/johnmarktaylor91/torchlens/commit/f97f19dc4f882b9f7e3b0fbd3df90bd618ab42ac))

Add ~64 new integration tests covering every remaining scenario in plan.md's Test Matrix that was
  not already covered by Phases 4-7. Two test files: - tests/test_conditional_integration.py (31
  atomic tests per matrix row) - tests/test_conditional_branches.py (33 cross-cutting tests that
  combine invariants + rendering + lifecycle checks on the same model)

Categories covered (missing from earlier phases): - Nested branches: NestedIfThenIfModel,
  NestedInElseModel, MultilinePredicateModel - Branch-entry via non-predicate ancestors, wrapped
  bool, reconvergence, scope resolution, false positives, compound/negation/walrus, ternary
  variants, false negatives, lifecycle, save_source_context=False gating.

Also extract _label_for_reference_removal helper in cleanup.py to consolidate the pass_finished
  label-namespace selection logic (required for consistent behavior across Phase 8 lifecycle tests).

Quality gates: 81+ conditional tests pass, smoke (26/26), tier-2 (929/929), mypy clean, ruff clean.

- **conditionals**: Graphviz rendering with THEN/ELIF/ELSE labels (Phase 7/10)
  ([`11e408c`](https://github.com/johnmarktaylor91/torchlens/commit/11e408c46f650147c21733dea2818d4fe2ae8e10))

Refactor torchlens/visualization/rendering.py edge-label logic into a precedence function
  (_compute_edge_label) per plan.md "Label precedence rules": 1. Arm-entry label (highest): THEN /
  ELIF N / ELSE, composite for multi-arm or mixed-pass rolled edges. 2. IF label (secondary): for
  condition-chain edges only. 3. Arg labels move from edge_dict["label"] to
  edge_dict["headlabel"]/xlabel so they do not compete with branch labels.

Covers if/elif/else ladders, ternary (ifexp) THEN/ELSE, multi-arm-entry edges (D13), rolled
  mixed-pass composite labels (D15 conditional_edge_passes), and branch-entry edge + arg label
  collision avoidance.

Tests (tests/test_conditional_rendering.py, 5 passing): -
  test_simple_if_else_graphviz_labels_then_and_else_edges -
  test_elif_ladder_graphviz_labels_elif_and_else_edges -
  test_basic_ternary_graphviz_labels_then_and_else_edges -
  test_branch_entry_with_arg_label_keeps_semantic_and_argument_labels_separate -
  test_rolled_mixed_arm_graphviz_shows_composite_pass_label

Dagua bridge and ELK remain deferred per user direction.

Quality gates: rendering tests (5/5), mypy clean, ruff clean.

- **conditionals**: Lifecycle wiring for new fields (Phase 3/10)
  ([`f4c59c1`](https://github.com/johnmarktaylor91/torchlens/commit/f4c59c15da3ce014567944219861eca15a5b6d90))

Modified files: - torchlens/postprocess/labeling.py - torchlens/data_classes/cleanup.py -
  torchlens/data_classes/interface.py - tests/test_conditional_lifecycle.py

Summary: - rewired Step 11 rename coverage for cond_branch_children_by_cond, conditional_arm_edges,
  conditional_edge_passes, conditional_events.bool_layers, and the derived elif/else views -
  scrubbed the same conditional surfaces during label removal, including keep_unsaved_layers=False
  pruning via the shared batch cleanup path - exported the new conditional lifecycle fields through
  to_pandas, including a compact conditional_branch_stack string and the missing derived child
  columns - added Phase 3 integration tests for rename, cleanup, and to_pandas

Assumption: - the spec's reference to _rename_raw_labels_in_place maps to the current Step 11
  helpers _replace_layer_names_for_layer_entry and _rename_model_history_layer_names in this branch

- **conditionals**: Multi-pass LayerLog aggregation (Phase 5/10)
  ([`7eb3baa`](https://github.com/johnmarktaylor91/torchlens/commit/7eb3baa34b5170465e745c553271f54ac7ec9e75))

Add ordered multi-pass LayerLog aggregation for conditional branch stacks, stack-to-pass maps, and
  pass-stripped cond-id child unions. Recompute aggregate LayerLog conditional compatibility views
  after merge, and rebuild rolled conditional_edge_passes from arm-entry edges with no-pass labels
  and sorted child-pass lists.\n\nCover the new behavior with dedicated multi-pass tests for
  alternating recurrent branches, mixed-arm rolled edges, looped alternating branches, and a
  non-conditional recurrent regression case.

- **conditionals**: Step 5 restructure into 5a-5f with AST-based classification and attribution
  (Phase 4/10)
  ([`c04fe8b`](https://github.com/johnmarktaylor91/torchlens/commit/c04fe8b8fd7c6d5a1dff8ebdd968911bed411fe0))

Restructure Step 5 in torchlens/postprocess/control_flow.py into the 5a-5f pipeline while preserving
  the _mark_conditional_branches entry point: build AST file indexes, classify terminal scalar
  bools, materialize dense ConditionalEvent records, run the IF backward flood, attribute forward
  arm-entry edges, and rebuild derived compatibility views.

Add Phase 4 integration coverage in tests/test_conditional_step5.py for simple if/else, elif
  ladders, assert classification, and save_source_context=False attribution.

Regression surface verified: - pytest tests/ -m smoke -x --tb=short -q - pytest
  tests/test_conditional_step5.py -x --tb=short - pytest tests/test_toy_models.py -k
  'conditional_branching or nested_conditional_loop or conditional_vae' -x --tb=short - pytest
  tests/ -m "not slow" -x --tb=short - mypy torchlens/ - ruff check . --fix

Conservative compatibility choice: preserve the existing in_cond_branch condition-chain semantics
  for public metadata/tests while treating conditional_branch_stack/conditional_arm_edges as the new
  primary executed-arm attribution. Also add a narrow decorated-function scope fallback in
  control_flow.py when attribute_op() fails closed on decorator-line co_firstlineno offsets, without
  modifying frozen ast_branches.py.

- **conditionals**: Step 5 six-phase restructure + Phase 4 integration tests (Phase 4/10)
  ([`71f976c`](https://github.com/johnmarktaylor91/torchlens/commit/71f976c89a9ec762f333c8d54e4e901fdb6b7f7b))

Restructure torchlens/postprocess/control_flow.py::_mark_conditional_branches into six sub-phases
  per plan.md: 5a. Build AST file indexes for files referenced by terminal scalar bools. 5b.
  Classify terminal bools into branch/non-branch contexts (via ast_branches). 5c. Materialize dense
  ConditionalEvent records from structural keys (sole key->id step). 5d. Backward-flood IF edges
  from branch-participating bools only. 5e. Forward branch attribution across every forward edge
  (D13 multi-entry) + cond-id-aware primary structures (conditional_arm_edges,
  cond_branch_children_by_cond). 5f. Materialize derived compatibility views (legacy
  cond_branch_*_children and conditional_*_edges fields).

Integration tests (tests/test_conditional_step5.py): SimpleIfElseModel, ElifLadderModel,
  AssertNotBranchModel, SaveSourceContextOffStillAttributes. All 4 pass on real models.

Quality gates: step5 tests (4/4), smoke (22/22), mypy clean, ruff clean.

Notes: - Old post-validation (D12) removed; pre-classification obsoletes it. - conditional_events
  dense IDs assigned deterministically (first-seen order). - Backward compat:
  conditional_branch_edges, conditional_then_edges, etc. still populated via derived views from
  primary structures.

### Testing

- Mark test_validation_250k as slow
  ([`a960cb2`](https://github.com/johnmarktaylor91/torchlens/commit/a960cb27c2bbd36340367f5e0dc20f18c7336f64))

250k-node validation takes too long for Tier 2 runs and was not excluded by `-m "not slow"` despite
  being marked rare.

- Mark test_validation_50k as rare
  ([`8070f8e`](https://github.com/johnmarktaylor91/torchlens/commit/8070f8eed80216b0bb07f4c38be94367ce405ce6))

Times out on slower machines even with 600s per-test limit. Validation of 50k-node random graph is
  too heavy for default runs.

- Skip test_validation_250k unconditionally
  ([`bd756b2`](https://github.com/johnmarktaylor91/torchlens/commit/bd756b2c679ee19f1f03bf1b0335f298ef91f089))

250k-node validation OOMs or hangs for hours on most machines. Slow/rare markers weren't enough --
  needs an explicit skip so `pytest tests/` (full run) doesn't block on it. Run manually with
  --no-skip when needed.

- **conditionals**: Isolate AST cache + linecache per test (Phase 10/10 prep)
  ([`eb00a71`](https://github.com/johnmarktaylor91/torchlens/commit/eb00a712292fc3a255d1868566dc16f59d050227))

Autouse fixture invalidates torchlens.postprocess.ast_branches file cache and Python linecache at
  the start of each test in test_conditional_branches.py.

Without this, test ordering dependencies surface in the full tier-2 suite:
  test_multiline_predicate_model_tracks_multiline_if_event fails when a prior test run populates
  stale state that the multi-line predicate assertion then reads. Running the file in isolation or
  with any smaller prefix passes cleanly; the full suite flakes depending on pytest's collection
  order.

Gates: ruff + mypy clean; smoke 28/28; tier-2 966 passed, 12 skipped, 0 failures (vs 1 failure
  pre-fixture under same collection order).


## v1.0.2 (2026-04-05)

### Bug Fixes

- **decoration**: Python 3.14 compat -- two-pass decoration + TypeError catch
  ([#138](https://github.com/johnmarktaylor91/torchlens/pull/138),
  [`e6f0f9a`](https://github.com/johnmarktaylor91/torchlens/commit/e6f0f9a33ba31504d6e402c117aac4a87ef46cb5))

Python 3.14 (PEP 649) evaluates annotations lazily. During decorate_all_once(), wrapping Tensor.bool
  before inspecting Tensor.dim_order caused inspect.signature() to resolve `bool` in `bool |
  list[torch.memory_format]` to the wrapper function instead of the builtin type, raising TypeError
  on first call only.

Three fixes: - Catch TypeError alongside ValueError in get_func_argnames (safety net) - Split
  decorate_all_once() into two passes: collect argnames from pristine namespace first, then decorate
  (eliminates root cause) - Replace _orig_to_decorated idempotency guard with _is_decorated flag so
  partial decoration failure allows retry instead of locking in incomplete state

6 new tests, gotchas.md updated.


## v1.0.1 (2026-03-23)

### Bug Fixes

- **decoration**: Clear stale sq_item C slot after wrapping Tensor.__getitem__
  ([`b2c6085`](https://github.com/johnmarktaylor91/torchlens/commit/b2c6085e31695b973b8d11c23c773af837a846cc))

When __getitem__ is replaced on a C extension type with a Python function, CPython sets the sq_item
  slot in tp_as_sequence. This makes PySequence_Check(tensor) return True (was False in clean
  PyTorch), causing torch.tensor([0-d_tensor, ...]) to iterate elements as sequences and call len()
  -- which raises TypeError for 0-d tensors. The slot is never cleared by restoring the original
  wrapper_descriptor or by delattr.

Fix: null sq_item via ctypes after every decoration/undecoration cycle (decorate_all_once,
  unwrap_torch, wrap_torch). Safe because tensor indexing uses mp_subscript (mapping protocol), not
  sq_item (sequence protocol). Verified via tp_name guard; fails silently on non-CPython.

Adds 9 regression tests covering all lifecycle paths.

### Chores

- Add secret detection pre-commit hooks
  ([`0e2889a`](https://github.com/johnmarktaylor91/torchlens/commit/0e2889ae90f822840895d9331e912a570d2a9acf))

Add detect-private-key (pre-commit-hooks) and detect-secrets (Yelp) to catch leaked keys, tokens,
  and high-entropy strings before they hit the repo.


## v1.0.0 (2026-03-13)

### Bug Fixes

- **decoration**: Cast mode.device to str for mypy return-value check
  ([`45c0ff3`](https://github.com/johnmarktaylor91/torchlens/commit/45c0ff3be8586675817905dcb273e9bad7ac0519))

CI mypy (stricter torch stubs) catches that mode.device returns torch.device, not str. Explicit
  str() cast satisfies the Optional[str] return type annotation.

### Features

- **decoration**: Lazy wrapping — import torchlens has no side effects
  ([`b5da8b8`](https://github.com/johnmarktaylor91/torchlens/commit/b5da8b8b15136f108534ba29022ab6579bdb9315))

BREAKING CHANGE: torch functions are no longer wrapped at import time. Wrapping happens lazily on
  first log_forward_pass() call and persists.

Three changes:

1. Lazy decoration: removed decorate_all_once() / patch_detached_references() calls from
  __init__.py. _ensure_model_prepared() triggers wrapping on first use via wrap_torch().

2. Public wrap/unwrap API: - torchlens.wrap_torch() — install wrappers (idempotent) -
  torchlens.unwrap_torch() — restore original torch callables - torchlens.wrapped() — context
  manager (wrap on enter, unwrap on exit) - log_forward_pass(unwrap_when_done=True) — one-shot
  convenience Old names (undecorate_all_globally, redecorate_all_globally) kept as internal aliases.

3. torch.identity fix: decorated identity function now stored on _state._decorated_identity instead
  of monkey-patching torch.identity (which doesn't exist in PyTorch type stubs). Eliminates 2 mypy
  errors.

Tests updated: 75 pass including 12 new lifecycle tests.

### Breaking Changes

- **decoration**: Torch functions are no longer wrapped at import time. Wrapping happens lazily on
  first log_forward_pass() call and persists.


## v0.22.0 (2026-03-13)

### Chores

- **types**: Remove stale typing noise
  ([`5ba099d`](https://github.com/johnmarktaylor91/torchlens/commit/5ba099df82f789f308f86d78a66c82b1baf3751d))

### Documentation

- **maintenance**: Refresh maintainer notes
  ([`685a358`](https://github.com/johnmarktaylor91/torchlens/commit/685a358caa12b3210de68e136c59959de4c3f94f))

- **maintenance**: Split CLAUDE.md/AGENTS.md into architect vs implementation roles
  ([`18f8ae9`](https://github.com/johnmarktaylor91/torchlens/commit/18f8ae9851e91be5ca7c335facb45209212f6d95))

Break the symlink mirroring convention: CLAUDE.md now holds architect-level context (what, why, how
  it connects) while AGENTS.md holds implementation-level context (conventions, gotchas, known bugs,
  test commands). Pure-implementation subdirs (.github, scripts, tests, utils) get AGENTS.md only.
  Also populates .project-context/ templates (architecture, conventions, gotchas, decisions).

### Features

- **decoration**: Add global undecorate override
  ([`b0bafeb`](https://github.com/johnmarktaylor91/torchlens/commit/b0bafeb51fbf9b406bd34a84ed826cd4b9c607bb))

- **viz**: Add dagua torchlens integration
  ([`35d5bcd`](https://github.com/johnmarktaylor91/torchlens/commit/35d5bcdb976420b86be54048ab4503cbbc146763))

### Refactoring

- **types**: Finish package mypy cleanup
  ([`6f9a3fe`](https://github.com/johnmarktaylor91/torchlens/commit/6f9a3fe423367c9bd9e32949cdf06945d90786ee))


## v0.21.3 (2026-03-11)

### Bug Fixes

- **tests**: Make SIGALRM signal safety test deterministic
  ([`b3fc461`](https://github.com/johnmarktaylor91/torchlens/commit/b3fc46154f0942c829c3bbeb9b07f3d900471b79))

Replace timer-based SIGALRM with direct os.kill() inside forward() so the signal always fires
  mid-logging. Eliminates flaky skips when the forward pass completes before the timer.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.21.2 (2026-03-09)

### Bug Fixes

- **vis**: Avoid graphviz.Digraph memory bomb when ELK fails on large graphs
  ([`f5563ee`](https://github.com/johnmarktaylor91/torchlens/commit/f5563eee457383772c9b148cca058b87e126b6b3))

When ELK layout fails (OOM/timeout) on 1M+ node graphs, the fallback path previously built a
  graphviz.Digraph in Python — nested subgraph body-list copies exploded memory and hung
  indefinitely. Now render_elk_direct handles the failure internally: reuses already-collected Phase
  1 data to generate DOT text without positions and renders directly with sfdp, bypassing
  graphviz.Digraph entirely.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Bypass ELK for large graphs — use Python topological layout
  ([`37cce3a`](https://github.com/johnmarktaylor91/torchlens/commit/37cce3ab5607591f622b4fd7f5916bca16736d59))

ELK's stress algorithm allocates TWO O(n²) distance matrices (n² × 16 bytes). At 100k nodes that's
  160 GB, at 1M nodes it's 16 TB — the root cause of the std::bad_alloc. The old >150k stress switch
  could never work.

For graphs above 100k nodes, we now skip ELK entirely and compute a topological rank layout in
  Python (Kahn's algorithm, O(n+m)). Module bounding boxes are computed from node positions. The
  result feeds into the same neato -n rendering path, preserving cluster boxes.

If ELK fails for smaller graphs, the Python layout is also used as a fallback instead of the old
  sfdp path that built a graphviz.Digraph (which exploded on nested subgraph body-list copies).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.21.1 (2026-03-09)

### Bug Fixes

- **postprocess**: Fix mypy type errors in _build_module_param_info
  ([`11ea006`](https://github.com/johnmarktaylor91/torchlens/commit/11ea006c9a683675c926a9b0d649a2fa24b46558))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- Trigger CI
  ([`99f4102`](https://github.com/johnmarktaylor91/torchlens/commit/99f4102fa34564868291214d6b6cf3dfc239fdce))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Performance Improvements

- **postprocess**: Optimize pipeline for large models
  ([`a211417`](https://github.com/johnmarktaylor91/torchlens/commit/a2114170b4f5b441d4c0ab7add26435820fc8f8f))

- Per-step verbose timing: unwrap grouped _vtimed blocks into individual step timing with
  graph-stats summary, enabling users to identify which specific step is slow (O16) - Cache
  module_str by containing_modules tuple to avoid redundant string joins in Step 6 (O8) -
  Early-continue guards in _undecorate_all_saved_tensors to skip BFS on layers with empty
  captured_args/kwargs (O5) - Pre-compute buffer_layers_by_module dict in _build_module_logs,
  eliminating O(modules × buffers) scan per module (O6) - Single-pass arglist rebuild in Step 11
  rename, replacing 3-pass enumerate + index set + filter pattern (O2) - Replace OrderedDict with
  dict in _trim_and_reorder (Python 3.7+ preserves insertion order) for lower allocation overhead
  (O4) - Reverse-index approach in _refine_iso_groups: O(members × neighbors) instead of O(members²)
  all-pairs combinations (O9) - Pre-compute param types per subgraph as frozenset before pair loop
  in _merge_iso_groups_to_layers (O10) - Set-based O(n) collision detection replacing O(n²) .count()
  calls in _find_isomorphic_matches (O12)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.21.0 (2026-03-09)

### Bug Fixes

- **capture**: Fix mypy type errors in output_tensors field dict
  ([`d54e9a9`](https://github.com/johnmarktaylor91/torchlens/commit/d54e9a99df47442ddc396ca20666203cfbeb5f06))

Annotate fields_dict as Dict[str, Any] and extract param_shapes with proper type to satisfy mypy
  strict inference.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Pass heap limits to ELK Worker thread to prevent OOM on 1M nodes
  ([`23ef8d8`](https://github.com/johnmarktaylor91/torchlens/commit/23ef8d8c175846fce1b48427a0c980a825486b9c))

The Node.js Worker running ELK layout had no explicit maxOldGenerationSizeMb in its resourceLimits —
  only stackSizeMb was set. The --max-old-space-size flag controls the main thread's V8 isolate, not
  the Worker's. This caused the Worker to OOM at ~16GB on 1M-node graphs despite the main thread
  being configured for up to 64GB.

- Add maxOldGenerationSizeMb and maxYoungGenerationSizeMb to Worker resourceLimits, passed via
  _TL_HEAP_MB env var - Add _available_memory_mb() to detect system RAM and cap heap allocation to
  (available - 4GB), preventing competition with Python process - Include available system memory in
  OOM diagnostic messages

Also includes field/param renames from feat/grand-rename branch.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Update all CLAUDE.md files with deepdive session 4 findings
  ([`b15c5bf`](https://github.com/johnmarktaylor91/torchlens/commit/b15c5bfa665011002a67cdcc7710f4737f1fc5d6))

Sync all project and subpackage documentation with current codebase: - Updated line counts across
  all 36 modules - Added elk_layout.py documentation to visualization/ - Added arg_positions.py and
  salient_args.py to capture/ - Documented 13 new bugs (ELK-IF-THEN, BFLOAT16-TOL, etc.) - Updated
  test counts (1,004 tests across 16 files) - Added known bugs sections to validation/, utils/,
  decoration/ - Updated data_classes/ with new fields and properties

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Rename all data structure fields and function args for clarity
  ([`f0d7452`](https://github.com/johnmarktaylor91/torchlens/commit/f0d7452e272bede7a874e1bda2dbce68dbb94697))

Rename ~68 fields across all 8 data structures (ModelLog, LayerPassLog, LayerLog, ParamLog,
  ModuleLog, BufferLog, ModulePassLog, FuncCallLocation) plus user-facing function arguments. Key
  changes:

- tensor_contents → activation, grad_contents → gradient - All *_fsize* → *_memory* (e.g.
  tensor_fsize → tensor_memory) - func_applied_name → func_name, gradfunc → grad_fn_name -
  is_bottom_level_submodule_output → is_leaf_module_output - containing_module_origin →
  containing_module - spouse_layers → co_parent_layers, orig_ancestors → root_ancestors -
  model_is_recurrent → is_recurrent, elapsed_time_* → time_* - vis_opt → vis_mode, save_only →
  vis_save_only - Fix typo: output_descendents → output_descendants

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.5 (2026-03-09)

### Bug Fixes

- **vis**: Prevent OOM kill on 1M-node ELK render
  ([#128](https://github.com/johnmarktaylor91/torchlens/pull/128),
  [`d9a1525`](https://github.com/johnmarktaylor91/torchlens/commit/d9a15259e452bf3c224732a0bc5a1671f9cbb56e))

The 1M-node render was OOM-killed at ~74GB RSS because: 1. Model params (~8-10GB) stayed alive
  during ELK subprocess 2. preexec_fn forced fork+exec, COW-doubling the 74GB process 3. Heap/stack
  formulas produced absurd values (5.6TB heap, 15GB stack) 4. No memory cleanup before subprocess
  launch

Changes: - render_large_graph.py: separate log_forward_pass from render_graph, free model/autograd
  before ELK render - elk_layout.py: cap heap at 64GB, stack floor 4096MB/cap 8192MB, write JSON to
  temp file (free string before subprocess), gc.collect before subprocess, set RLIMIT_STACK at
  module level (removes preexec_fn and the forced fork+exec)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.4 (2026-03-09)

### Bug Fixes

- **postprocess**: Backward-only flood in conditional branch detection + THEN labeling
  ([#88](https://github.com/johnmarktaylor91/torchlens/pull/88),
  [`d737828`](https://github.com/johnmarktaylor91/torchlens/commit/d7378281a4be5e56602902cd4cf1aa555b391d44))

Bug #88: _mark_conditional_branches flooded bidirectionally (parents + children), causing
  non-conditional nodes' children to be falsely marked as in_cond_branch. Fix restricts flooding to
  parent_layers only.

Additionally adds THEN branch detection via AST analysis when save_source_context=True, with IF/THEN
  edge labels in visualization. Includes 8 new test models, 22 new tests, and fixes missing
  'verbose' in MODEL_LOG_FIELD_ORDER.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Use Worker thread for ELK layout to fix stack overflow on large graphs
  ([`3fe6a84`](https://github.com/johnmarktaylor91/torchlens/commit/3fe6a84c3d9d8a5856dfadcf451ef85700a2385c))

V8's --stack-size flag silently caps at values well below what's requested, causing "Maximum call
  stack size exceeded" on 1M+ node graphs. Switch to Node.js Worker threads with
  resourceLimits.stackSizeMb, which reliably delivers the requested stack size at the V8 isolate
  level.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.3 (2026-03-08)

### Bug Fixes

- **vis**: Increase ELK Node.js stack floor to 4GB for large graphs
  ([`29af94e`](https://github.com/johnmarktaylor91/torchlens/commit/29af94ed51b9018ba214f2de15c48e5454a773b3))

128MB was insufficient for ELK's recursive layout on 500k+ node graphs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Raise OS stack limit for ELK Node.js subprocess
  ([`da82c9d`](https://github.com/johnmarktaylor91/torchlens/commit/da82c9d0fd66c2fdcbaabb86b31750811693a930))

The OS soft stack limit (ulimit -s) was smaller than the --stack-size value passed to Node.js,
  causing a segfault on large graphs (500k+ nodes) instead of allowing V8 to use the requested
  stack. Uses preexec_fn to set RLIMIT_STACK to unlimited in the child process only.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Performance Improvements

- **decoration**: Optimize model prep and move session attrs to ModelLog dicts
  ([`b63a4fa`](https://github.com/johnmarktaylor91/torchlens/commit/b63a4fa3871113a5cc73554c576fb11c5eac2cd2))

Five performance fixes for _prepare_model_session and related setup code:

- PERF-38: Replace O(N²) list concat in _traverse_model_modules with deque - PERF-37: Cache
  user_methods per class in _get_class_metadata; move _pytorch_internal set to module-level
  frozenset - PERF-36: Iterate module._parameters directly instead of rsplit on named_parameters
  addresses + lookup dict - PERF-39: Skip patch_model_instance for already-prepared models - Move 4
  session-scoped module attrs (tl_module_pass_num, tl_module_pass_labels, tl_tensors_entered_labels,
  tl_tensors_exited_labels) from nn.Module instances to ModelLog dicts keyed by id(module). Remove
  tl_source_model_log (dead code). Eliminates per-module cleanup iteration in
  _cleanup_model_session.

At 10K modules: ensure_prepared repeat calls drop from ~48ms to ~0.4ms (111x), session setup ~1.3x
  faster, cleanup ~1.4x faster.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.2 (2026-03-08)

### Bug Fixes

- **vis**: Increase ELK Node.js stack size to prevent overflow
  ([`b8edbc8`](https://github.com/johnmarktaylor91/torchlens/commit/b8edbc8c779dda44b046c9efa4855b3a6fad46f7))

Bump --stack-size floor from 64MB to 128MB and multiplier from 16x to 48x (matching heap scaling) to
  prevent "Maximum call stack size exceeded" in elkjs on large graphs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **scripts**: Enable loop detection in render_large_graph
  ([`803e16f`](https://github.com/johnmarktaylor91/torchlens/commit/803e16f3ecec7e2fbb1c662ce963f7de51f4d16c))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.1 (2026-03-08)

### Chores

- **scripts**: Use log_forward_pass vis_opt instead of separate render call
  ([`d2aea0f`](https://github.com/johnmarktaylor91/torchlens/commit/d2aea0fab326e02444d99aadc7dcf12abf9c8b10))

Let verbose mode handle all phase timing instead of manual timestamps. Use log_forward_pass's
  built-in vis_opt to render in one call.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Performance Improvements

- **model_prep**: Optimize _prepare_model_session for large models
  ([`2892323`](https://github.com/johnmarktaylor91/torchlens/commit/2892323ac2910532c9bbde894280ba917a72e966))

- Hoist set(dir(nn.Module)) to module-level constant _NN_MODULE_ATTRS - Replace dir(module) MRO walk
  with __dict__ scans for attrs and methods - Pre-build address→module dict to eliminate
  per-parameter tree walks - Use model.modules() with cached tl_module_address instead of second DFS

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.0 (2026-03-08)

### Chores

- **scripts**: Unify large graph render scripts into single parameterized script
  ([`07a8186`](https://github.com/johnmarktaylor91/torchlens/commit/07a8186537cf88bb39f4abae55a05dc01ec16457))

Replace `run_250k.py` and `run_1M.py` with `render_large_graph.py` that accepts any node count as a
  CLI argument, plus --format, --seed, and --outdir options.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **logging**: Add verbose mode for timed progress messages
  ([`0603f10`](https://github.com/johnmarktaylor91/torchlens/commit/0603f1035b8be7c2c44a416099fac282eaafa686))

Add `verbose: bool = False` parameter to `log_forward_pass`, `show_model_graph`, and internal
  pipeline functions. When enabled, prints `[torchlens]`-prefixed progress at each major pipeline
  stage with timing. Also fixes `_trim_and_reorder_model_history_fields` to preserve all non-ordered
  attributes (not just private ones).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.19.0 (2026-03-08)

### Chores

- Add large graph render scripts to scripts/
  ([`d02233b`](https://github.com/johnmarktaylor91/torchlens/commit/d02233bc050657ed6088b8338e056c2c531d45bd))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Update RESULTS.md and tests/CLAUDE.md with current counts
  ([`75fa346`](https://github.com/johnmarktaylor91/torchlens/commit/75fa3463b16505d21b3498a29c848e5da248bd36))

- Total tests: 892 → 951, test files: 14 → 15, toy models: 249 → 250 - Add test_large_graphs.py (51
  tests) to file tables - Add decoration overhead benchmark table to profiling baselines - Add large
  graph scaling section (100 to 1M nodes) - Update all per-file test counts to current values

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **capture**: Add func_config field for lightweight hyperparameter extraction
  ([`7144d6d`](https://github.com/johnmarktaylor91/torchlens/commit/7144d6d5200bb17377956ffb8c579553ae34ddfe))

Adds a `func_config` dict to every LayerPassLog/LayerLog containing computation-defining
  hyperparameters (kernel_size, stride, in/out channels, dropout p, etc.) extracted at capture time
  with zero tensor cloning. Empty for source tensors and output nodes.

Also fixes pre-existing test failures in test_validation.py (read-only property assignments) and
  adds detect_loops to MODEL_LOG_FIELD_ORDER.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **profiling**: Add decoration overhead benchmark to profiling report
  ([`c8fbffd`](https://github.com/johnmarktaylor91/torchlens/commit/c8fbffdc43b4f97a2e7ab90cecd78a8d59950592))

Measures per-call overhead of TorchLens's toggle-gated wrappers when logging is disabled. Benchmarks
  11 functions from cheap (relu, add) to heavy (conv2d, SDPA) — confirms ~600ns overhead on cheap
  ops, <1% on real compute.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.18.0 (2026-03-08)

### Features

- **data**: Add MACs properties to LayerPassLog, LayerLog, ModelLog, ModuleLog
  ([`c60e7b9`](https://github.com/johnmarktaylor91/torchlens/commit/c60e7b9a2fd29de09d82c5be076e75c6e4c8b39d))

MACs (multiply-accumulate operations) = FLOPs / 2. Added: - LayerPassLog: macs_forward,
  macs_backward properties - LayerLog: macs_forward, macs_backward properties - ModelLog:
  total_macs_forward, total_macs_backward, total_macs, macs_by_type() - ModuleLog: flops_forward,
  flops_backward, flops, macs_forward, macs_backward, macs

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **data**: Convert 23 stored fields to computed @properties across data classes
  ([`2c22208`](https://github.com/johnmarktaylor91/torchlens/commit/2c22208a72636bb796268510a3529cf1183519eb))

Replace redundant stored fields with computed @property methods that derive their values from
  existing data. Eliminates ~155 lines of write-site code across 13 files while preserving identical
  behavior and passing all invariants.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.17.0 (2026-03-08)

### Bug Fixes

- **types**: Add mypy annotations for defaultdict and deque in elk_layout
  ([`98478a3`](https://github.com/johnmarktaylor91/torchlens/commit/98478a33a28e890554bad6325478efb8a2ff5f85))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **vis**: Scale ELK rendering to 250k+ nodes, add detect_loops option
  ([`4707631`](https://github.com/johnmarktaylor91/torchlens/commit/470763146ce30334b5853cfc7cfd104035fd9922))

ELK's layered algorithm (Sugiyama) uses O(n²) memory for crossing minimization, causing
  std::bad_alloc at ~150k+ nodes. This adds:

- Auto-switch to ELK stress algorithm above 150k nodes (O(n) memory) - Topological position seeding
  for stress to preserve directional flow - Increased Node.js heap allocation (16GB floor, 48x JSON
  size) - Better error messages when Node.js OOM-kills (was silent empty stderr) - `detect_loops`
  parameter on log_forward_pass/show_model_graph to skip expensive isomorphic subgraph expansion,
  keeping only same-param grouping (Rule 1). Default True (existing behavior unchanged). - 8 loop
  comparison tests rendering with/without loop detection

Successfully renders 250k-node graphs in ~19 minutes (was impossible). 1M-node render in progress.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.16.4 (2026-03-08)

### Bug Fixes

- **validation**: Use call nesting depth not address depth in invariant Q, fix test inputs
  ([#117](https://github.com/johnmarktaylor91/torchlens/pull/117),
  [`98660ff`](https://github.com/johnmarktaylor91/torchlens/commit/98660ff114f31a2324d3268c0d925cf0b31c02c9))


## v0.16.3 (2026-03-08)

### Bug Fixes

- **tests**: Rename model_kwargs to input_kwargs and fix test configs
  ([`8768339`](https://github.com/johnmarktaylor91/torchlens/commit/87683393e2c38a2703a73643c7b975517bffd589))

- Rename model_kwargs= to input_kwargs= in 17 test call sites to match API - Skip test_gpt_bigcode
  (JIT-compiled attention incompatible with TorchLens) - Fix test_deformable_detr backbone
  out_features to match 2-layer ResNet - Add index, symint, checkkeypaddingmask to _UNARY_FUNCS
  ArgSpec entries

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **types**: Resolve 14 pre-existing mypy errors across 4 files
  ([`7adcc36`](https://github.com/johnmarktaylor91/torchlens/commit/7adcc362552697cf8961024d9bc1dcd7a4fb347f))

- rng.py: annotate rng_dict as Dict[str, object] for mixed-type values - elk_layout.py: annotate
  out_shape as tuple to avoid index-out-of-range - model_prep.py: annotate meta as Dict[str,
  object], add type: ignore for dynamic tl_buffer_address attrs on Tensor - output_tensors.py: add
  type: ignore for ArgSpec arg-type and assignment

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.16.2 (2026-03-08)

### Bug Fixes

- **validation**: Exempt exponential_ from perturbation check
  ([`a6dce40`](https://github.com/johnmarktaylor91/torchlens/commit/a6dce40dc314a36a0296a7c65dd4faf04647b126))

In-place RNG op — output is determined by RNG state, not input values. Fixes test_gumbel_vq_model.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Exempt maximum/minimum from perturbation check
  ([`9d658bc`](https://github.com/johnmarktaylor91/torchlens/commit/9d658bc1d80222523ec7e410c0cf1d38665b16f2))

torch.maximum/minimum with extreme-valued args (e.g. RWKV's negative infinity masks) are insensitive
  to perturbation — same as max/min.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Fix perturbation precision and add exemptions for C++ ops
  ([`7f86655`](https://github.com/johnmarktaylor91/torchlens/commit/7f86655705a8d914126a174ff8b95cf3cc123cee))

- Scale constant-tensor perturbation by ±10% of magnitude to ensure float32-distinguishable values
  while staying in safe range - Add posthoc magnitude-ratio exemption: when non-perturbed parent's
  magnitude dwarfs the perturbed parent's (>100x), float32 arithmetic swallows the perturbation —
  exempt rather than fail - Add maximum/minimum to posthoc exemption (element-wise binary ops
  insensitive to perturbation when one arg dominates) - Skip perturbation for _op (torchvision C++
  PyCapsule ops like nms, roi_align) — perturbed coordinates segfault native extensions - Fix
  nystromformer test: seq_len must equal num_landmarks² (64)

Fixes: test_fcos_resnet50_train, test_retinanet_resnet50_train, test_nystromformer,
  test_maskrcnn_resnet50_train

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Scale perturbation range by magnitude for large constants
  ([`efcd876`](https://github.com/johnmarktaylor91/torchlens/commit/efcd8766b38c6ea9f5d3715fac4994509985299b))

Constant tensors with large values (e.g. 1e8) were perturbed by ±1.0, which is indistinguishable in
  float32 precision. Now scales expansion by abs(value) * 1e-4 to ensure perturbation is always
  detectable.

Fixes test_fcos_resnet50_train validation failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Use relative threshold for near-constant tensor perturbation
  ([`e0c82c7`](https://github.com/johnmarktaylor91/torchlens/commit/e0c82c7a63e63d2f6e033274d152afc68ecb060b))

Near-constant float tensors (e.g. [2.6785714, 2.6785717]) had a value range smaller than float32
  precision, so perturbation within [min, max] produced the same value after rounding. Changed exact
  `lo == hi` check to relative threshold `hi - lo < max(1e-6, abs(lo) * 1e-6)` to trigger range
  expansion for these cases.

Fixes test_ssd300_vgg16_train validation failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Probe nvm paths when node is not on PATH
  ([`4d3ae24`](https://github.com/johnmarktaylor91/torchlens/commit/4d3ae2414bbcc0835d4cb01da230f158004da690))

Non-interactive shells (IDE test runners, cron, subprocesses) often lack nvm's PATH additions,
  causing elk_available() to return False even when elkjs is installed. Add _find_node_binary() to
  probe ~/.nvm/versions/node/ as a fallback, and inject the node binary's directory into the
  subprocess PATH so elkjs detection works regardless of shell configuration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.16.1 (2026-03-07)

### Bug Fixes

- **tests**: Use relative import for example_models in test_large_graphs
  ([`e2d0ae4`](https://github.com/johnmarktaylor91/torchlens/commit/e2d0ae476e4d54c2211a3629e8958032c82b1c0f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Harden ELK heap scaling and fix flaky signal safety test
  ([`41b9f89`](https://github.com/johnmarktaylor91/torchlens/commit/41b9f893692bc39a4ea0342092df62b2f2ee2b38))

- Bump ELK Node.js heap scaling from 8x to 16x JSON size to prevent OOM on 250k+ node graphs - Mark
  100k node tests as @rare (too slow for regular runs) - Fix flaky TestSignalSafety: use
  setitimer(50ms) instead of alarm(1s), increase model iterations to 50k, skip if alarm doesn't fire

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.16.0 (2026-03-07)

### Bug Fixes

- **validation**: Batch of bug fixes for edge-case models
  ([`dca5a7e`](https://github.com/johnmarktaylor91/torchlens/commit/dca5a7e11d31c81823601b5b93f79e143cd33c55))

- Fix vmap/functorch compatibility: skip logging inside functorch transforms to avoid missing
  batching rules (torch_funcs.py) - Fix tensor_nanequal infinite recursion: wrap decorated tensor
  ops (.isinf, .resolve_conj, etc.) in pause_logging() (tensor_utils.py) - Fix perturbation for
  range-restricted functions: use uniform random within original value range instead of scaled
  normal (core.py) - Fix atomic_bool_val crash inside vmap context (output_tensors.py) - Fix output
  node initialized_inside_model flag (graph_traversal.py) - Add scatter_ full-overwrite exemption
  (exemptions.py) - Add max/min indices exemption for integer dtype outputs - Add bernoulli scalar-p
  exemption, constant-output exemption - Add non-perturbed parent special-value check for nested
  args (einsum) - Fix buffer_xrefs invariant: accept ancestor module matches - Fix real-world model
  configs: CvT, CLAP, EnCodec, SpeechT5, Informer, Autoformer, MobileBERT kwarg names

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Fix ELK rendering and scale for large graphs
  ([`ea96a85`](https://github.com/johnmarktaylor91/torchlens/commit/ea96a85a766068527366cbe0a77ecd7c7f467eef))

- Fix neato -n position format: use points (not inches) — fixes empty nodes, missing edges, and
  invisible labels in ELK output - Auto-scale Node.js heap and stack with graph JSON size -
  Auto-scale ELK and neato timeouts with node count - Use straight-line edges for graphs > 1k nodes
  (spline routing is O(n^2)) - Warn users to use SVG format for graphs > 25k nodes (PDF renders
  empty) - Add dot-vs-ELK aesthetic comparison tests at 15/100/500/1k/3k nodes - Add 1M node test
  (rare marker) for trophy-file rendering

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **vis**: Add ELK layout engine for large graph visualization
  ([`5245872`](https://github.com/johnmarktaylor91/torchlens/commit/5245872746ee07dddd41a222392aa7548e33ce3f))

- New elk_layout.py: ELK-based node placement via Node.js/elkjs subprocess, with graceful fallback
  to sfdp when unavailable - New vis_node_placement parameter ("auto"/"dot"/"elk"/"sfdp") threaded
  through show_model_graph, log_forward_pass, and render_graph - Auto mode uses dot for <3500 nodes,
  ELK (or sfdp fallback) for larger - RandomGraphModel in example_models.py: seeded random model
  generator with calibrated node counts (within ~5% of target up to 100k+) - 39 tests in
  test_large_graphs.py: node count accuracy, validation, engine selection, ELK utilities, rendering
  at 3k-100k scales, dot threshold benchmark - Increased Node.js stack size (--stack-size=65536) to
  handle graphs up to 250k+ nodes without stack overflow

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Add hierarchical module grouping to ELK layout
  ([`bcc1ad2`](https://github.com/johnmarktaylor91/torchlens/commit/bcc1ad207aa2d9512960201594bfa9afc2495fb0))

- New build_elk_graph_hierarchical(): builds nested ELK compound nodes from module containment
  structure (containing_modules_origin_nested) - ELK's "INCLUDE_CHILDREN" hierarchy handling
  preserves module grouping in the layout — nodes within the same module cluster together -
  inject_elk_positions() now recurses into compound nodes, accumulating absolute positions from
  nested ELK coordinates - render_with_elk() passes entries_to_plot for hierarchical layout, falls
  back to flat DOT parsing when entries not available - Tests for hierarchical graph building and
  nested position injection

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **vis**: Add rare marker for 250k-node tests, fill test coverage gaps
  ([`c749b94`](https://github.com/johnmarktaylor91/torchlens/commit/c749b94e69a1eee18f9084c1f46efc472fb3fe72))

- New pytest marker "rare": always excluded by default via addopts, run explicitly with `pytest -m
  rare` - Add 250k node count, validation, and ELK render tests (marked rare) - Fill gaps:
  validation tests at 5k/10k/20k/50k/100k, ELK render tests at 5k/20k, node count test at 20k

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.15 (2026-03-07)

### Bug Fixes

- **gc**: Release ParamLog._param_ref on cleanup, add GC test suite (#GC-1, #GC-12)
  ([`83e1bd2`](https://github.com/johnmarktaylor91/torchlens/commit/83e1bd2df4304b55bff7d59bea8ffe2728ee7c7f))

- Add ParamLog.release_param_ref() to cache grad metadata then null _param_ref - cleanup() now nulls
  all _param_ref before clearing entries - Add ModelLog.release_param_refs() public API for early
  param release - Add _param_logs_by_module to cleanup's internal containers list - New test_gc.py
  with 10 tests covering ModelLog/param GC, memory growth, save_new_activations stability, and
  transient data clearing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Move RESULTS.md to repo root for visibility
  ([`2e814dd`](https://github.com/johnmarktaylor91/torchlens/commit/2e814ddbd3421b3c81d7f2518d169a3470c8e9a8))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **tests**: Add public test results summary
  ([`fd7b33f`](https://github.com/johnmarktaylor91/torchlens/commit/fd7b33ff4f8bf1b4b74548642f70dc58a98f85f7))

Committed tests/RESULTS.md with suite overview, model compatibility matrix (121 toy + 85
  real-world), profiling baselines, and pointers to generated reports. Transparent scoreboard for
  the repo.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **models**: Add autoencoders, state space models, and architecture coverage
  ([`716cc9d`](https://github.com/johnmarktaylor91/torchlens/commit/716cc9dc425eb6245886eb7873c7a212db16d4d5))

Toy models (18 new in example_models.py): - Autoencoders: VanillaAutoencoder, ConvAutoencoder,
  SparseAutoencoder, DenoisingAutoencoder, VQVAE, BetaVAE, ConditionalVAE - State space: SimpleSSM,
  SelectiveSSM (Mamba-style), GatedSSMBlock, StackedSSM - Additional: SiameseNetwork, MLPMixer,
  SimpleGCN, SimpleGAT, SimpleDiffusion, SimpleNormalizingFlow, CapsuleNetwork

Real-world models (5 new in test_real_world_models.py): - SSMs: Mamba, Mamba-2, RWKV, Falcon-Mamba
  (via transformers) - Autoencoders: ViT-MAE ForPreTraining (via transformers)

All 22 new tests pass. Updated RESULTS.md to reflect 736 total tests, 139 toy models, 92 real-world
  models.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Close remaining stress-test gaps — MAML, NeRF, RecurrentGemma, VOLO
  ([`a1f2254`](https://github.com/johnmarktaylor91/torchlens/commit/a1f22543bcc2cbb22b2008121a5082e64a534860))

Toy models (+2): - MAMLInnerLoop: higher-order gradients (torch.autograd.grad inside forward) -
  TinyNeRF: differentiable volumetric rendering (ray marching + alpha compositing)

Real-world models (+2): - RecurrentGemma: Griffin architecture (linear recurrence + local attention
  hybrid) - VOLO: outlooker attention (distinct from standard self-attention)

Closes 37/38 stress-test patterns from taxonomy. Only remaining gap is test-time training (TTT
  layers) which requires gradient computation within inference — fundamentally incompatible with
  TorchLens forward-pass logging.

Total: 249 toy models, 185 real-world models, 892 tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Exhaustive architecture coverage across 30+ categories
  ([`8411a72`](https://github.com/johnmarktaylor91/torchlens/commit/8411a725439819de55a54f4339afa21410f67686))

Add 32 new toy models (Groups M-R) covering distinct computational patterns: - Group M: Attention
  variants (MQA, GQA, RoPE, ALiBi, slot, cross-attention) - Group N: Gating & skip patterns
  (highway, SE, depthwise-sep, inverted-residual, FPN) - Group O: Generative & self-supervised
  (hierarchical VAE, gated conv, masked conv, SimCLR, stop-gradient/BYOL, AdaIN) - Group P: Exotic
  architectures (hypernetwork, DEQ, neural ODE, NTM memory, SwiGLU) - Group Q: Graph neural networks
  (GraphSAGE, GIN, EdgeConv, graph transformer) - Group R: Additional patterns (MoE, spatial
  transformer, dueling DQN, RMS norm, sparse pruning, Fourier mixing)

Add 37 new real-world model tests: - Decoder-only LLMs: LLaMA, Mistral, Phi, Gemma, Qwen2, Falcon,
  BLOOM, OPT - Encoder-only: ALBERT, DeBERTa-v2, XLM-RoBERTa - Encoder-decoder: Pegasus, LED -
  Efficient transformers: FNet, Nystromformer, BigBird - MoE: Mixtral, Switch Transformer - Vision
  transformers: DeiT, CvT, SegFormer - Detection: DETR, Mask R-CNN (train+eval) - Perceiver IO,
  PatchTST, Decision Transformer - timm: HRNet, EfficientNetV2, LeViT, CrossViT, PVT-v2, Twins-SVT,
  FocalNet - GNN (PyG): GraphSAGE, GIN, Graph Transformer

Total: 805 tests, 213 toy models, 129 real-world tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Exhaustive coverage expansion — 20 toy + 33 real-world architectures
  ([`4f4e7ae`](https://github.com/johnmarktaylor91/torchlens/commit/4f4e7aeb958e35ed7845b03b209367d8d55e5ab0))

Toy models (+20): GRU, NiN, ChannelShuffle, PixelShuffle, PartialConv, FiLM, CoordinateAttention,
  DifferentialAttention, RelativePositionAttention, EarlyExit, MultiScaleParallel, GumbelVQ,
  EndToEndMemoryNetwork, RBFNetwork, SIREN, MultiTask, WideAndDeep, ChebGCN, PrototypicalNetwork,
  ECA.

Real-world models (+33): GPT-J, GPTBigCode, GPT-NeoX, FunnelTransformer, CANINE, MobileBERT, mBART,
  ProphetNet, WavLM, Data2VecAudio, UniSpeech, ConvNeXt-v2, NFNet, DaViT, CoAtNet, RepVGG, ReXNet,
  PiT, Visformer, GC-ViT, EfficientFormer, FastViT, NesT, Sequencer2D, TResNet, SigLIP, BLIP-2,
  Deformable DETR, LayoutLM, TimeSeriesTransformer, ChebConv, SGConv, TAGConv.

Total: 241 toy models, 183 real-world models, 882 tests. RESULTS.md updated with all new entries and
  pattern coverage table.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Final coverage pass — 6 novel computational patterns
  ([`e33c71a`](https://github.com/johnmarktaylor91/torchlens/commit/e33c71a0cec2a64fce944aa36a42690fe3f6fe15))

New toy models targeting genuinely missing graph patterns: - LinearAttentionModel: kernel-based
  phi(Q)(phi(K)^T V), no softmax - SimpleFNO: FFT -> learned spectral weights -> iFFT (Fourier
  Neural Operator) - PerceiverModel: cross-attention to fixed learned latent bottleneck - ASPPModel:
  multi-rate parallel dilated convolutions (DeepLab ASPP) - ControlNetModel: parallel encoder copy +
  zero-conv injection - SimpleEGNN: E(n) equivariant message passing with coordinate updates

Total: 247 toy models, 183 real-world models, 888 tests. RESULTS.md updated with new patterns and
  counts.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Gap-fill 8 toy models + 21 real-world models for exhaustive coverage
  ([`7d5f879`](https://github.com/johnmarktaylor91/torchlens/commit/7d5f879bdefc9547d7341eea8b9c8adda3fd7e9f))

Toy models (8 new, 221 total): - LeNet5, BiLSTM, Seq2SeqWithAttention, TripletNetwork -
  BarlowTwinsModel, DeepCrossNetwork, AxialAttentionModel, CBAMBlock

Real-world models (21 new, 150 total): - TorchVision: MobileNetV3, Keypoint R-CNN (train+eval) -
  timm: Res2Net, gMLP, ResMLP, EVA-02 - HF decoder-only: OLMo - HF vision: DINOv2 - HF efficient:
  Longformer, Reformer - HF audio: AST, CLAP, EnCodec, SEW, SpeechT5, VITS - HF time series:
  Informer, Autoformer - PyG GNN: GATv2, R-GCN

834 total tests, 221 toy models, 150 real-world tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.14 (2026-03-07)

### Performance Improvements

- **capture**: Lazy _fsize_nice properties, remove _trim_and_reorder, batch pause_logging, drop
  deepcopy
  ([`986bd91`](https://github.com/johnmarktaylor91/torchlens/commit/986bd914a317be6acde0a4c0898a417fb9f6dcbb))

Four low-risk optimizations targeting remaining allocation pressure and per-operation overhead in
  the instrumentation path:

1. Lazy _fsize_nice properties — tensor_fsize_nice, grad_fsize_nice, parent_params_fsize_nice,
  total_params_fsize_nice, params_fsize_nice, and fsize_nice converted from eagerly computed strings
  to @property methods. Eliminates ~2700 human_readable_size() calls per Swin-T pass.

2. Remove _trim_and_reorder from postprocess — the OrderedDict rebuild of every LayerPassLog's
  __dict__ (685 calls, ~0.04s on Swin-T) is purely cosmetic. Python 3.7+ dicts maintain insertion
  order. Function definition kept for opt-in use.

3. Batch pause_logging for tensor memory — inline nelement() * element_size() at the two hottest
  call sites (_build_param_fields, _log_output_tensor_info) inside a single pause_logging() context.
  Eliminates per-call context manager overhead (~1088 calls).

4. Remove activation_postfunc deepcopy — copy.deepcopy() on a callable is unnecessary; callables are
  effectively immutable.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.13 (2026-03-07)

### Performance Improvements

- **capture**: Speed-optimized defaults and remaining bottleneck elimination
  ([#110](https://github.com/johnmarktaylor91/torchlens/pull/110),
  [`7c9a00c`](https://github.com/johnmarktaylor91/torchlens/commit/7c9a00cdec26258090cf6be1c93869cb82aa351a))

Seven targeted optimizations that reduce Swin-T log_forward_pass from 5.91s to 1.55s (3.8x):

1. Unified save_source_context flag (was save_call_stacks) — controls both per-function call stacks
  AND module source/signature fetching. Default: False. 2. save_rng_states=False default — skips
  per-op RNG state capture. Auto-enabled by validate_forward_pass. Uses torch_only=True when enabled
  (skips Python/NumPy RNG). 3. Inline isinstance in wrapped_func — _collect_tensor_args() and
  _collect_output_tensors() replace BFS crawls for flat arg/output cases. Falls back to BFS only for
  nested containers. 4. __dict__ scan for buffer prep/cleanup — replaces iter_accessible_attributes
  (dir() + MRO walk) with direct __dict__ iteration. 10x faster for buffer tagging and tensor
  cleanup. 5. Hoisted warnings.catch_warnings() — moved from per-attribute (46K entries) to caller
  level. 6. Lazy module metadata — _get_class_metadata skips
  inspect.getsourcelines/inspect.signature when save_source_context=False. Only captures class name
  and docstrings. 7. Module-level import weakref — moved from per-call in _trim_and_reorder to
  module level.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.12 (2026-03-06)

### Performance Improvements

- **capture**: O(1) tensor/param extraction via per-function ArgSpec lookup table
  ([`b1d6c56`](https://github.com/johnmarktaylor91/torchlens/commit/b1d6c56f4f1c5cf1176232e8fc33e229f8555baf))

Replace expensive 3-level BFS crawl (~1.44s, 39% self-time, ~1.9M getattr calls) with O(1)
  position-based lookups using a static ArgSpec table of 350+ entries. Three-tier strategy: static
  table for known torch functions, dynamic cache for user-defined modules, BFS fallback (fires at
  most once per unique class).

Also hoists warnings.catch_warnings() from per-attribute (~77K entries) to per-call level, and adds
  usage stats collection + coverage test infrastructure.

Benchmark: Swin-T log_forward_pass 5.91s → 4.41s (-25%).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.11 (2026-03-06)

### Bug Fixes

- **gc**: Convert back-references to weakrefs and add optional call stack collection
  ([`2bed167`](https://github.com/johnmarktaylor91/torchlens/commit/2bed167d78fed3a143060f8f5d75007cd3a06214))

Convert 5 circular back-references from strong to weakref.ref() so ModelLog and its children
  (LayerPassLog, LayerLog, ModuleLog, BufferAccessor, LayerAccessor) no longer prevent timely
  garbage collection. GPU tensors are now freed immediately when the last strong reference to
  ModelLog is dropped, instead of waiting for Python's gen-2 GC cycle.

Also add save_call_stacks parameter to log_forward_pass() (default True). When False, skips
  _get_func_call_stack() on every tensor operation, which is the main per-op overhead in production
  use. Call stacks remain on by default for pedagogical use.

Fixes: GC-2, GC-3, GC-4, PERF-19

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Add folder-wise CLAUDE.md files and comprehensive inline documentation
  ([`deaec2d`](https://github.com/johnmarktaylor91/torchlens/commit/deaec2d4b3fb3c7ffee1fe7f61c8701bdc36dc70))

- Add CLAUDE.md to every package directory (torchlens/, capture/, data_classes/, decoration/,
  postprocess/, utils/, validation/, visualization/, tests/, scripts/, .github/) with file maps, key
  concepts, gotchas, and cross-references - Add module-level docstrings, function/class docstrings,
  and inline comments across all 39 source files explaining non-obvious logic, ordering
  dependencies, design decisions, and invariants - Fix coverage HTML output directory in
  pyproject.toml to point to tests/test_outputs/reports/coverage_html (matching conftest.py)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.10 (2026-03-05)

### Bug Fixes

- **tests**: Move coverage HTML output to reports directory
  ([`916b2d1`](https://github.com/johnmarktaylor91/torchlens/commit/916b2d19a0aec297546230a87db1290aea1f9984))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.9 (2026-03-05)

### Bug Fixes

- **tests**: Use render_graph return value instead of reading cleaned-up .gv file
  ([`7998776`](https://github.com/johnmarktaylor91/torchlens/commit/7998776401752eeedb5693b665114c29234c893d))

Commit 147c7b7 added cleanup=True to dot.render(), which deletes the intermediate .gv source file
  after rendering. The TestVisualizationParams tests were reading that source file and all 15 failed
  with FileNotFoundError.

render_graph now returns dot.source so tests can inspect the graphviz source without depending on
  the intermediate file.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.8 (2026-03-05)

### Bug Fixes

- **tests**: Generate HTML coverage report in sessionfinish hook
  ([`29095f9`](https://github.com/johnmarktaylor91/torchlens/commit/29095f91e19bfbfd0a66a855232247679183d85f))

The pyproject.toml configured coverage_html output directory but the pytest_sessionfinish hook only
  generated the text report. Add cov.html_report() call so HTML reports are written alongside the
  text summary.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.7 (2026-03-04)

### Bug Fixes

- **vis**: Clean up intermediate .gv source files after rendering
  ([`147c7b7`](https://github.com/johnmarktaylor91/torchlens/commit/147c7b7329ae3e86ee1c65dbe61e30a849dd719f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **tests**: Add @pytest.mark.smoke to 18 critical-path tests
  ([`d26f4ec`](https://github.com/johnmarktaylor91/torchlens/commit/d26f4ec86fbe64f26deb4c40880da4fb5882718c))

18 fast tests across 9 files marked as smoke tests for quick validation during development. Run with
  `pytest tests/ -m smoke` (~6s).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.6 (2026-03-04)

### Bug Fixes

- **tests**: Add explicit vis_outpath to render_graph calls
  ([`7d7926b`](https://github.com/johnmarktaylor91/torchlens/commit/7d7926b22781e3e78ca9cde166efc9ead7b0ec57))

Two tests in TestVisualizationBugfixes called render_graph() without vis_outpath, causing stray
  modelgraph/modelgraph.pdf files in whatever directory tests were run from.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **tests**: Rename test_outputs/graphs to test_outputs/visualizations
  ([`fbf1fe6`](https://github.com/johnmarktaylor91/torchlens/commit/fbf1fe66786cf829ed4a26e76a9a934708e92a75))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.5 (2026-03-04)

### Bug Fixes

- **loop-detection**: Group param-sharing ops by func_applied_name + param_barcodes
  ([`f1e795b`](https://github.com/johnmarktaylor91/torchlens/commit/f1e795b05365b6929fbf3a7b3a30446d93073674))

Previously _merge_iso_groups_to_layers Pass 2 grouped operations by parent_param_barcodes alone,
  causing different ops on the same params (e.g. __getitem__ vs __add__) to be incorrectly merged
  into one layer. Now groups by (func_applied_name, sorted(parent_param_barcodes)). Also updates
  docstrings to reflect the correct grouping rule.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **tests**: Tidy test_outputs structure and remove private bug numbers
  ([`03817a2`](https://github.com/johnmarktaylor91/torchlens/commit/03817a25f1091f4cb0bf16b9f547cab08ecc515d))

- Restructure test_outputs/ into reports/ and graphs/ subdirectories - Update all path references in
  conftest, test_output_aesthetics, test_profiling, and ui_sandbox notebook - Strip private bug
  number references (Bug #N, #N:) from comments, docstrings, and class names across 7 test files -
  Rename TestBug* classes to descriptive names

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.4 (2026-03-04)

### Bug Fixes

- **validation**: Resolve 14 test failures across validation, labeling, and warnings
  ([`ffc51a5`](https://github.com/johnmarktaylor91/torchlens/commit/ffc51a5be57033ffc1b0506af5a330bd97a343b0))

- Fix param sharing invariant: include func_applied_name in grouping key so different operations
  consuming the same parameter aren't incorrectly required to share labels (12 model failures: vit,
  beit, cait, coat, convit, poolformer, xcit, t5, clip, blip, etc.) - Fix unused input validation:
  guard against func_applied=None for unused model inputs like DistilBert's token_type_ids (1
  failure) - Fix pass labeling consistency: inherit layer_type from first pass for pass>1 entries,
  preventing label mismatches within same_layer_operations groups (SSD300 train failure) - Fix numpy
  2.0 deprecation: use cpu_data.detach().numpy() instead of np.array(cpu_data) - Suppress 6 external
  third-party warnings in pyproject.toml filterwarnings - Add 4 regression tests with 2 helper model
  classes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.3 (2026-03-04)

### Bug Fixes

- **types**: Move type: ignore to correct lines after ruff reformat
  ([`8205ca4`](https://github.com/johnmarktaylor91/torchlens/commit/8205ca4002ab2f0d0ea5f4b09411735b8ff159a4))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **types**: Resolve all 181 mypy errors across 18 source files
  ([`546fd94`](https://github.com/johnmarktaylor91/torchlens/commit/546fd9438241da86cf394a03716d8b3dd32391ab))

- Auto-fix implicit Optional with no_implicit_optional tool (24 errors) - Add type annotations for
  untyped variables (20 errors) - Add type: ignore[attr-defined] for dynamic tl_* attributes (24
  errors) - Add type: ignore[assignment] for fields_dict and cross-type assignments (79 errors) -
  Add type: ignore[union-attr] for ModuleLog|ModulePassLog unions (19 errors) - Add type:
  ignore[arg-type] for acceptable type mismatches (19 errors) - Fix callable type annotation in
  exemptions.py (Callable[..., bool]) - Fix found_ids parameter type in introspection.py (List ->
  Set)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **types**: Suppress torch.device return-value mismatch in CI mypy
  ([`11922b3`](https://github.com/johnmarktaylor91/torchlens/commit/11922b32aaa6ca5f9769785b4afe737121ae3fae))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **ci**: Remove continue-on-error from mypy check — now a blocking gate
  ([`0e76db0`](https://github.com/johnmarktaylor91/torchlens/commit/0e76db01d0016e1a2b033af29b0b5e4fbc7933b8))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.2 (2026-03-04)

### Bug Fixes

- **core**: Resolve remaining open bugs — FLOPs accuracy, PERF-21, mypy types (#19, #154-161)
  ([`da02dc5`](https://github.com/johnmarktaylor91/torchlens/commit/da02dc56b4940cf32c5e319dc4310ffa7e15c0e3))

Bug #19: Verified fast-pass buffer orphan resolved by Bug #116 guard; added regression tests FLOPs:
  Fix addbmm/baddbmm batch dimension, add einsum handler, fix pool kernel_size extraction

PERF-21: Cache get_ignored_functions()/get_testing_overrides() in _get_torch_overridable_functions

Bug #154: Type _SENTINEL as Any in func_call_location.py Bug #155: Eliminate reused variable across
  str/bytes types in hashing.py Bug #156: Accept float in human_readable_size signature Bug #157:
  Use setattr/getattr for dynamic tl_tensor_label_raw in safe_copy Bug #158: Add List[Any]
  annotation to AutocastRestore._contexts Bug #160: Add _FlopsHandler type alias and typed
  SPECIALTY_HANDLERS dict Bug #161: Add ModuleLog TYPE_CHECKING import and type annotations in
  invariants.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **ci**: Switch CI from pip to uv for faster installs
  ([`d619ef2`](https://github.com/johnmarktaylor91/torchlens/commit/d619ef212e43c397336a5e0479aa52a47e82569a))

Replace pip with uv pip in quality and lint workflows. uv resolves and installs packages 10-20x
  faster. Also adds explicit torch CPU index-url to dep-audit job (was missing, pulling full CUDA
  bundle).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.1 (2026-03-04)

### Bug Fixes

- **core**: Resolve 52 bugs across 9 waves of correctness, safety, and polish fixes
  ([`d4ca75d`](https://github.com/johnmarktaylor91/torchlens/commit/d4ca75d13defe15f558799be05cb4dcd9d217a45))

Wave 1 — Critical Correctness: - Rewrite safe_copy() to eliminate numpy round-trip, fix bfloat16
  overflow, handle sparse/meta/quantized dtypes (#103, #137, #128, #139) - Guard print_override
  numpy conversion (#140), meta/sparse tensor memory (#24) - Fix validation crash on None
  parent_values (#150), silent exception pass (#151) - Add buffer duplicate guard in fast path
  (#116) - Fix postprocess_fast shared reference and unguarded parent_layers (#8, #152) - Add empty
  graph guard (#153), zombie label cleanup (#75) - Handle defaultdict in _safe_copy_arg (#127)

Wave 2 — Exception Safety: - Make module_forward_decorator exception-safe (#122) - Guard
  _handle_module_entry for untracked tensors (#117) - Clean up output tensors on fast-pass exception
  (#110) - Wrap activation_postfunc in pause_logging (#96) - Guard validation with
  save_function_args=False (#131) - Fix CUDA perturbation device consistency (#36)

Wave 3 — Fast-Pass State Reset: - Reset timing and stale lookup caches in save_new_activations (#87,
  #97, #98) - Pass raw label (not final) to gradient hook in fast path (#86) - Raise descriptive
  error for unexpected missing parents (#111)

Wave 4 — Argument Handling: - Deep-copy nested tuples in creation_args (#44) - Slice before cloning
  in display helper (#73) - Use logged shape in display (#45)

Wave 5 — Interface Polish: - Key layer_num_passes by no-pass label (#53) - Use rsplit for
  colon-split safety (#54) - Add slice indexing support (#78) - Guard to_pandas before pass finished
  (#124) - List candidates for ambiguous substring (#125) - Add ModuleLog string indexing (#120) -
  Support int/short-name in ParamAccessor.__contains__ (#84)

Wave 6 — Decoration/Model Prep: - Only add mapper entries if setattr succeeded (#31) - Add hasattr
  guard for identity asymmetry (#39) - Guard buffer setattr (#40) - Store dunder argnames stripped
  (#82) - Use inspect.Parameter.kind for varargs (#123)

Wave 7 — Visualization: - Fix vis_nesting_depth=0 crash (#94) - Use rsplit for colon-split (#104) -
  Fix collapsed node variable (#100) - Guard None tensor_shape (#118) - Guard LayerLog nesting depth
  (#138) - Use exact equality instead of substring match (#129)

Wave 8 — Control Flow/Loop Detection: - Remove dead sibling_layers iteration (#2) - Prevent shared
  neighbor double-add (#148)

Wave 9 — Cleanup + GC: - Lazy import IPython (#72) - Filter callables from _trim_and_reorder (#134)
  - Complete cleanup() with internal containers (GC-5, GC-12) - Use weakref in backward hook closure
  (GC-8) - Clear ModulePassLog forward_args/kwargs after build (GC-11)

585 tests passing across 10 test files, 47 new regression tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Resolve final remaining bugs #23, #83, #95, #99
  ([`6302662`](https://github.com/johnmarktaylor91/torchlens/commit/630266251861ca505fcd0f33b33d3aa3b21f7c66))

- #95: fix mixed LayerLog/LayerPassLog format in vis module containment check - #83:
  LayerLog.parent_layer_arg_locs returns strings (not sets) for consistency - #99: warn on tensor
  shape mismatch in fast-path source tensor logging - #23: add case-insensitive exact match and
  substring lookup for layers/modules - #85: confirmed not-a-bug (any special arg correctly explains
  output invariance) - #39, #41, #42: confirmed already fixed or correct as-is - Add 6 regression
  tests for the above fixes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Resolve remaining low-risk bugs #28, #107, #108, #147
  ([`d8616a0`](https://github.com/johnmarktaylor91/torchlens/commit/d8616a0b1dfa133cd5aeb6bd92af6cdfb2bc566a))

- #147: descriptive ValueError in log_source_tensor_fast on graph change - #108: document fast-path
  module log preservation (no rebuild needed) - #107: handle both tuple and string formats in module
  hierarchy info - #28: remove torch.Tensor from dead type-check list in introspection - Add 4
  regression tests for the above fixes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **reorganize**: Redistribute bugfix tests into domain-specific test files
  ([`e9c64cc`](https://github.com/johnmarktaylor91/torchlens/commit/e9c64cc58a6bbd8f5d1b9942d234af3e0eed94bd))

Move all 57 regression tests from test_bugfixes.py into their natural homes: - test_internals.py:
  safe_copy, tensor utils, exception safety, cleanup/GC - test_validation.py: validation
  correctness, posthoc perturb check - test_save_new_activations.py: fast-path state reset, graph
  consistency - test_layer_log.py: interface polish, case-insensitive lookup, arg locs -
  test_module_log.py: string indexing, tuple/string normalization - test_param_log.py: ParamAccessor
  contains, param ref cleanup - test_output_aesthetics.py: visualization smoke tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.0 (2026-03-04)

### Bug Fixes

- **validation**: Resolve 8 test failures and Bug #79 in metadata invariants
  ([`14b2e5b`](https://github.com/johnmarktaylor91/torchlens/commit/14b2e5bbb8ac96d0dc45d8f8108db948dbb20097))

Three root-cause fixes for failures exposed by the full test suite:

1. Module hierarchy aliasing (finalization.py): shared nn.Module instances (same Conv2d registered
  as both self.proj1 and self.conv_list[0]) had metadata lookup fail because _prepare_model_once
  overwrites tl_module_address to the last-visited alias while _capture_module_metadata stores under
  the first-visited alias. Built alias→meta map and address_children prefix rewriting so all aliases
  resolve correctly.

2. Loop detection Bug #79 (loop_detection.py): _merge_iso_groups_to_layers only compared pairs
  within iso-groups, missing Rule 1 (param sharing → same layer) across iso-groups. Rewrote with
  union-find + path compression and added cross-iso-group param barcode merge pass. Also unify
  operation_equivalence_type for merged nodes whose module path suffixes differ.

3. Adjacency invariant (invariants.py): removed incorrect operation-level neighbor check —
  subgraph-level adjacency is verified during BFS in loop_detection.py, not reconstructable
  post-hoc. Non-param multi-pass groups can be the only multi-pass group (param-free loops).
  Upgraded param sharing violation back from warning to error now that Bug #79 is fixed.

Also: boolean flag fixes in control_flow.py and graph_traversal.py for buffer/output layers;
  @pytest.mark.slow on vit and beit tests; profiling test streamlined.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **ci**: Add mypy type checking and pip-audit dependency auditing
  ([`2a0e0e7`](https://github.com/johnmarktaylor91/torchlens/commit/2a0e0e725f3551db15221d2bea326018423df880))

Add quality.yml workflow with two jobs: - mypy (advisory, continue-on-error) — 199 existing errors
  to fix later - pip-audit (blocking) — fails on known CVEs in dependencies

Mypy configured leniently in pyproject.toml; both tools added to dev deps.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Add pip caching to quality workflow
  ([`d7573b1`](https://github.com/johnmarktaylor91/torchlens/commit/d7573b18d57bc3e9d20c4c524875ffdb07e4de1f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **coverage**: Add pytest-cov with HTML and text report generation
  ([`c1ac2c9`](https://github.com/johnmarktaylor91/torchlens/commit/c1ac2c900d334e0ba8e0cff2e6f248e506d15686))

Configure branch coverage for torchlens/ source. HTML report writes to
  tests/test_outputs/coverage_html/, text summary to coverage_report.txt. Reports auto-generate when
  running pytest --cov via a sessionfinish hook.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **dev**: Add UI sandbox notebook and nbstripout for clean diffs
  ([`5247361`](https://github.com/johnmarktaylor91/torchlens/commit/524736119646202be37bb8fb30b18d584d0a5a26))

Add tests/ui_sandbox.ipynb — interactive workbench for tinkering with torchlens during development:
  logging, accessors, reprs, visualization, validation. Set up nbstripout via .gitattributes to
  auto-strip cell outputs on commit.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **profiling**: Expand profiling report to 12 models across architecture families
  ([`60e983a`](https://github.com/johnmarktaylor91/torchlens/commit/60e983a4d47c05141bbe6f47929708018b3f5727))

Added 9 models to the profiling test for illustrative coverage: - SimpleBranching (diverging/merging
  flow) - ResidualBlock (conv-BN-relu skip connection) - MultiheadAttention (nn.MultiheadAttention)
  - LSTM (recurrent + classifier) - ResNet18, MobileNetV2, EfficientNet_B0 (real CNNs) - Swin_T
  (shifted-window vision transformer, 671 layers) - VGG16 (deep sequential CNN, 138M params)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **modules**: Add shared module alias lookup and explicit alias fields
  ([`c2fb14e`](https://github.com/johnmarktaylor91/torchlens/commit/c2fb14ef15fd27bdd352b2e27b99c4866c5706bd))

Shared nn.Module instances (same object registered under multiple addresses) can now be looked up by
  any alias:

ml.modules["shared_conv"] # primary address ml.modules["alias_list.0"] # alias — returns same
  ModuleLog

New fields: - ModuleLog.is_shared: bool, True when len(all_addresses) > 1 -
  ModulePassLog.all_module_addresses: list of all addresses for the module -
  ModulePassLog.is_shared_module: bool, same as ModuleLog.is_shared - ModuleAccessor builds
  alias→ModuleLog map for __getitem__/__contains__ - ModuleLog.__repr__ shows aliases when
  is_shared=True

Invariant checks (module_hierarchy) now account for shared modules where address_parent refers to
  the primary alias's parent path.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Add complex semantic invariants M-R to check_metadata_invariants
  ([`b745db8`](https://github.com/johnmarktaylor91/torchlens/commit/b745db826135e0470df75d70eaa6dcce48feae42))

Add 6 new invariant categories verifying loop detection, graph ordering, distance/reachability,
  connectivity, module containment, and lookup key consistency. Fix existing checks H and L for
  recurrent model compatibility. 17 new corruption tests (50 total in test_validation.py).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Add validate_forward_pass with metadata invariant checks
  ([`8a25ede`](https://github.com/johnmarktaylor91/torchlens/commit/8a25ede7f90b4e924f06e55ad15bb46d148f7384))

Rename validate_saved_activations → validate_forward_pass (old name kept as deprecated alias). Add
  comprehensive metadata invariant checker (check_metadata_invariants) covering 12 categories:
  ModelLog self-consistency, special layer lists, graph topology, LayerPassLog fields, recurrence,
  branching, LayerLog cross-refs, module-layer containment, module hierarchy, param cross-refs,
  buffer cross-refs, and equivalence symmetry.

validate_metadata=True by default, so every existing validation call now exercises the full
  invariant suite automatically.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **profiling**: Add automated profiling report for overhead measurement
  ([`a483011`](https://github.com/johnmarktaylor91/torchlens/commit/a483011785da9ee4f5dd5957c083d9e67f9f7e82))

Profiles raw forward pass, log_forward_pass, save_new_activations, and validate_saved_activations
  across toy and real-world models. Generates tests/test_outputs/profiling_report.txt with absolute
  times and overhead ratios.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.14.0 (2026-03-04)

### Features

- **core**: Add buffer name/module_address properties to LayerLog
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`e3a1ce2`](https://github.com/johnmarktaylor91/torchlens/commit/e3a1ce21927af0459f1388ed559b0a28a7646bb6))

Ensures buffer_address, name, and module_address are accessible on both buffer LayerPassLogs (via
  BufferLog subclass) and buffer LayerLogs (via computed properties derived from buffer_address).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Add LayerAccessor and log.layers property
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`b639e50`](https://github.com/johnmarktaylor91/torchlens/commit/b639e50d793d319d83fc51ecd86c3ac367ad3161))

Adds LayerAccessor (dict-like accessor for LayerLog objects) with support for
  label/index/pass-notation lookup and to_pandas(). Accessible via log.layers, mirroring log.modules
  for modules.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Add LayerLog aggregate class with per-layer metadata
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`9c6472d`](https://github.com/johnmarktaylor91/torchlens/commit/9c6472db6c71c10dbed02428844972f23ebb96b3))

Introduces LayerLog as the aggregate per-layer object grouping one or more LayerPassLog entries (one
  per invocation). For non-recurrent models every LayerLog has exactly one pass; for recurrent
  models, multiple passes are grouped under a single LayerLog with union-based aggregate graph
  properties.

Key changes: - New LayerLog class with ~52 aggregate fields, single-pass delegation via @property
  and __getattr__, aggregate graph properties (child_layers, parent_layers unions) -
  _build_layer_logs() postprocessing step wired into both exhaustive and fast pipelines -
  ModelLog.layer_logs OrderedDict populated after postprocessing - __getitem__ routing: log["label"]
  returns LayerLog for multi-pass layers - parent_layer_log back-reference on LayerPassLog - 27 new
  tests covering construction, delegation, multi-pass behavior, display, and integration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **core**: Modulelog.all_layers now stores unique no-pass labels
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`214e161`](https://github.com/johnmarktaylor91/torchlens/commit/214e1616e87ab89a9f7e6a44de8d7e6453bdf430))

Establishes aggregate↔aggregate symmetry: - ModuleLog.all_layers → no-pass labels → resolve to
  LayerLog - ModulePassLog.layers → pass-qualified labels → resolve to LayerPassLog

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Remove buffer name/module_address from LayerLog
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`f896dd4`](https://github.com/johnmarktaylor91/torchlens/commit/f896dd41c24f4174b70395db0fb174c8b70f1e44))

These properties are too generic on LayerLog (name returns "" for non-buffers). Keep them only on
  BufferLog(LayerPassLog); single-pass buffer LayerLogs still access them via __getattr__
  delegation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Remove RolledTensorLog, use LayerLog for rolled visualization
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`118b3ec`](https://github.com/johnmarktaylor91/torchlens/commit/118b3ec67be88274d80634a79f38e87f1c70e27e))

RolledTensorLog was a visualization-only aggregate that duplicated ~30 fields from LayerPassLog with
  its own merge logic. Now that LayerLog provides a proper aggregate abstraction, RolledTensorLog is
  unnecessary.

Key changes: - Delete RolledTensorLog class (~175 lines) and _roll_graph() function - Remove
  layer_list_rolled and layer_dict_rolled from ModelLog - Visualization "rolled" mode now uses
  self.layer_logs (LayerLog objects) - Add vis-specific computed properties to LayerLog:
  child_layers_per_pass, parent_layers_per_pass, child_passes_per_layer, parent_passes_per_layer,
  edges_vary_across_passes, bottom_level_submodule_passes_exited, parent_layer_arg_locs -
  LayerLog.child_layers/parent_layers always return no-pass labels - Merge multi-pass aggregate
  fields (has_input_ancestor, input_output_address, is_bottom_level_submodule_output) in
  _build_layer_logs - Update tests and aesthetic report to use LayerLog

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Rename TensorLog → LayerPassLog + nomenclature sweep
  ([`d328476`](https://github.com/johnmarktaylor91/torchlens/commit/d32847669bc0d2410d6f1070865f53c81de84b98))

Phase 1 of LayerLog hierarchy redesign. Mechanical rename of the class, file (tensor_log.py →
  layer_pass_log.py), constant (TENSOR_LOG_FIELD_ORDER → LAYER_PASS_LOG_FIELD_ORDER), and all
  imports. Also sweeps internal nomenclature: field names (_raw_tensor_dict → _raw_layer_dict,
  etc.), function names (_make_tensor_log_entry → _make_layer_log_entry, etc.), and local variables
  (tensor_entry → layer_entry, etc.) that referred to log entries rather than actual tensors.
  Backward-compat aliases preserved.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Replace 17 flat module_* dicts on ModelLog with transient _module_build_data
  ([`ef3f20e`](https://github.com/johnmarktaylor91/torchlens/commit/ef3f20ef16422a4c997a2727cf9c9df941e56030))

These intermediate fields (module_addresses, module_types, module_passes, etc.) were only used
  during postprocessing to build structured ModuleLog objects, but persisted on the public API. Now
  consolidated into a single _module_build_data dict that is populated during logging, consumed by
  _build_module_logs (step 17), and then re-initialized. Reduces ModelLog from ~143 to ~127 public
  fields.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.13.1 (2026-03-03)

### Bug Fixes

- **core**: Handle DataParallel, device mismatch, and embedding perturbation
  ([#91](https://github.com/johnmarktaylor91/torchlens/pull/91),
  [`1a0f8b8`](https://github.com/johnmarktaylor91/torchlens/commit/1a0f8b802f738a399d74876f39d121336a1fe855))

- Unwrap nn.DataParallel in log_forward_pass, show_model_graph, validate_saved_activations -
  Auto-move inputs to model device (supports dict, UserDict, BatchEncoding) - Exempt embedding index
  arg from perturbation (prevents CUDA OOB)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **batch1**: Aesthetic cleanup — names, types, docstrings for small files
  ([`52efdc1`](https://github.com/johnmarktaylor91/torchlens/commit/52efdc1e8ec55e1b5047589903f3a601523dfe14))

Files: _state.py, buffer_log.py, cleanup.py, postprocess/__init__.py, param_log.py,
  func_call_location.py

- Add docstrings to dunder methods across BufferAccessor, ParamAccessor, ParamLog, FuncCallLocation
  - Rename _UNSET -> _SENTINEL in func_call_location.py - Fix single-letter vars in cleanup.py
  (x->label, tup->edge, k/v->descriptive) - Add return type hints (-> None) to all cleanup functions
  - Add docstring and return type to postprocess_fast() - Add property docstrings and setter type
  hints to ParamLog

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **batch2**: Aesthetic cleanup — names, types, docstrings for medium files
  ([`8610c46`](https://github.com/johnmarktaylor91/torchlens/commit/8610c464a2106d8037198852fb7fa7acba6ed808))

Files: trace_model.py, control_flow.py, model_log.py, module_log.py, graph_traversal.py,
  finalization.py, interface.py, user_funcs.py

- Add docstrings to _get_input_arg_names, _find_output_ancestors, _update_node_distance_vals,
  _log_internally_terminated_tensor, _log_time_elapsed, _single_pass_or_error - Fix truncated
  docstring on _flood_graph_from_input_or_output_nodes - Add return type hints to all private
  functions - Rename loop vars: a->arg_idx, t->tensor, pn->pass_num, mpl->module_pass_log,
  cc->child_pass_label, sp->child_pass_label - Rename pretty_print_list_w_line_breaks ->
  _format_list_with_line_breaks - Rename _module_hierarchy_str_helper ->
  _module_hierarchy_str_recursive - Mark run_model_and_save_specified_activations as private (_) -
  Rename x->input_data in validate_batch_of_models_and_inputs - Add docstrings to all dunder methods
  on ModuleLog, ModulePassLog, ModuleAccessor

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **batch3**: Aesthetic cleanup — names, types, docstrings for medium-large files
  ([`74c0850`](https://github.com/johnmarktaylor91/torchlens/commit/74c085095276e0361607d6d3226df85ffa8e8187))

Files: decorate_torch.py, labeling.py, constants.py, validation/core.py, loop_detection.py

- Rename open_ind/close_ind -> paren_start/paren_end in decorate_torch.py - Rename
  namespace_name_notorch -> namespace_key - Rename my_get_overridable_functions ->
  _get_torch_overridable_functions - Expand shadow set abbrevs: _ml_seen -> _module_labels_seen,
  etc. - Rename p_label -> perturbed_label in validation/core.py - Parameterize bare Dict type hints
  in loop_detection.py - Expand single-letter vars: m->module_index, n->pass_index, s->subgraph_key
  - Add docstrings to _merge_iso_groups_to_layers, _validate_layer_against_arg,
  _copy_validation_args, _get_torch_overridable_functions - Add return type hints to all private
  functions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **batch4**: Aesthetic cleanup — names, types, docstrings for large files
  ([`ce19825`](https://github.com/johnmarktaylor91/torchlens/commit/ce19825710472895b0b644881d6db3bacea50e84))

tensor_log.py: add docstrings to _str_during_pass, _str_after_pass, _tensor_family_str_helper
  model_funcs.py: rename log_whether_exited_submodule_is_bottom_level →
  _is_bottom_level_submodule_exit, fwd → forward_func, mid → module_id helper_funcs.py: rename
  make_var_iterable → ensure_iterable, tuple_tolerant_assign → assign_to_sequence_or_dict,
  remove_attributes_starting_with_str → remove_attributes_with_prefix flops.py: add type hints to
  all 25 private functions, add Args sections vis.py: rename _check_whether_to_mark_* →
  _should_mark_*, _construct_* → _build_*, fix _setup_subgraphs consistency logging_funcs.py: rename
  _log_info_specific_to_single_function_output_tensor → _log_output_tensor_info,
  _get_parent_tensor_function_call_location → _locate_parent_tensors_in_args

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **design**: Algorithmic review pass — efficiency, correctness, robustness
  ([`78924f8`](https://github.com/johnmarktaylor91/torchlens/commit/78924f825f2b604fe7dcd1df234ce76d23c51ad6))

Efficiency (21 fixes): - Hot path: eliminate all_args allocation via kwargs short-circuit - Barcode
  gen: secrets.choice → random.choices with pre-computed alphabet - Type scanning: any([...]) →
  any(...) generator, found_ids list → set - Model prep: recursive get_all_submodules →
  list(model.modules()) - Source/output tensors: eliminate double get_tensor_memory_amount() calls -
  Output tensors: copy only mutables instead of full 131-key dict copy - Gradient hooks: O(N²)
  per-hook rebuild → _saved_gradients_set O(1) dedup - Cleanup: dir() → list(__dict__), skip O(n²)
  reference removal in full teardown - Labeling: dir(self) → self.__dict__, inds_to_remove list →
  set - Finalization: pre-computed reverse mappings, _roll_graph idempotency guard - Graph
  traversal/validation: min([a,b]) → min(a,b), list.pop(0) → deque.popleft() - Visualization:
  any([...]) → any(...), min([...]) → min(...) - TensorLog: frozenset for FIELD_ORDER,
  view-then-clone-slice for display - Display: np.round → round(), removed numpy dependency

Correctness (4 FLOPs fixes): - SDPA: new _sdpa_flops handler (Q@K^T + softmax + attn@V) -
  addbmm/baddbmm: _matmul_flops → _addmm_flops (correct shape extraction) - MHA: return None to
  avoid double-counting with sub-op FLOPs - scatter/index ops: moved from ZERO_FLOPS to
  ELEMENTWISE_FLOPS

Robustness (4 fixes): - Buffer dedup: for...else pattern fixes incorrect inner-loop append -
  Parameter fingerprinting: reorder isinstance checks (Parameter before Tensor) - Duplicate
  siblings: added not-in guards before sibling append - _roll_graph: early-return guard if already
  populated

Test coverage: - New tests/test_internals.py: 13 tests for FIELD_ORDER sync and constants

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **design**: Mid-level design pass — dataclasses, decomposition, parameter bundles
  ([`0849edb`](https://github.com/johnmarktaylor91/torchlens/commit/0849edb605bb63a469831dd1959c1bd8236ce964))

Replace implicit parameter clusters with explicit dataclasses (FuncExecutionContext,
  VisualizationOverrides, IsomorphicExpansionState, ModuleParamInfo), decompose 100+ line functions
  into coordinator + helpers, and extract shared patterns (model traversal visitor, buffer tagging).

No public API changes. All 471 non-slow tests pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **qa**: Address naive reader findings — docstrings, comments, variable names
  ([`22724af`](https://github.com/johnmarktaylor91/torchlens/commit/22724afb26e6511365eee5c6ce6bdf97888eeb12))

Fix 14 issues flagged by whole-codebase naive reader review: - make_random_barcode: fix misleading
  "integer hash" docstring - _getitem_after_pass: replace TODO-like docstring with proper docs -
  _is_bottom_level_submodule_exit: define "bottom-level" in docstring -
  search_stack_for_vars_of_type: rename tensor-specific accumulators to generic names -
  validate_parents_of_saved_layer: fill in empty param docs, note mutation - _append_arg_hash:
  explain operation_equivalence_type concept - nested_assign: add docstring and type hints -
  _capture_module_metadata: fix misleading wording - extend_search_stack_from_item: document missing
  address/address_full params - safe_copy: add bfloat16 round-trip comment - _roll_graph: define
  "rolled" in docstring - vis.py: add docstrings to _get_node_bg_color, _make_node_label,
  _get_max_nesting_depth - _give_user_feedback_about_lookup_key: document mode parameter -
  _get_torch_overridable_functions: comment the ignore flag - RolledTensorLog.update_data: replace
  vacuous docstring

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **structure**: Reorganize into focused subpackages
  ([`2cfa90a`](https://github.com/johnmarktaylor91/torchlens/commit/2cfa90ac15e8e325c9e417202b0d884450c43915))

Split grab-bag root modules into focused subpackages without changing any public API, class names,
  or core algorithmic logic.

- Split helper_funcs.py into 7 utils/ modules (rng, tensor_utils, arg_handling, introspection,
  collections, hashing, display) - Moved cleanup.py + interface.py into data_classes/ - Created
  decoration/ from decorate_torch.py + model_funcs.py - Split logging_funcs.py into capture/
  (source_tensors, output_tensors, tensor_tracking); moved trace_model.py + flops.py into capture/ -
  Created visualization/ from vis.py - Replaced hardcoded _TORCHLENS_SUFFIXES with directory-based
  stack filtering - Added module-level docstrings to all 27 files that lacked them - Decomposed 3
  oversized functions into named helpers - 9 root files deleted, 22 new files created across
  subpackages

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.13.0 (2026-03-03)

### Bug Fixes

- **logging**: Prevent label-steal on no-op same-object returns
  ([`c8ec165`](https://github.com/johnmarktaylor91/torchlens/commit/c8ec1653919660d7b93e26046d3fe3706e40bc42))

Functions like `to(same_dtype)` and `contiguous()` on already-contiguous tensors return the same
  Python object without modifying it. TorchLens was treating these as in-place ops and propagating
  the new label back to the original tensor, breaking module thread cleanup in T5 and other
  encoder-decoder models with pre-norm residual patterns.

Fix 1 (decorate_torch.py): Split `was_inplace` into `same_object_returned` (always safe_copy) vs
  true in-place (func ends with `_` or starts with `__i`) — only true in-place ops propagate the
  label back.

Fix 2 (model_funcs.py): Module post-hook thread cleanup now uses labels captured at pre-hook time
  rather than reading from live tensors that may have been overwritten by true in-place ops
  mid-forward.

Also removes the cycle-protection band-aid in vis.py (visited sets in _get_max_nesting_depth and
  _set_up_subgraphs) since the root cause of module_pass_children cycles is now fixed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **multi**: Resolve 16 test failures across suite
  ([`6fcf72f`](https://github.com/johnmarktaylor91/torchlens/commit/6fcf72f6677ac30cd0f63bfc62250b2c133f028e))

- Add soxr to test deps (transformers 5.2 unconditionally imports it) - Fix test_no_grad_block and
  test_model_calling_model shape mismatch (use inline torch.rand(2, 32) instead of small_input
  fixture) - Fix _get_input_arg_names for *args signatures (generate synthetic names) - Add empty
  tensor perturbation exemption in validation core - Add topk/sort integer indices posthoc exemption
  - Add type_as structural arg position exemption - Fix _append_arg_hash infinite recursion (tensor
  formatting triggers wrapped methods; use isinstance check + shape-only representation) - Fix
  TorchFunctionMode/DeviceContext bypass in decorated wrappers (Python wrappers skip C-level
  dispatch, so torch.device('meta') context wasn't injecting device kwargs into factory functions,
  causing corrupt buffers during transformers from_pretrained) - Rewrite test_autocast_mid_forward
  to skip validation (autocast context not captured during logging)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Exempt setitem when perturbing all-zeros/all-ones destination
  ([`a3a24ad`](https://github.com/johnmarktaylor91/torchlens/commit/a3a24ad75ba36ffcd1a87994d2c0abc5dce97102))

BART creates position embeddings by new_zeros() then __setitem__ to fill. Perturbing the all-zeros
  destination has no effect since setitem overwrites it. Add Case 3 to _check_setitem_exempt: when
  the perturbed tensor is the destination (args[0]) and it's a special value (all-zeros/all-ones),
  exempt.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Add cycle protection to _get_max_nesting_depth and subgraph setup
  ([`8779af9`](https://github.com/johnmarktaylor91/torchlens/commit/8779af9bc95743f55232660432dcf6bcb0055842))

call_children can contain cycles in models like T5 where residual connections cross module
  boundaries (e.g. layer_norm -> next block). _get_max_nesting_depth looped infinitely, causing
  >600s timeout.

Also: - Add visited set to _set_up_subgraphs while-loop for same reason - Add min with multiple args
  to posthoc exemption (same as max) - Remove @pytest.mark.slow from test_t5_small (now 15s, was
  infinite)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **validation**: Capture and restore autocast context during logging and replay
  ([`19bbcb6`](https://github.com/johnmarktaylor91/torchlens/commit/19bbcb665c55f9e022263b51f673a781d5b80410))

Autocast state (enabled/dtype per device) is now captured alongside RNG state when each function
  executes during logging. During validation replay, the saved autocast context is restored so
  operations run at the same precision as the original forward pass.

- Add log_current_autocast_state() and AutocastRestore context manager - Thread func_autocast_state
  through exhaustive and fast logging paths - Wrap validation replay in AutocastRestore - Restore
  validate_saved_activations in test_autocast_mid_forward

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **validation**: Restructure into subpackage with data-driven exemption registries
  ([`ff85efd`](https://github.com/johnmarktaylor91/torchlens/commit/ff85efd282df188e5202d35944c0782ed1e16c3f))

Replace the monolithic validation.py (814 lines) with a clean torchlens/validation/ subpackage. The
  141-line elif chain and three separate exemption mechanisms are now consolidated into four
  declarative registries in exemptions.py (SKIP_VALIDATION_ENTIRELY, SKIP_PERTURBATION_ENTIRELY,
  STRUCTURAL_ARG_POSITIONS, CUSTOM_EXEMPTION_CHECKS), with posthoc checks kept as
  belt-and-suspenders.

Secondary fixes: - mean==0/mean==1 heuristic replaced with exact torch.all() checks - Bare except
  Exception now logs error details when verbose=True - _copy_validation_args uses recursive
  _deep_clone_tensors helper - While-loop perturbation bounded by MAX_PERTURB_ATTEMPTS=100

Adds 23 new tests covering imports, registry consistency, perturbation unit tests, deep clone
  helpers, and integration tests through specific exemption paths. All 399 existing tests pass with
  zero regressions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **suite**: Exhaustive test suite expansion (~559 tests)
  ([`b489120`](https://github.com/johnmarktaylor91/torchlens/commit/b489120b91af4e41496cd206ce9ad20ab0a37092))

Add ~76 new tests covering major gaps in architecture and edge-case coverage: - 64 new toy model
  tests (attention, transformers, containers, conditionals, normalization variants, residual/param
  sharing, in-place/type ops, scalar/ broadcasting, exemption stress tests, architecture patterns,
  adversarial edge cases) - 12 new real-world model tests (DistilBERT, ELECTRA, MobileViT, MoE, T5,
  BART, RoBERTa, BLIP, Whisper, ViT-MAE, Conformer, SentenceTransformer)

Merge test_real_world_models_slow.py into test_real_world_models.py organized by
  architecture/modality, using @pytest.mark.slow for heavy tests. Register the slow marker in
  pyproject.toml.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.12.0 (2026-03-02)

### Features

- **data_classes**: Add BufferLog subclass and BufferAccessor for first-class buffer support
  ([`5a9b4c6`](https://github.com/johnmarktaylor91/torchlens/commit/5a9b4c6bc7079176eaa4b259144b7316ef13699c))

Buffers now get their own dedicated class (BufferLog) that subclasses TensorLog, giving them focused
  repr, convenience properties (name, module_address), and ergonomic access via BufferAccessor on
  both ModelLog and ModuleLog.

- BufferLog(TensorLog) subclass with clean __repr__ and computed properties - BufferAccessor with
  indexing by address, short name, or ordinal position - Scoped mh.modules["addr"].buffers for
  per-module buffer access - Fix vis.py type(node) == TensorLog checks to use isinstance (supports
  subclasses) - Fix TensorLog.copy() to preserve subclass type via type(self)(fields_dict) - Fix
  cleanup.py property-before-callable check order to prevent AttributeError - Add buffer repr
  sections to aesthetic test reports (text + PDF)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.2 (2026-03-02)

### Performance Improvements

- **postprocess**: Optimize 6 remaining hot-path bottlenecks in log_forward_pass
  ([`ed62350`](https://github.com/johnmarktaylor91/torchlens/commit/ed623508d8f8353634066cc4702d39bf8f0ca981))

Cache dir() per type, replace sorted deque with heapq in loop detection, use __dict__ + empty-field
  skip in label renaming, two-phase stack capture, batch orphan removal, and shadow sets for module
  hierarchy membership checks. 8-33% wall time reduction across models.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.1 (2026-03-02)

### Performance Improvements

- **logging**: Optimize 5 hot-path bottlenecks in log_forward_pass
  ([`d4a5d3a`](https://github.com/johnmarktaylor91/torchlens/commit/d4a5d3a57e52a22c6b312e8b77f3ec6a6d896efc))

- Cache torch.cuda.is_available() (called per-op, ~2ms each) - Replace tensor CPU copy + numpy +
  getsizeof with nelement * element_size - Hoist warnings.catch_warnings() out of per-attribute loop
  - Cache class-level module metadata (inspect.getsourcelines, signature) - Replace
  inspect.stack()/getframeinfo() with sys._getframe() chain; defer source context + signature
  loading in FuncCallLocation to first property access via linecache

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **decoration**: Comprehensive tests for permanent decoration architecture
  ([`f6305c8`](https://github.com/johnmarktaylor91/torchlens/commit/f6305c8c0f30337bdb2e5f6f265b0d253b7bb05e))

61 tests across 14 test classes covering: - Toggle state (on/off/exception/KeyboardInterrupt
  recovery) - Passthrough behavior when logging disabled - Detached import patching (module-level,
  class attrs, lists, dicts, late imports) - Permanent model preparation (WeakSet caching, session
  cleanup) - pause_logging context manager (nesting, exception safety) - Wrapper transparency
  (functools.wraps, __name__, __doc__) - torch.identity installation and graph presence - JIT
  builtin table registration and shared-original wrapper reuse - Decoration consistency
  (bidirectional mapper, idempotency) - In-place ops, property descriptors, edge cases - Signal
  safety (SIGALRM during forward) - Session isolation (no cross-session leakage)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.0 (2026-03-02)

### Features

- **core**: Permanent toggle-gated decoration of torch functions
  ([`cbeaa83`](https://github.com/johnmarktaylor91/torchlens/commit/cbeaa83aa08098ea721e319a2f1af9a559279892))

Replace the per-call decorate/undecorate cycle (~2000 torch functions) with one-time permanent
  decoration at import time. A single boolean toggle (_state._logging_enabled) gates whether
  wrappers log or pass through, eliminating repeated overhead and fixing detached import blindness
  (e.g. `from torch import cos`).

Key changes: - New _state.py: global toggle, active_logging/pause_logging context managers -
  decorate_all_once(): one-time decoration with JIT builtin table registration -
  patch_detached_references(): sys.modules crawl for detached imports - Split model prep:
  _prepare_model_once (WeakSet) + _prepare_model_session - Delete clean_* escape hatches;
  safe_copy/safe_to use pause_logging() - Fix tensor_log.py activation_postfunc exception-safety bug
  (#89) - Fix GC issues: closures no longer capture ModelLog (GC-6, GC-7)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.10.1 (2026-03-02)

### Bug Fixes

- **core**: Fix one-liners and small guard additions across codebase
  ([`3d1591b`](https://github.com/johnmarktaylor91/torchlens/commit/3d1591b003675d91191e7a0abc728ad5eb562fae))

- Remove duplicate "contiguous" from ZERO_FLOPS_OPS (flops.py) - Fix strip("*") → rstrip("*") to
  avoid stripping leading chars (tensor_log.py) - Fix return type annotation to include str return
  (trace_model.py) - Replace isinstance(attr, Callable) with callable(attr) (model_funcs.py) -
  Remove dead pass_num=1 overwrite of parameterized value (logging_funcs.py) - Fix
  parent_params=None → [] for consistency (finalization.py) - Add int key guard in
  _getitem_after_pass (interface.py) - Add max(0, ...) guard for empty model module count
  (interface.py) - Add empty list guard in _get_lookup_help_str (interface.py) - Add explicit
  KeyError raise for failed lookups (interface.py) - Remove dead unreachable module address check
  (interface.py) - Add str() cast for integer layer keys (trace_model.py) - Add list() copy to
  prevent getfullargspec mutation (trace_model.py) - Reset has_saved_gradients and unlogged_layers
  in save_new_activations (logging_funcs.py)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **misc**: Fix decoration, param detection, vis forwarding, and guards
  ([`4269c4d`](https://github.com/johnmarktaylor91/torchlens/commit/4269c4d35beeabf54468fbc99b083290e830ff98))

- Remove dead double-decoration guard in decorate_torch.py - Add hasattr guard for torch.identity
  installation - Add None guard for code_context in FuncCallLocation.__getitem__ - Fix is_quantized
  to check actual qint dtypes, not all non-float - Forward show_buffer_layers to rolled edge check
  in vis.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **safety**: Add try/finally cleanup and exception state resets
  ([`c9bdb7f`](https://github.com/johnmarktaylor91/torchlens/commit/c9bdb7fc354c0b44f3cfcf6bde17e623a3b6f6db))

- Wrap validate_saved_activations in try/finally for cleanup (user_funcs.py) - Wrap show_model_graph
  render_graph in try/finally with cleanup (user_funcs.py) - Reset _track_tensors and _pause_logging
  in exception handler (trace_model.py) - Update test to expect [] instead of None for cleared
  parent_params

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Fix isinstance checks and misleading variable name
  ([`581f286`](https://github.com/johnmarktaylor91/torchlens/commit/581f286fc7c25f50b8b82b44d1f3fcbe03b4f01c))

- Replace type(val) == torch.Tensor with isinstance() (12 occurrences) - Rename mean_output to
  output_std for accuracy

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **postprocess**: Split monolithic postprocess.py into thematic package
  ([`cd14d27`](https://github.com/johnmarktaylor91/torchlens/commit/cd14d2744ea60db3a911905c377fc6c52123a1cc))

Break 2,115-line postprocess.py into a postprocess/ package with 5 modules: - graph_traversal.py:
  Steps 1-4 (output nodes, ancestry, orphans, distances) - control_flow.py: Steps 5-7 (conditional
  branches, module fixes, buffers) - loop_detection.py: Step 8 (loop detection, isomorphic subgraph
  expansion) - labeling.py: Steps 9-12 (label mapping, final info, renaming, cleanup) -
  finalization.py: Steps 13-18 (undecoration, timing, params, modules, finish)

No behavioral changes — pure file reorganization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **aesthetic**: Add aesthetic testing infrastructure for visual inspection
  ([`73cdf01`](https://github.com/johnmarktaylor91/torchlens/commit/73cdf017c29f2385cebb721e7b08b88177807365))

Add regenerable human-inspectable outputs in tests/test_outputs/: - Comprehensive text report
  (aesthetic_report.txt) covering all user-facing reprs, accessors, DataFrames, error messages, and
  field dumps for every major data structure (ModelLog, TensorLog, RolledTensorLog, ModuleLog,
  ModulePassLog, ParamLog, ModuleAccessor, ParamAccessor) - 28 visualization PDFs exercising nesting
  depth, rolled/unrolled views, buffer visibility, graph direction, frozen params, and loop
  detection - 6 new aesthetic test models (AestheticDeepNested, AestheticSharedModule,
  AestheticBufferBranch, AestheticKitchenSink, AestheticFrozenMix) - Rename visualization_outputs/ →
  test_outputs/ for cleaner structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **aesthetic**: Add gradient visualization coverage
  ([`085dcd9`](https://github.com/johnmarktaylor91/torchlens/commit/085dcd9caf95b1ca364f83a7f97bd020f611c521))

Add gradient backward arrows (blue edges) to aesthetic testing: - New _vis_gradient helper using
  log_forward_pass(save_gradients=True) + backward() - 5 gradient vis PDFs: deep nested, frozen mix,
  kitchen sink (various configs) - Gradient section (G) in text report: TensorLog/ParamLog grad
  fields, frozen contrast - GRADIENT_VIS_GALLERY integrated into LaTeX PDF report as Section 3

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **aesthetic**: Add LaTeX PDF report with embedded visualizations
  ([`d101326`](https://github.com/johnmarktaylor91/torchlens/commit/d101326906573b95d62f5d518b13ee4be7025cb9))

Generate a comprehensive PDF report (aesthetic_report.pdf) alongside the text report, with: -
  tcolorbox-formatted sections for each model's outputs - Full field dumps for all data structures -
  All 28 visualization PDFs embedded inline with captions - Table of contents, figure numbering,
  proper typography - Skips gracefully if pdflatex not installed

Also saves the .tex source for customization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.10.0 (2026-03-02)

### Features

- **data**: Add ModuleLog, ModulePassLog, and ModuleAccessor
  ([`5b0baa8`](https://github.com/johnmarktaylor91/torchlens/commit/5b0baa84903cd7e89f4192fa94fb444eebf5ceb5))

Introduce structured per-module metadata classes following the ParamLog/ParamAccessor pattern.
  log.modules["features.3"] now returns a rich ModuleLog with class, params, layers, source info,
  hierarchy, hooks, forward signature, and nesting depth. Multi-pass modules support per-call access
  via passes dict and pass notation (e.g. "fc1:2").

- ModulePassLog: per-(module, pass) lightweight container - ModuleLog: per-module-object user-facing
  class with delegating properties for single-pass modules - ModuleAccessor: dict-like accessor with
  summary()/to_pandas() - Metadata captured in prepare_model() before cleanup strips tl_* attrs -
  Forward args/kwargs captured per pass in module_forward_decorator() - _build_module_logs() in
  postprocess Step 17 assembles everything - Old module_* dicts kept alive for vis.py backward
  compat - 44 new tests, 315 total passing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **data**: Move ModelHistory & TensorLogEntry into data_classes/
  ([`6ecc7e5`](https://github.com/johnmarktaylor91/torchlens/commit/6ecc7e5880eadffd8c13e82bd39d3aa9c5544672))

Move model_history.py and tensor_log.py into torchlens/data_classes/ alongside the existing
  FuncCallLocation and ParamLog data classes. Update all relative imports across 12 consumer files.
  Also fixes a missing-dot bug in decorate_torch.py's TYPE_CHECKING import.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **data**: Rename ModelHistory/TensorLogEntry to ModelLog/TensorLog
  ([`f36718e`](https://github.com/johnmarktaylor91/torchlens/commit/f36718e6a471838ef40af0e25e8e41357d9b9a30))

- Rename classes: ModelHistory → ModelLog, TensorLogEntry → TensorLog, RolledTensorLogEntry →
  RolledTensorLog - Rename file: data_classes/model_history.py → data_classes/model_log.py - Rename
  variables: model_history → model_log, source_model_history → source_model_log - Rename constants:
  MODEL_HISTORY_FIELD_ORDER → MODEL_LOG_FIELD_ORDER, TENSOR_LOG_ENTRY_FIELD_ORDER →
  TENSOR_LOG_FIELD_ORDER - Fold test_meta_validation.py into test_metadata.py

- **vis**: Migrate vis.py and interface.py from module_* dicts to ModuleAccessor API
  ([`8e64d43`](https://github.com/johnmarktaylor91/torchlens/commit/8e64d439d78081551c4112c0cc49b8c300609296))

Replace all direct accesses to old module_types, module_nparams, module_num_passes,
  module_pass_layers, module_pass_children, module_children, module_layers, module_num_tensors,
  module_pass_num_tensors, top_level_module_passes, and top_level_modules dicts with the new
  ModuleAccessor API (self.modules[addr]).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **validation**: Add meta-validation tests for corruption detection
  ([`11de9da`](https://github.com/johnmarktaylor91/torchlens/commit/11de9da7209b3f504dd3ed6433a0d04367c38d22))

Add 9 tests that deliberately corrupt saved activations and verify validate_saved_activations()
  catches each corruption: output/intermediate replacement, layer swap, zeroing, noise, scaling,
  wrong shape, and corrupted creation_args.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.9.0 (2026-03-01)

### Features

- **data**: Add ParamLog class for structured parameter metadata
  ([`f21f3e9`](https://github.com/johnmarktaylor91/torchlens/commit/f21f3e9c7a57facd2dfd0bcf7b5c76da66b62fe2))

Introduce ParamLog — a dedicated data class for parameter metadata — and ParamAccessor for
  convenient dict-like access by address, index, or short name. Parameters are now first-class
  objects with full metadata including trainable/frozen status, module info, linked params, gradient
  tracking, and optional optimizer tagging.

Key additions: - ParamLog class with lazy gradient detection via _param_ref - ParamAccessor on both
  ModelHistory (mh.params) and TensorLogEntry (entry.params) - Trainable/frozen tallies on MH, TLE,
  and per-module - Optional optimizer= param on log_forward_pass() - Visualization: param names with
  bracket convention, trainable/frozen color coding, gradient fills for mixed layers, caption
  breakdown - 68 new tests covering all functionality

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.8.0 (2026-03-01)

### Features

- **data**: Add FuncCallLocation class for structured call stack metadata
  ([`49ada0e`](https://github.com/johnmarktaylor91/torchlens/commit/49ada0ef1415441562c2f3dd5b71d36de8478472))

Replace unstructured List[Dict] func_call_stack with List[FuncCallLocation] providing clean repr
  with source context arrows, __getitem__/__len__ support, and optional
  func_signature/func_docstring extraction. Introduces torchlens/data_classes/ package and threads a
  configurable num_context_lines parameter through the full call chain.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.2 (2026-03-01)

### Bug Fixes

- **tests**: Reduce r2plus1d_18 input size to prevent OOM
  ([`2c73fcd`](https://github.com/johnmarktaylor91/torchlens/commit/2c73fcd7af620a29749fd55f753f1b6d3d915b7a))

The test_video_r2plus1_18 test used a (16,3,16,112,112) input (~96M elements) which consistently got
  OOM-killed. Reduced to (1,3,1,112,112) which passes reliably while still exercising the full
  model.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.1 (2026-02-28)

### Bug Fixes

- **logging**: Handle complex-dtype tensors in tensor_nanequal
  ([`b425cbe`](https://github.com/johnmarktaylor91/torchlens/commit/b425cbed76fdaf7bef6f2fc94fa6fdc02404b4c2))

torch.nan_to_num does not support complex tensors, which caused test_qml to fail when PennyLane
  quantum ops produced complex outputs. Use view_as_real/view_as_complex to handle NaN replacement
  for complex dtypes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.0 (2026-02-28)

### Bug Fixes

- **logging**: Capture/restore RNG state for two-pass stochastic models
  ([#58](https://github.com/johnmarktaylor91/torchlens/pull/58),
  [`271cef3`](https://github.com/johnmarktaylor91/torchlens/commit/271cef345d8c94e709bc5eeccc8e4d64b951c64d))

Move RNG state capture/restore before pytorch decoration to prevent internal .clone() calls from
  being intercepted by torchlens' decorated torch functions. Also speed up test_stochastic_loop by
  using a higher starting value.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Ensure output layer parents are saved with layers_to_save
  ([#46](https://github.com/johnmarktaylor91/torchlens/pull/46),
  [`b104094`](https://github.com/johnmarktaylor91/torchlens/commit/b10409445dc036ce8957e4e52e314b3b4a38dc4f))

When layers_to_save is a subset, the fast pass now automatically includes parents of output layers
  in the save list. This ensures output layer tensor_contents is populated in postprocess_fast
  (which copies from parent).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Replace copy.deepcopy with safe_copy to prevent infinite loops
  ([#18](https://github.com/johnmarktaylor91/torchlens/pull/18),
  [`e1ad9ae`](https://github.com/johnmarktaylor91/torchlens/commit/e1ad9ae08937b11ccffe83c47efc45a052397c6c))

copy.deepcopy hangs on complex tensor wrappers with circular references (e.g. ESCNN
  GeometricTensor). Replace with safe_copy_args/safe_copy_kwargs that clone tensors, recurse into
  standard containers, and leave other objects as references.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Use argspec to disambiguate tuple/list input args
  ([#43](https://github.com/johnmarktaylor91/torchlens/pull/43),
  [`a91e378`](https://github.com/johnmarktaylor91/torchlens/commit/a91e378125e1d3ee0a9483b9240c3f7730faba3a))

When a model's forward() expects a single arg that IS a tuple/list of tensors, torchlens incorrectly
  unpacked it into multiple positional args. Now uses inspect.getfullargspec to detect single-arg
  models and wraps the tuple/list as a single arg. Also handles immutable tuples in
  _fetch_label_move_input_tensors device-move logic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Functional ops at end of container modules rendered as ovals
  ([#48](https://github.com/johnmarktaylor91/torchlens/pull/48),
  [`7d1c68c`](https://github.com/johnmarktaylor91/torchlens/commit/7d1c68c502c3274bf1418bfcac2de9d81eef9d77))

_check_if_only_non_buffer_in_module was too broad — it returned True for functional ops (like
  torch.relu) at the end of container modules with child submodules, causing them to render as
  boxes. Added a leaf-module check: only apply box rendering for modules with no child submodules.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **metadata**: Track module training mode per submodule
  ([#52](https://github.com/johnmarktaylor91/torchlens/pull/52),
  [`e5197ca`](https://github.com/johnmarktaylor91/torchlens/commit/e5197ca2e48d4510df750ea420196e68f038c0b6))

Capture module.training in module_forward_decorator and store in ModelHistory.module_training_modes
  dict (keyed by module address). This lets users check whether each submodule was in train or eval
  mode during the forward pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.2 (2026-02-28)

### Bug Fixes

- **postprocess**: Correct _merge_buffer_entries call convention
  ([#47](https://github.com/johnmarktaylor91/torchlens/pull/47),
  [`07df691`](https://github.com/johnmarktaylor91/torchlens/commit/07df6911dae2d6aa22e13cf2aa2fe4793320e314))

_merge_buffer_entries is a module-level function, not a method on ModelHistory. Fixes AttributeError
  when processing recurrent models with duplicate buffer entries.

Based on gilmoright's contribution in PR #56.

Co-authored-by: gilmoright <artem.dahaka@gmail.com>


## v0.6.1 (2026-02-28)

### Bug Fixes

- **tensor_log**: Use getattr default in TensorLogEntry.copy()
  ([`538288d`](https://github.com/johnmarktaylor91/torchlens/commit/538288d6170ab95ca24e7b88ada37a7e4196d6d2))

Prevents AttributeError when copying entries that predate newly added fields (e.g., deserialized
  from an older version).

Co-Authored-By: whisperLiang <whisperLiang@users.noreply.github.com>


## v0.6.0 (2026-02-28)

### Features

- **flops**: Add per-layer FLOPs computation for forward and backward passes
  ([`31b43e6`](https://github.com/johnmarktaylor91/torchlens/commit/31b43e6659d521f0332b99d02e969b4ac19f1abd))

Compute forward and backward FLOPs at logging time for every traced operation. Uses category-based
  dispatch: zero-cost ops (view, reshape, etc.), element-wise ops with per-element cost, and
  specialty handlers for matmul, conv, normalization, pooling, reductions, and loss functions.
  Unknown ops return None rather than guessing.

- New torchlens/flops.py with compute_forward_flops / compute_backward_flops - ModelHistory gains
  total_flops_forward, total_flops_backward, total_flops properties and flops_by_type() method -
  TensorLogEntry and RolledTensorLogEntry gain flops_forward / flops_backward fields - 28 new tests
  in test_metadata.py (unit + integration) - scripts/check_flops_coverage.py dev utility for
  auditing op coverage - Move test_video_r2plus1_18 to slow test file

Based on whisperLiang's contribution in PR #53.

Co-Authored-By: whisperLiang <whisperLiang@users.noreply.github.com>


## v0.5.0 (2026-02-28)

### Bug Fixes

- **logging**: Revert children_tensor_versions to proven simpler detection
  ([`ade9c39`](https://github.com/johnmarktaylor91/torchlens/commit/ade9c39f15604459af9134ffcb770ae08238f5cf))

The refactor version applied device/postfunc transforms to the stored value in
  children_tensor_versions, but validation compares against creation_args which are always raw. This
  caused fasterrcnn validation to fail. Revert to the simpler approach that stores raw arg copies
  and was verified passing twice.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **postprocess**: Gate output node variations on has_child_tensor_variations
  ([`4106be9`](https://github.com/johnmarktaylor91/torchlens/commit/4106be93168cbff4b346a70f06417556c3444490))

Don't unconditionally store children_tensor_versions for output nodes. Gate on
  has_child_tensor_variations (set during exhaustive logging) to avoid false positives and preserve
  postfunc-applied tensor_contents.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **postprocess**: Rebuild pass assignments after loop detection and fix output handler
  ([`847b2a7`](https://github.com/johnmarktaylor91/torchlens/commit/847b2a79202cee13bf2ba231153b71068cf6311a))

Two fixes:

1. _rebuild_pass_assignments: Multiple rounds of _expand_isomorphic_subgraphs can reassign a node to
  a new group while leaving stale same_layer_operations in the old group's members. This caused
  multiple raw tensors to map to the same layer:pass label, producing validation failures (e.g.
  fasterrcnn). The cleanup step groups tensors by their authoritative layer_label_raw and rebuilds
  consistent pass numbers.

2. Output node handler: Replaced the has_child_tensor_variations gate with a direct comparison of
  actual output (with device/postfunc transforms) against tensor_contents using tensor_nanequal.
  This correctly handles in-place mutations through views (e.g. InPlaceZeroTensor) while preserving
  postfunc values for unmodified outputs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Handle bool and complex tensor perturbation properly
  ([`dfd2d7b`](https://github.com/johnmarktaylor91/torchlens/commit/dfd2d7be8dce252312235f954446e98357ccfe35))

- Generate proper complex perturbations using torch.complex() instead of casting away imaginary part
  - Fix bool tensor crash by reordering .float().abs() (bool doesn't support abs, but float
  conversion handles it) - Add ContextUnet diffusion model to example_models.py for self-contained
  stable_diffusion test - Update test_stable_diffusion to use example_models.ContextUnet

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **logging**: Generalize was_getitem_applied to has_child_tensor_variations
  ([`1177f60`](https://github.com/johnmarktaylor91/torchlens/commit/1177f607bf6406d47424c8b05755aec86d992dcb))

Replace the getitem-specific parent detection with runtime mismatch detection that catches any case
  where a parent's tensor_contents diverges from what children actually received (getitem slicing,
  view mutations through shared storage, in-place ops after logging, etc.).

Key changes: - Rename was_getitem_applied → has_child_tensor_variations - Detection now compares arg
  copies against parent tensor_contents at child-creation time, with transform-awareness (device +
  postfunc) - Output nodes now detect value changes vs parent tensor_contents - Use tensor_nanequal
  (not torch.equal) for dtype/NaN consistency - Fix fast-mode: clear stale state on re-run, prevent
  double-postfunc - Use clean_to and try/finally for _pause_logging safety - Add 6 view-mutation
  stress tests (unsqueeze, reshape, transpose, multiple, chained, false-positive control)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **loops**: Cherry-pick approved changes from feat/loop-detection-hardening
  ([`33e0f22`](https://github.com/johnmarktaylor91/torchlens/commit/33e0f223d12aae806799f3853ed6e221b1232274))

- Rename 10 loop detection functions to clearer names (e.g.
  _assign_corresponding_tensors_to_same_layer → _detect_and_label_loops,
  _fetch_and_process_next_isomorphic_nodes → _advance_bfs_frontier) - Rename 6 local variables for
  clarity (e.g. node_to_iso_group_dict → node_to_iso_leader, subgraphs_dict → subgraph_info) - Add
  SubgraphInfo dataclass replacing dict-based subgraph bookkeeping - Replace list.pop(0) with
  deque.popleft() in BFS traversals - Remove ungrouped sweep in _merge_iso_groups_to_layers - Remove
  safe_copy in postprocess_fast (direct reference suffices) - Rewrite _get_hash_from_args to
  preserve positional indices, kwarg names, and dict keys via recursive _append_arg_hash helper -
  Remove vestigial index_in_saved_log field from TensorLogEntry, constants.py, logging_funcs.py, and
  postprocess.py - Fix PEP8: type(x) ==/!= Y → type(x) is/is not Y in two files - Split
  test_real_world_models.py into fast and slow test files - Add 12 new edge-case loop detection test
  models and test functions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.4.1 (2026-02-26)

### Bug Fixes

- **loops**: Refine iso groups to prevent false equivalence in loop detection
  ([`5aaad70`](https://github.com/johnmarktaylor91/torchlens/commit/5aaad70196073364ecfa7351fba08fed013e81b4))

When operations share the same equivalence type but occupy structurally different positions (e.g.,
  sin(x) in a loop body vs sin(y) in a branch), the BFS expansion incorrectly groups them together.
  Add _refine_iso_groups to split such groups using direction-aware neighbor connectivity.

Also adds NestedParamFreeLoops test model and prefixes intentionally unused variables in test models
  with underscores to satisfy linting.

### Code Style

- Auto-format with ruff
  ([`b9413f1`](https://github.com/johnmarktaylor91/torchlens/commit/b9413f1b4e60511eba117e032df60dea15849354))


## v0.4.0 (2026-02-26)

### Chores

- **tests**: Move visualization_outputs into tests/ directory
  ([`22ed018`](https://github.com/johnmarktaylor91/torchlens/commit/22ed018d80ce274225bb65bae9a0dda3aed1561b))

Anchor vis_outpath to tests/ via VIS_OUTPUT_DIR constant in conftest.py so test outputs don't
  pollute the project root.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **core**: Generalize in-place op handling, fix loop grouping, and harden validation
  ([`c72d6a6`](https://github.com/johnmarktaylor91/torchlens/commit/c72d6a6ecabd42471c1badd1b6cb8b0b6958e518))

- Generalize in-place op detection in decorate_torch.py: use `was_inplace` flag based on output
  identity instead of hardcoded function name list, and propagate tensor labels for all in-place ops
  - Fix ungrouped isomorphic nodes in postprocess.py: sweep for iso nodes left without a layer group
  after pairwise grouping (fixes last-iteration loop subgraphs with no params) - Deduplicate ground
  truth output tensors by address in user_funcs.py to match trace_model.py extraction behavior - Add
  validation exemptions: bernoulli_/full arg mismatches from in-place RNG ops,
  meshgrid/broadcast_tensors multi-output perturbation, *_like ops that depend only on
  shape/dtype/device - Gracefully handle invalid perturbed arguments (e.g. pack_padded_sequence)
  instead of raising - Guard empty arg_labels in vis.py edge label rendering - Fix test stability:
  reduce s3d batch size, add eval mode and bool mask dtype for StyleTTS

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.3.1 (2026-02-25)

### Bug Fixes

- **core**: Fix in-place op tracking, validation, and test stability
  ([`326b8a9`](https://github.com/johnmarktaylor91/torchlens/commit/326b8a907a170f8c9e4c7ec16296faaecfa56d51))

- Propagate tensor labels back to original tensor after in-place ops (__setitem__, zero_,
  __delitem__) so subsequent operations see updated labels - Add validation exemption for scalar
  __setitem__ assignments - Fix torch.Tensor → torch.tensor for correct special value detection -
  Remove xfail marker from test_varying_loop_noparam2 (now passes) - Add ruff lint ignores for
  pre-existing E721/F401 across codebase - Includes prior bug-blitz fixes across logging,
  postprocessing, cleanup, helper functions, visualization, and model tracing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.3.0 (2026-02-25)

### Chores

- **ci**: Replace black with ruff auto-format on push
  ([`e0cb9e1`](https://github.com/johnmarktaylor91/torchlens/commit/e0cb9e1c20b347cc2ff2579cb10b1beb959f8252))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **config**: Add ruff config and replace black/isort with ruff in pre-commit
  ([`c27ced8`](https://github.com/johnmarktaylor91/torchlens/commit/c27ced8f7f8fa9e555dfa6a8313db53575b00305))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **config**: Set major_on_zero to true
  ([`d63451c`](https://github.com/johnmarktaylor91/torchlens/commit/d63451cdd3360fa2ce6979b2e371a806310ec6ca))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **tests**: Restructure test suite into focused files with metadata coverage
  ([`31b0353`](https://github.com/johnmarktaylor91/torchlens/commit/31b0353813353a0eaa0dc94e9659f9e84bbaab00))

Split monolithic test_validation_and_visuals.py (3130 lines, 155 tests) into: - conftest.py: shared
  fixtures and deterministic seeding - test_toy_models.py: 78 tests (66 migrated + 12 new API
  coverage tests) - test_metadata.py: 44 comprehensive metadata field tests (7 test classes) -
  test_real_world_models.py: 75 tests with local imports and importorskip

New tests cover: log_forward_pass parameters (layers_to_save, save_function_args,
  activation_postfunc, mark_distances), get_model_metadata, ModelHistory access patterns,
  TensorLogEntry field validation, recurrent metadata, and GeluModel.

Removed 13 genuine size-duplicate tests (ResNet101/152, VGG19, etc.). All optional dependencies now
  use pytest.importorskip for graceful skipping.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.2.0 (2026-02-24)

### Bug Fixes

- **ci**: Enable verbose PyPI upload and disable attestations for debugging
  ([`579506d`](https://github.com/johnmarktaylor91/torchlens/commit/579506df2260dd6c7dcf153b52ad6ee42e282191))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Fetch tags in release workflow so semantic-release finds v0.1.36
  ([`0f0a3d0`](https://github.com/johnmarktaylor91/torchlens/commit/0f0a3d0895239152b9093c9f0752f957e27724d7))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Pin semantic-release to v9 and add debug output
  ([`b68388c`](https://github.com/johnmarktaylor91/torchlens/commit/b68388cfead9d624c569df0c3687bed823cb0ce4))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Prevent major version bump on 0.x releases
  ([`db4a31e`](https://github.com/johnmarktaylor91/torchlens/commit/db4a31edfc626a3e38b46e11d644b8b02b06d264))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Remove direct URL dependency rejected by PyPI and clean up workflow
  ([`500368b`](https://github.com/johnmarktaylor91/torchlens/commit/500368ba22d50f2235b9be3b87a455b2c81b9891))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Use GitHub App token to bypass branch protection in release workflow
  ([`f2cf8ae`](https://github.com/johnmarktaylor91/torchlens/commit/f2cf8ae7450712bb54fcf58a1f2bbb8d3760f076))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **build**: Migrate to pyproject.toml with semantic-release and GitHub Actions
  ([`f8e01c9`](https://github.com/johnmarktaylor91/torchlens/commit/f8e01c9f7821a3dd54c9df805bc2181f3340e9ea))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.1.36 (2025-09-26)
