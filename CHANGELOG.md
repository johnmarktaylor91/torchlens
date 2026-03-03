# CHANGELOG


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
