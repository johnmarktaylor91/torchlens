# CHANGELOG


## v0.7.2 (2026-02-28)

### Bug Fixes

- **tests**: Reduce r2plus1d_18 input size to prevent OOM
  ([`952e34e`](https://github.com/johnmarktaylor91/torchlens/commit/952e34e38d940eeff0f77a7f8f16e1912121bfdd))

The test_video_r2plus1_18 test used a (16,3,16,112,112) input (~96M elements) which consistently got
  OOM-killed. Reduced to (1,3,1,112,112) which passes reliably while still exercising the full
  model.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.1 (2026-02-28)

### Bug Fixes

- **logging**: Handle complex-dtype tensors in tensor_nanequal
  ([`fe58f25`](https://github.com/johnmarktaylor91/torchlens/commit/fe58f25c27a246ff4c2f29ef1deaa8809fd269fa))

torch.nan_to_num does not support complex tensors, which caused test_qml to fail when PennyLane
  quantum ops produced complex outputs. Use view_as_real/view_as_complex to handle NaN replacement
  for complex dtypes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.0 (2026-02-28)

### Bug Fixes

- **logging**: Capture/restore RNG state for two-pass stochastic models
  ([#58](https://github.com/johnmarktaylor91/torchlens/pull/58),
  [`2b3079f`](https://github.com/johnmarktaylor91/torchlens/commit/2b3079f00e89e91bd8a84652bae381a6ab6813df))

Move RNG state capture/restore before pytorch decoration to prevent internal .clone() calls from
  being intercepted by torchlens' decorated torch functions. Also speed up test_stochastic_loop by
  using a higher starting value.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Ensure output layer parents are saved with layers_to_save
  ([#46](https://github.com/johnmarktaylor91/torchlens/pull/46),
  [`a3c74fa`](https://github.com/johnmarktaylor91/torchlens/commit/a3c74fa92215dcd0bd5b300b32559ecf47b0ef97))

When layers_to_save is a subset, the fast pass now automatically includes parents of output layers
  in the save list. This ensures output layer tensor_contents is populated in postprocess_fast
  (which copies from parent).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Replace copy.deepcopy with safe_copy to prevent infinite loops
  ([#18](https://github.com/johnmarktaylor91/torchlens/pull/18),
  [`aa841fe`](https://github.com/johnmarktaylor91/torchlens/commit/aa841fe7def7def929e29cc612f01976d4b12f62))

copy.deepcopy hangs on complex tensor wrappers with circular references (e.g. ESCNN
  GeometricTensor). Replace with safe_copy_args/safe_copy_kwargs that clone tensors, recurse into
  standard containers, and leave other objects as references.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Use argspec to disambiguate tuple/list input args
  ([#43](https://github.com/johnmarktaylor91/torchlens/pull/43),
  [`34f6687`](https://github.com/johnmarktaylor91/torchlens/commit/34f6687770573935257201e99c97e8da70f6e8bf))

When a model's forward() expects a single arg that IS a tuple/list of tensors, torchlens incorrectly
  unpacked it into multiple positional args. Now uses inspect.getfullargspec to detect single-arg
  models and wraps the tuple/list as a single arg. Also handles immutable tuples in
  _fetch_label_move_input_tensors device-move logic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Functional ops at end of container modules rendered as ovals
  ([#48](https://github.com/johnmarktaylor91/torchlens/pull/48),
  [`f886b36`](https://github.com/johnmarktaylor91/torchlens/commit/f886b36c104416120a9a4de8f70a7fbc52848bfe))

_check_if_only_non_buffer_in_module was too broad — it returned True for functional ops (like
  torch.relu) at the end of container modules with child submodules, causing them to render as
  boxes. Added a leaf-module check: only apply box rendering for modules with no child submodules.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **metadata**: Track module training mode per submodule
  ([#52](https://github.com/johnmarktaylor91/torchlens/pull/52),
  [`6ede005`](https://github.com/johnmarktaylor91/torchlens/commit/6ede0059f5d420d69cd6c65fa2ba744dac248c95))

Capture module.training in module_forward_decorator and store in ModelHistory.module_training_modes
  dict (keyed by module address). This lets users check whether each submodule was in train or eval
  mode during the forward pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.2 (2026-02-28)

### Bug Fixes

- **postprocess**: Correct _merge_buffer_entries call convention
  ([#47](https://github.com/johnmarktaylor91/torchlens/pull/47),
  [`0e539c4`](https://github.com/johnmarktaylor91/torchlens/commit/0e539c4dbd113e066e02c447cd3f5744247c2f27))

_merge_buffer_entries is a module-level function, not a method on ModelHistory. Fixes AttributeError
  when processing recurrent models with duplicate buffer entries.

Based on gilmoright's contribution in PR #56.


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
