## BLOCKING (must resolve before building P1)

### 1. `facet.grad = transform(op.grad)` is not available by default, and the P1 API does not specify the failure contract.

**Evidence**
- Public trace defaults leave gradient capture off: `save_gradients: bool = False` and `backward_ready: bool = False` in `torchlens/user_funcs.py:914` and `torchlens/user_funcs.py:932`.
- `Op.grad` starts as `None`; the Op model says grad is populated by a backward hook, not by forward capture: `torchlens/data_classes/op.py:855`-`torchlens/data_classes/op.py:859`.
- Tensor grad hooks are only registered at forward time if `trace.save_gradients` is true: `torchlens/backends/torch/model_prep.py:1174`-`torchlens/backends/torch/model_prep.py:1177`.
- `Trace.log_backward()` can lazily register hooks after capture, but only for saved, in-memory tensor outs: `_ensure_layer_grad_hooks()` scans `layer.out` and registers only if it is a `torch.Tensor`, `torchlens/backends/torch/backward.py:876`-`torchlens/backends/torch/backward.py:892`.
- Grad storage is selection-gated. `_log_tensor_grad()` skips layers outside `_grad_layer_nums_to_save`, `torchlens/backends/torch/tensor_tracking.py:97`-`torchlens/backends/torch/tensor_tracking.py:105`. GradFn hook payload storage is gated by the same selection, `torchlens/backends/torch/backward.py:301`-`torchlens/backends/torch/backward.py:310`.

**Why it matters**
The design describes per-facet gradients as "lazy, cheap, no capture-time blowup" from "already-saved gradients." That is only true for traces that either captured with `save_gradients` / explicit `gradients_to_save` or later call `log_backward()` while the home op's raw out is still saved and graph-connected. In default traces, `op.grad` is `None`. In selective traces, `op.grad` may be absent for the home op even though sibling facets exist. In streamed/evicted traces, `_ensure_layer_grad_hooks()` cannot hook missing `layer.out` tensors.

**Suggested resolution**
Define `facet.grad` as an explicit capability:
- `available` only when the home op has `has_grad=True`, or when the trace can still run `log_backward()` and the home op's raw out is saved in memory.
- return a typed `MissingFacet` / `MissingGradient` with the exact fix: recapture with `backward_ready=True` plus `gradients_to_save='all'` or include the home op in `gradients_to_save`.
- do not advertise default gradient availability. P1 tests must cover default trace, selective `gradients_to_save`, streamed grads with eviction, and `save_raw_gradients=False` plus `grad_transform`.

### 2. Paired `grad_fn` sourcing is underspecified and cannot currently map cleanly to facet slices.

**Evidence**
- `GradFnCall` stores only opaque `grad_inputs` and `grad_outputs` payloads: `torchlens/data_classes/grad_fn_call.py:31`-`torchlens/data_classes/grad_fn_call.py:35`.
- Forward-to-backward pairing is only by autograd object identity: `_layer_by_grad_fn_id()` maps `grad_fn_object_id -> layer_label`, `torchlens/backends/torch/backward.py:400`-`torchlens/backends/torch/backward.py:418`.
- The backward graph walk discards PyTorch's `next_functions` input index: `_iter_next_grad_fns()` iterates `(next_fn, _input_num)` and ignores `_input_num`, `torchlens/backends/torch/backward.py:138`-`torchlens/backends/torch/backward.py:140`.
- The GradFn record has `has_op` and `op_label`, but no forward argument index map or output index map: `torchlens/data_classes/grad_fn.py:171`-`torchlens/data_classes/grad_fn.py:187`.
- Multi-output metadata exists on `Op` (`multi_output_index`, `container_path`, `container_spec`), `torchlens/data_classes/op.py:894`-`torchlens/data_classes/op.py:898`, but no code ties that to a particular element of `grad_outputs`.

**Why it matters**
The proposal says paired grad_fns give "input-side" gradients and fallback output-side gradients, with "index bookkeeping for multi-in/out." That bookkeeping does not exist. For autograd Functions, `grad_inputs` tuple ordering is the backward function's contract, not TorchLens's parent-label order. `None` entries, optional tensor arguments, fused ops, views, and generated backward nodes all break any assumption that `grad_inputs[i]` maps to `op.parents[i]` or `saved_args[i]`. For multi-output ops, `grad_outputs` tuple order may be recoverable, but TorchLens currently stores only forward container paths on Ops, not a verified mapping from an Op leaf to a grad output tuple element.

**Suggested resolution**
Drop paired-grad_fn input-side gradients from P1, or make it a separate design with a verified mapping table:
- store `next_functions` input numbers, forward parent labels, parent arg paths, and forward output paths in GradFn metadata;
- for every supported op class, assert tuple length/order against forward metadata;
- expose output-side grad_fn sourcing only when it is proven equivalent to `op.grad`, and use it as a fallback only after validation.

### 3. The structural transform vocabulary is too broad for the claimed gradient and write-back guarantees.

**Evidence**
- Current facets return concrete tensors and callables, not provenance specs. `FacetView._compute()` merges `recipe.func(self._record)` dictionaries directly, `torchlens/semantic/facets.py:238`-`torchlens/semantic/facets.py:255`.
- Current attention head slicing assumes a concrete tensor layout and hard-codes `value[:, :, head_index, :]`, including GQA index remapping for K/V, `torchlens/semantic/facets.py:118`-`torchlens/semantic/facets.py:131`.
- Existing intervention hooks validate and replace the whole tensor, not a slice: `_execute_hook()` passes the entire `out` to the hook, `torchlens/intervention/runtime.py:120`-`torchlens/intervention/runtime.py:167`, and `validate_hook_output()` requires the replacement to match the whole output shape unless forced, `torchlens/intervention/runtime.py:201`-`torchlens/intervention/runtime.py:228`.
- `TensorSliceSpec` exists, but only carries ad hoc axes/index metadata and is not used as an inverse transform chain, `torchlens/intervention/types.py:54`-`torchlens/intervention/types.py:67`.

**Why it matters**
The proposal treats `.reshape`, `.transpose`, `.split`, `.select`, and `__getitem__` as automatically safe for read, grad, and intervention. Read is easy. Grad is safe only for bijective/permutation-style reshapes and pure selections; it is not safe for transforms that duplicate, broadcast, reduce, concatenate, apply nonlinear postprocessing, or combine multiple homes. Write-back is harder: a slice/spec inverse must compose with existing hooks, preserve dtype/device/contiguity semantics, and merge multiple edits targeting the same home op.

Examples that break the model:
- GPT-2 `c_attn` q/k/v are slices of one home op. Current recipe splits one tensor into q/k/v at `torchlens/semantic/recipes/attention.py:137`-`torchlens/semantic/recipes/attention.py:144`. If a user edits both q and k, P1 must define cumulative write-back ordering and conflict detection for the shared home.
- GQA K/V use fewer heads than Q. Current read-time remap collapses query heads to KV groups, `torchlens/semantic/facets.py:125`-`torchlens/semantic/facets.py:130`. A K/V facet addressed through query head `h` is not a unique storage slice when multiple Q heads map to one KV head. Write-back needs alias semantics.
- MLP `intermediate` is computed as `gated_out * up_out`, not anchored to one real op in current recipes, `torchlens/semantic/recipes/mlp.py:31`-`torchlens/semantic/recipes/mlp.py:36`. This must degrade to read-only unless TorchLens anchors to the captured multiply op, not the recipe computation.

**Suggested resolution**
Split transform ops into capability classes:
- `bijective_view`: reshape/transpose/permute with exact inverse; read+grad+write.
- `selection`: slice/select/split; read+grad+write with scatter-back and conflict policy.
- `aliasing_selection`: GQA repeated KV, expand, broadcast; read+grad only unless a write policy is explicitly chosen.
- `computed`: arbitrary callable or multi-home expression; read-only unless anchored to a captured op.

Also specify same-home edit composition before P2, because P1's spec object becomes the ABI for intervention.

### 4. Attention reconstruction cannot rely on "captured inputs saved by default."

**Evidence**
- `save_arg_values` defaults to false, `torchlens/user_funcs.py:913`-`torchlens/user_funcs.py:914`; the docs state it stores non-tensor args and is needed for validation, `torchlens/user_funcs.py:1331`-`torchlens/user_funcs.py:1332`.
- `Op.save_activation()` stores `saved_args` / `saved_kwargs` only when `save_arg_values` is true, else sets them to `None`, `torchlens/data_classes/op.py:2085`-`torchlens/data_classes/op.py:2095`.
- SDPA parent extraction knows Q/K/V and tensor mask positions, `torchlens/capture/arg_positions.py:977`-`torchlens/capture/arg_positions.py:981`, but non-tensor flags like `dropout_p`, `is_causal`, and `scale` live in `non_tensor_kwargs` / `func_config`.
- `func_config` intentionally drops default dropout and default scale, and only includes causal when true, `torchlens/capture/salient_args.py:293`-`torchlens/capture/salient_args.py:305`.
- Probe result from this repo: default SDPA trace recorded parent labels for Q/K/V, `saved_args=None`, `saved_kwargs=None`, and `func_config={'is_causal': True}` for `dropout_p=0.0, is_causal=True`. With `save_arg_values=True`, the same op saved only three positional tensors plus non-tensor kwargs.

**Why it matters**
Reconstruction requires exact Q, K, V, mask, scale, causal behavior, dropout state, backend dtype/softmax convention, and any GQA expansion semantics. Default TorchLens can usually find Q/K/V as parents, but it does not save a full callable argument snapshot by default. If Q/K/V are the same tensor, parent labels repeat. If a tensor mask is omitted because it is built inside an untraced path, reconstruction cannot infer it. If `scale=None`, the implementation must know PyTorch's default scale convention and shape-dependent head dimension.

Replay validation can also fail for correct reconstructions:
- SDPA may compute softmax in fp32 while inputs/outputs are bf16/fp16.
- dropout inside attention makes `pattern @ V` not equal the fused output unless the dropout mask and RNG state are captured.
- causal mask construction differs for non-square query/key lengths and KV-cache offsets.
- GQA/MQA may internally repeat/interleave K/V heads; reconstructing with unexpanded K/V can validate only if the implementation applies the same `enable_gqa` semantics.

**Suggested resolution**
Make fused attention reconstruction opt-in to a "reconstruction-ready" capture mode, or require `save_arg_values=True` plus `save_rng_states=True` for stochastic attention. The reconstruction record should persist:
- resolved Q/K/V/mask source labels and exact argument paths;
- effective scale;
- `is_causal`, `dropout_p`, training mode, RNG state if dropout is nonzero;
- GQA expansion convention and head mapping.

Validation tolerances and dtype policy must be specified per backend/dtype. Validation failure should report which prerequisite is missing, not just return a generic `MissingFacet`.

### 5. The registration model is a new concurrency/reproducibility subsystem, not a small replacement of the current registry.

**Evidence**
- Current global registry is a process-global mutable list, `_REGISTRY`, `torchlens/semantic/facets.py:258`.
- `register()` appends to `_REGISTRY` in registration order, `torchlens/semantic/facets.py:305`-`torchlens/semantic/facets.py:320`.
- Matching recipes are computed when a `FacetView` is constructed, not snapped at trace capture: `self._recipes = _matching_recipes(record)`, `torchlens/semantic/facets.py:137`-`torchlens/semantic/facets.py:148`.
- Module and Op facet accessors cache the `FacetView` on the record object, `torchlens/data_classes/op.py:1907`-`torchlens/data_classes/op.py:1915` and `torchlens/data_classes/module.py:1525`-`torchlens/data_classes/module.py:1533`.
- Duplicate facet names are last-wins with a warning at compute time, `torchlens/semantic/facets.py:244`-`torchlens/semantic/facets.py:253`.

**Why it matters**
The locked design wants built-ins, user overrides, contextvars, per-trace recipes, entry points, snapshot-at-capture, provenance, specificity ordering, and stable lazy facets. Current lazy caching creates races:
- if `record.facets` is first accessed before a registry mutation, it pins old `_recipes`;
- if first accessed after mutation, the same trace sees new recipes;
- per-trace snapshots cannot be implemented by only changing `_matching_recipes()` unless all records read from a trace-owned immutable recipe set;
- contextvar overrides need clear behavior for nested traces and for `FacetView` objects created inside/outside the context.

**Suggested resolution**
Before migrating recipes, implement a trace-owned immutable `facet_registry_snapshot` with version/provenance IDs. `FacetView` must read only that snapshot. `tl.facets.using()` and `trace(..., recipes=...)` must affect capture-time snapshot construction only, not lazy access after capture. Define exact conflict ordering and diagnostics before recipe migration, because otherwise P1 facet names will not be reproducible.

### 6. Multi-output "fix" is already partly implemented; touching it blindly risks loop detection regressions.

**Evidence**
- Container specs already preserve dict keys, NamedTuple fields, dataclass fields, and HF ModelOutput keys: `torchlens/backends/torch/ops.py:474`-`torchlens/backends/torch/ops.py:544` and the `ContainerSpec` fields at `torchlens/intervention/types.py:206`-`torchlens/intervention/types.py:216`.
- Output traversal emits typed path components for HF keys, named fields, dataclass fields, dict keys, and tuple/list indices: `torchlens/backends/torch/ops.py:547`-`torchlens/backends/torch/ops.py:656`.
- Module output role hints already cover LSTM/GRU/RNN/MultiheadAttention: `torchlens/data_classes/_module_role_hints.py:19`-`torchlens/data_classes/_module_role_hints.py:39`.
- Module exit assigns `multi_output_name` for multi-output module returns, `torchlens/backends/torch/model_prep.py:1322`-`torchlens/backends/torch/model_prep.py:1327`, and finalization fills missing roles, `torchlens/postprocess/finalization.py:319`-`torchlens/postprocess/finalization.py:349`.
- Equivalence class construction includes `_outindex{i}` whenever `in_multi_output` is true, `torchlens/backends/torch/tensor_tracking.py:340`-`torchlens/backends/torch/tensor_tracking.py:380`, and op creation feeds `output_index` into parameterized op barcodes, `torchlens/backends/torch/ops.py:2171`-`torchlens/backends/torch/ops.py:2189`.
- Probe result from this repo: a simple `nn.LSTM` currently logs three module outputs with names `output`, `h_n`, `c_n`, each `pass_index=1`, `num_passes=1`, and equivalence classes containing `outindex0/1/2`.

**Why it matters**
The sprint plan says P1 must add container name preservation and fix "LSTM outputs mislabeled as recurrent passes." The code already has name-preserving `ContainerSpec` and LSTM role hints. If there is still a bug, it is narrower than the plan states. Any change to the `in_multi_output` flag, output index assignment, or equivalence class keying can regress loop detection because loop detection groups by equivalence class (`torchlens/postprocess/loop_detection.py:194`-`torchlens/postprocess/loop_detection.py:203`) and matches frontier nodes by equivalence class (`torchlens/postprocess/loop_detection.py:604`-`torchlens/postprocess/loop_detection.py:633`).

**Suggested resolution**
First write a failing regression that demonstrates the exact LSTM mislabel condition, including `pass_index`, `num_passes`, `recurrent_ops`, and `equivalence_class`. If the bug no longer reproduces, remove this from P1 or reframe it as "verify and lock regression coverage." Do not rewrite container traversal as part of facets unless the failing test proves the specific missing behavior.

## MAJOR risks

### 7. `outs -> facets` unification has no clear mapping for non-identifier names and nested paths.

**Evidence**
- `multi_output_role_from_path()` stringifies normalized path components with dots, `torchlens/data_classes/_module_role_hints.py:145`-`torchlens/data_classes/_module_role_hints.py:151`.
- Dict and HF keys can be arbitrary values, not necessarily valid Python identifiers, `torchlens/intervention/types.py:21`-`torchlens/intervention/types.py:47`.
- `FacetView.__getattr__` exposes attribute-style access and maps missing keys to `AttributeError`, `torchlens/semantic/facets.py:220`-`torchlens/semantic/facets.py:226`.

**Why it matters**
The design promises `model.facets.attentions` and dotted names like `out1.0`. Attribute access cannot represent `out1.0`, integer dict keys, keys with dots/spaces, duplicate names after stringification, or keys that collide with `FacetView` methods (`keys`, `get`, `head`, `recipe_source`). The current role stringification loses type information (`TupleIndex(1)` and dict key `"1"` can both become `"1"`).

**Suggested resolution**
Make item access canonical and typed. Attribute access should be best-effort sugar only for valid identifiers that do not collide with accessor methods. Store structural facets under a path object or escaped path string with a reversible grammar, and specify collision rules.

### 8. In-place ops and view aliases make "home op reference" ambiguous.

**Evidence**
- Op logging marks in-place outputs by checking whether the output tensor already has a label, `torchlens/backends/torch/ops.py:2219`-`torchlens/backends/torch/ops.py:2220`.
- Non-parameterized equivalence class names distinguish in-place functions by appending `_inplace`, `torchlens/backends/torch/ops.py:2194`-`torchlens/backends/torch/ops.py:2199`.
- Parent tensor variation tracking exists because children can receive different raw versions, `torchlens/data_classes/op.py:849`-`torchlens/data_classes/op.py:853`.

**Why it matters**
For a facet anchored to a view or an in-place-mutated tensor, `home.out` may not be the value that a downstream child consumed. Write-back to the current `home.out` can edit the wrong version. Gradient slicing can also be misleading when the selected view aliases storage mutated later.

**Suggested resolution**
FacetSpec must carry not just `home op`, but the relevant value version: raw home out vs transformed out vs child-specific `out_versions_by_child`. Mark in-place/view-derived facets as not intervention-safe until replay/rerun semantics are explicitly validated.

### 9. Current recipe coverage includes multi-home and computed facets that P1 says should degrade, but the migration plan does not classify them.

**Evidence**
- Embedding `weight` returns the live module parameter, not an Op output, `torchlens/semantic/recipes/embedding.py:26`-`torchlens/semantic/recipes/embedding.py:29`.
- Norm `gamma` and `beta` return parameter values, not Op outputs, `torchlens/semantic/recipes/norm.py:31`-`torchlens/semantic/recipes/norm.py:43`.
- MLP activation facets recompute GELU/SiLU on child outputs, `torchlens/semantic/recipes/_helpers.py:101`-`torchlens/semantic/recipes/_helpers.py:115`, and MLP `intermediate` multiplies two tensors in recipe code, `torchlens/semantic/recipes/mlp.py:31`-`torchlens/semantic/recipes/mlp.py:36`.

**Why it matters**
If P1 "migrates built-in recipes to FacetSpec form" without a per-facet capability table, some current useful facets will silently disappear or be incorrectly upgraded to grad/intervention capable. Parameter facets need a different home type; computed facets need either a captured op anchor or read-only status.

**Suggested resolution**
Inventory every built-in facet and assign one of: `op_structural`, `parameter`, `module_input`, `module_output`, `computed_read_only`, `missing`. P1 should fail closed when a recipe tries to claim grad/write capability for anything but `op_structural`.

### 10. Replay validation for reconstructed attention is underspecified and may be vacuous at module boundaries.

**Evidence**
- Existing fused pattern recipe only returns `MissingFacet`, with an eager-recapture message, `torchlens/semantic/recipes/_helpers.py:117`-`torchlens/semantic/recipes/_helpers.py:125`.
- Current attention recipes often set `attn_out` to the module output or output projection child, not necessarily the raw SDPA output: `torchlens/semantic/recipes/attention.py:40`-`torchlens/semantic/recipes/attention.py:51`, `torchlens/semantic/recipes/attention.py:145`, and `torchlens/semantic/recipes/attention.py:174`.

**Why it matters**
The proposal says validate `pattern @ V (+ output proj) == captured attn output`. Which captured output? SDPA op output, attention module output, post-output-projection tensor, or residual-added block output? Each recipe has different boundaries. If validation compares against the wrong boundary, it either fails for correct reconstruction or passes for the wrong reason after extra transforms cancel error.

**Suggested resolution**
Define validation target per reconstructed facet:
- `z`: validate against the fused SDPA op output exactly.
- `pattern/scores`: validate by recomputing `z` and comparing to the fused SDPA op output.
- `attn_out/result`: validate against the output projection child only when the recipe supplies the projection op and weights.

### 11. FacetSpec portability is not addressed, but `.tlspec` currently drops executable function objects.

**Evidence**
- `Op.PORTABLE_STATE_SPEC` drops `func`, `grad_fn_handle`, and `grad_fn`, `torchlens/data_classes/op.py:572`-`torchlens/data_classes/op.py:594`.
- `Trace.PORTABLE_STATE_SPEC` drops `backward_ready`, `module_filter`, and transform callables, `torchlens/data_classes/trace.py:973`-`torchlens/data_classes/trace.py:976`.
- Current `FacetView.recipe_source` is just recipe function names, not a durable recipe hash or import ref, `torchlens/semantic/facets.py:151`-`torchlens/semantic/facets.py:160`.

**Why it matters**
The design says every facet records which recipe produced it and traces snapshot active recipes. If a trace is saved and loaded, can facet specs still resolve? A structural transform chain can be portable, but raw callable read-only facets cannot unless represented as non-executable provenance. Entry-point recipes are code and may not exist in the loading environment.

**Suggested resolution**
P1 must define portable fields for `FacetSpec`: home label, transform primitive chain, capability flags, recipe ID, recipe version/source, and missing-recipe behavior on load. Raw callables should never be serialized as executable facets.

## Recommendations / underspecified points

### A. Define the FacetSpec data model as an ABI before recipe migration.

Required fields:
- home kind: op, module output, module input, parameter, or computed;
- home label/address and pass/call index;
- output path for structural outputs;
- transform primitive chain with shape assertions before/after each step;
- capability flags: read, grad, write, portable, reconstructed;
- alias/conflict group for shared-home edits;
- recipe source ID and version.

### B. Specify transform inverse semantics with tests.

Every structural primitive needs:
- `apply(x)`;
- `project_grad(g)`;
- `scatter_update(home, edited, mode='replace'|'add'?)`;
- shape/dtype/device checks;
- alias conflict behavior for overlapping writes.

Do not call lossy selection "invertible"; call it "scatter-back capable."

### C. Separate gradient coordinates from activation coordinates.

For pure views, `transform(op.grad)` is a reasonable coordinate projection. For aliased/broadcast facets, gradients may need reduction or may be undefined. The public docs should say "facet.grad is the parent-output gradient expressed in the facet's coordinate system" and list the transforms where this is mathematically valid.

### D. Add non-negotiable P1 tests before implementation.

Minimum test matrix:
- default trace: `facet.grad` is a typed missing value with instructions;
- `backward_ready=True` plus `log_backward`: facet grad matches manual slice;
- selective `gradients_to_save`: missing on unselected home;
- GPT-2 fused QKV: q/k/v share one home; read works; conflicting write specs are detected;
- GQA K/V: query-head-to-KV alias semantics are explicit;
- LSTM/GRU/RNN multi-output roles: names, indices, paths, `num_passes`, `recurrent_ops`;
- NamedTuple/dataclass/HF ModelOutput/dict key collisions;
- SDPA reconstruction with mask, `is_causal`, non-default scale, dropout zero, dropout nonzero, bf16/fp16 tolerance.

### E. Reduce P1 scope.

P1 currently contains four hard projects: FacetSpec ABI, recipe migration, gradient capability, fused attention reconstruction, and registry reproducibility. Build order should be:
1. immutable registry snapshot and structural output facets only;
2. FacetSpec primitives for op-anchored read and grad only;
3. recipe migration with explicit capability inventory;
4. attention reconstruction after argument-capture prerequisites are solved.

### F. Do not treat "container_spec name-preservation" as missing.

The code already preserves structural names in `ContainerSpec`. The likely missing piece is exposing those names through `.facets` with a reversible naming scheme and collision policy. P1 should not rewrite container capture unless a failing test proves the current implementation loses a specific name.
