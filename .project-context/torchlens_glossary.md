# TorchLens Glossary
> Public API reference. Reflects the locked naming targets from the rename
> sprint. Generated for human reference and as a working tool for
> implementation.
## How to read this glossary
This glossary describes the target public API, not the current implementation names.
Names come from `.project-context/notebook_audit_notes.md`.
When the audit notes say "Was X, Now Y", this glossary documents `Y`.
An `Op` is one captured tensor-producing operation.
A `Layer` is the stable label that can aggregate one or more Ops at the same graph position.
A `ModuleCall` is one invocation of an `nn.Module`.
A `Module` aggregates all calls of the same module address.
Labels are TorchLens structured identifiers. Backend addresses are backend-native paths or
handles; torch module and parameter addresses are PyTorch dotted paths.
Example: `conv2d_1_2` is a Layer label, while `conv2d_1_2:1` is an Op label.
All user-facing type, overall, pass, call, op, and creation indices are 1-based unless an entry explicitly says otherwise.
Private storage names, dropped aliases, and underscore-prefixed locked names are omitted from the main entries.
Deferred items are listed at the end instead of promoted as final API.
## Top-level vocabulary
- `tl.trace(model, x, *, backend=None)`: Resolve a backend, run a captured forward pass, and return a `Trace`. `None` keeps torch eager capture as the default and MLX module auto-routing as a technical preview; explicit `backend="jax"` and `backend="tinygrad"` enable their preview functional captures.
- `BackendName`: Public backend identifier used by `backend=` and `Trace.backend`. Shipped names are `"torch"`, `"mlx"`, `"jax"`, and `"tinygrad"`; `"fake"` is test-only.
- `BackendSpec`: Registry object owning backend resolution, capture entry, validation entry, serialization policy, capability flags, and canonical backend errors.
- `Trace`: Top-level object for one captured model execution, including graph, tensors, modules, params, buffers, gradients, backend tag, and metadata.
- `trace.derived_grads`: Backend-neutral accessor for leaf-level gradients derived outside true backward capture. JAX populates it through a second `jax.value_and_grad` run; tinygrad populates it through a bracketed `DEV=PYTHON` leaf-gradient run.
- `trace.intermediate_derived_grads`: tinygrad-only accessor for exact unambiguous per-op gradients from the separate no-realize intermediate-gradient pass. It is not true backward capture and does not populate `trace.saved_grad_ops`.
- `op.derived_grad`: tinygrad-only read-only convenience property for an Op's entry in `trace.intermediate_derived_grads`; `op.grads` remains true-backward-only and raises on non-torch traces.
- `tl.backends.jax.GradOptions`: JAX preview options object passed to `tl.trace(..., backend="jax", grad_options=...)`; declares params, optional loss function, and input-relative gradient argnums.
- `DtypeRef`: Backend-neutral dtype reference stored beside legacy dtype fields.
- `DeviceRef`: Backend-neutral device reference stored beside legacy torch device fields when a backend exposes device metadata.
- `Trace.backward(loss)`: Add backward-pass data to an existing Trace by backpropagating an explicit loss.
- `Layer`: A stable graph position such as `conv2d_1_2`; it aggregates the Ops for that position.
- `Op`: One tensor operation invocation such as `conv2d_1_2:1`; this is where saved `out`, `grad`, args, timing, and function data live.
- `Module`: Aggregate record for one `nn.Module` address in the model.
- `ModuleCall`: One actual call of an `nn.Module.forward`.
- `Param`: Record for one `nn.Parameter` tensor.
- `Buffer`: Record for one PyTorch buffer value, with buffer-specific identity and update metadata.
- `GradFn`: Record for one autograd `grad_fn` node in the backward graph.
- `GradFnCall`: One hook firing or backward call event for a `GradFn`.
- `Bundle`: Coordinated container for comparing multiple Traces.
- `TraceAccessor`: Dict-like Bundle accessor for named Traces.
- `SuperOp`: Cross-trace view of the same Op label across Bundle members.
- `SuperLayer`: Cross-trace view of the same Layer label across Bundle members.
- `SuperOpAccessor`: Dict-like Bundle accessor returning `SuperOp` objects.
- `SuperLayerAccessor`: Dict-like Bundle accessor returning `SuperLayer` objects.
## Trace (per `tl.trace(model, x)`)
### Identity
- `model_name`: Short display name for the source model, usually the class name unless overridden.
- `model_qualname`: Fully qualified class name for the source model, useful for verifying reruns.
- `model_class`: Runtime Python type object for the source model; unavailable after portable load if the class cannot be resolved.
- `model_id`: Runtime `id(model)` value for in-process identity checks.
- `trace_name`: Name of this Trace within forks or Bundles; defaults to `model_name`.
- `parent_trace`: Immediate parent Trace when this Trace is forked; `None` for an original Trace.
- `root_trace`: Root Trace of a fork tree; `None` on the root itself.
- `run_state`: Enum-like state indicator describing whether the Trace is pristine, rerun, appended, or directly modified.
- `io_format_version`: Portable file format version used for save/load compatibility.
- `backend`: Public `BackendName` tag identifying the capture backend, e.g. `"torch"`, `"mlx"`, `"jax"`, or `"tinygrad"`. Survives portable save/load.
- `module_identity_mode`: Module-record interpretation for the trace: `"torch_module"`, `"pytree_module"`, `"object_module"`, or `"function_root"`.
- `jax_control_flow`: Public `trace()` option for JAX control-flow handling; `lax.scan`/`cond`/`while` can be unrolled for raw JAX function-root captures.
- `jax_max_control_flow_unroll`: Public `trace()` option for the JAX control-flow unroll safety limit.
- `payload_policy`: Backend payload codec/materialization policy. JAX, tinygrad, and MLX portable writes use `"array_payloads"` for non-torch array payloads, which is narrower than torch `"full"` payload portability; MLX currently covers saved forward arrays.
- `jax_sharding_*` codec metadata: Audit-only `.tlspec` manifest fields for fully addressable JAX array payloads. They describe saved sharding provenance with JSON primitives such as sharding kind, mesh axis names, partition spec strings, and device counts; load materializes values on the default/current device and does not reconstruct the saved topology.
- `save_preview`: Declared public `trace()` option for future non-torch save preview flags; the `save=` kwarg itself supports static-label selectors on JAX, tinygrad, and MLX.
- `param_source`: Parameter-record source for the trace: `"native-module"`, `"pytree-derived"`, or `"none"`.
- `FIELD_DEFAULTS`: Class-level defaults applied at initialization and cleanup.
- `FIELD_FORK_POLICY`: Class-level per-field policy for `fork()`.
- `FIELD_SAVE_POLICY`: Class-level per-field policy for portable save/load.
### Counts
- `num_layers`: Number of distinct Layer records, including input, output, and buffer boundary Layers.
- `num_ops`: Number of Op invocations captured in this Trace.
- `num_saved_ops`: Number of Ops whose forward tensor was saved by TorchLens.
- `num_streamed_passes`: Number of streaming-mode chunks or appended streamed passes represented in this Trace.
- `num_grad_fns`: Number of unique autograd grad-fn nodes captured for backward analysis.
- `num_grad_fns_without_op`: Number of grad-fn nodes that have no corresponding forward Op.
- `num_params`: Total scalar parameter count across the model.
- `num_params_trainable`: Total scalar parameter count with `requires_grad=True`.
- `num_params_frozen`: Total scalar parameter count with `requires_grad=False`.
- `num_param_tensors`: Number of `nn.Parameter` tensor objects.
- `num_layers_with_params`: Number of Layers that use parameters.
- `num_ops_with_params`: Number of Op invocations that use parameters.
- `num_backward_passes`: Number of backward passes logged into this Trace.
### Memory
- `total_activation_memory`: Bytes for all Op outputs, whether saved or not.
- `total_activation_memory_str`: Human-readable form of `total_activation_memory`.
- `saved_activation_memory`: Bytes for Op outputs actually saved by TorchLens.
- `saved_activation_memory_str`: Human-readable form of `saved_activation_memory`.
- `total_gradient_memory`: Bytes for all Op output gradients computed during backward.
- `total_gradient_memory_str`: Human-readable form of `total_gradient_memory`.
- `saved_gradient_memory`: Bytes for Op output gradients actually saved by TorchLens.
- `saved_gradient_memory_str`: Human-readable form of `saved_gradient_memory`.
- `param_memory`: Bytes used by all model parameter tensors.
- `param_memory_str`: Human-readable form of `param_memory`.
- `total_param_gradient_memory`: Bytes for all populated parameter gradients.
- `total_param_gradient_memory_str`: Human-readable form of `total_param_gradient_memory`.
- `forward_peak_memory`: Peak observed memory during forward; exact peak semantics depend on backend.
- `forward_peak_memory_str`: Human-readable form of `forward_peak_memory`.
- `backward_peak_memory`: Peak observed memory during backward; exact peak semantics depend on backend.
- `backward_peak_memory_str`: Human-readable form of `backward_peak_memory`.
- `autograd_saved_memory`: Bytes PyTorch autograd retained for backward computation.
- `autograd_saved_memory_str`: Human-readable form of `autograd_saved_memory`.
### Layer and Op Collections
- `layer_labels`: Layer labels in execution order, without `:N` pass suffixes.
- `op_labels`: Op labels in execution order, always including `:N` pass suffixes.
- `layers`: Accessor for Layers; accepts bare Layer labels and pass-qualified Op labels.
- `ops`: Accessor for Ops; use pass-qualified labels such as `conv2d_1_2:1`.
Example: `trace.layers["conv2d_1_2"]` returns the Layer; `trace.ops["conv2d_1_2:1"]` returns its first Op.
- `input_layers`: Layers representing external model inputs.
- `output_layers`: Layers representing model outputs.
- `buffer_layers`: Layers representing buffers as graph boundary values.
- `internal_source_ops`: Ops that begin an internal subgraph without normal external input ancestry.
- `internal_sink_ops`: Ops that end an internal subgraph before reaching normal outputs.
- `internally_terminated_bool_ops`: Internal sink Ops that produced terminal booleans for control flow.
- `orphan_ops`: Ops that TorchLens could not connect to the main captured graph.
- `unlogged_ops`: Ops known to exist but not fully logged.
- `layers_with_params`: Layer labels for Layers that use parameters.
- `ops_with_saved_activations`: Op labels whose forward outputs were saved.
- `ops_with_saved_gradients`: Op labels whose gradients were saved.
- `layers_with_saved_activations`: Derived Layer-label view of `ops_with_saved_activations`.
- `layers_with_saved_gradients`: Derived Layer-label view of `ops_with_saved_gradients`.
- `equivalent_ops`: Groups of Op labels that loop detection considers structurally equivalent.
### Module, Param, Buffer, and Grad Accessors
- `root_module`: The Module record for the top-level model, usually at address `self`.
- `modules`: Accessor for Module records by address, alias, call label, or ordinal index.
- `uncalled_modules`: Module addresses registered on the source model but not called in this capture.
- `params`: Accessor for Param records.
- `buffers`: Accessor for Buffer records.
- `grad_fns`: Accessor for GradFn records.
### Topology
- `is_branching`: True when any Layer has more than one child in the captured graph.
- `is_recurrent`: True when any Layer has more than one Op.
- `max_layer_op_count`: Maximum number of Ops aggregated by any single Layer.
- `is_dynamic_graph`: True when execution path depends on tensor values.
- `has_backward_pass`: True when backward-pass data has been logged.
- `has_gradients`: True when gradient values are present.
### Backward and GradFn Anchors
- `backward_memory_backend`: Backend used to measure backward memory, such as `cuda`, `cpu`, `mps`, or `unknown`.
- `backward_root_grad_fn_id`: Runtime Python id of the root grad-fn from the loss tensor.
### Compute
- `total_flops`: Approximate total forward plus backward FLOPs represented by the Trace.
- `total_flops_forward`: Approximate forward FLOPs.
- `total_flops_backward`: Approximate backward FLOPs.
- `total_flops_str`: Human-readable form of `total_flops`.
- `total_flops_forward_str`: Human-readable form of `total_flops_forward`.
- `total_flops_backward_str`: Human-readable form of `total_flops_backward`.
- `total_macs`: Approximate total multiply-accumulate count, currently derived from FLOPs.
- `total_macs_forward`: Approximate forward MACs.
- `total_macs_backward`: Approximate backward MACs.
- `total_macs_str`: Human-readable form of `total_macs`.
- `total_macs_forward_str`: Human-readable form of `total_macs_forward`.
- `total_macs_backward_str`: Human-readable form of `total_macs_backward`.
- `flops_by_type`: Dictionary summarizing FLOPs by normalized operation type.
- `macs_by_type`: Dictionary summarizing MACs by normalized operation type.
### Capture State
- `trace_annotations`: User-attached metadata about the whole Trace.
- `input_id`: Runtime `id()` of the input tensor or primary input structure at capture time.
- `input_shape_hash`: Hash of input shapes, dtypes, and devices for comparability checks.
- `input_annotations`: User-attached metadata about the inputs.
- `random_seed`: RNG seed actually applied for capture, if any.
- `param_hash_quick`: Quick hash of parameter values for relationship checks.
- `param_hash_full`: Full hash of parameter values for stronger verification.
- `graph_shape_hash`: Hash of captured graph topology and shape-relevant structure.
- `forward_source_file`: File where the model's `forward()` method is defined, when available.
- `forward_source_line`: Source line number for the model `forward()` definition, when available.
- `forward_source_location`: Combined `forward_source_file:forward_source_line` editor jump string.
- `model_source_file`: File where the model class is defined, when available.
- `model_source_line`: Line where the model class is defined, when available.
- `model_source_location`: Combined `model_source_file:model_source_line` editor jump string.
- `ledger`: Audit trail of fork, replay, rerun, and direct-write operations.
- `last_run_ctx`: Inspectable context from the most recent replay or rerun operation.
- `is_appended`: True when this Trace currently holds appended chunks. Cleared on non-append `rerun()`; preserved across `replace_run_state_from`.
- `append_history`: List of per-append provenance dicts recording inputs, hashes, and metadata for each `rerun(append=True)` chunk.
- `_append_sequence_id` (?): Monotonic id incremented on each append. Cleared on non-append rerun, preserved by `replace_run_state_from`. Underscore-prefixed but `FieldPolicy.KEEP` for save.
- `_grad_fn_param_refs` (?): `dict[str, str]` mapping AccumulateGrad grad_fn labels to parameter addresses. Built at capture; `FieldPolicy.KEEP`. Used by visualization to annotate AccumulateGrad nodes with the parameter they accumulate into.
- `_param_log_by_pid` (?): `dict[int, str]` mapping `id(param)` to parameter address at capture time. `FieldPolicy.DROP` (capture-time only). Used by backward capture to attribute grad_fns to parameter sinks.
### Capture Config
- `save_raw_activations`: Whether raw untransformed activation tensors are saved.
- `save_raw_gradients`: Whether raw untransformed gradient tensors are saved.
- `save_grads`: Whether TorchLens should retain observed operation gradients after backward.
- `save_arg_values`: Whether Op function arguments are saved.
- `save_rng_states`: Whether RNG state snapshots are saved.
- `save_code_context`: Whether source-code context around calls is saved.
- `save_grads_policy`: Predicate or boolean policy snapshot used for backward gradient saving.
- `differentiable`: Whether the Trace retains autograd connectivity for later backward.
- `intervention_ready`: Whether the Trace was captured with enough state for interventions.
- `intervention_spec`: Current intervention specification associated with the Trace.
- `capture_mode`: Capture behavior mode, replacing old logging vocabulary.
- `detach_saved_activations`: Whether saved tensors are detached from autograd.
- `recurrence_detection`: Whether TorchLens performs loop detection.
- `emit_nvtx`: Whether capture emits NVIDIA NVTX ranges.
- `mark_layer_depths`: Whether topological input/output depth metadata is marked on Layers.
- `raise_on_nan`: Whether capture raises when NaNs are detected.
- `verbose`: Whether capture prints progress or diagnostic messages.
- `output_device`: Device where saved outputs should be placed.
- `module_filter`: Callable that decides which modules are eligible for module logging.
- `unsupported_ops`: Placeholder metadata for unsupported operation reporting.
- `activation_transform`: Callable applied to saved forward outputs before transformed storage.
- `gradient_transform`: Callable applied to saved gradients before transformed storage.
- `capture_args_template`: Whether full capture arguments are retained for cache/replay workflows.
### Timing
- `start_time`: Epoch timestamp when capture started.
- `end_time`: Epoch timestamp when capture ended.
- `start_time_str`: Human-readable form of `start_time`.
- `end_time_str`: Human-readable form of `end_time`.
- `setup_duration`: Seconds spent in capture setup.
- `forward_duration`: Seconds spent running the model forward pass.
- `func_calls_duration`: Seconds spent inside user-visible function calls during capture.
- `cleanup_duration`: Seconds spent in cleanup.
- `setup_duration_str`: Human-readable form of `setup_duration`.
- `forward_duration_str`: Human-readable form of `forward_duration`.
- `func_calls_duration_str`: Human-readable form of `func_calls_duration`.
- `cleanup_duration_str`: Human-readable form of `cleanup_duration`.
- `total_duration`: Total elapsed capture time.
- `total_duration_str`: Human-readable form of `total_duration`.
- `overhead_duration`: TorchLens overhead duration.
- `overhead_duration_str`: Human-readable form of `overhead_duration`.
### Trace Methods
- `backward(loss)`: Run backward from an explicit loss and populate grad-related fields.
- `find_layers(query, *, limit=10)`: Find Layer labels matching a query.
- `fork(name=None)`: Duplicate the Trace with a fresh intervention spec and optional new name.
- `rerun(model, x, **kwargs)`: Re-execute the model and update or append Trace state.
- `replay(**kwargs)`: Replay saved activations without re-executing the model.
- `draw(...)`: Draw the forward graph, either for display or saved output.
- `draw_backward(...)`: Draw the backward grad-fn graph.
- `cleanup()`: Clear circular references and runtime-only heavyweight objects.
- `summary(level, ...)`: Return a textual summary of the Trace.
- `save(path, **kwargs)`: Save the Trace to portable TorchLens I/O.
- `load(path, **kwargs)`: Load a saved Trace.
- `to_pandas()`: Export Trace fields to a pandas DataFrame.
- `to_csv(path, **kwargs)`: Write Trace export data to CSV.
- `to_parquet(path, **kwargs)`: Write Trace export data to Parquet.
- `to_json(path, **kwargs)`: Write Trace export data to JSON.
### Conditional Flow
- `conditionals`: `ConditionalAccessor` for conditional structures by integer ordinal or stable conditional id.
- `has_conditionals`: True when this Trace captured at least one conditional.
- `num_conditionals`: Number of conditionals captured in this Trace.
- `Conditional`: One if-chain at one source location.
- `Conditional.id`: Stable id such as `cond_gt_1_4`, derived from the leading terminal bool Op label.
- `Conditional.arms`: Ordered list of `ConditionalArm` records: leading `then`, zero or more `elif`, and optional `else`.
- `Conditional.fired_arm_index`: Index of the arm whose body ran, or `None` if no arm fired.
- `Conditional.fired_arm_kind`: Denormalized fired arm kind: `then`, `elif`, `else`, or `None`.
- `Conditional.source_file`: File containing the if-statement, when known.
- `Conditional.source_line`: Line of the `if` keyword, when known.
- `Conditional.source_location`: Combined `source_file:source_line`, or `None` if incomplete.
- `Conditional.fired_arm`: Direct access to the fired `ConditionalArm`, or `None`.
- `Conditional.has_else`: True when an else arm exists.
- `Conditional.has_elif`: True when at least one elif arm exists.
- `Conditional.num_arms`: Number of arms.
- `Conditional.num_elifs`: Number of elif arms.
- `ConditionalArm.kind`: Arm kind: `then`, `elif`, or `else`.
- `ConditionalArm.evaluation_op_labels`: Op labels computing this arm's condition; empty for `else`.
- `ConditionalArm.terminal_bool_op_label`: Final scalar bool Op for the arm, or `None` for `else`.
- `ConditionalArm.bool_value_at_run`: Bool value observed when evaluated, or `None`.
- `ConditionalArm.condition_evaluated`: True when this arm's condition was reached.
- `ConditionalArm.evaluation_entry_edge`: Edge into the first evaluation Op; used by visualization for IF/elif labels.
- `ConditionalArm.execution_op_labels`: Op labels in this arm's body; empty if the body did not run.
- `ConditionalArm.fired`: True when this arm's body ran.
- `ConditionalArm.execution_entry_edge`: Edge into the first body Op; used by visualization for THEN/ELSE labels.
- `ConditionalRoleRef`: One Op's participation in a conditional arm.
- `ConditionalRoleRef.conditional_id`: Conditional this Op participates in.
- `ConditionalRoleRef.arm_index`: Index into `conditional.arms`.
- `ConditionalRoleRef.arm_kind`: Denormalized arm kind: `then`, `elif`, or `else`.
- `ConditionalRoleRef.role`: `evaluation` or `body`.
### Trace Intervention Methods
Intervention method names are intentionally not promoted as final here; the audit deferred them to the integrated `Site` concept review.
## Op (one captured tensor operation)
### Identity and Labeling
- `label`: Pass-qualified Op label such as `conv2d_1_2:1`.
- `label_short`: Short label omitting the overall execution index where available.
- `type`: Normalized operation type token such as `conv2d`.
- `type_index`: 1-based position among Ops or Layers of the same type.
- `trace_index`: 1-based position in full forward execution order.
- `pass_index`: 1-based pass number within the parent Layer.
- `compute_index`: 1-based Op invocation number.
- `capture_index`: Position in tensor creation order.
- `num_passes`: Number of Op passes in the parent Layer.
- `layer_label`: Parent Layer label without `:N`.
- `trace`: Back-pointer to the owning Trace; runtime-only.
### Function Identity
- `func`: Runtime callable that produced this Op's output.
- `func_name`: Short callable name, preserving in-place suffixes when present.
- `func_qualname`: Fully qualified callable name for verification and display.
- `is_inplace`: True when the underlying PyTorch operation was in-place.
- `arg_names`: Names of callable arguments.
- `num_args_total`: Total number of positional and keyword arguments.
- `num_pos_args`: Number of positional arguments.
- `num_kwargs`: Number of keyword arguments.
- `grad_fn_name`: Name of the PyTorch autograd node attached to this Op output.
- `grad_fn_id`: Runtime id of the PyTorch autograd node.
- `grad_fn`: Runtime autograd handle, matching PyTorch `tensor.grad_fn`.
- `grad_fn_log`: TorchLens GradFn record corresponding to this Op, when backward data exists.
### Tensor Properties
- `out`: Saved forward tensor output of this Op when TorchLens saved it.
- `shape`: Shape of `out`.
- `dtype`: Dtype of `out`.
- `memory`: Bytes used by `out`.
- `memory_str`: Human-readable form of `memory`.
- `grad`: Saved gradient of the Op output after backward, when available.
- `derived_grad`: tinygrad-only derived intermediate gradient from `trace.intermediate_derived_grads`, when `GradOptions(intermediate_grads=True)` produced an exact unambiguous match. This is not a saved true-backward gradient.
- `grad_shape`: Shape of `grad`.
- `grad_dtype`: Dtype of `grad`.
- `grad_memory`: Bytes used by `grad`.
- `grad_memory_str`: Human-readable form of `grad_memory`.
- `transformed_out`: Saved transformed version of `out`, after `activation_transform`.
- `transformed_out_shape`: Shape of `transformed_out`.
- `transformed_out_dtype`: Dtype of `transformed_out`.
- `transformed_out_memory`: Bytes used by `transformed_out`.
- `transformed_out_memory_str`: Human-readable form of `transformed_out_memory`.
- `transformed_grad`: Saved transformed version of `grad`, after `gradient_transform`.
- `transformed_grad_shape`: Shape of `transformed_grad`.
- `transformed_grad_dtype`: Dtype of `transformed_grad`.
- `transformed_grad_memory`: Bytes used by `transformed_grad`.
- `transformed_grad_memory_str`: Human-readable form of `transformed_grad_memory`.
- `autograd_saved_memory`: Bytes retained by PyTorch autograd for this Op.
- `autograd_saved_memory_str`: Human-readable form of `autograd_saved_memory`.
- `num_autograd_saved_tensors`: Number of tensors PyTorch autograd retained for this Op.
- `multi_output_index`: 0-based position when this Op came from a multi-output call; `None` otherwise.
- `multi_output_role` (?): Optional semantic role tag for this Op's position in a multi-output container (e.g., tuple element name or dict key). `None` when not from a multi-output call or when no role label was captured.
- `is_multi_output`: True when `multi_output_index` is not `None`.
### Per-Op Config and Saved State
- `output_device`: Device where saved tensors for this Op were placed.
- `activation_transform`: Transform used for the Op's saved output.
- `gradient_transform`: Transform used for the Op's saved gradient.
- `annotations`: User-attached metadata about this Op.
- `detach_saved_activations`: Whether saved tensors were detached.
- `save_grads`: Whether gradient saving was requested for this Op.
- `has_saved_activation`: True when this Op's forward output was saved.
- `has_saved_gradient`: True when this Op's gradient was saved.
- `has_saved_args`: True when function arguments were saved.
- `saved_args`: Saved positional argument values.
- `saved_kwargs`: Saved keyword argument values.
- `args_template`: Structured template for saved positional arguments, used by replay/intervention.
- `kwargs_template`: Structured template for saved keyword arguments.
- `non_tensor_pos_args`: Non-tensor positional arguments passed to the function.
- `non_tensor_kwargs`: Non-tensor keyword arguments passed to the function.
- `container_path`: Path inside a structured return container for this output; for example, index `1` of a tuple return or key `"hidden"` in a dict return.
- `container_spec`: Structural description of the return container, used to rebuild where this output came from.
### Timing, RNG, and Call Context
- `code_context`: Python call-stack frames showing where in user/PyTorch code this Op function ran.
- `func_duration`: Time spent in the function call for this Op.
- `func_rng_states`: RNG state snapshots associated with this Op.
- `func_autocast_state`: Autocast state active during the function call.
- `lookup_keys`: Alternate lookup keys that can resolve to this Op.
### Compute
- `flops_forward`: Approximate forward FLOPs for this Op.
- `flops_forward_str`: Human-readable form of `flops_forward`.
- `flops_backward`: Approximate backward FLOPs for this Op.
- `flops_backward_str`: Human-readable form of `flops_backward`.
- `flops_total`: Approximate forward plus backward FLOPs for this Op.
- `flops_total_str`: Human-readable form of `flops_total`.
- `macs_forward`: Approximate forward multiply-accumulate operations (MACs).
- `macs_forward_str`: Human-readable form of `macs_forward`.
- `macs_backward`: Approximate backward multiply-accumulate operations (MACs).
- `macs_backward_str`: Human-readable form of `macs_backward`.
- `macs_total`: Approximate forward plus backward multiply-accumulate operations (MACs).
- `macs_total_str`: Human-readable form of `macs_total`.
### Parameters
- `param_shapes`: Shapes of parameters consumed by this Op.
- `param_names`: Short parameter names consumed by this Op.
- `param_dtypes`: Dtypes of parameters consumed by this Op.
- `num_params`: Number of scalar parameters consumed by this Op.
- `num_params_trainable`: Number of trainable scalar parameters consumed by this Op.
- `num_params_frozen`: Number of frozen scalar parameters consumed by this Op.
- `num_param_tensors`: Number of parameter tensors consumed by this Op.
- `param_memory`: Bytes used by consumed parameters.
- `param_memory_str`: Human-readable form of `param_memory`.
- `params`: Accessor or list-like view of Param records consumed by this Op.
- `uses_params`: True when this Op consumes any parameters.
### Equivalence and Recurrence
- `equivalence_class`: Loop-detection equivalence class for this Op.
- `equivalent_ops`: Other Op labels in the same equivalence class.
- `recurrent_ops`: Op labels confirmed as repeated executions of the same recurrent graph position.
### Role Flags
- `is_input`: True when this Op represents an external input boundary.
- `is_output`: True when this Op represents an output boundary node.
- `is_final_output`: True when this Op is the final model output marker.
- `is_buffer`: True when this Op represents a buffer boundary.
- `buffer_address`: Dotted path for the buffer when `is_buffer` is true.
- `buffer_source`: Source Layer or Op that wrote this buffer value, or `None` for static buffers.
- `is_internal_source`: True when this Op starts an internally generated graph region.
- `is_internal_sink`: True when this Op terminates an internal graph region.
- `is_terminal_bool`: True when this Op is a terminal boolean used by control flow.
- `is_scalar_bool`: True when the Op output is a scalar boolean.
- `bool_value`: Boolean value when `is_scalar_bool` is true.
- `in_conditionals`: List of `ConditionalRoleRef` records describing this Op's conditional roles.
- `is_in_conditional`: True when this Op participates in any conditional role.
- `is_in_conditional_evaluation`: True when this Op computes a conditional arm condition.
- `is_in_conditional_body`: True when this Op is in a conditional arm body.
- `conditional_depth`: Number of distinct conditionals this Op participates in.
- `terminal_bool_for`: `(conditional_id, arm_index)` when this Op is the terminal bool of an arm, else `None`.
- `is_atomic_module_op`: True when this Op is the sole operation output of an atomic module call.
- `is_submodule_output`: True when this Op is an output of any submodule.
### Module Containment
- `module`: Module containing this Op.
- `module_call_stack`: Dynamic stack of Module calls active for this Op.
- `output_of_modules`: Modules for which this Op is an output value.
- `input_to_modules`: Modules for which this Op is an input value.
- `output_of_module_calls`: ModuleCall labels for calls this Op outputs from.
- `input_to_module_calls`: ModuleCall labels for calls this Op inputs to.
- `module_entry_argnames`: Argument names used when entering a module boundary.
- `atomic_module_call`: ModuleCall for an atomic module output, when applicable.
- `atomic_module`: Module for an atomic module output, when applicable.
### Graph Relations
At Layer scope, relation lists use bare Layer labels; at Op scope, relation lists may use pass-qualified Op labels when per-pass precision matters.
- `parents`: Immediate parent Layer or Op labels feeding this Op.
- `children`: Immediate child Layer or Op labels consuming this Op.
- `siblings`: Ops or Layers sharing graph parents with this Op.
- `co_parents`: Other parents that jointly feed this Op's children.
- `has_parents`: True when this Op has parents.
- `has_children`: True when this Op has children.
- `has_siblings`: True when this Op has siblings.
- `has_co_parents`: True when this Op has co-parents.
- `has_input_ancestor`: True when an input boundary reaches this Op.
- `has_output_descendant`: True when this Op reaches an output boundary.
- `parent_arg_positions`: Positions in the function call where parent tensors were used.
- `is_output_parent`: True when this tensor was directly returned from `forward`.
- `output_descendants`: Output nodes reachable from this Op.
- `min_distance_from_output`: Shortest graph distance from this Op to an output.
- `max_distance_from_output`: Longest graph distance from this Op to an output.
- `input_ancestors`: Input nodes that can reach this Op.
- `min_distance_from_input`: Shortest graph distance from an input to this Op.
- `max_distance_from_input`: Longest graph distance from an input to this Op.
- `root_ancestors`: Root graph ancestors for this Op.
- `has_internal_source_ancestor`: True when this Op descends from an internal source.
- `internal_source_parents`: Immediate parents that are internal sources.
- `internal_source_ancestors`: Internal-source ancestors reachable upstream.
### Output Variations and Interventions
- `has_output_variations`: True when different children observed different versions of this Op output.
- `output_versions_per_child`: Per-child record of output versions seen by downstream consumers.
- `interventions`: Intervention records applied to this Op.
### Op Methods
- `save_activation(...)`: Public method for saving tensor payloads; final naming deferred if intervention survey changes tensor-save workflow.
- `to_pandas()`: Export this Op as a one-row DataFrame.
- `to_csv(path, **kwargs)`: Write this Op row to CSV.
- `to_parquet(path, **kwargs)`: Write this Op row to Parquet.
- `to_json(path, **kwargs)`: Write this Op row to JSON.
## Layer (aggregate over one or more ops)
### Identity and Labeling
- `label`: Layer label without a pass suffix, such as `conv2d_1_2`.
- `label_short`: Short Layer label without the overall execution index.
- `type`: Normalized operation type token for this Layer.
- `type_index`: 1-based position within Layers of the same type.
- `trace_index`: 1-based position in full forward execution order.
- `num_ops`: Number of Ops aggregated by this Layer.
- `trace`: Back-pointer to the owning Trace; runtime-only.
### Function Identity Passthroughs
These entries follow the single-Op passthrough rule: they read naturally on one-Op Layers and direct users to `layer.ops[n]` for recurrent or otherwise multi-Op Layers.
- `func`: Callable for the single Op when this Layer has one Op; raises for multi-Op Layers.
- `func_name`: Short callable name for the Layer's Op or stable callable identity.
- `func_qualname`: Fully qualified callable name when available.
- `is_inplace`: True when the underlying Op was in-place.
- `arg_names`: Function argument names.
- `num_args_total`: Total positional plus keyword argument count.
- `num_pos_args`: Number of positional arguments.
- `num_kwargs`: Number of keyword arguments.
- `grad_fn_name`: Autograd node name associated with this Layer's Op.
- `grad_fn_id`: Autograd node id for a single-Op Layer; use `layer.ops[n].grad_fn_id` for multi-Op Layers.
- `grad_fn`: Runtime autograd handle for a single-Op Layer.
- `grad_fn_log`: TorchLens GradFn record for a single-Op Layer.
### Tensor Properties
- `out`: Saved forward output for a single-Op Layer; raises for multi-Op Layers.
- `shape`: Output shape when stable or single-Op.
- `dtype`: Output dtype when stable or single-Op.
- `memory`: Output memory in bytes.
- `memory_str`: Human-readable memory.
- `grad`: Saved gradient for a single-Op Layer.
- `grad_shape`: Gradient shape.
- `grad_dtype`: Gradient dtype.
- `grad_memory`: Gradient memory in bytes.
- `grad_memory_str`: Human-readable gradient memory.
- `transformed_out`: Transformed saved output for a single-Op Layer.
- `transformed_out_shape`: Shape of transformed output.
- `transformed_out_dtype`: Dtype of transformed output.
- `transformed_out_memory`: Memory for transformed output.
- `transformed_out_memory_str`: Human-readable transformed output memory.
- `transformed_grad`: Transformed gradient for a single-Op Layer.
- `transformed_grad_shape`: Shape of transformed gradient.
- `transformed_grad_dtype`: Dtype of transformed gradient.
- `transformed_grad_memory`: Memory for transformed gradient.
- `transformed_grad_memory_str`: Human-readable transformed gradient memory.
- `autograd_saved_memory`: Bytes retained by autograd for this Layer's Op or Ops.
- `autograd_saved_memory_str`: Human-readable autograd saved memory.
- `num_autograd_saved_tensors`: Number of autograd-saved tensors.
- `multi_output_index`: Multi-output index for a single-Op Layer.
- `is_multi_output`: True when the output came from a multi-output call.
### Per-Layer Config and Saved State
- `output_device`: Device where saved Layer tensors were placed.
- `activation_transform`: Transform applied to saved outputs.
- `gradient_transform`: Transform applied to saved gradients.
- `annotations`: User-attached metadata for this Layer.
- `detach_saved_activations`: Whether saved tensors were detached for this Layer.
- `save_grads`: Whether gradient saving was requested for this Layer.
- `has_saved_activations`: True when any Op in this Layer has a saved forward output.
- `has_saved_gradients`: True when any Op in this Layer has a saved gradient.
- `saved_args`: Saved positional arguments for a single-Op Layer.
- `saved_kwargs`: Saved keyword arguments for a single-Op Layer.
- `args_template`: Structured argument template for replay/intervention.
- `kwargs_template`: Structured keyword template for replay/intervention.
- `code_context`: Python call-stack frames for a single-Op Layer.
- `func_duration`: Function-call duration for a single-Op Layer.
- `func_rng_states`: RNG state snapshots for a single-Op Layer.
- `lookup_keys`: Alternate lookup keys for this Layer.
### Compute
- `flops_forward`: Approximate forward FLOPs for this Layer.
- `flops_forward_str`: Human-readable forward FLOPs.
- `flops_backward`: Approximate backward FLOPs for this Layer.
- `flops_backward_str`: Human-readable backward FLOPs.
- `flops_total`: Approximate forward plus backward FLOPs.
- `flops_total_str`: Human-readable total FLOPs.
- `macs_forward`: Approximate forward MACs.
- `macs_forward_str`: Human-readable forward MACs.
- `macs_backward`: Approximate backward MACs.
- `macs_backward_str`: Human-readable backward MACs.
- `macs_total`: Approximate total MACs.
- `macs_total_str`: Human-readable total MACs.
### Parameters
- `param_shapes`: Shapes of parameters used by this Layer.
- `param_names`: Short names of parameters used by this Layer.
- `param_dtypes`: Dtypes of parameters used by this Layer.
- `num_params`: Number of scalar parameters used by this Layer.
- `num_params_trainable`: Number of trainable scalar parameters used by this Layer.
- `num_params_frozen`: Number of frozen scalar parameters used by this Layer.
- `num_param_tensors`: Number of parameter tensors used by this Layer.
- `param_memory`: Bytes used by parameters for this Layer.
- `param_memory_str`: Human-readable parameter memory.
- `params`: Param accessor for parameters used by this Layer.
- `uses_params`: True when this Layer uses parameters.
### Equivalence and Roles
- `equivalence_class`: Loop-detection equivalence class for this Layer.
- `equivalent_ops`: Op labels grouped with this Layer's Ops by loop-detection equivalence analysis.
- `is_input`: True when this Layer is an input boundary.
- `is_output`: True when this Layer is an output boundary.
- `is_final_output`: True when this Layer marks the final model output.
- `is_buffer`: True when this Layer represents a buffer.
- `buffer_address`: Buffer path when this Layer is a buffer boundary.
- `buffer_overwrite_index`: Which use/update of the buffer this Layer represents.
- `is_internal_source`: True when this Layer starts an internal graph region.
- `is_internal_sink`: True when this Layer ends an internal graph region.
- `is_terminal_bool`: True when this Layer is a terminal boolean for control flow.
- `is_scalar_bool`: True when this Layer output is a scalar boolean.
- `bool_value`: Boolean value when `is_scalar_bool` is true.
- `in_conditionals`: List of `ConditionalRoleRef` records aggregated from this Layer's Ops.
- `is_in_conditional`: True when this Layer participates in any conditional role.
- `is_in_conditional_evaluation`: True when this Layer computes a conditional arm condition.
- `is_in_conditional_body`: True when this Layer is in a conditional arm body.
- `conditional_depth`: Number of distinct conditionals this Layer participates in.
- `terminal_bool_for`: `(conditional_id, arm_index)` when this Layer is the terminal bool of an arm, else `None`.
- `is_atomic_module_op`: True when this Layer is the sole output Op of an atomic module call.
### Module Containment
- `module`: Module containing this Layer.
- `module_call_stack`: Dynamic Module call stack containing this Layer.
- `output_of_modules`: Modules for which this Layer is an output.
- `input_to_modules`: Modules for which this Layer is an input.
- `output_of_module_calls`: ModuleCall labels this Layer outputs from.
- `input_to_module_calls`: ModuleCall labels this Layer inputs to.
- `in_submodule`: True when this Layer was computed inside a submodule rather than directly in top-level forward.
- `module_call_depth`: Depth in the dynamic Module call stack.
### Graph Relations
- `children`: Child Layer labels.
- `parents`: Parent Layer labels.
- `siblings`: Sibling Layer labels that share parents.
- `co_parents`: Other parent Layers that share children.
- `has_children`: True when this Layer has children.
- `has_parents`: True when this Layer has parents.
- `has_siblings`: True when this Layer has siblings.
- `has_co_parents`: True when this Layer has co-parents.
- `has_input_ancestor`: True when this Layer descends from an input.
- `has_output_descendant`: True when this Layer reaches an output.
### Pass Management
- `ops`: Scoped `OpAccessor` for this Layer's Ops, keyed by 1-based pass index or Op label.
- `op_labels`: Op labels belonging to this Layer.
### Layer Methods
- `to_pandas()`: Export this Layer as a one-row DataFrame.
- `to_csv(path, **kwargs)`: Write this Layer row to CSV.
- `to_parquet(path, **kwargs)`: Write this Layer row to Parquet.
- `to_json(path, **kwargs)`: Write this Layer row to JSON.
## Module / ModuleCall
### ModuleCall Identity
- `call_index`: 1-based invocation index for this Module call.
- `call_label`: Pass-qualified ModuleCall label, usually `address:N`.
- `address`: Primary module address for this call.
- `all_addresses`: All module addresses sharing the same module object.
- `has_multiple_addresses`: True when the module object appears at multiple addresses.
### ModuleCall Layers and Args
- `layers`: Pass-qualified Op labels computed during this ModuleCall.
- `num_layers`: Number of Layers or Ops associated with this call.
- `input_layers`: Pass-qualified input Layer labels for this call.
- `output_layers`: Pass-qualified output Layer labels for this call.
- `outputs`: List of `OpLog` records that are the actual outputs of this call. Stored alongside `output_layers` because the OpLog list captures structural information (container path, multi-output role) that string labels alone cannot.
- `output_structure`: `ContainerSpec | None` describing the shape of the call's return container. Used by intervention rerun to rebuild tuple/dict/dataclass outputs.
- `forward_args`: Positional arguments passed to `forward`.
- `forward_kwargs`: Keyword arguments passed to `forward`.
- `forward_args_summary`: Human-readable summary of forward positional arguments.
- `forward_kwargs_summary`: Human-readable summary of forward keyword arguments.
- `call_parent`: Parent ModuleCall label in the dynamic call tree.
- `call_children`: Child ModuleCall labels in the dynamic call tree.
### ModuleCall Output Passthroughs
These properties resolve through the output Layers/Ops. Singular forms require exactly one output and raise `MultiOutputModuleError` on multi-output calls; plural forms return lists in container-path order.
- `out` / `outs`: Saved output tensor or output tensors.
- `out_shape` / `out_shapes`: Output shape or shapes.
- `out_dtype` / `out_dtypes`: Output dtype or dtypes.
- `out_memory` / `out_memories`: Output memory or memories in bytes.
- `out_memory_str` / `out_memories_str`: Human-readable output memory.
- `transformed_out` / `transformed_outs`: Transformed saved output tensor or tensors.
- `transformed_out_shape` / `transformed_out_shapes`: Transformed output shape or shapes.
- `transformed_out_dtype` / `transformed_out_dtypes`: Transformed output dtype or dtypes.
- `transformed_out_memory` / `transformed_out_memories`: Transformed output memory or memories.
- `transformed_out_memory_str` / `transformed_out_memories_str`: Human-readable transformed output memory.
- `grad` / `grads`: Saved output gradient or gradients.
- `grad_shape` / `grad_shapes`: Gradient shape or shapes.
- `grad_dtype` / `grad_dtypes`: Gradient dtype or dtypes.
- `grad_memory` / `grad_memories`: Gradient memory or memories.
- `grad_memory_str` / `grad_memories_str`: Human-readable gradient memory.
- `transformed_grad` / `transformed_grads`: Transformed saved gradient or gradients.
- `transformed_grad_shape` / `transformed_grad_shapes`: Transformed gradient shape or shapes.
- `transformed_grad_dtype` / `transformed_grad_dtypes`: Transformed gradient dtype or dtypes.
- `transformed_grad_memory` / `transformed_grad_memories`: Transformed gradient memory or memories.
- `transformed_grad_memory_str` / `transformed_grad_memories_str`: Human-readable transformed gradient memory.
### Module Identity
- `address`: Primary dotted PyTorch module address.
- `all_addresses`: All addresses for a shared module object.
- `has_multiple_addresses`: True when a module object has multiple addresses.
- `name`: Last address segment, exposed as a derived property.
- `class_name`: Short Python class name for this Module.
- `cls`: Runtime Python type object for the Module; spelled `cls` because `class` is a Python keyword.
- `class_qualname`: Fully qualified Python class name.
- `trace`: Back-pointer to the owning Trace.
### Module Source Info
- `source_file`: File where the module class or source was defined, when available.
- `source_line`: Line number for the module source, when available.
- `source_location`: Combined `source_file:source_line` string for editor-style jumping.
- `class_docstring`: Docstring for the module class.
- `init_signature`: Signature of `__init__`, when introspectable.
- `init_docstring`: Docstring of `__init__`.
- `forward_signature`: Signature of `forward`, when introspectable.
- `forward_docstring`: Docstring of `forward`.
### Module Hierarchy
The address tree is the static `nn.Module` registration hierarchy; the call tree is what actually happened as modules called one another during `forward`.
- `address_parent`: Parent module address in the static `nn.Module` tree.
- `address_children`: Child module addresses in the static tree.
- `address_depth`: Depth in the static address tree.
- `call_parent`: Parent Module in the dynamic call tree for single-call Modules.
- `call_children`: Child Modules in the dynamic call tree.
- `call_depth`: Depth in the dynamic call stack.
- `num_calls`: Number of times this Module was called.
- `calls`: Scoped `ModuleCallAccessor` for this Module's calls.
- `call_labels`: Labels for this Module's calls.
### Module Parameters and Buffers
- `params`: ParamAccessor for parameters owned by this Module.
- `num_params`: Scalar parameter count owned by this Module.
- `num_param_tensors`: Number of parameter tensors owned by this Module.
- `num_params_trainable`: Trainable scalar parameter count.
- `num_params_frozen`: Frozen scalar parameter count.
- `params_memory`: Bytes used by this Module's parameters.
- `params_memory_str`: Human-readable parameter memory.
- `has_trainable_params`: True when any owned parameter is trainable.
- `buffer_layers`: Deferred to the integrated buffer rethink.
### Module State
- `is_train_mode`: True when the Module was in PyTorch train mode at capture.
- `forward_pre_hook_info`: `HookInfo` for registered forward pre-hooks.
- `forward_hook_info`: `HookInfo` for registered forward hooks.
- `backward_pre_hook_info`: `HookInfo` for registered backward pre-hooks.
- `backward_hook_info`: `HookInfo` for registered backward hooks.
- `full_backward_pre_hook_info`: `HookInfo` for registered full backward pre-hooks.
- `full_backward_hook_info`: `HookInfo` for registered full backward hooks.
- `has_forward_hooks`: Derived predicate for forward hook registries.
- `has_backward_hooks`: Derived predicate for backward hook registries.
- `HookInfo.count`: Number of hooks in a registry.
- `HookInfo.names`: Hook function `__name__` values.
- `HookInfo.qualnames`: Fully qualified hook names.
- `HookInfo.source_locations`: `FuncCallLocation` records for hook definitions, when introspectable.
- `custom_attributes`: User-defined instance attributes beyond standard `nn.Module` state.
- `custom_methods`: User-defined methods beyond inherited `nn.Module` methods.
### Module Compute
- `total_flops_forward`: Sum of forward FLOPs across Layers in this Module.
- `total_flops_backward`: Sum of backward FLOPs across Layers in this Module.
- `total_flops`: Sum of forward and backward FLOPs for this Module.
- `total_macs_forward`: Approximate forward MACs across this Module.
- `total_macs_backward`: Approximate backward MACs across this Module.
- `total_macs`: Approximate total MACs for this Module.
### Module Layer Access
- `layers`: Aggregate Layer labels inside this Module, using no-pass Layer labels.
- `num_layers`: Number of aggregate Layers inside this Module.
- `input_layers`: Aggregate union of input Layers across this Module's calls.
- `output_layers`: Aggregate union of output Layers across this Module's calls.
- `outputs`: Aggregate list of `OpLog` outputs across this Module's calls.
- `output_structure`: `ContainerSpec | None` for single-call Modules; reads through to the call's `output_structure`.
### Module Forward Args
- `forward_args`: Positional args for a single-call Module; raises for multi-call Modules.
- `forward_kwargs`: Keyword args for a single-call Module; raises for multi-call Modules.
- `forward_args_summary`: Human-readable args for a single-call Module.
- `forward_kwargs_summary`: Human-readable kwargs for a single-call Module.
### Module Output Passthroughs
These mirror ModuleCall output passthroughs for single-call Modules; multi-call Modules raise on singular access and require `module.calls[N]`.
- `out` / `outs`: Saved output tensor or output tensors.
- `out_shape` / `out_shapes`: Output shape or shapes.
- `out_dtype` / `out_dtypes`: Output dtype or dtypes.
- `out_memory` / `out_memories`: Output memory or memories in bytes.
- `out_memory_str` / `out_memories_str`: Human-readable output memory.
- `transformed_out` / `transformed_outs`: Transformed saved output tensor or tensors.
- `transformed_out_shape` / `transformed_out_shapes`: Transformed output shape or shapes.
- `transformed_out_dtype` / `transformed_out_dtypes`: Transformed output dtype or dtypes.
- `transformed_out_memory` / `transformed_out_memories`: Transformed output memory or memories.
- `transformed_out_memory_str` / `transformed_out_memories_str`: Human-readable transformed output memory.
- `grad` / `grads`: Saved output gradient or gradients.
- `grad_shape` / `grad_shapes`: Gradient shape or shapes.
- `grad_dtype` / `grad_dtypes`: Gradient dtype or dtypes.
- `grad_memory` / `grad_memories`: Gradient memory or memories.
- `grad_memory_str` / `grad_memories_str`: Human-readable gradient memory.
- `transformed_grad` / `transformed_grads`: Transformed saved gradient or gradients.
- `transformed_grad_shape` / `transformed_grad_shapes`: Transformed gradient shape or shapes.
- `transformed_grad_dtype` / `transformed_grad_dtypes`: Transformed gradient dtype or dtypes.
- `transformed_grad_memory` / `transformed_grad_memories`: Transformed gradient memory or memories.
- `transformed_grad_memory_str` / `transformed_grad_memories_str`: Human-readable transformed gradient memory.
### Module Methods
- `draw(**kwargs)`: Draw this Module's subgraph.
- `to_pandas()`: Export this Module or ModuleCall as tabular data.
- `to_csv(path, **kwargs)`: Write Module or ModuleCall data to CSV.
- `to_parquet(path, **kwargs)`: Write Module or ModuleCall data to Parquet.
- `to_json(path, **kwargs)`: Write Module or ModuleCall data to JSON.
## Param
### Identity
- `address`: Primary dotted PyTorch parameter address, matching `named_parameters()` style.
- `name`: Last segment of `address`, exposed as a derived property.
- `all_addresses`: All parameter addresses sharing the same `nn.Parameter`.
- `has_multiple_addresses`: True when the same parameter tensor is registered at multiple addresses.
- `trace`: Back-pointer to the owning Trace.
### Tensor Properties
- `shape`: Shape of the parameter tensor.
- `dtype`: Dtype of the parameter tensor.
- `num_params`: Number of scalar parameters in this tensor, equivalent to `numel()`.
- `memory`: Bytes used by the parameter tensor.
- `memory_str`: Human-readable parameter memory.
### Status Flags
- `is_trainable`: True when this parameter has `requires_grad=True`.
- `is_quantized`: True when this parameter uses a quantized dtype or representation.
- `has_optimizer`: True when optimizer metadata is associated with this parameter.
### Module Ownership
- `module_address`: Primary owning Module address.
- `module_class_name`: Short class name of the owning Module.
- `module_class_qualname`: Full Python class name of the owning Module.
- `all_module_addresses`: All owning Module addresses for shared parameters.
- `module`: Primary owning ModuleLog.
- `modules`: All owning ModuleLogs.
### Usage Tracking
- `num_uses`: Number of Op uses that referenced this parameter.
- `used_by_layers`: Layer or Op labels that used this parameter.
- `co_parent_params`: Parameters that co-occur with this one in the same operation; this is different from `all_addresses`, which means the same parameter object has multiple names.
### Gradient Family
- `has_grad`: True when a gradient was observed for this parameter.
- `grad`: Live `nn.Parameter.grad` tensor when the runtime parameter reference is available.
- `grad_shape`: Shape of the parameter gradient.
- `grad_dtype`: Dtype of the parameter gradient.
- `grad_memory`: Bytes used by the parameter gradient.
- `grad_memory_str`: Human-readable parameter-gradient memory.
### Param Methods
- `release_param_ref()`: Release the live parameter reference while keeping portable metadata.
- `to_pandas()`: Export this Param as a one-row DataFrame.
- `to_csv(path, **kwargs)`: Write this Param row to CSV.
- `to_parquet(path, **kwargs)`: Write this Param row to Parquet.
- `to_json(path, **kwargs)`: Write this Param row to JSON.
## Buffer
### Buffer Identity
- `buffer_address`: Dotted PyTorch buffer address; prefix is intentional because Buffer also behaves like an Op.
- `name`: Last segment of `buffer_address`.
- `module_address`: Owning module address derived from `buffer_address`.
- `all_buffer_addresses`: All addresses sharing the same buffer object.
- `has_multiple_addresses`: True when the same buffer appears at multiple addresses.
### Buffer Dynamics
- `buffer_overwrite_index`: 1-based index of this buffer use or update for its address.
- `buffer_source`: Layer or Op that wrote this buffer value, or `None` for static buffers.
### Inherited Op-Like Fields
These fields come from Buffer's Op-like role in the graph. `pass_index` is the generic Op pass index; `buffer_overwrite_index` is the buffer-specific "which use/update of this buffer" index.
- `label`: Op-like label for this buffer boundary record.
- `shape`: Shape of the buffer value.
- `dtype`: Dtype of the buffer value.
- `memory`: Bytes used by the buffer value.
- `memory_str`: Human-readable buffer memory.
- `has_saved_activation`: True when the buffer value was saved.
- `has_saved_gradient`: True when a gradient was saved for the buffer record.
- `pass_index`: 1-based pass index inherited from Op semantics.
- `module`: Module containing the buffer use.
- `module_call_stack`: Dynamic module call stack at the buffer use.
- `layer_label`: Layer label associated with the buffer record.
- `grad_shape`: Shape of saved gradient, if any.
- `grad_dtype`: Dtype of saved gradient, if any.
- `grad_memory`: Bytes used by saved gradient, if any.
- `grad_memory_str`: Human-readable gradient memory.
### Buffer Methods
- `to_pandas()`: Export this Buffer as tabular data.
- `to_csv(path, **kwargs)`: Write Buffer data to CSV.
- `to_parquet(path, **kwargs)`: Write Buffer data to Parquet.
- `to_json(path, **kwargs)`: Write Buffer data to JSON.
## GradFn / GradFnCall
### GradFnCall
- `call_index`: 1-based hook-firing or backward-call index for this GradFn.
- `duration`: Time spent in this GradFn call when duration capture is implemented.
- `grad_inputs`: Gradient inputs observed by the backward hook; current public field with no rename locked.
- `grad_outputs`: Gradient outputs observed by the backward hook; current public field with no rename locked.
- `to_pandas()`: Export this GradFnCall as a one-row DataFrame.
- `to_csv(path, **kwargs)`: Write GradFnCall data to CSV.
- `to_parquet(path, **kwargs)`: Write GradFnCall data to Parquet.
- `to_json(path, **kwargs)`: Write GradFnCall data to JSON.
### GradFn Identity
- `class_name`: Short class name of the PyTorch grad-fn object.
- `cls`: Runtime Python type object for the grad-fn, when introspectable; may be `None` after load for C++ autograd internals.
- `class_qualname`: Fully qualified class name of the grad-fn object.
- `grad_fn_label`: Stable TorchLens label for this GradFn.
- `trace`: Back-pointer to the owning Trace.
### GradFn Type Info
- `type`: Normalized grad-fn type token.
- `type_index`: 1-based index among grad-fns of the same type.
- `trace_index`: 1-based index across all grad-fns.
- `is_custom`: True when this GradFn comes from a custom autograd Function.
### GradFn Source Info
These fields are best-effort introspection data; built-in PyTorch grad-fns may have `None` values or only stub-file locations.
- `source_file`: File defining the grad-fn class or stub, when introspectable.
- `source_line`: Line defining the grad-fn class or stub, when introspectable.
- `source_location`: Combined `source_file:source_line` string.
- `class_docstring`: Docstring for the grad-fn class.
- `forward_signature`: Signature of the custom Function forward method, when available.
- `forward_docstring`: Docstring of the custom Function forward method.
- `backward_signature`: Signature of the custom Function backward method, when available.
- `backward_docstring`: Docstring of the custom Function backward method.
### GradFn Graph Relations
- `parents`: Upstream GradFn labels in the original forward graph orientation.
- `children`: Downstream GradFn labels, derived by reverse traversal.
- `siblings`: GradFns sharing a parent grad fn.
- `co_parents`: GradFns sharing a child grad fn.
- `has_parents`: True when this GradFn has parents.
- `has_children`: True when this GradFn has children.
- `has_siblings`: True when this GradFn has siblings.
- `has_co_parents`: True when this GradFn has co-parents.
- `op_label`: Stable Op label corresponding to this GradFn, if one exists.
- `op`: Op corresponding to `op_label`, if one exists.
- `is_intervening`: True when a grad-fn node has no corresponding forward op. Replaces the legacy `has_op` field (now a deprecated property returning `not is_intervening`).
- `has_op` (deprecated property): True when `op` is not None. Issues `DeprecationWarning`; prefer `not grad_fn.is_intervening`.
### GradFn Calls
- `num_calls`: Number of times this GradFn hook fired.
- `calls`: Scoped `GradFnCallAccessor` for this GradFn's calls.
- `call_labels`: Labels for this GradFn's calls.
### GradFn Methods
- `to_pandas()`: Export this GradFn as a one-row DataFrame.
- `to_csv(path, **kwargs)`: Write this GradFn row to CSV.
- `to_parquet(path, **kwargs)`: Write this GradFn row to Parquet.
- `to_json(path, **kwargs)`: Write this GradFn row to JSON.
## Bundle (cross-trace coordination)
### Bundle Members
- `traces`: TraceAccessor mapping Bundle member names to Trace objects.
- `trace_names`: Names of Traces in the Bundle.
- `baseline_name`: Optional name of the designated baseline Trace.
- `baseline`: Baseline Trace object, or `None` when no baseline is set.
- `capacity`: Maximum number of Traces retained by the Bundle; assign to set it.
### Bundle Graph Properties
- `supergraph`: Cross-trace graph that aligns comparable labels across Traces.
- `relationships`: Pairwise relationship dictionary for Trace comparisons.
- `relationship(a, b)`: Return the relationship for one pair of Trace names.
### Bundle State Management
- `add(trace, name=None)`: Add a Trace to the Bundle.
- `remove(name)`: Remove a named Trace from the Bundle.
- `remove_except(keep)`: Remove every Trace except the named Trace or names.
- `clear()`: Remove all Traces from the Bundle.
### Bundle Cross-Member Access
- `layers`: SuperLayerAccessor; `bundle.layers[label]` returns a SuperLayer.
- `ops`: SuperOpAccessor; `bundle.ops[label]` returns a SuperOp.
- `at(label)`: Dispatch to `layers` or `ops` based on label format.
- `compare_at(label)`: Compare all Bundle members at one Layer or Op label.
- `diff_pair(a, b)`: Rank label-level differences between two named Traces.
- `most_changed(baseline, ...)`: Return labels with the largest divergence from a baseline.
- `cluster(...)`: Placeholder clustering entry point; details remain implementation-specific.
### Bundle Metric Operations
- `apply(fn)`: Apply a callable to each Trace and return a name-keyed result dict.
- `save(path, level, overwrite)`: Save all Bundle members in unified TorchLens format.
- `do(*args, **kwargs)`: Apply intervention-style operations across members.
- `fork(name)`: Duplicate the Bundle.
- `attach_hooks(*args, **kwargs)`: Attach hooks across member Traces.
- `replay(**kwargs)`: Replay all member Traces.
- `rerun(model, x, **kwargs)`: Rerun all member Traces.
- `draw(...)`: Draw forward graphs for Bundle member Traces.
- `help()`: Return or print a readiness summary for Bundle operations.
### TraceAccessor
- `TraceAccessor`: Dict-like accessor for Bundle Traces by name or ordinal position.
- `TraceAccessor.__getitem__(key)`: Return a Trace for a name or integer index.
- `TraceAccessor.__contains__(key)`: Test whether a Trace name exists.
- `TraceAccessor.__iter__()`: Iterate over Trace records in insertion order.
- `TraceAccessor.keys()`: Return Trace names.
- `TraceAccessor.values()`: Return Trace objects.
- `TraceAccessor.items()`: Return `(name, trace)` pairs.
- `TraceAccessor.to_pandas()`: Export member Trace summaries to a DataFrame.
### SuperOp
SuperOp and SuperLayer are cross-member graph entities; plural tensor-map names below follow the locked `out`/`grad` vocabulary.
- `SuperOp`: Cross-member aggregate for one Op label.
- `label`: Op label represented by this SuperOp.
- `members`: Mapping of Trace name to that Trace's Op at this label.
- `labels`: Mapping of Trace name to the resolved Op label.
- `traces`: Set or view of Trace names represented.
- `coverage`: Fraction of Bundle members that contain this Op.
- `type`: Normalized Op type when consistent across members.
- `module`: Module context when comparable across members.
- `shape`: Common output shape when compatible.
- `outs`: Mapping of Trace name to saved Op output.
- `grads`: Mapping of Trace name to saved Op gradient.
- `out`: Stacked or single output convenience when all members are compatible.
- `grad`: Stacked or single gradient convenience when all members are compatible.
- `diff(metric=...)`: Pairwise or baseline-relative differences for this Op.
- `aggregate(fn)`: Apply an aggregation over this Op's member tensors.
### SuperLayer
- `SuperLayer`: Cross-member aggregate for one Layer label.
- `label`: Layer label represented by this SuperLayer.
- `members`: Mapping of Trace name to that Trace's Layer at this label.
- `labels`: Mapping of Trace name to the resolved Layer label.
- `traces`: Set or view of Trace names represented.
- `coverage`: Fraction of Bundle members that contain this Layer.
- `type`: Normalized Layer type when consistent across members.
- `module`: Module context when comparable across members.
- `shape`: Common output shape when compatible.
- `outs`: Mapping of Trace name to saved Layer output when single-Op passthrough is valid.
- `grads`: Mapping of Trace name to saved Layer gradient when valid.
- `out`: Convenience output when all members expose compatible single-Layer outputs.
- `grad`: Convenience gradient when all members expose compatible gradients.
- `diff(metric=...)`: Differences for this Layer across members.
- `aggregate(fn)`: Apply an aggregation over member Layer tensors.
### Super Accessors
- `SuperOpAccessor`: Dict-like accessor returning SuperOps by Op label.
- `SuperLayerAccessor`: Dict-like accessor returning SuperLayers by Layer label.
- `SuperOpAccessor.__getitem__(label)`: Return a SuperOp.
- `SuperLayerAccessor.__getitem__(label)`: Return a SuperLayer.
- `SuperOpAccessor.__iter__()`: Iterate SuperOps in label order.
- `SuperLayerAccessor.__iter__()`: Iterate SuperLayers in label order.
- `SuperOpAccessor.to_pandas()`: Export aligned Op summaries.
- `SuperLayerAccessor.to_pandas()`: Export aligned Layer summaries.
## Cross-class accessor patterns
- `trace["conv2d_1_2"]`: Delegates to `trace.layers` and returns a Layer.
- `trace["conv2d_1_2:1"]`: Delegates to Op resolution and returns an Op.
- `trace.layers["conv2d_1_2"]`: Returns a Layer.
- `trace.layers["conv2d_1_2:1"]`: Returns the corresponding Op for convenience.
- `trace.ops["conv2d_1_2:1"]`: Returns an Op.
- `trace.modules["encoder.block.0"]`: Returns a Module by PyTorch dotted address.
- `trace.params["encoder.block.0.weight"]`: Returns a Param by parameter address.
- `trace.buffers["bn.running_mean"]`: Returns a Buffer by buffer address.
- `trace.grad_fns[...]`: Returns a GradFn; stable human-readable grad-fn label syntax remains deferred.
- `bundle["baseline"]`: Delegates to `bundle.traces` and returns a Trace.
- `bundle.traces["baseline"]`: Returns a Trace.
- `bundle.layers["conv2d_1_2"]`: Returns a SuperLayer.
- `bundle.ops["conv2d_1_2:1"]`: Returns a SuperOp.
- `bundle.at("conv2d_1_2")`: Returns a SuperLayer because the label has no `:N` pass suffix.
- `bundle.at("conv2d_1_2:1")`: Returns a SuperOp because the label is pass-qualified.
- Single-Op passthrough rule: Layer fields such as `out` and `grad` work directly only when the Layer has one Op.
- Multi-Op rule: use `layer.ops[n].field` or a pass-qualified label when a Layer has multiple Ops.
- Module aggregate rule: Module fields describe all calls unless explicitly documented as single-call passthrough.
- ModuleCall rule: ModuleCall fields describe one invocation only.
- GradFn aggregate rule: GradFn fields describe one autograd node; GradFnCall fields describe each hook firing.
## Function and method signatures
This section lists the signatures of the main public callables. Parameter names
follow current implementation; some are explicitly slated for renaming and are
flagged. Keyword-only arguments appear after the `*` marker, matching Python
syntax.
### Top-level capture
- `tl.trace(model, input_args, input_kwargs=None, *, layers_to_save="all", transform=None, output_transform=None, layer_visualizers=None, save_visualizations=False, save_raw_input="small", save_raw_output="small", batch_render="auto", keep_unsaved_layers=False, keep_orphans=False, output_device="cpu", out_transform=None, grad_transform=None, gradient_postfunc=None, save_raw_outs=True, save_raw_grads=True, mark_layer_depths=False, detach_saved_activations=True, save_arg_values=True, save_grads=False, save_code_context=True, save_rng_states=False, save_outs_to=None, keep_outs_in_memory=True, intervention_ready=False, hooks=None, unwrap_when_done=False, verbose=False, source_context_lines=2, compute_input_output_distances=True, recurrence_detection=True, capture=None, save=None, visualization=None, streaming=None, train_mode=False, name=None, cache=False, cache_dir=None, module_filter=None, stop_after=None, raise_on_nan=False, backend=None, ...)`: Resolve a backend, run a forward pass with capture, and return a Trace. Explicit `backend=` wins; `None` preserves current torch/MLX auto-resolution. Parameters prefixed `vis_*` (e.g. `vis_outpath`, `vis_fileformat`, `vis_node_mode`, `vis_save_only`, `vis_direction`, `vis_graph_overrides`, `vis_edge_overrides`, `vis_grad_edge_overrides`, `vis_module_overrides`, `vis_node_placement`, `vis_renderer`, `vis_theme`, `vis_intervention_mode`, `vis_show_cone`, `vis_call_depth`, `vis_buffers`, `vis_mode`, `vis_opt`) are accepted but slated to drop the prefix; rename target deferred. Legacy aliases also accepted: `view`, `depth`, `renderer`, `layout`, `node_style`, `out_postfunc`. `gradient_postfunc` is a silent alias for `grad_transform` (the rename pass should pick one). MLX models route through `BackendSpec` to `MLXBackend`, which hard-rejects `intervention_ready=True`, pre-attached `hooks`, and `save_grads=True`.
- `tl.peek(model, x, layer, stop_after=None)`: Return the saved out for one layer. Convenience over `trace()`.
- `tl.extract(model, x, layers)`: Return saved outs for many layers. `layers` is an iterable of lookups or a `{user_label: lookup}` mapping.
- `tl.batched_extract(model, stimuli, layers, batch_size=32, device=None, output_dir=None, postfunc=None, progress=True)`: Extract outs from a batched stimulus set.
- `tl.fastlog.record(model, input_args, input_kwargs=None, *, keep_op=None, keep_module=None, default_op=MISSING, default_module=MISSING, history_size=8, include_source_events=False, max_predicate_failures=32, on_predicate_error="auto", streaming=None, return_output=False, postprocess="none", random_seed=None, out_transform=None, save_raw_outs=True, train_mode=False)`: Predicate-driven sparse capture, returns a `Recording`. If a predicate calls `tl.halt(reason)`, the recording stops at that pass; `Recording.halted`, `Recording.halt_reason`, and `Recording.halts_by_pass` capture the state.
- `tl.fastlog.halt(reason="")`: Raise `HaltSignal` to halt the active fastlog recording at this predicate call site. Returns `NoReturn`.
- `tl.fastlog.HaltSignal(BaseException)`: Signal class raised by `halt`. Inherits from `BaseException` so it bypasses generic `except Exception:` handlers in user predicates.
- `tl.fastlog.Recorder.__enter__() -> Recorder`: Begin a fastlog rollout context. `Recorder.log_backward(loss, *, save_grads=None, default_grad=None, retain_graph=None, create_graph=False)` runs backward for the active recorder and retains selected gradients. Returns the live `Recording`.
- `Recording.log_backward(loss, *, save_grads=None, default_grad=None, retain_graph=None, create_graph=False)`: Run backward on the captured forward, retaining selected gradients into `grad_records`. Raises `RecorderStateError` if the recording is halted.
- `Recording.halted: bool`: True when the recording was stopped by `HaltSignal`.
- `Recording.halt_reason: str | None`: Reason from the halt call site, or None.
- `Recording.halts_by_pass: dict[int, str]`: Per-pass halt reasons for multi-pass recordings.
- `Recording.grad_records: list[GradientRecord]`: Captured gradient records.
- `Recording.grad_by_pass: dict[int, list[int]]`, `Recording.grad_by_label: dict[str, list[int]]`, `Recording.grad_by_grad_fn_label: dict[str, list[int]]`: Indexes into `grad_records`.
- `Recording.keep_grad_repr: str | None`: Backward-compatibility storage name for the repr of the `save_grads` predicate used.
- `GradientRecord`: One captured backward gradient sample (analogue of `ActivationRecord`).
- `GradRecordContext`: Predicate context object passed to `save_grads` callables; mirrors `RecordContext` for the backward graph.
### Save, load, validation, bundle construction
- `tl.save(trace, path, *, level="portable", include_outs=True, include_grads=True, include_saved_args=False, include_rng_states=False, strict=True, overwrite=False)`: Persist a Trace to a `.tlspec` directory.
- `tl.load(path, **kwargs)`: Load a `.tlspec` Trace or Bundle.
- `tl.validate(model, input_args, input_kwargs=None, *, scope, random_seed=None, verbose=False, validate_metadata=True, loss_fn=None, perturb_saved_grads=False, atol=1e-5, rtol=1e-4, validate_layer_grads=False, layer_grad_atol=None, layer_grad_rtol=None, backend=None)`: Resolve a backend and run validation. `scope` is `"forward"`, `"backward"`, `"saved"`, or `"intervention"`. Backward scope additionally accepts `validate_layer_grads` to also compare module-output gradients (returns aggregated bool); `layer_grad_atol`/`layer_grad_rtol` default to the top-level `atol`/`rtol`. Backward scope state hygiene: state_dict snapshot/restore around capture, deterministic `random_seed` (auto-assigned if None), `model.zero_grad(set_to_none=True)` between stock and candidate runs.
- `tl.aggregate(model, dataloader, metrics, *, target="out", loss_fn=None)`: Stream outs (or grads) through metric accumulators; `metrics` maps layer label to a streaming statistic. `target="out"` (default) accumulates forward outs; `target="grad"` requires `loss_fn` and accumulates gradients of layer outs via `log.log_backward(loss_fn(output, *batch_tail))`.
- `tl.bundle(*args, **kwargs)`: Construct a Bundle. Forwards to `Bundle.__init__`.
### Selectors
- `tl.label(name)`: Selector matching an exact Layer or Op label.
- `tl.func(name, *, output=None)`: Selector matching a function name token (e.g. `"relu"`). `output` disambiguates multi-output calls by index or semantic role.
- `tl.output(target)`: Selector matching by output index or semantic output role; used for multi-output function and module disambiguation. `target` is an int index or string role name.
- `tl.module(address)`: Selector matching a dotted module address.
- `tl.contains(substring)`: Selector matching any label containing the substring.
- `tl.where(predicate, *, name_hint=None)`: Selector wrapping a custom predicate.
- `tl.in_module(address_or_layer, address=None)`: When called with one argument, returns a selector restricted to a module address. When called with two arguments, returns a `bool` indicating whether the first argument lives inside the named module.
- `tl.sites(layer_pattern, ops=None, modes=None)`: Build a structured `SiteCollection` for parameter sweeps.
### Backward selectors
These selectors match the backward grad-fn graph and compose with direction-agnostic selectors (`label`, `module`, `output`, `contains`, `where`, `in_module`) but cannot be combined with forward-only selectors (`func`).
- `tl.grad_fn(type=None, *, label=None, is_custom=None)`: Match grad_fns by class name (string or class object), label substring, or custom-autograd flag. `type` is normalized via `__name__` when a class is passed.
- `tl.intervening()`: Match grad_fns whose `is_intervening` is True (no paired forward op). Used for backward-only nodes such as `AccumulateGrad` parameter sinks.
- `tl.grad_fn_label(name)`: Match a grad_fn by its exact stable label.
### Selector composition rules
- `a & b` (intersection), `a | b` (union), `~a` (negation).
- Cross-graph composition is rejected by `SelectorCompositionError`: a forward selector (`func`) cannot intersect a backward selector (`grad_fn`, `intervening`, `grad_fn_label`).
- Direction-agnostic selectors compose freely with either direction.
- Selectors without a registered direction taxonomy raise `UnclassifiedSelectorError` at composition time.
### Intervention helpers (all return `HelperSpec`)
- `tl.zero_ablate(*, force_shape_change=False)`: Replace value with zeros.
- `tl.mean_ablate(source=None, *, over="self", force_shape_change=False)`: Replace value with a mean over `source`.
- `tl.resample_ablate(source=None, *, from_=None, seed=None, force_shape_change=False)`: Replace value with a resample.
- `tl.steer(direction, magnitude=1.0, *, coef=None, feature_axis=None, force_shape_change=False)`: Add a steering vector.
- `tl.scale(factor, *, force_shape_change=False)`: Multiply by a scalar.
- `tl.clamp(*, min=None, max=None, force_shape_change=False)`: Clamp values to a range.
- `tl.noise(std, *, seed=None, force_shape_change=False)`: Add Gaussian noise.
- `tl.project_onto(direction, *, feature_axis=None, force_shape_change=False)`: Project onto a direction.
- `tl.project_off(direction, *, feature_axis=None, force_shape_change=False)`: Project orthogonal to a direction.
- `tl.swap_with(other_label, *, force_shape_change=False)`: Swap activation with another site's saved value.
- `tl.splice_module(module, *, input="out", output="out", force_shape_change=False)`: Splice an `nn.Module` into the forward pass.
### Backward intervention helpers (all return `HelperSpec` with `kind="backward"`)
These helpers operate during a backward pass. Mount-shape metadata: `bwd_hook`/`grad_zero`/`grad_scale` use the tensor-level mount; `grad_clip`/`grad_noise`/`grad_clamp` use `mount_shape="tuple"` and apply to the grad_input tuple at a grad_fn hook.
- `tl.bwd_hook(fn)`: Wrap a backward callback as a HelperSpec. `fn` first positional arg may be named `g`; keyword-only `hook` argument required.
- `tl.grad_zero(*, force_shape_change=False)`: Replace a backward gradient tensor with zeros.
- `tl.grad_scale(factor, *, force_shape_change=False)`: Multiply a backward gradient tensor by `factor`.
- `tl.grad_clip(max_norm, norm_type=2.0)`: Per-tensor norm clipping over a grad_input tuple; uses `torch.linalg.vector_norm`. Mount shape: `tuple`.
- `tl.grad_noise(std, *, seed=None)`: Add Gaussian noise to each tensor in a grad_input tuple. Mount shape: `tuple`.
- `tl.grad_clamp(min=None, max=None)`: Elementwise clamp on each tensor in a grad_input tuple. Mount shape: `tuple`. Requires `min`, `max`, or both.
### Errors raised by intervention helpers and selectors
- `HelperMountError(HookSiteCoverageError)`: Raised when a helper is mounted on an incompatible selector universe (e.g., forward helper on a backward selector site).
- `HookSignatureError(ConfigurationError, TypeError)`: Raised when a hook callable does not accept the required signature.
- `HookValueError(InterventionError, ValueError)`: Raised when a hook returns an invalid replacement value.
- `HookSiteCoverageError(SiteResolutionError)`: Raised when hook normalization cannot associate a hook with any site.
- `SelectorCompositionError(SiteResolutionError)`: Raised when forward and backward selectors are composed.
- `UnclassifiedSelectorError(SiteResolutionError)`: Raised when a selector lacks an explicit direction taxonomy bucket.
- `AxisAmbiguityError(ConfigurationError, ValueError)`: Raised when a helper cannot infer a feature axis safely.
- `SpliceModuleDtypeError(CompatibilityError, RuntimeError)`: Raised when `splice_module` returns an unexpected dtype.
- `SpliceModuleDeviceError(CompatibilityError, RuntimeError)`: Raised when `splice_module` returns a tensor on an unexpected device.
- `MultiOutputModuleError(ValidationError, ValueError)`: Raised by singular-output access on a multi-output ModuleCall.
- `AppendMismatchError(ValidationError, ValueError)`: Raised when a chunked append candidate is incompatible with the base log.
- `AppendStreamingNotSupportedError(ValidationError, ValueError)`: Raised when append rerun would mutate active streamed activation blobs.
- `AppendBatchDependenceError(ValidationError, ValueError)`: Raised when append cannot prove helper or grad batch independence.
- `AppendStateValidationWarning(TorchLensInterventionWarning)`: Warning when validators skip fresh checks on stacked appended traces.
- `BatchNormTrainModeWarning(TorchLensInterventionWarning)`: Warning for append reruns through batch-sensitive train-mode modules.
### Standalone intervention verbs
- `tl.do(log, hooks_or_site, value_or_hook=None, *, model=None, x=None, engine=MISSING, confirm_mutation=MISSING, strict=MISSING, intervention=None)`: One-shot intervention call mirroring `Trace.do`.
- `tl.replay(log, *, strict=MISSING, hooks=MISSING, replay=None)`: Replay saved values without re-executing the model.
- `tl.replay_from(log, site, *, strict=MISSING, replay=None)`: Replay starting from a site.
- `tl.rerun(log, model, x=None, *, append=MISSING, strict=MISSING, replay=None, output_transform=None)`: Re-execute the model and update or append Trace state. Pending rename (see baton).
### Observers
- `tl.tap(site, *, direction="forward")`: Create a `TapObserver` for a site. `direction` is `"forward"`, `"backward"`, or `"both"`. Backward direction records `grad_input`/`grad_output` snapshots via `record_backward`.
- `tl.record_span(name, *, direction="both")`: Context manager creating a named observer span scoped to a direction (`"forward"`, `"backward"`, or `"both"`).
- `TapObserver.records: list[TapRecord]`: Captured observations.
- `TapObserver.values() -> list[torch.Tensor]`: Detached out snapshots in observation order.
- `TapObserver.record_backward(grad_input, *, grad_output, grad_fn_log, call_index, run_ctx)`: Backward callback signature for taps with `direction="backward"` or `"both"`.
- `TapRecord(value, site_label, span_names, timestamp, direction, grad_kind=None, backward_call_index=None)`: One observed tensor value. `direction` is `"forward"` or `"backward"`; `grad_kind` is `"grad_input"` or `"grad_output"` for backward records.
### Visualization helpers
- `tl.viz.heatmap(max_size=200)`: Returns a tensor-to-`PIL.Image` callable for `layer_visualizers`.
- `tl.viz.channel_grid(n=16, max_size=300)`: Returns a per-channel grid visualizer.
- `tl.viz.histogram(bins=30, width=240, height=160)`: Returns a histogram visualizer.
- `tl.debug.bisect_nan(trace)`: Scans saved activations in execution order and returns a result object for the first op whose output contains NaN or Inf. If the suspect region was not saved, the result message tells the user to re-trace with a wider `save=` predicate.
- `tl.debug.hot_path(trace, by="flops"|"memory"|"duration")`: Returns a pandas DataFrame ranking source lines by aggregate `flops_forward`, `activation_memory`, or `func_duration`; missing metrics are excluded and reported in `DataFrame.attrs`.
- `tl.show_bundle_graph(bundle, vis_outpath="bundle_modelgraph", vis_mode="unrolled", direction="forward", vis_direction="bottomup", vis_graph_overrides=None, vis_node_overrides=None, vis_edge_overrides=None, vis_save_only=False, vis_fileformat="pdf")`: Draw a Bundle graph. `direction` accepts `"forward"`, `"backward"`, `"both"`, or `"overlay"`. The `"backward"` direction is now functional (renders the union of bundle backward graphs).
- `tl.draw_backward(trace, vis_outpath=MISSING, vis_save_only=MISSING, vis_fileformat=MISSING, vis_direction=MISSING, vis_graph_overrides=MISSING, vis_edge_overrides=MISSING, node_spec_fn=None, collapsed_node_spec_fn=None, node_style=MISSING, vis_node_mode=MISSING, code_panel=False, visualization=None)`: Draw a backward grad-fn graph. Deprecated top-level wrapper; prefer `Trace.draw_backward`.
- `tl.draw_combined(trace, ...)`: Deprecated top-level wrapper for `Trace.draw_combined`.
### Bridges
- `tl.bridge.hf.trace_text(model, text, *, tokenizer=None, chat_template=False, **kwargs)`: Trace a Hugging Face model with raw text input. Auto-resolves the tokenizer and forwards remaining kwargs to `tl.trace`.
### Trace methods
- `Trace.backward(loss, **backward_kwargs)`: Run backward from a loss and populate grad fields. Implementation method is `log_backward`; `backward` is the target public name.
- `Trace.find_layers(query, *, limit=10)`: Return Layer labels matching a query.
- `Trace.find_sites(query, *, strict=False, max_fanout=8)`: Return intervention sites matching a query. Deferred for naming review (see Deferred items).
- `Trace.fork(name=None)`: Duplicate the Trace with a fresh intervention spec.
- `Trace.set(site, value, *, strict=False, confirm_mutation=False)`: Set a site value.
- `Trace.attach_hooks(hooks_or_site, hook=None, *extra_hooks, strict=False, prepend=False, confirm_mutation=False)`: Attach hooks to one or many sites.
- `Trace.do(hooks_or_site, value_or_hook=None, *, model=None, x=None, engine=MISSING, confirm_mutation=MISSING, strict=MISSING, intervention=None)`: One-shot intervention application.
- `Trace.replay(*, strict=MISSING, hooks=MISSING, replay=None)`: Replay saved values without re-executing the model.
- `Trace.replay_from(site, *, strict=MISSING, replay=None)`: Replay starting from a site.
- `Trace.rerun(model=None, x=None, *, append=MISSING, strict=MISSING, replay=None, transform=USE_STORED, output_transform=USE_STORED)`: Re-execute and update or append Trace state.
- `Trace.draw(vis_mode="unrolled", vis_call_depth=1000, vis_outpath="modelgraph", vis_graph_overrides=None, module=None, node_mode="default", node_spec_fn=None, collapsed_node_spec_fn=None, collapse_fn=None, skip_fn=None, vis_edge_overrides=None, vis_grad_edge_overrides=None, vis_module_overrides=None, vis_save_only=False, vis_fileformat="pdf", show_buffer_layers="meaningful", direction="bottomup", vis_node_placement="auto", vis_renderer="graphviz", vis_theme="torchlens", vis_intervention_mode="node_mark", vis_show_cone=True, code_panel=False, node_overlay=None, node_label_fields=None, show_legend=False, font_size=None, dpi=None, for_paper=False, ...)`: Draw the forward graph. `vis_*` parameter rename target deferred.
  Rolled-view labels report each node's own count: a Layer shows `num_passes` and a collapsed `@module` shows module calls. Extra suffixes appear only when needed: `k sites` means one Layer bundles multiple separable structural positions; `×P` is shown only when an enclosing module call rectangle certifies those sites; `calls 1,2-4` surfaces hidden call partitions without splitting a shared module address; buffer nodes show flat version sets such as `@state v1-6`. The internal site signature uses repeated-body neighbor topology with open-chain boundary quotienting, inbound argument slots, the child's top-level consumer slot, and bracketing context. The chain/fan-out/mixed facet is metadata by default and does not change the count.
- `Trace.draw_backward(vis_outpath="backward_modelgraph", vis_graph_overrides=None, node_spec_fn=None, collapsed_node_spec_fn=None, vis_node_mode="default", vis_edge_overrides=None, vis_save_only=False, vis_fileformat="pdf", vis_direction="topdown", code_panel=False)`: Draw the backward grad-fn graph.
- `Trace.draw_combined(vis_outpath="combined_modelgraph", vis_graph_overrides=None, node_spec_fn=None, backward_node_spec_fn=None, vis_edge_overrides=None, vis_save_only=False, vis_fileformat="pdf", vis_direction="leftright", vis_mode="unrolled", intervening_cluster="upstream", show_buffer_layers="meaningful")`: Render forward ops and backward grad_fns in a single graph. `intervening_cluster` is `Literal["upstream", "outside", "downstream", "own"]` and controls where backward-only nodes are clustered relative to forward graph.
- `Trace.log_backward(loss, **backward_kwargs)`: Backward capture implementation method. `Trace.backward(loss)` is the locked public name; `log_backward` remains because validation, observers, and stats internals call it directly.
- `Trace.replace_run_state_from(new_log)`: Atomically replace this Trace's run state from a freshly-built Trace. Preserves identity fields, intervention spec, ledger, `is_appended`, `_append_sequence_id`, and `append_history`. Used by the intervention rerun engine.
- `Trace.append_run_state_from(new_log)`: Merge compatible chunk outs from `new_log` into this Trace. Used by append rerun for chunked execution.
- `Trace.preview_fastlog(predicate=None, keep_op=None, keep_module=None, **kwargs)`: Render a fastlog predicate preview over this graph; uses `torchlens.visualization.fastlog_preview.preview_fastlog`.
- `Trace.find_sites(query, *, strict=False, max_fanout=8)`: Find intervention sites matching a query. (Naming still deferred pending the Site rethink.)
- `Trace.resolve_sites(query, *, strict=False, max_fanout=8)`: Resolve intervention sites for a query. (Naming still deferred.)
- `Trace.summary(level="overview", *, fields=None, mode="auto", show_ops=False, preset=None, columns=None, include_ops=None, max_rows=200, print_to=None, count_fma_as_two=False)`: Return a textual summary.
- `Trace.save(path, **kwargs)`: Save the Trace; forwards to `tl.save`.
- `Trace.cleanup()`: Clear circular references and runtime-only heavyweight objects.
- `Trace.to_pandas() / to_csv(filepath, **kwargs) / to_parquet(filepath, **kwargs) / to_json(filepath, **kwargs)`: Tabular export. The same export quartet exists on Op, Layer, Module, ModuleCall, Param, Buffer, GradFn, GradFnCall, TraceAccessor, SuperOpAccessor, and SuperLayerAccessor.
### Bundle methods
- `Bundle(traces=None, *, baseline=None, capacity=None, ...)`: Construct a Bundle. Accepts an iterable of Traces or a name-keyed mapping.
- `Bundle.add(trace, name=None)`: Add a Trace.
- `Bundle.remove(name)`: Remove a Trace by name; returns the removed Trace.
- `Bundle.remove_except(keep)`: Remove every Trace except the named Traces.
- `Bundle.clear()`: Remove all Traces.
- `Bundle.fork(name=None)`: Duplicate the Bundle.
- `Bundle.do(*args, **kwargs)`: Apply intervention-style operations across members. Forwards to per-Trace `do`.
- `Bundle.attach_hooks(*args, **kwargs)`: Attach hooks across member Traces.
- `Bundle.replay(**kwargs)`: Replay all member Traces.
- `Bundle.rerun(model, x=None, **kwargs)`: Rerun all member Traces.
- `Bundle.save(path, *, level="portable", overwrite=False)`: Save all members to a unified `.tlspec` directory.
- `Bundle.apply(fn)`: Call `fn(trace)` for each member; returns a name-keyed dict.
- `Bundle.at(label)`: Resolve a label to the matching Super accessor (`SuperLayer` or `SuperOp`).
- `Bundle.compare_at(site)`: Stack member tensors at one site.
- `Bundle.diff_pair(a, b=None)`: Out differences between two members or between two sites.
- `Bundle.most_changed(baseline=None, *, top_k=10, metric="cosine")`: Rank sites by distance from a baseline. `metric` is `"cosine"`, `"l2"`, or a callable.
- `Bundle.cluster(*args, **kwargs)`: Placeholder; raises `NotImplementedError`.
- `Bundle.relationship(a, b)`: Return the recorded relationship between two named Traces.
- `Bundle.help()`: Return or print a per-member readiness summary.
### Notes on signatures
- `MISSING` (and `_USE_STORED_TRANSFORM` for `Trace.rerun`) are sentinels meaning "use the captured default"; users pass concrete values to override.
- The `to_pandas`/`to_csv`/`to_parquet`/`to_json` quartet is uniform across record classes and accessors. Where individual signatures differ they accept `**kwargs` forwarded to the underlying writer.
- Parameters slated for renaming (e.g. all `vis_*` draw parameters, `peek -> pluck`, the `replay` verb) appear here under their current names so the rename pass can review them in one place.

## Validation reports
- `LayerGradReport` (?): Dataclass returned by `validate_backward_pass(..., validate_layer_grads=True)` summarising per-module-output gradient parity. Fields: `mode` (`Literal["module_output"]`), `overall_passed: bool`, `coverage: dict[str, str]` (module-call label -> bucket), `covered_count`, `mismatched_count`, `skipped_no_first_leaf_count`, `skipped_module_less_count`, `skipped_no_grad_count`, `skipped_identity_output_count`, `skipped_root_module_count`, `unexpected_count`, `candidate_grad_count`, `atol`, `rtol`, `mismatched_labels: tuple[str, ...]`, `max_abs_diffs: dict[str, float]`, `max_rel_diffs: dict[str, float]`. `__bool__` returns `overall_passed`.
- Coverage buckets:
  - `covered`: candidate grad matched stock grad within tolerance.
  - `mismatched`: candidate grad differed from stock or shapes disagreed.
  - `skipped_no_first_leaf`: call has no resolvable first output leaf layer.
  - `skipped_module_less`: candidate layer has a grad but no module containment (counter only; not stored in `coverage`).
  - `skipped_no_grad`: candidate or stock side missing a grad for this module-call.
  - `skipped_identity_output`: stock module output is identity-equivalent to its input; skipped to avoid double-counting.
  - `skipped_root_module`: the top-level model (address `"self"`); skipped because it has no enclosing module.
- `MIN_MODULE_OUTPUT_COVERAGE: float = 0.80`: Minimum covered ratio required for `overall_passed`.

## Backend integration
- `Trace.backend`: Public `BackendName` tag for the capture backend (`"torch"`, `"mlx"`, `"jax"`, or `"tinygrad"` in shipped code). Survives portable save/load.
- `Trace.module_identity_mode`: Status field declaring whether module accessors expose torch modules, pytree-derived modules, object-discovered modules, or only a function root.
- `Trace.param_source`: Status field declaring whether `Trace.params` comes from native module parameters, pytree-derived leaves, or no declared params.
- `Trace.payload_load_status`: Load-time payload materialization status. Values include `"loaded"`, `"loaded_device_best_effort"`, `"audit_only"`, and `"audit_only_missing_runtime"`.
- `Trace.validation_replay_status`: `ValidationReplayStatus` machine-readable replay-validation status. Loaded JAX/tinygrad traces whose runtime replay captures were stripped and loaded MLX traces without replay support report `state="unavailable"` and `reason="loaded_trace_runtime_capture_stripped"` instead of a pass/fail bool; live JAX/tinygrad traces still run real replay validation, including static-label selectively saved traces via runtime-only hidden replay payloads.
- `dtype_ref` / `device_ref`: Neutral mirror fields on tensor/parameter records. Torch keeps existing `dtype` and device-facing fields byte-stable; the mirrors let non-torch backends expose dtype/device metadata without torch objects.
- `backend_address` / `resolver_status`: Neutral address metadata on records. `backend_address` stores the backend-native address or handle used by the builder; `resolver_status` records whether that address is currently resolved.
- Backend canonical errors: `UnknownBackendError`, `BackendMismatchError`, `BackendAmbiguityError`, `BackendUnsupportedError`, `BackendPayloadUnsupportedError`, and `BackendRuntimeCompatibilityError`.
- Public option capability epochs: ordered patches that update `trace()` signature, `CaptureOptions`, backend registry specs, per-backend capability mirrors, cache-key inclusion, docs, and tests together whenever a declared backend option changes capability status.
- MLX capability contract: the MLX `BackendSpec` reports `validation_replay=False`, `backward_capture=False`, `fastlog=False`, `interventions=False`, `rng_replay=False`, `payload_materialization=True`, `payload_policy="array_payloads"`, and `module_identity_modes=("function_root", "object_module")`. `mlx.nn.Module` roots use public `named_modules()` traversal for object-module hierarchy by default; `module_identity_mode="function_root"` keeps a root-only module surface. Static-label `save=` supports `tl.func`, `tl.label`, `tl.module`, `tl.in_module`, `tl.contains`, and boolean composites. Portable saves round-trip saved forward payloads as `mlx.core.array` values when the MLX runtime is installed; loaded MLX traces report replay validation unavailable rather than pass/fail. Unsupported public surfaces raise canonical backend errors where the registry owns dispatch; backend-private defensive checks may still raise existing concrete errors during the cutover.
- JAX capability contract: the JAX `BackendSpec` reports `validation_replay=True`, `backward_capture=False`, `fastlog=False`, `interventions=False`, `payload_materialization=True`, `payload_policy="array_payloads"`, and `module_identity_modes=("function_root", "pytree_module")`. Raw callables use `function_root`; pure nested `jax.jit` calls inline through the concrete per-equation replay path, including JIT-wrapped control flow that then unrolls under the existing `jax_control_flow` policy. Nested JIT calls with closed constants, effects, donated inputs, or explicit shardings remain rejected. Equinox and Flax NNX roots use `pytree_module` with strict rejection of inner JAX transforms/control flow and NNX state rebinding. Static-label `save=` selectors are post-finalization public-payload filters; value-dependent save predicates, `intervene=`, and `halt=` are rejected because jaxpr tracing does not expose public TorchLens labels plus concrete predicate-time mutation/replay semantics at one boundary. `tl.backends.jax.GradOptions` populates `trace.derived_grads`; true backward surfaces raise with guidance to that accessor.
- tinygrad capability contract: the tinygrad `BackendSpec` reports `validation_replay=True`, `backward_capture=False`, `fastlog=False`, `interventions=False`, `payload_materialization=True`, `payload_policy="array_payloads"`, and `module_identity_modes=("function_root", "object_module")`. Raw callables use `function_root`; callable object graphs with discoverable tinygrad module attributes use `object_module` unless explicitly forced back to `function_root`. Static-label `save=` selectors are post-finalization public-payload filters; value-dependent save predicates, `intervene=`, and `halt=` are rejected because lazy UOps have no concrete predicate-time value and no stable replace-one-UOp descendant-rewrite surface before realize. UOp attribution follows first observed construction when tinygrad reuses a UOp identity.
- MLX wrapped surface (since post-backward megasprint): Conv2d, normalization layers, Embedding, Dropout, MultiHeadAttention, reductions, shape ops, activations. Pinned to `mlx>=0.26,<0.27`.

## Portable save scrub
- `_io.scrub.scrub_for_save(trace, *, include_outs=True, include_grads=True, include_saved_args=False, include_rng_states=False) -> tuple[dict, list[BlobSpec], list[dict]]`: Scrub a Trace into portable metadata, tensor blob specs, and an unsupported-tensor audit list. Third tuple element collects tensors whose dtype/device combination cannot be serialised by the active backend payload codec.
- `_io.scrub._ScrubOptions.unsupported_tensor_records: list[dict[str, str]]` (?): Sidecar audit collector populated during scrub.

## Deferred items
- `edge_uses`: deferred until EdgeUseRecord's public role is decided.
- Rolled visualization properties such as `children_per_op`, `parents_per_op`, `child_ops_per_layer`, `parent_ops_per_layer`, `edges_vary_across_ops`, `atomic_module_ops`, and `parent_arg_positions`: deferred to a report/visualization namespace move.
- `buffer_num_passes`: deferred at Trace scope; likely follows Buffer vocabulary as `buffer_use_counts`.
- `recording_kept`: deferred because `recording` conflicts with fastlog `Recording` vocabulary.
- `keep_unsaved_layers`: deferred until notebook walkthrough clarifies user-facing semantics.
- `capture_cache_hit`, `capture_cache_key`, `capture_cache_path`: deferred to the save/cache design review.
- `capture_args_template`: included as config, but deeper cache semantics remain deferred.
- `manual_tensor_connections`: deferred until manual graph-link workflows are reviewed.
- `observer_spans`: deferred until span recording surface is reviewed.
- `relationship_evidence`: deferred until rerun relationship reporting is reviewed.
- `report_values`: deferred and likely unified with `trace_annotations`.
- `streaming_pass_logs`: deferred to streaming/rerun-append design.
- `find_sites` and `resolve_sites`: deferred to the integrated intervention `Site` concept survey. Both methods exist on `Trace` and `tl.intervention.resolve_sites` is publicly exported; the rename pass should confirm the final naming.
- `set`, `attach_hooks`, `do`, `clear_hooks`, `remove`, `detach_hooks`, `save_intervention`, and `intervention_spec`: deferred to intervention API review. (Most landed live in the 2.16 intervention sprint; the rename pass should review names.)
- `preview_fastlog`: moved to `tl.fastlog.preview`; top-level `tl.preview_fastlog` is a deprecated alias for one minor cycle. Final naming inside fastlog namespace still deferred.
- `vis_*` draw parameters: target direction is to drop the prefix, but exact parameter list is deferred.
- `peek -> pluck`: top-level `tl.peek` rename target. Still deferred; no implementation alias yet.
- `tl.rerun(append=...)` rename target: still deferred. `append` kwarg accepts `MISSING` sentinel.
- `replay` / `replay_from` verb rename: still deferred. No `replay` -> X alias in code.
- Bundle dynamic helpers `aligned_pairs`, `compare`, `delta_map`, `norm_delta`, `output_delta`, and `show_diff`: deferred to Bundle helper redesign.
- `save_activations` and other workflow-level save kwargs: deferred pending decision on workflow vocabulary versus `out` field vocabulary.
- Fast-path naming such as `fastlog`: deferred because fast output is intentionally not a full Trace.
- `multi_output_role` taxonomy: field exists on OpLog but the vocabulary for role tags (tuple-element name vs dict key vs custom role) is not yet fully exercised by tests. Defer final naming until the taxonomy stabilizes.
- `tl.bridge.hf` extras footprint: only `trace_text` is documented; the rest of the bridge surface (Captum, SHAP, SAE Lens, LIT, profiler) remains under `tl.bridge.*` namespaces with extras-gated imports.
- `gradient_postfunc` vs `grad_transform`: silent alias landed in alpha.3. Rename pass should pick one canonical name.
- `_backend_name`, `_grad_fn_param_refs`, `_param_log_by_pid`, `_append_sequence_id`: underscore-prefixed but partially load-bearing for portable save and visualization. Rename pass should decide which to promote (drop underscore) versus keep private.
