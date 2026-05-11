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
Labels are TorchLens structured identifiers; addresses are PyTorch dotted paths.
Example: `conv2d_1_2` is a Layer label, while `conv2d_1_2:1` is an Op label.
All user-facing type, overall, pass, call, op, and creation indices are 1-based unless an entry explicitly says otherwise.
Private storage names, dropped aliases, and underscore-prefixed locked names are omitted from the main entries.
Deferred items are listed at the end instead of promoted as final API.
## Top-level vocabulary
- `tl.trace(model, x)`: Run a real PyTorch forward pass and return a `Trace`.
- `Trace`: Top-level object for one captured model execution, including graph, tensors, modules, params, buffers, gradients, and metadata.
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
### Capture Config
- `save_raw_activations`: Whether raw untransformed activation tensors are saved.
- `save_raw_gradients`: Whether raw untransformed gradient tensors are saved.
- `save_gradients`: Whether TorchLens should save gradients after backward.
- `save_arg_values`: Whether Op function arguments are saved.
- `save_rng_states`: Whether RNG state snapshots are saved.
- `save_code_context`: Whether source-code context around calls is saved.
- `gradients_to_save`: Which gradients are requested for saving.
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
- `is_multi_output`: True when `multi_output_index` is not `None`.
### Per-Op Config and Saved State
- `output_device`: Device where saved tensors for this Op were placed.
- `activation_transform`: Transform used for the Op's saved output.
- `gradient_transform`: Transform used for the Op's saved gradient.
- `annotations`: User-attached metadata about this Op.
- `detach_saved_activations`: Whether saved tensors were detached.
- `save_gradients`: Whether gradient saving was requested for this Op.
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
- `save_gradients`: Whether gradient saving was requested for this Layer.
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
- `forward_args`: Positional arguments passed to `forward`.
- `forward_kwargs`: Keyword arguments passed to `forward`.
- `forward_args_summary`: Human-readable summary of forward positional arguments.
- `forward_kwargs_summary`: Human-readable summary of forward keyword arguments.
- `call_parent`: Parent ModuleCall label in the dynamic call tree.
- `call_children`: Child ModuleCall labels in the dynamic call tree.
### ModuleCall Output Passthroughs
These properties resolve through the output Layers/Ops. Singular forms require exactly one output; plural forms return lists.
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
- `has_op`: True when `op_label is not None`.
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
- `find_sites` and `resolve_sites`: deferred to the integrated intervention `Site` concept survey.
- `set`, `attach_hooks`, `do`, `clear_hooks`, `remove`, `detach_hooks`, `save_intervention`, and `intervention_spec`: deferred to intervention API review.
- `replace_run_state_from` and `append_run_state_from`: deferred to streaming/rerun-append review.
- `preview_fastlog`: deferred to fastlog namespace review.
- `vis_*` draw parameters: target direction is to drop the prefix, but exact parameter list is deferred.
- Bundle dynamic helpers `aligned_pairs`, `compare`, `delta_map`, `norm_delta`, `output_delta`, and `show_diff`: deferred to Bundle helper redesign.
- `save_activations` and other workflow-level save kwargs: deferred pending decision on workflow vocabulary versus `out` field vocabulary.
- Fast-path naming such as `fastlog`: deferred because fast output is intentionally not a full Trace.
