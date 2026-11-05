# Op: one operation on one pass

An `Op` is one call of one function on one pass (`trace.ops["relu_1_2"]`). A `Layer` groups the
ops of all passes; see `layer.md`. Every field here is per op, so on the loop fixture each pass
has its own value.

## Labels and position

### op-labels-facts
Label fields of one op (per op: the label carries this op's pass).

- covers: `Op.label`, `Op.layer_label`, `Op.short_label`, `Op.short_layer_label`, `Op.raw_label`, `Op.type`, `Op.type_index`, `Op.raw_index`, `Op.ordinal_index`, `Op.step_index`
- tasks: inspect-op, labels
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print("label", op.label)
print("layer_label", op.layer_label)
print("short_label", op.short_label)
print("short_layer_label", op.short_layer_label)
print("raw_label", op.raw_label)
print("type", op.type)
print("type_index", op.type_index)
print("raw_index", op.raw_index)
print("ordinal_index", op.ordinal_index)
print("step_index", op.step_index)
# Output:
# label relu_1_2:1
# layer_label relu_1_2
# short_label relu_1:1
# short_layer_label relu_1
# raw_label relu_1_3
# type relu
# type_index 1
# raw_index 3
# ordinal_index 2
# step_index 2
```

### op-pass-facts
Each pass of a looped module is its own op with its own pass number (per op).

- covers: `Op.pass_number`, `Op.num_passes`, `Op.step_index`, `Op.graph.recurrent_ops`, `Op.graph.equivalent_ops`, `Op.graph.equivalence_class`
- tasks: loops, multi-pass, inspect-op
- fixture: loop

```python
trace = tl.trace(model, x)
op = trace.ops["tanh_1_2:2"]
print("pass_number", op.pass_number)
print("num_passes", op.num_passes)
print("step_index", op.step_index)
print("recurrent_ops", sorted(op.graph.recurrent_ops))
print("equivalent_ops", sorted(op.graph.equivalent_ops))
print("equivalence_class", op.graph.equivalence_class)
# Output:
# pass_number 2
# num_passes 3
# step_index 4
# recurrent_ops ['tanh_1_2:1', 'tanh_1_2:2', 'tanh_1_2:3']
# equivalent_ops ['tanh_1_2:1', 'tanh_1_2:2', 'tanh_1_2:3']
# equivalence_class tanh_966d0e7792d95d96
```

### op-lookup-keys
Every key that finds this op in `trace.ops`, and its addresses (per op).

- covers: `Op.lookup_keys`, `Op.buffer_address`, `Op.backend_address`, `Op.resolver_status`, `Op.graph.site_key`, `Op.fx_label`, `Op.source.fx_path`, `Op.source.fx_call_index`
- tasks: labels, lookup
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("lookup_keys", sorted(map(str, op.lookup_keys)))
print("address", op.buffer_address)
print("backend_address", op.backend_address)
print("resolver_status", op.resolver_status)
print("site_key", op.graph.site_key)
print("fx_label", op.fx_label)
print("fx_qualpath", op.source.fx_path)
print("fx_call_index", op.source.fx_call_index)
# Output:
# lookup_keys ['-4', '1', 'fc1', 'fc1:1', 'fc1_linear_0', 'linear_1', 'linear_1:1', 'linear_1_1', 'linear_1_1:1', 'linear_1_2', 'linear_1_2_raw']
# address fc1
# backend_address fc1
# resolver_status resolved
# site_key s1|fc1|linear||1
# fx_label fc1_linear_0
# fx_qualpath fc1.linear
# fx_call_index 0
```

### op-back-references
An op points back to its layer and to the trace that holds it.

- covers: `Op.layer`, `Op.trace`, `Op.source_trace`
- tasks: inspect-op, navigate
- fixture: loop

```python
trace = tl.trace(model, x)
op = trace.ops["tanh_1_2:3"]
print("layer", op.layer.layer_label, op.layer.num_passes)
print("trace", op.trace is trace)
print("source_trace", op.source_trace is trace)
# Output:
# layer tanh_1_2 3
# trace True
# source_trace True
```

## Saved output

### op-out-facts
The saved output tensor of one op and its description (per op).

- covers: `Op.out`, `Op.shape`, `Op.dtype`, `Op.output_device`, `Op.has_saved_activation`, `Op.detach_saved_activations`, `Op.cost.activation_memory`, `Op.device_ref`, `Op.dtype_ref`
- tasks: inspect-op, read-activation
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print("out", tuple(op.out.shape), bool((op.out >= 0).all()))
print("shape", tuple(op.shape))
print("dtype", op.dtype)
print("output_device", op.output_device)
print("has_saved_activation", op.has_saved_activation)
print("detach_saved_activations", op.detach_saved_activations)
print("activation_memory", op.cost.activation_memory)
print("device_ref", op.device_ref)
print("dtype_ref", op.dtype_ref)
# Output:
# out (3, 8) True
# shape (3, 8)
# dtype torch.float32
# output_device same
# has_saved_activation True
# detach_saved_activations False
# activation_memory 96 B
# device_ref cpu
# dtype_ref torch.float32
```

### op-autograd-memory
Memory the autograd graph keeps for this op's call (per op).

- covers: `Op.cost.autograd_memory`, `Op.autograd.num_autograd_tensors`
- tasks: memory
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("autograd_memory", op.cost.autograd_memory)
print("num_autograd_tensors", op.autograd.num_autograd_tensors)
# Output:
# autograd_memory 48 B
# num_autograd_tensors 1
```

### op-activation-transform
An activation transform stores a changed copy beside the raw output (per op).

- covers: `Op.activation_transform`, `Op.payload.transformed_out`, `Op.payload.transformed_out_shape`, `Op.payload.transformed_out_dtype`, `Op.cost.transformed_activation_memory`
- tasks: transform-activations
- fixture: toy

```python
opts = tl.options.CaptureOptions(activation_transform=lambda t: t.mean(dim=0))
trace = tl.trace(model, x, capture=opts)
op = trace.ops["relu_1_2"]
print("activation_transform", callable(op.activation_transform))
print("transformed_out", tuple(op.payload.transformed_out.shape))
print("transformed_out_shape", tuple(op.payload.transformed_out_shape))
print("transformed_out_dtype", op.payload.transformed_out_dtype)
print("transformed_activation_memory", op.cost.transformed_activation_memory)
# Output:
# activation_transform True
# transformed_out (8,)
# transformed_out_shape (8,)
# transformed_out_dtype torch.float32
# transformed_activation_memory 32 B
```

### op-lazy-out
A lazy load keeps outputs on disk until you ask for one.

- covers: `Op.out_ref`, `Op.payload.materialize_out`
- tasks: save-load, lazy
- fixture: toy

```python
trace = tl.trace(model, x)
trace.save("run.tlspec")
lazy = tl.load("run.tlspec", lazy=True)
op = lazy.ops["relu_1_2"]
print("out_ref", op.out_ref is not None)
print("materialize_out", tuple(op.payload.materialize_out().shape))
# Output:
# out_ref True
# materialize_out (3, 8)
```

## Function call

### op-func-facts
Which function an op ran and its non-tensor settings (per op).

- covers: `Op.func`, `Op.func_name`, `Op.source.func_qualname`, `Op.call.config`, `Op.call.call_id`, `Op.call.func_id`, `Op.call.non_tensor_args`, `Op.call.non_tensor_pos_args`, `Op.call.non_tensor_kwargs`, `Op.is_inplace`
- tasks: inspect-op, function
- fixture: cnn

```python
trace = tl.trace(model, x)
op = trace.ops["conv2d_1_1"]
print("func", op.func.__name__)
print("func_name", op.func_name)
print("func_qualname", op.source.func_qualname)
print("func_config", op.call.config)
print("func_call_id", op.call.call_id)
print("func_id", op.call.func_id)
print("func_non_tensor_args", op.call.non_tensor_args)
print("non_tensor_pos_args", op.call.non_tensor_pos_args)
print("non_tensor_kwargs", op.call.non_tensor_kwargs)
print("is_inplace", op.is_inplace)
# Output:
# func conv2d
# func_name conv2d
# func_qualname _VariableFunctionsClass.conv2d
# func_config {'out_channels': 4, 'in_channels': 3, 'kernel_size': (3, 3), 'padding': (1, 1)}
# func_call_id 1
# func_id None
# func_non_tensor_args [(1, 1), (1, 1), (1, 1), 1]
# non_tensor_pos_args [(1, 1), (1, 1), (1, 1), 1]
# non_tensor_kwargs {}
# is_inplace False
```

### op-args-facts
How an op's call arguments were spelled and counted (per op).

- covers: `Op.call.arg_names`, `Op.call.num_args_total`, `Op.call.num_pos_args`, `Op.call.num_kwargs`, `Op.args_summary`, `Op.kwargs_summary`, `Op.source.arg_expressions`, `Op.source.var_names`, `Op.call.parent_arg_positions`
- tasks: inspect-op, function
- fixture: none

```python
class Scale(torch.nn.Module):
    def forward(self, x):
        y = torch.mul(x, 2.0)
        return torch.clamp(y, min=-1.0)
trace = tl.trace(Scale(), torch.randn(2, 3))
op = trace.ops["clamp_1_2"]
print("arg_names", op.call.arg_names)
print("num_args_total", op.call.num_args_total)
print("num_pos_args", op.call.num_pos_args)
print("num_kwargs", op.call.num_kwargs)
print("args_summary", trace.ops["mul_1_1"].args_summary)
print("kwargs_summary", op.kwargs_summary)
print("arg_expressions", op.source.arg_expressions)
print("var_names", trace.ops["mul_1_1"].source.var_names)
print("parent_arg_positions", op.call.parent_arg_positions)
# Output:
# arg_names ('input', 'min', 'max', 'out')
# num_args_total 2
# num_pos_args 1
# num_kwargs 1
# args_summary 2.0
# kwargs_summary min=-1.0
# arg_expressions ['y', 'min=-1.0']
# var_names []
# parent_arg_positions {'args': {0: 'mul_1_1'}, 'kwargs': {}}
```

### op-arg-templates
The structural templates TorchLens keeps to rebuild an op's call.

- covers: `Op.call.args_template`, `Op.call.kwargs_template`
- tasks: replay, function
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("args_template", type(op.call.args_template).__name__, len(op.call.args_template.args))
print("kwargs_template", type(op.call.kwargs_template).__name__)
# Output:
# args_template CapturedArgTemplate 3
# kwargs_template NoneType
```

### op-saved-args
With `save_args=True` each op keeps the argument values it was called with.

- covers: `Op.call.saved_args`, `Op.call.saved_kwargs`, `Op.call.has_saved_args`
- tasks: replay, function
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True))
op = trace.ops["linear_1_1"]
print("has_saved_args", op.call.has_saved_args)
print("saved_args", [tuple(a.shape) for a in op.call.saved_args])
print("saved_kwargs", op.call.saved_kwargs)
# Output:
# has_saved_args True
# saved_args [(3, 4), (8, 4), (8,)]
# saved_kwargs {}
```

### op-input-activations
The saved outputs of an op's parents, the values it consumed (per op).

- covers: `Op.input_activations`, `Op.input_shapes`, `Op.input_dtypes`, `Op.cost.input_memory`
- tasks: inspect-op, read-activation
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_2_3"]
print("input_activations", [tuple(t.shape) for t in op.input_activations])
print("input_shapes", op.input_shapes)
print("input_dtypes", op.input_dtypes)
print("input_memory", op.cost.input_memory)
# Output:
# input_activations [(3, 8)]
# input_shapes (torch.Size([3, 8]),)
# input_dtypes (torch.float32,)
# input_memory 96 B
```

### op-call-environment
Timing, RNG and autocast state recorded around an op's call (per op).

- covers: `Op.cost.duration`, `Op.call.rng_states`, `Op.call.autocast_state`, `Op.cost.bytes_delta_at_call`, `Op.cost.bytes_peak_at_call`
- tasks: timing, inspect-op
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("func_duration", type(op.cost.duration).__name__)
print("func_rng_states", type(op.call.rng_states).__name__)
print("func_autocast_state", sorted(op.call.autocast_state))
print("bytes_delta_at_call", op.cost.bytes_delta_at_call)
print("bytes_peak_at_call", type(op.cost.bytes_peak_at_call).__name__)
# Output:
# func_duration Duration
# func_rng_states dict
# func_autocast_state ['__execution__', 'cpu', 'cuda']
# bytes_delta_at_call None
# bytes_peak_at_call NoneType
```

### op-code-context
Where in your code an op ran: the last frame is the line that called the function.

- covers: `Op.source.frames`
- tasks: source, inspect-op
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
frame = op.source.frames[-1]
print(len(op.source.frames) > 0)
print(frame.func_name)
print(frame.call_line.strip())
# Output:
# True
# forward
# return self.fc2(torch.relu(self.fc1(x)))
```

### op-grad-fn
The autograd node behind an op's output (per op; the live handle is not saved to disk).

- covers: `Op.autograd.grad_fn_class_name`, `Op.autograd.grad_fn_class_qualname`, `Op.autograd.grad_fn_handle`, `Op.grad_fn_object_id`
- tasks: autograd, inspect-op
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
op = trace.ops["linear_1_1"]
print("grad_fn_class_name", op.autograd.grad_fn_class_name)
print("grad_fn_class_qualname", op.autograd.grad_fn_class_qualname)
print("grad_fn_handle", type(op.autograd.grad_fn_handle).__name__)
print("grad_fn_object_id", isinstance(op.grad_fn_object_id, int))
# Output:
# grad_fn_class_name AddmmBackward0
# grad_fn_class_qualname builtins.AddmmBackward0
# grad_fn_handle AddmmBackward0
# grad_fn_object_id True
```

### op-grad-fn-after-backward
After a backward pass the op links to its backward record and the live handle is released.

- covers: `Op.autograd.grad_fn`, `Op.grad_fn_label`, `Op.autograd.grad_fn_handle`
- tasks: autograd, backward
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
trace.backward(trace.ops["output_1"].out.sum())
op = trace.ops["linear_1_1"]
print("grad_fn", type(op.autograd.grad_fn).__name__, op.autograd.grad_fn.label, op.autograd.grad_fn.op_label)
print("grad_fn_label", op.grad_fn_label)
print("grad_fn_handle", op.autograd.grad_fn_handle)
# Output:
# grad_fn GradFn addmm_back_2_7 linear_1_1
# grad_fn_label addmm_back_2_7
# grad_fn_handle None
```

## Cost

### op-cost-facts
FLOPs, MACs and ideal memory traffic of one conv call (per op).

- covers: `Op.cost.flops_forward`, `Op.cost.flops_backward`, `Op.cost.flops_total`, `Op.cost.macs_forward`, `Op.macs_backward`, `Op.macs_total`, `Op.cost.bytes_read`, `Op.cost.bytes_written`, `Op.cost.arithmetic_intensity`, `Op.cost.compute_record`
- tasks: flops, cost
- fixture: cnn

```python
trace = tl.trace(model, x)
op = trace.ops["conv2d_1_1"]
print("flops_forward", op.cost.flops_forward)
print("flops_backward", op.cost.flops_backward)
print("flops_total", op.cost.flops_total)
print("macs_forward", op.cost.macs_forward)
print("macs_backward", op.macs_backward)
print("macs_total", op.macs_total)
print("bytes_read", op.cost.bytes_read)
print("bytes_written", op.cost.bytes_written)
print("arithmetic_intensity", round(op.cost.arithmetic_intensity, 3))
print("compute_record", op.cost.compute_record)
# Output:
# flops_forward 14.1 KFLOPs
# flops_backward 28.2 KFLOPs
# flops_total 42.2 KFLOPs
# macs_forward 6.91 KMACs
# macs_backward None
# macs_total None
# bytes_read 1.19 KiB
# bytes_written 1.00 KiB
# arithmetic_intensity 6.286
# compute_record ComputeRecord(fma_macs=6912, other_flops=256, mac_applicability='mac', evidence='formula_exact', reason=None)
```

## Parameters

### op-params-facts
Parameter counts of a linear op after freezing its bias (per op).

- covers: `Op.has_params`, `Op.num_params`, `Op.num_params_trainable`, `Op.num_params_frozen`, `Op.num_param_tensors_trainable`, `Op.num_param_tensors_frozen`, `Op.has_trainable_params`, `Op.has_frozen_params`, `Op.param_names`, `Op.param_shapes`, `Op.param_dtypes`, `Op.cost.param_memory`
- tasks: params, inspect-op
- fixture: toy

```python
model.fc1.bias.requires_grad_(False)
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("has_params", op.has_params)
print("num_params", op.num_params)
print("num_params_trainable", op.num_params_trainable)
print("num_params_frozen", op.num_params_frozen)
print("num_param_tensors_trainable", op.num_param_tensors_trainable)
print("num_param_tensors_frozen", op.num_param_tensors_frozen)
print("has_trainable_params", op.has_trainable_params)
print("has_frozen_params", op.has_frozen_params)
print("param_names", op.param_names)
print("param_shapes", op.param_shapes)
print("param_dtypes", op.param_dtypes)
print("param_memory", op.cost.param_memory)
# Output:
# has_params True
# num_params 40
# num_params_trainable 32
# num_params_frozen 8
# num_param_tensors_trainable 1
# num_param_tensors_frozen 1
# has_trainable_params True
# has_frozen_params True
# param_names ['weight', 'bias']
# param_shapes [(8, 4), (8,)]
# param_dtypes [torch.float32, torch.float32]
# param_memory 160 B
```

### op-param-objects
`Op.params` lists the parameters an op used; `parent_params` is emptied after capture to free memory.

- covers: `Op.params`, `Op.graph.parent_params`, `Op.graph.parent_param_ops`
- tasks: params
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("params", [p.address for p in op.params])
print("parent_param_ops", sorted(op.graph.parent_param_ops.values()))
print("parent_params", op.graph.parent_params)
# Output:
# params ['fc1.weight', 'fc1.bias']
# parent_param_ops [1, 1]
# parent_params ()
```

## Graph neighbours

### op-graph-facts
An op's direct neighbours in the graph, as pass-qualified labels (per op).

- covers: `Op.parents`, `Op.children`, `Op.graph.has_parents`, `Op.graph.has_children`, `Op.graph.siblings`, `Op.graph.has_siblings`, `Op.graph.co_parents`, `Op.graph.has_co_parents`, `Op.graph.num_inputs`, `Op.input_ops`, `Op.graph.root_ancestors`
- tasks: graph, navigate
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print("parents", op.parents)
print("children", op.children)
print("has_parents", op.graph.has_parents)
print("has_children", op.graph.has_children)
print("siblings", op.graph.siblings)
print("has_siblings", op.graph.has_siblings)
print("co_parents", op.graph.co_parents)
print("has_co_parents", op.graph.has_co_parents)
print("num_inputs", op.graph.num_inputs)
print("input_ops", [o.label for o in op.input_ops])
print("root_ancestors", op.graph.root_ancestors)
# Output:
# parents ('linear_1_1',)
# children ('linear_2_3',)
# has_parents True
# has_children True
# siblings []
# has_siblings False
# co_parents []
# has_co_parents False
# num_inputs 1
# input_ops ['linear_1_1:1']
# root_ancestors frozenset({'input_1'})
```

### op-get-parents-children
Walk from an op to its neighbour `Op` objects instead of their labels.

- covers: `Op.get_parents`, `Op.get_children`
- tasks: graph, navigate
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print([p.label for p in op.get_parents()])
print([c.label for c in op.get_children()])
# Output:
# ['linear_1_1:1']
# ['linear_2_3:1']
```

### op-ancestry-facts
Which inputs and outputs an op connects to, and how far away they are (per op).

- covers: `Op.input_ancestors`, `Op.output_descendants`, `Op.graph.has_input_ancestor`, `Op.graph.has_output_descendant`, `Op.graph.min_distance_from_input`, `Op.graph.max_distance_from_input`, `Op.graph.min_distance_to_output`, `Op.graph.max_distance_to_output`, `Op.graph.is_orphan`, `Op.is_internal_sink`
- tasks: graph
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print("input_ancestors", op.input_ancestors)
print("output_descendants", op.output_descendants)
print("has_input_ancestor", op.graph.has_input_ancestor)
print("has_output_descendant", op.graph.has_output_descendant)
print("min_distance_from_input", op.graph.min_distance_from_input)
print("max_distance_from_input", op.graph.max_distance_from_input)
print("min_distance_to_output", op.graph.min_distance_to_output)
print("max_distance_to_output", op.graph.max_distance_to_output)
print("is_orphan", op.graph.is_orphan)
print("is_internal_sink", op.is_internal_sink)
# Output:
# input_ancestors frozenset({'input_1'})
# output_descendants frozenset({'output_1'})
# has_input_ancestor True
# has_output_descendant True
# min_distance_from_input 2
# max_distance_from_input 2
# min_distance_to_output 2
# max_distance_to_output 2
# is_orphan False
# is_internal_sink False
```

### op-io-role
Which ops are inputs, outputs or real computations, one row per op.

- covers: `Op.graph.io_role`, `Op.is_input`, `Op.is_output`, `Op.graph.is_final_output`, `Op.graph.is_output_parent`, `Op.graph.is_compute`, `Op.input_was_parameter`
- tasks: graph, io
- fixture: toy

```python
trace = tl.trace(model, x)
for label in trace.op_labels:
    op = trace.ops[label]
    print(label, op.graph.io_role, op.is_input, op.is_output, op.graph.is_final_output,
          op.graph.is_output_parent, op.graph.is_compute, op.input_was_parameter)
# Output:
# input_1:1 input.x True False False False False False
# linear_1_1:1 None False False False False True False
# relu_1_2:1 None False False False False True False
# linear_2_3:1 None False False False True True False
# output_1:1 output False True True True False False
```

### op-internal-source
A tensor made inside `forward` (here `torch.ones`) is an internal source, not an input.

- covers: `Op.graph.is_internal_source`, `Op.call.is_internally_initialized`, `Op.is_constant`, `Op.graph.internal_source_parents`, `Op.graph.internal_source_ancestors`, `Op.graph.has_internal_source_ancestor`, `Op.call.tl_authored_root`
- tasks: graph, internal-source
- fixture: none

```python
class AddOnes(torch.nn.Module):
    def forward(self, x):
        return x + torch.ones(4)
trace = tl.trace(AddOnes(), torch.randn(2, 4))
ones, add = trace.ops["ones_1_1"], trace.ops["add_1_2"]
print("is_internal_source", ones.graph.is_internal_source)
print("is_internally_initialized", ones.call.is_internally_initialized)
print("is_constant", ones.is_constant, add.is_constant)
print("internal_source_parents", add.graph.internal_source_parents)
print("internal_source_ancestors", add.graph.internal_source_ancestors)
print("has_internal_source_ancestor", add.graph.has_internal_source_ancestor)
print("tl_authored_root", ones.call.tl_authored_root)
# Output:
# is_internal_source True
# is_internally_initialized True
# is_constant True False
# internal_source_parents ('ones_1_1',)
# internal_source_ancestors frozenset({'ones_1_1'})
# has_internal_source_ancestor True
# tl_authored_root None
```

### op-edge-records
Per-edge records: how each parent fed this op and which tensor args made no edge.

- covers: `Op.edge_uses`, `Op.call.dropped_edge_tensor_args`, `Op.call.unattributed_tensor_args`, `Op.graph.edge_substitutions`, `Op.edge_replacement_stamps`, `Op.out_versions_by_child`, `Op.has_out_variations`
- tasks: graph, replay
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_2_3"]
print("edge_uses", op.edge_uses)
print("dropped_edge_tensor_args", op.call.dropped_edge_tensor_args)
print("unattributed_tensor_args", op.call.unattributed_tensor_args)
print("edge_substitutions", op.graph.edge_substitutions)
print("edge_replacement_stamps", op.edge_replacement_stamps)
print("out_versions_by_child", op.out_versions_by_child)
print("has_out_variations", op.has_out_variations)
# Output:
# edge_uses (EdgeUseRecord: positional argument 0,)
# dropped_edge_tensor_args ()
# unattributed_tensor_args ()
# edge_substitutions None
# edge_replacement_stamps None
# out_versions_by_child {}
# has_out_variations False
```

## Modules

### op-module-facts
Which module an op ran inside, and whether it is that module's only op (per op).

- covers: `Op.module`, `Op.modules`, `Op.module_context.call_stack`, `Op.module_context.call_depth`, `Op.in_submodule`, `Op.module_context.is_atomic_module`, `Op.module_context.atomic_module`, `Op.module_context.atomic_module_address`, `Op.module_context.atomic_module_call`, `Op.atomic_module_call_label`
- tasks: modules, inspect-op
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("module", op.module)
print("modules", op.modules)
print("module_call_stack", op.module_context.call_stack)
print("module_call_depth", op.module_context.call_depth)
print("in_submodule", op.in_submodule)
print("is_atomic_module", op.module_context.is_atomic_module)
print("atomic_module", type(op.module_context.atomic_module).__name__, op.module_context.atomic_module.address)
print("atomic_module_address", op.module_context.atomic_module_address)
print("atomic_module_call", op.module_context.atomic_module_call)
print("atomic_module_call_label", op.atomic_module_call_label)
# Output:
# module fc1:1
# modules ('fc1:1',)
# module_call_stack ('fc1:1',)
# module_call_depth 1
# in_submodule True
# is_atomic_module True
# atomic_module Module fc1
# atomic_module_address fc1
# atomic_module_call fc1:1
# atomic_module_call_label fc1:1
```

### op-module-io
Which module calls an op's output entered or left (per op).

- covers: `Op.module_context.input_to_modules`, `Op.module_context.input_to_module_calls`, `Op.module_context.output_of_modules`, `Op.module_context.output_of_module_calls`, `Op.module_context.is_module_input`, `Op.module_context.is_module_output`, `Op.module_context.entry_arg_keys`
- tasks: modules, io
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print("input_to_modules", op.module_context.input_to_modules)
print("input_to_module_calls", op.module_context.input_to_module_calls)
print("is_module_input", op.module_context.is_module_input)
print("module_entry_arg_keys", dict(op.module_context.entry_arg_keys))
fc1 = trace.ops["linear_1_1"]
print("output_of_modules", fc1.module_context.output_of_modules)
print("output_of_module_calls", fc1.module_context.output_of_module_calls)
print("is_module_output", fc1.module_context.is_module_output)
# Output:
# input_to_modules ['fc2']
# input_to_module_calls ('fc2:1',)
# is_module_input True
# module_entry_arg_keys {'fc2:1': [0]}
# output_of_modules ('fc1',)
# output_of_module_calls ('fc1:1',)
# is_module_output True
```

## Multi-output and containers

### op-multi-output
`torch.chunk` returns a tuple: each piece is its own op with its position in the tuple.

- covers: `Op.in_multi_output`, `Op.multi_output_index`, `Op.multi_output_name`, `Op.multi_output_type`
- tasks: multi-output
- fixture: none

```python
class Halves(torch.nn.Module):
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return a * b
trace = tl.trace(Halves(), torch.randn(2, 4))
for label in trace.op_labels:
    op = trace.ops[label]
    if op.in_multi_output:
        print(label, op.multi_output_index, op.multi_output_name, op.multi_output_type)
# Output:
# chunk_1_1:1 0 None None
# chunk_2_2:1 1 None None
```

### op-containers
With container capture on, output ops know the dict they were returned in.

- covers: `Op.container`, `Op.container_path`, `Op.container_spec`, `Op.containers`, `Op.input_containers`, `Op.output_containers`
- tasks: containers, multi-output
- fixture: none

```python
class Pair(torch.nn.Module):
    def forward(self, x):
        return {"pos": torch.relu(x), "neg": torch.relu(-x)}
opts = tl.options.CaptureOptions(capture_container_structure=True)
trace = tl.trace(Pair(), torch.randn(2, 4), capture=opts)
op = trace.ops[trace.output_layers[0]]
print("container", op.container.kind)
print("container_path", op.container_path)
print("container_spec", op.container_spec.kind, op.container_spec.keys)
print("containers", [c.kind for c in op.containers])
print("input_containers", op.input_containers)
print("output_containers", [c.root_kind for c in op.output_containers])
# Output:
# container dict
# container_path (DictKey(key='pos'),)
# container_spec dict ('pos', 'neg')
# containers ['dict']
# input_containers ()
# output_containers ['final_output']
```

## Gradients

### op-grad-facts
After one backward pass each op holds its output gradient (per op).

- covers: `Op.grad`, `Op.has_grad`, `Op.has_saved_grad`, `Op.autograd.grad_shape`, `Op.autograd.grad_dtype`, `Op.gradient_memory`, `Op.save_grads`, `Op.autograd.derived_grad`
- tasks: gradients, backward
- fixture: toy

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all")
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum())
op = trace.ops["relu_1_2"]
print("grad", tuple(op.grad.shape))
print("has_grad", op.has_grad)
print("has_saved_grad", op.has_saved_grad)
print("grad_shape", tuple(op.autograd.grad_shape))
print("grad_dtype", op.autograd.grad_dtype)
print("gradient_memory", op.gradient_memory)
print("save_grads", op.save_grads)
print("derived_grad", op.autograd.derived_grad)
# Output:
# grad (3, 8)
# has_grad True
# has_saved_grad True
# grad_shape (3, 8)
# grad_dtype torch.float32
# gradient_memory 96 B
# save_grads True
# derived_grad None
```

### op-grads-per-backward
Two backward passes give each op one gradient record per pass; `grad_for` picks one.

- covers: `Op.autograd.grads`, `Op.autograd.grad_for`
- tasks: gradients, backward
- fixture: toy

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all")
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum(), retain_graph=True)
trace.backward(trace.ops["output_1"].out.sum() * 2)
op = trace.ops["relu_1_2"]
print("grads", len(op.autograd.grads))
first, second = op.autograd.grad_for(bwd=1), op.autograd.grad_for(bwd=2)
print("grad_for", tuple(first.shape), bool(torch.allclose(second, 2 * first)))
# Output:
# grads 2
# grad_for (3, 8) True
```

### op-gradient-transform
A gradient transform stores a changed copy of each saved gradient (per op).

- covers: `Op.gradient_transform`, `Op.payload.transformed_grad`, `Op.payload.transformed_grad_shape`, `Op.payload.transformed_grad_dtype`, `Op.transformed_gradient_memory`
- tasks: gradients, transform-activations
- fixture: toy

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all", grad_transform=lambda g: g.abs().sum(dim=0))
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum())
op = trace.ops["relu_1_2"]
print("gradient_transform", callable(op.gradient_transform))
print("transformed_grad", tuple(op.payload.transformed_grad.shape))
print("transformed_grad_shape", tuple(op.payload.transformed_grad_shape))
print("transformed_grad_dtype", op.payload.transformed_grad_dtype)
print("transformed_gradient_memory", op.transformed_gradient_memory)
# Output:
# gradient_transform True
# transformed_grad (8,)
# transformed_grad_shape (8,)
# transformed_grad_dtype torch.float32
# transformed_gradient_memory 32 B
```

### op-lazy-grad
A lazy load keeps saved gradients on disk until you ask for one.

- covers: `Op.grad_ref`, `Op.payload.materialize_grad`
- tasks: save-load, lazy, gradients
- fixture: toy

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all")
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum())
trace.save("run.tlspec")
op = tl.load("run.tlspec", lazy=True).ops["relu_1_2"]
print("grad_ref", op.grad_ref is not None)
print("materialize_grad", tuple(op.payload.materialize_grad().shape))
# Output:
# grad_ref True
# materialize_grad (3, 8)
```

## Conditionals

### op-conditional-test
The `h.sum() > 0` test of an `if` is a terminal bool op (per op).

- covers: `Op.control_flow.is_scalar_bool`, `Op.control_flow.bool_value`, `Op.control_flow.is_terminal_bool`, `Op.control_flow.is_terminal_conditional_bool`, `Op.control_flow.terminal_bool_for`, `Op.terminal_conditional_id`, `Op.control_flow.context_kind`, `Op.control_flow.wrapper_kind`, `Op.control_flow.is_in_conditional_evaluation`
- tasks: conditionals, branch
- fixture: branch

```python
trace = tl.trace(model, x)
op = next(trace.ops[l] for l in trace.op_labels if trace.ops[l].control_flow.is_terminal_bool)
print("label", op.label)
print("is_scalar_bool", op.control_flow.is_scalar_bool)
print("bool_value", op.control_flow.bool_value)
print("is_terminal_bool", op.control_flow.is_terminal_bool)
print("is_terminal_conditional_bool", op.control_flow.is_terminal_conditional_bool)
print("terminal_bool_for", op.control_flow.terminal_bool_for)
print("terminal_conditional_id", op.terminal_conditional_id)
print("conditional_context_kind", op.control_flow.context_kind)
print("conditional_wrapper_kind", op.control_flow.wrapper_kind)
print("is_in_conditional_evaluation", op.control_flow.is_in_conditional_evaluation)
# Output:
# label gt_1_3:1
# is_scalar_bool True
# bool_value True
# is_terminal_bool True
# is_terminal_conditional_bool True
# terminal_bool_for ('cond_gt_1_3', 0)
# terminal_conditional_id 0
# conditional_context_kind if_test
# conditional_wrapper_kind None
# is_in_conditional_evaluation True
```

### op-conditional-body
Ops in the `if` test or the taken arm carry their conditional roles and branch stack (per op).

- covers: `Op.control_flow.is_in_conditional`, `Op.control_flow.in_conditionals`, `Op.control_flow.depth`, `Op.control_flow.branch_depth`, `Op.control_flow.branch_stack`
- tasks: conditionals, branch
- fixture: branch

```python
trace = tl.trace(model, x)
for label in trace.op_labels:
    op = trace.ops[label]
    print(label, op.control_flow.is_in_conditional, op.control_flow.in_conditionals,
          op.control_flow.depth, op.control_flow.branch_depth, op.control_flow.branch_stack)
# Output:
# input_1:1 False () 0 0 ()
# linear_1_1:1 False () 0 0 ()
# sum_1_2:1 False () 0 0 ()
# gt_1_3:1 True (ConditionalRoleRef(conditional_id='cond_gt_1_3', arm_index=0, arm_kind='then', role='evaluation'),) 1 0 ()
# relu_1_4:1 True (ConditionalRoleRef(conditional_id='cond_gt_1_3', arm_index=0, arm_kind='then', role='body'),) 1 1 ((0, 'then'),)
# output_1:1 False () 0 0 ()
```

### op-conditional-children
The op before an `if` lists which of its children sit in which arm (per op).

- covers: `Op.control_flow.arm_children`, `Op.control_flow.then_children`, `Op.control_flow.else_children`, `Op.control_flow.elif_children`, `Op.control_flow.entry_children`
- tasks: conditionals, branch
- fixture: branch

```python
trace = tl.trace(model, x)
op = trace.ops["linear_1_1"]
print("conditional_arm_children", op.control_flow.arm_children)
print("conditional_then_children", op.control_flow.then_children)
print("conditional_else_children", op.control_flow.else_children)
print("conditional_elif_children", op.control_flow.elif_children)
print("conditional_entry_children", op.control_flow.entry_children)
# Output:
# conditional_arm_children {0: {'then': ['relu_1_4']}}
# conditional_then_children ('relu_1_4',)
# conditional_else_children ()
# conditional_elif_children {}
# conditional_entry_children ('sum_1_2',)
```

## Buffers

### op-buffer-writes
A training-mode batch norm writes its running stats: each write is a buffer op.

- covers: `Op.graph.is_buffer`, `Op.is_buffer_source`, `Op.buffer_pass`, `Op.buffer_write_kind`, `Op.graph.buffer_source`, `Op.buffer_value_changed`, `Op.buffer_replay_validated`, `Op.graph.buffer_source_func_name`
- tasks: buffers
- fixture: none

```python
torch.manual_seed(0)
trace = tl.trace(torch.nn.BatchNorm1d(4), torch.randn(8, 4))
for label in trace.op_labels:
    op = trace.ops[label]
    if op.graph.is_buffer:
        print(label, op.buffer_address, op.is_buffer_source, op.graph.buffer_pass, op.buffer_write_kind,
              op.graph.buffer_source, op.buffer_value_changed, op.buffer_replay_validated,
              op.graph.buffer_source_func_name)
# Output:
# buffer_1:1 num_batches_tracked True 1 None None None None None
# buffer_2:1 num_batches_tracked True 2 inplace add_1_1 True None add_
# buffer_3:1 running_mean True 1 None None None None None
# buffer_4:1 running_var True 1 None None None None None
# buffer_5:1 running_mean True 2 fused batchnorm_1_2 True False batch_norm
# buffer_6:1 running_var True 2 fused batchnorm_1_2 True False batch_norm
```

### op-buffer-links
Walk from a buffer write to the op that produced it, and from an op to the writes it feeds.

- covers: `Op.graph.buffer_source_ops`, `Op.graph.buffer_sink_ops`
- tasks: buffers, navigate
- fixture: none

```python
torch.manual_seed(0)
trace = tl.trace(torch.nn.BatchNorm1d(4), torch.randn(8, 4))
for label in trace.op_labels:
    op = trace.ops[label]
    sources, sinks = list(op.graph.buffer_source_ops), list(op.graph.buffer_sink_ops)
    if sources or sinks:
        print(label, [o.label for o in sources], [o.label for o in sinks])
# Output:
# add_1_1:1 ['buffer_1:1'] ['buffer_2:1']
# batchnorm_1_2:1 ['buffer_3:1', 'buffer_4:1'] ['buffer_5:1', 'buffer_6:1']
```

## Function transforms

### op-transform-boundary
`torch.func.vmap` shows up as one boundary op that names the function it mapped.

- covers: `Op.call.is_func_transform`, `Op.call.func_transform_kind`, `Op.call.func_transform_chain`, `Op.call.func_transform_config`, `Op.call.func_transform_fn_name`, `Op.call.func_transform_fn_qualname`, `Op.call.func_transform_fn_source`
- tasks: transforms, vmap
- fixture: none

```python
def double(v):
    return v * 2
class Mapped(torch.nn.Module):
    def forward(self, x):
        return torch.func.vmap(double)(x)
trace = tl.trace(Mapped(), torch.randn(3, 2))
op = next(trace.ops[l] for l in trace.op_labels if trace.ops[l].call.is_func_transform)
print("label", op.label)
print("transform_kind", op.call.func_transform_kind)
print("transform_chain", op.call.func_transform_chain)
print("transform_config", {k: op.call.func_transform_config[k] for k in ("in_dims", "out_dims")})
print("transform_fn_name", op.call.func_transform_fn_name)
print("transform_fn_qualname", op.call.func_transform_fn_qualname)
print("transform_fn_source", op.call.func_transform_fn_source.func_name)
# Output:
# label vmap_1_1:1
# transform_kind vmap
# transform_chain ('vmap',)
# transform_config {'in_dims': 0, 'out_dims': 0}
# transform_fn_name double
# transform_fn_qualname double
# transform_fn_source double
```

## Interventions and annotations

### op-do-ablate
`Op.do` edits this op through its trace; the edited op records the edit. Edit a fork (on the root trace it warns).

- covers: `Op.do`, `Op.is_edited`, `Op.edits`
- tasks: intervene, ablate
- fixture: toy

```python
trace = tl.trace(model, x)
fork = trace.fork()
edited = fork.ops["relu_1_2"].do(tl.zero_ablate())
op = edited.ops["relu_1_2"]
print("is_edited", op.is_edited)
print("edits", len(op.edits))
print("zeroed", float(op.out.abs().sum()))
# Output:
# is_edited True
# edits 1
# zeroed 0.0
```

### op-injection-provenance
Ordinary model ops carry no injection record; only intervention-made ops do.

- covers: `Op.call.injection_provenance`, `Op.episode_step`
- tasks: intervene, provenance
- fixture: toy

```python
trace = tl.trace(model, x)
op = trace.ops["relu_1_2"]
print("injection_provenance", op.call.injection_provenance)
print("episode_step", op.episode_step)
# Output:
# injection_provenance None
# episode_step None
```

### op-annotations
Attach your own data to an op with `Trace.annotate`, then read it back on the op.

- covers: `Op.annotations`
- tasks: annotate
- fixture: toy

```python
trace = tl.trace(model, x)
trace.annotate("relu_1_2", data={"note": "after fc1"})
print(trace.ops["relu_1_2"].annotations)
print(trace.ops["linear_1_1"].annotations)
# Output:
# {'user': {'data': {'note': 'after fc1'}}}
# {}
```

### op-facets
An op's facet view: named read-outs registered for its function.

- covers: `Op.facets`
- tasks: facets
- fixture: toy

```python
trace = tl.trace(model, x)
facets = trace.ops["linear_1_1"].facets
print(type(facets).__name__)
print(facets.keys())
# Output:
# FacetView
# ['out']
```

## Fields, views and display

### op-receptive-field
Which input pixels a conv output can see, and which outputs one input pixel reaches.

- covers: `Op.receptive_field`, `Op.projective_field`
- tasks: receptive-field
- fixture: cnn

```python
trace = tl.trace(model, x)
op = trace.ops["conv2d_2_3"]
print("receptive_field", op.receptive_field.size, op.receptive_field.status)
projective = trace.ops["conv2d_1_1"].projective_field
print("projective_field", projective.size, projective.status)
# Output:
# receptive_field (5, 5) ReceptiveFieldStatus.EXACT
# projective_field (3, 3) ReceptiveFieldStatus.EXACT
```

### op-to-pandas
One op as a one-row table of its fields.

- covers: `Op.to_pandas`
- tasks: table, export
- fixture: toy

```python
trace = tl.trace(model, x)
frame = trace.ops["relu_1_2"].to_pandas()
print(type(frame).__name__, frame.shape[0])
print(frame["label"].iloc[0])
# Output:
# DataFrame 1
# relu_1_2:1
```

### op-show
Draw an op's saved output; without plotting available it returns a text note instead.

- covers: `Op.show`, `Op.visualizer_path`
- tasks: plot, visualize
- fixture: cnn

```python
trace = tl.trace(model, x)
op = trace.ops["conv2d_1_1"]
picture = op.show()
print(type(picture).__name__)
print("visualizer_path", op.visualizer_path)
# Output:
# Figure
# visualizer_path None
```
