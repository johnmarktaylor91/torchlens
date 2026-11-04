# Layer: one operation across all its passes

A `Layer` groups the ops that repeat the same operation at the same place, one op per pass
(`trace.layers["tanh_1_2"]`). Layer-level fields describe the whole layer (all passes);
per-pass fields such as `label` or `out` read through to the op only when the layer has one
pass, and raise on a looped layer: read them on `layer.ops[k]` instead. Per-op fields: `op.md`.

## Labels and passes

### layer-label-facts
Layer-level labels and pass counts of a looped layer (per layer: all passes).

- covers: `Layer.layer_label`, `Layer.short_layer_label`, `Layer.layer_type`, `Layer.type_index`, `Layer.num_passes`, `Layer.__len__`, `Layer.op_labels`, `Layer.call_labels`, `Layer.raw_index`, `Layer.ordinal_index`
- tasks: inspect-layer, loops, multi-pass
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["tanh_1_2"]
print("layer_label", layer.layer_label)
print("short_layer_label", layer.short_layer_label)
print("layer_type", layer.layer_type)
print("type_index", layer.type_index)
print("num_passes", layer.num_passes)
print("len", len(layer))
print("op_labels", layer.op_labels)
print("call_labels", layer.call_labels)
print("raw_index", layer.raw_index)
print("ordinal_index", layer.ordinal_index)
# Output:
# layer_label tanh_1_2
# short_layer_label tanh_1
# layer_type tanh
# type_index 1
# num_passes 3
# len 3
# op_labels ['tanh_1_2:1', 'tanh_1_2:2', 'tanh_1_2:3']
# call_labels ['tanh_1_2:1', 'tanh_1_2:2', 'tanh_1_2:3']
# raw_index 3
# ordinal_index 2
```

### layer-ops-per-pass
`layer.ops[k]` is the op of one pass; read per-pass values there on a looped layer.

- covers: `Layer.ops`
- tasks: inspect-layer, loops, multi-pass
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["tanh_1_2"]
for op in layer.ops:
    print(op.label, op.pass_number, round(float(op.out.abs().mean()), 3))
print(layer.ops[1].label)
# Output:
# tanh_1_2:1 1 0.357
# tanh_1_2:2 2 0.371
# tanh_1_2:3 3 0.323
# tanh_1_2:2
```

### layer-single-pass-reads
On a single-pass layer the per-pass fields read through to its one op.

- covers: `Layer.label`, `Layer.short_label`, `Layer.pass_index`, `Layer.step_index`, `Layer.lookup_keys`, `Layer.out`, `Layer.has_saved_activation`, `Layer.is_module_input`
- tasks: inspect-layer, read-activation
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace.layers["relu_1_2"]
print("label", layer.label)
print("short_label", layer.short_label)
print("pass_index", layer.pass_index)
print("step_index", layer.step_index)
print("lookup_keys", sorted(map(str, layer.lookup_keys)))
print("out", tuple(layer.out.shape))
print("has_saved_activation", layer.has_saved_activation)
print("is_module_input", layer.is_module_input)
# Output:
# ...
```

### layer-addresses
A layer's module address and backend locators (per layer).

- covers: `Layer.address`, `Layer.backend_address`, `Layer.resolver_status`, `Layer.site_peers`, `Layer.fx_label`, `Layer.fx_qualpath`, `Layer.fx_call_index`
- tasks: labels, lookup
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("address", layer.address)
print("backend_address", layer.backend_address)
print("resolver_status", layer.resolver_status)
print("site_peers", layer.site_peers)
print("fx_label", layer.fx_label)
print("fx_qualpath", layer.fx_qualpath)
print("fx_call_index", layer.fx_call_index)
# Output:
# address fc1
# backend_address fc1
# resolver_status resolved
# site_peers ()
# fx_label fc1_linear_0
# fx_qualpath fc1.linear
# fx_call_index 0
```

### layer-back-references
A layer points back to the trace that holds it.

- covers: `Layer.trace`, `Layer.source_trace`
- tasks: navigate
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace.layers["relu_1_2"]
print("trace", layer.trace is trace)
print("source_trace", layer.source_trace is trace)
# Output:
# trace True
# source_trace True
```

## Saved tensors

### layer-tensor-facts
Shape, dtype and memory of a looped layer: per layer, with totals over all passes.

- covers: `Layer.shape`, `Layer.dtype`, `Layer.shape_summary`, `Layer.output_device`, `Layer.device_ref`, `Layer.dtype_ref`, `Layer.activation_memory`, `Layer.total_activation_memory`, `Layer.autograd_memory`, `Layer.total_autograd_memory`, `Layer.num_autograd_tensors`, `Layer.detach_saved_activations`
- tasks: inspect-layer, memory
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["tanh_1_2"]
print("shape", tuple(layer.shape))
print("dtype", layer.dtype)
print("shape_summary", layer.shape_summary)
print("output_device", layer.output_device)
print("device_ref", layer.device_ref)
print("dtype_ref", layer.dtype_ref)
print("activation_memory", layer.activation_memory)
print("total_activation_memory", layer.total_activation_memory)
print("autograd_memory", layer.autograd_memory)
print("total_autograd_memory", layer.total_autograd_memory)
print("num_autograd_tensors", layer.num_autograd_tensors)
print("detach_saved_activations", layer.detach_saved_activations)
# Output:
# shape (2, 4)
# dtype torch.float32
# shape_summary None
# output_device same
# device_ref cpu
# dtype_ref torch.float32
# activation_memory 32 B
# total_activation_memory 96 B
# autograd_memory 96 B
# total_autograd_memory 96 B
# num_autograd_tensors 3
# detach_saved_activations False
```

### layer-activation-transform
An activation transform stores a changed copy of the saved output (single-pass layer).

- covers: `Layer.activation_transform`, `Layer.transformed_out`, `Layer.transformed_out_shape`, `Layer.transformed_out_dtype`, `Layer.transformed_activation_memory`
- tasks: transform-activations
- fixture: toy

```python
opts = tl.options.CaptureOptions(activation_transform=lambda t: t.mean(dim=0))
trace = tl.trace(model, x, capture=opts)
layer = trace.layers["relu_1_2"]
print("activation_transform", callable(layer.activation_transform))
print("transformed_out", tuple(layer.transformed_out.shape))
print("transformed_out_shape", tuple(layer.transformed_out_shape))
print("transformed_out_dtype", layer.transformed_out_dtype)
print("transformed_activation_memory", layer.transformed_activation_memory)
# Output:
# activation_transform True
# transformed_out (8,)
# transformed_out_shape (8,)
# transformed_out_dtype torch.float32
# transformed_activation_memory 32 B
```

## Function call

### layer-func-facts
Which function a layer calls and how it is called (per layer: the same on every pass).

- covers: `Layer.func`, `Layer.func_name`, `Layer.func_qualname`, `Layer.func_config`, `Layer.arg_names`, `Layer.num_args_total`, `Layer.num_pos_args`, `Layer.num_kwargs`, `Layer.is_inplace`
- tasks: inspect-layer, function
- fixture: cnn

```python
trace = tl.trace(model, x)
layer = trace.layers["conv2d_1_1"]
print("func", layer.func.__name__)
print("func_name", layer.func_name)
print("func_qualname", layer.func_qualname)
print("func_config", layer.func_config)
print("arg_names", layer.arg_names)
print("num_args_total", layer.num_args_total)
print("num_pos_args", layer.num_pos_args)
print("num_kwargs", layer.num_kwargs)
print("is_inplace", layer.is_inplace)
# Output:
# func conv2d
# func_name conv2d
# func_qualname _VariableFunctionsClass.conv2d
# func_config {'out_channels': 4, 'in_channels': 3, 'kernel_size': (3, 3), 'padding': (1, 1)}
# arg_names ('input', 'weight', 'bias', 'stride', 'padding', 'dilation', 'groups')
# num_args_total 7
# num_pos_args 7
# num_kwargs 0
# is_inplace False
```

### layer-grad-fn
The autograd node of a looped layer, read from its first pass (per layer).

- covers: `Layer.grad_fn_class_name`, `Layer.grad_fn_class_qualname`, `Layer.grad_fn_handle`, `Layer.grad_fn_object_id`
- tasks: autograd, inspect-layer
- fixture: loop

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
layer = trace.layers["linear_1_1"]
print("grad_fn_class_name", layer.grad_fn_class_name)
print("grad_fn_class_qualname", layer.grad_fn_class_qualname)
print("grad_fn_handle", type(layer.grad_fn_handle).__name__)
print("grad_fn_object_id", isinstance(layer.grad_fn_object_id, int))
# Output:
# grad_fn_class_name AddmmBackward0
# grad_fn_class_qualname builtins.AddmmBackward0
# grad_fn_handle AddmmBackward0
# grad_fn_object_id True
```

### layer-grad-fn-after-backward
After a backward pass a single-pass layer links to its backward record.

- covers: `Layer.grad_fn`
- tasks: autograd, backward
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
trace.backward(trace.ops["output_1"].out.sum())
layer = trace.layers["linear_1_1"]
print("grad_fn", layer.grad_fn)
print("ops grad_fn", layer.ops[0].grad_fn)
# Output:
# ...
```

### layer-call-records
Saved arguments, timing, RNG and source frames of a single-pass layer's call.

- covers: `Layer.saved_args`, `Layer.saved_kwargs`, `Layer.func_duration`, `Layer.total_func_duration`, `Layer.func_rng_states`, `Layer.code_context`
- tasks: replay, timing, source
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True))
layer = trace.layers["linear_1_1"]
print("saved_args", [tuple(a.shape) for a in layer.saved_args])
print("saved_kwargs", layer.saved_kwargs)
print("func_duration", type(layer.func_duration).__name__)
print("total_func_duration", type(layer.total_func_duration).__name__)
print("func_rng_states", type(layer.func_rng_states).__name__)
print("code_context", len(layer.code_context) > 0)
# Output:
# saved_args [(3, 4), (8, 4), (8,)]
# saved_kwargs {}
# func_duration Duration
# total_func_duration Duration
# func_rng_states dict
# code_context True
```

### layer-total-duration-loop
On a looped layer `func_duration` refuses; `total_func_duration` sums all passes.

- covers: `Layer.total_func_duration`, `Layer.func_duration`
- tasks: timing, loops
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print(type(layer.total_func_duration).__name__)
try:
    layer.func_duration
except Exception as error:
    print(type(error).__name__)
# Output:
# Duration
# InvalidArgumentError
```

## Cost

### layer-cost-facts
FLOPs and MACs of a looped layer: plain names are the first pass, `total_*` sum all passes.

- covers: `Layer.flops_forward`, `Layer.flops_backward`, `Layer.flops_total`, `Layer.total_flops_forward`, `Layer.total_flops_backward`, `Layer.total_flops_total`, `Layer.macs_forward`, `Layer.macs_backward`, `Layer.macs_total`, `Layer.total_macs_forward`, `Layer.total_macs_backward`, `Layer.total_macs_total`
- tasks: flops, cost, loops
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("flops_forward", layer.flops_forward)
print("flops_backward", layer.flops_backward)
print("flops_total", layer.flops_total)
print("total_flops_forward", layer.total_flops_forward)
print("total_flops_backward", layer.total_flops_backward)
print("total_flops_total", layer.total_flops_total)
print("macs_forward", layer.macs_forward)
print("macs_backward", layer.macs_backward)
print("macs_total", layer.macs_total)
print("total_macs_forward", layer.total_macs_forward)
print("total_macs_backward", layer.total_macs_backward)
print("total_macs_total", layer.total_macs_total)
# Output:
# flops_forward 72.0 FLOPs
# flops_backward 144 FLOPs
# flops_total 216 FLOPs
# total_flops_forward 216 FLOPs
# total_flops_backward 432 FLOPs
# total_flops_total 648 FLOPs
# macs_forward 32.0 MACs
# macs_backward None
# macs_total None
# total_macs_forward 96.0 MACs
# total_macs_backward None
# total_macs_total None
```

## Parameters

### layer-params-facts
Parameter counts of the looped linear layer after freezing its bias (per layer).

- covers: `Layer.has_params`, `Layer.num_params`, `Layer.num_params_trainable`, `Layer.num_params_frozen`, `Layer.num_param_tensors_trainable`, `Layer.num_param_tensors_frozen`, `Layer.has_trainable_params`, `Layer.has_frozen_params`, `Layer.param_names`, `Layer.param_shapes`, `Layer.param_dtypes`, `Layer.total_param_memory`
- tasks: params, inspect-layer
- fixture: loop

```python
model.cell.bias.requires_grad_(False)
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("has_params", layer.has_params)
print("num_params", layer.num_params)
print("num_params_trainable", layer.num_params_trainable)
print("num_params_frozen", layer.num_params_frozen)
print("num_param_tensors_trainable", layer.num_param_tensors_trainable)
print("num_param_tensors_frozen", layer.num_param_tensors_frozen)
print("has_trainable_params", layer.has_trainable_params)
print("has_frozen_params", layer.has_frozen_params)
print("param_names", layer.param_names)
print("param_shapes", layer.param_shapes)
print("param_dtypes", layer.param_dtypes)
print("total_param_memory", layer.total_param_memory)
# Output:
# has_params True
# num_params 20
# num_params_trainable 16
# num_params_frozen 4
# num_param_tensors_trainable 1
# num_param_tensors_frozen 1
# has_trainable_params True
# has_frozen_params True
# param_names ['weight', 'bias']
# param_shapes [(4, 4), (4,)]
# param_dtypes [torch.float32, torch.float32]
# total_param_memory 80 B
```

### layer-param-objects
`Layer.params` lists the parameters a layer used; `parent_params` is emptied after capture to free memory.

- covers: `Layer.params`, `Layer.parent_params`
- tasks: params
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_2_3"]
print("params", [p.address for p in layer.params])
print("parent_params", layer.parent_params)
# Output:
# params ['fc2.weight', 'fc2.bias']
# parent_params ()
```

## Graph neighbours

### layer-graph-facts
Neighbours of a looped layer: unions over all passes, plus per-pass maps.

- covers: `Layer.parents`, `Layer.children`, `Layer.has_parents`, `Layer.has_children`, `Layer.siblings`, `Layer.has_siblings`, `Layer.co_parents`, `Layer.has_co_parents`, `Layer.parents_per_pass`, `Layer.children_per_pass`, `Layer.parent_ops_per_layer`, `Layer.child_ops_per_layer`
- tasks: graph, loops
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("parents", layer.parents)
print("children", layer.children)
print("has_parents", layer.has_parents)
print("has_children", layer.has_children)
print("siblings", layer.siblings)
print("has_siblings", layer.has_siblings)
print("co_parents", layer.co_parents)
print("has_co_parents", layer.has_co_parents)
print("parents_per_pass", layer.parents_per_pass)
print("children_per_pass", layer.children_per_pass)
print("parent_ops_per_layer", layer.parent_ops_per_layer)
print("child_ops_per_layer", layer.child_ops_per_layer)
# Output:
# parents ('input_1', 'tanh_1_2')
# children ('tanh_1_2',)
# has_parents True
# has_children True
# siblings []
# has_siblings False
# co_parents []
# has_co_parents False
# parents_per_pass {1: ['input_1'], 2: ['tanh_1_2'], 3: ['tanh_1_2']}
# children_per_pass {1: ['tanh_1_2'], 2: ['tanh_1_2'], 3: ['tanh_1_2']}
# parent_ops_per_layer {'input_1': [1], 'tanh_1_2': [2, 3]}
# child_ops_per_layer {'tanh_1_2': [1, 2, 3]}
```

### layer-edges-across-passes
Whether a looped layer's edges change between passes, and its loop-detection key.

- covers: `Layer.edges_vary_across_ops`, `Layer.parent_arg_positions`, `Layer.equivalence_class`, `Layer.equivalent_ops`
- tasks: graph, loops
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("edges_vary_across_ops", layer.edges_vary_across_ops)
print("parent_arg_positions", layer.parent_arg_positions)
print("equivalence_class", layer.equivalence_class)
print("equivalent_ops", sorted(layer.equivalent_ops))
# Output:
# ...
```

### layer-get-parents-children
Walk from a layer to its neighbour `Layer` objects instead of their labels.

- covers: `Layer.get_parents`, `Layer.get_children`
- tasks: graph, navigate
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["tanh_1_2"]
print([p.layer_label for p in layer.get_parents()])
print([c.layer_label for c in layer.get_children()])
# Output:
# ['linear_1_1']
# ['linear_1_1', 'output_1']
```

### layer-io-role
Input, output and compute flags, one row per layer (per layer).

- covers: `Layer.io_role`, `Layer.is_input`, `Layer.is_output`, `Layer.is_final_output`, `Layer.is_compute_layer`, `Layer.input_was_parameter`, `Layer.is_orphan`, `Layer.is_internal_sink`, `Layer.is_internal_source`
- tasks: graph, io
- fixture: toy

```python
trace = tl.trace(model, x)
for layer in trace.layers:
    print(layer.layer_label, layer.io_role, layer.is_input, layer.is_output,
          layer.is_final_output, layer.is_compute_layer, layer.input_was_parameter,
          layer.is_orphan, layer.is_internal_sink, layer.is_internal_source)
# Output:
# input_1 input.x True False False False False False False False
# linear_1_1 None False False False True False False False False
# relu_1_2 None False False False True False False False False
# linear_2_3 None False False False True False False False False
# output_1 output False True True False False False False False
```

### layer-ancestry
Which model inputs feed a layer and which outputs it reaches (per layer).

- covers: `Layer.input_ancestors`, `Layer.output_descendants`, `Layer.has_input_ancestor`
- tasks: graph
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace.layers["relu_1_2"]
print("input_ancestors", layer.input_ancestors)
print("output_descendants", layer.output_descendants)
print("has_input_ancestor", layer.has_input_ancestor)
# Output:
# input_ancestors frozenset({'input_1'})
# output_descendants frozenset({'output_1'})
# has_input_ancestor True
```

### layer-internal-source
A tensor made inside `forward` is an internal-source layer, not an input.

- covers: `Layer.is_internal_source`, `Layer.internal_source_parents`, `Layer.has_input_ancestor`
- tasks: graph, internal-source
- fixture: none

```python
class AddOnes(torch.nn.Module):
    def forward(self, x):
        return x + torch.ones(4)
trace = tl.trace(AddOnes(), torch.randn(2, 4))
ones, add = trace.layers["ones_1_1"], trace.layers["add_1_2"]
print("is_internal_source", ones.is_internal_source, add.is_internal_source)
print("has_input_ancestor", ones.has_input_ancestor, add.has_input_ancestor)
print("internal_source_parents", add.internal_source_parents)
# Output:
# is_internal_source True False
# has_input_ancestor False True
# internal_source_parents ('ones_1_1',)
```

## Modules

### layer-module-facts
Which module a single-pass layer runs inside and which module calls it feeds or ends.

- covers: `Layer.module`, `Layer.modules`, `Layer.module_call_stack`, `Layer.module_call_depth`, `Layer.in_submodule`, `Layer.is_atomic_module`, `Layer.leaf_module_ops`, `Layer.output_of_modules`, `Layer.output_of_module_calls`, `Layer.input_to_module_calls`
- tasks: modules, inspect-layer
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("module", layer.module)
print("modules", layer.modules)
print("module_call_stack", layer.module_call_stack)
print("module_call_depth", layer.module_call_depth)
print("in_submodule", layer.in_submodule)
print("is_atomic_module", layer.is_atomic_module)
print("leaf_module_ops", layer.leaf_module_ops)
print("output_of_modules", layer.output_of_modules)
print("output_of_module_calls", layer.output_of_module_calls)
print("input_to_module_calls", trace.layers["relu_1_2"].input_to_module_calls)
# Output:
# ...
```

### layer-multi-output
`torch.chunk` returns a tuple: each piece is its own layer with its position in the tuple.

- covers: `Layer.in_multi_output`, `Layer.multi_output_index`, `Layer.multi_output_name`
- tasks: multi-output
- fixture: none

```python
class Halves(torch.nn.Module):
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return a * b
trace = tl.trace(Halves(), torch.randn(2, 4))
for layer in trace.layers:
    if layer.in_multi_output:
        print(layer.layer_label, layer.multi_output_index, layer.multi_output_name)
# Output:
# chunk_1_1 0 None
# chunk_2_2 1 None
```

## Gradients

### layer-grad-facts
After one backward pass a single-pass layer holds its output gradient.

- covers: `Layer.grad`, `Layer.has_grad`, `Layer.save_grads`, `Layer.total_gradient_memory`
- tasks: gradients, backward
- fixture: toy

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all")
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum())
layer = trace.layers["relu_1_2"]
print("grad", tuple(layer.grad.shape))
print("has_grad", layer.has_grad)
print("save_grads", layer.save_grads)
print("total_gradient_memory", layer.total_gradient_memory)
# Output:
# grad (3, 8)
# has_grad True
# save_grads True
# total_gradient_memory 96 B
```

### layer-grad-loop
On a looped layer read each pass's gradient on `layer.ops[k]`; the total sums all passes.

- covers: `Layer.ops`, `Layer.total_gradient_memory`
- tasks: gradients, backward, loops
- fixture: loop

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all")
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum())
layer = trace.layers["tanh_1_2"]
print([tuple(op.grad.shape) for op in layer.ops])
print("total_gradient_memory", layer.total_gradient_memory)
# Output:
# [(2, 4), (2, 4), (2, 4)]
# total_gradient_memory 96 B
```

### layer-gradient-transform
A gradient transform stores a changed copy of the saved gradient (single-pass layer).

- covers: `Layer.transformed_grad`, `Layer.transformed_grad_shape`, `Layer.transformed_grad_dtype`, `Layer.transformed_gradient_memory`
- tasks: gradients, transform-activations
- fixture: toy

```python
opts = tl.options.CaptureOptions(backward_ready=True, save_grads="all", grad_transform=lambda g: g.abs().sum(dim=0))
trace = tl.trace(model, x, capture=opts)
trace.backward(trace.ops["output_1"].out.sum())
layer = trace.layers["relu_1_2"]
print("transformed_grad", tuple(layer.transformed_grad.shape))
print("transformed_grad_shape", tuple(layer.transformed_grad_shape))
print("transformed_grad_dtype", layer.transformed_grad_dtype)
print("transformed_gradient_memory", layer.transformed_gradient_memory)
# Output:
# transformed_grad (8,)
# transformed_grad_shape (8,)
# transformed_grad_dtype torch.float32
# transformed_gradient_memory 32 B
```

## Conditionals

### layer-conditional-test
The `if` test is a terminal bool layer (per layer).

- covers: `Layer.is_scalar_bool`, `Layer.bool_value`, `Layer.is_terminal_bool`, `Layer.is_in_conditional_evaluation`
- tasks: conditionals, branch
- fixture: branch

```python
trace = tl.trace(model, x)
layer = next(l for l in trace.layers if l.is_terminal_bool)
print("layer_label", layer.layer_label)
print("is_scalar_bool", layer.is_scalar_bool)
print("bool_value", layer.bool_value)
print("is_terminal_bool", layer.is_terminal_bool)
print("is_in_conditional_evaluation", layer.is_in_conditional_evaluation)
# Output:
# layer_label gt_1_3
# is_scalar_bool True
# bool_value True
# is_terminal_bool True
# terminal_bool_for None
# is_in_conditional_evaluation True
```

### layer-conditional-body
Which layers sit in the taken arm, and the branch stacks their passes ran under (per layer).

- covers: `Layer.is_in_conditional`, `Layer.is_in_conditional_body`, `Layer.in_conditionals`, `Layer.conditional_depth`, `Layer.conditional_role_stacks`, `Layer.conditional_branch_stack_ops`
- tasks: conditionals, branch
- fixture: branch

```python
trace = tl.trace(model, x)
for layer in trace.layers:
    print(layer.layer_label, layer.is_in_conditional, layer.is_in_conditional_body,
          layer.in_conditionals, layer.conditional_depth, layer.conditional_role_stacks,
          layer.conditional_branch_stack_ops)
# Output:
# input_1 False False () 0 ([],) {(): [1]}
# linear_1_1 False False () 0 ([],) {(): [1]}
# sum_1_2 False False () 0 ([],) {(): [1]}
# gt_1_3 True False (ConditionalRoleRef(conditional_id='cond_gt_1_3', arm_index=0, arm_kind='then', role='evaluation'),) 1 ([],) {(): [1]}
# relu_1_4 True True (ConditionalRoleRef(conditional_id='cond_gt_1_3', arm_index=0, arm_kind='then', role='body'),) 1 ([(0, 'then')],) {((0, 'then'),): [1]}
# output_1 False False () 0 ([],) {(): [1]}
```

### layer-conditional-children
The layer before an `if` lists which of its children sit in which arm (per layer).

- covers: `Layer.conditional_arm_children`, `Layer.conditional_then_children`, `Layer.conditional_else_children`, `Layer.conditional_elif_children`, `Layer.conditional_entry_children`
- tasks: conditionals, branch
- fixture: branch

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print("conditional_arm_children", layer.conditional_arm_children)
print("conditional_then_children", layer.conditional_then_children)
print("conditional_else_children", layer.conditional_else_children)
print("conditional_elif_children", layer.conditional_elif_children)
print("conditional_entry_children", layer.conditional_entry_children)
# Output:
# conditional_arm_children {0: {'then': ['relu_1_4']}}
# conditional_then_children ('relu_1_4',)
# conditional_else_children ()
# conditional_elif_children {}
# conditional_entry_children ('sum_1_2',)
```

## Buffers

### layer-buffer-writes
A training-mode batch norm writes its running stats: each write is a buffer layer.

- covers: `Layer.is_buffer`, `Layer.is_buffer_source`, `Layer.buffer_pass`, `Layer.buffer_overwrite_index`, `Layer.buffer_write_kind`, `Layer.buffer_source`, `Layer.buffer_value_changed`, `Layer.buffer_replay_validated`, `Layer.buffer_source_func_name`
- tasks: buffers
- fixture: none

```python
torch.manual_seed(0)
trace = tl.trace(torch.nn.BatchNorm1d(4), torch.randn(8, 4))
for layer in trace.layers:
    if layer.is_buffer:
        print(layer.layer_label, layer.is_buffer_source, layer.buffer_pass,
              layer.buffer_overwrite_index, layer.buffer_write_kind, layer.buffer_source,
              layer.buffer_value_changed, layer.buffer_replay_validated,
              layer.buffer_source_func_name)
# Output:
# buffer_1 True 1 1 None None None None None
# buffer_2 True 2 2 inplace add_1_1 True None add_
# buffer_3 True 1 1 None None None None None
# buffer_4 True 1 1 None None None None None
# buffer_5 True 2 2 fused batchnorm_1_2 True False batch_norm
# buffer_6 True 2 2 fused batchnorm_1_2 True False batch_norm
```

## Interventions and annotations

### layer-do-ablate
`Layer.do` edits this layer through its trace and returns the edited trace.

- covers: `Layer.do`, `Layer.is_edited`
- tasks: intervene, ablate
- fixture: toy

```python
trace = tl.trace(model, x)
edited = trace.layers["relu_1_2"].do(tl.zero_ablate())
layer = edited.layers["relu_1_2"]
print("is_edited", layer.is_edited)
print("zeroed", float(layer.out.abs().sum()))
# Output:
# is_edited True
# zeroed 0.0
```

### layer-annotations
Attach your own data with `Trace.annotate`, then read it on the layer.

- covers: `Layer.annotations`
- tasks: annotate
- fixture: toy

```python
trace = tl.trace(model, x)
trace.annotate("relu_1_2", data={"note": "after fc1"})
print(trace.layers["relu_1_2"].annotations)
# Output:
# {'user': {'data': {'note': 'after fc1'}}}
```

## Fields, views and display

### layer-receptive-field
Which input pixels a conv layer can see, and which outputs one input pixel reaches.

- covers: `Layer.receptive_field`, `Layer.projective_field`
- tasks: receptive-field
- fixture: cnn

```python
trace = tl.trace(model, x)
receptive = trace.layers["conv2d_2_3"].receptive_field
print("receptive_field", receptive.size, receptive.status)
projective = trace.layers["conv2d_1_1"].projective_field
print("projective_field", projective.size, projective.status)
# Output:
# receptive_field (5, 5) ReceptiveFieldStatus.EXACT
# projective_field (3, 3) ReceptiveFieldStatus.EXACT
```

### layer-to-pandas
A looped layer as a one-row table of its layer-level fields.

- covers: `Layer.to_pandas`
- tasks: table, export, loops
- fixture: loop

```python
trace = tl.trace(model, x)
frame = trace.layers["tanh_1_2"].to_pandas()
print(type(frame).__name__, frame.shape[0])
print(list(frame.columns)[:6])
# Output:
# DataFrame 1
# ['layer_label', 'short_layer_label', 'layer_type', 'type_index', 'step_index', 'ordinal_index']
```

### layer-show
Draw a layer's saved output; without plotting available it returns a text note instead.

- covers: `Layer.show`, `Layer.visualizer_path`
- tasks: plot, visualize
- fixture: cnn

```python
trace = tl.trace(model, x)
layer = trace.layers["conv2d_1_1"]
picture = layer.show()
print(type(picture).__name__)
print("visualizer_path", layer.visualizer_path)
# Output:
# Figure
# visualizer_path None
```
