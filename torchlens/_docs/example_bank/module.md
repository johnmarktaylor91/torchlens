# Module and ModuleCall: one module, and one call of it

## Where a module sits

### module-address-facts
Read where a module sits in the model's attribute tree.

- covers: `Module.address`, `Module.name`, `Module.address_parent`, `Module.address_children`, `Module.address_depth`, `Module.all_addresses`, `Module.has_multiple_addresses`
- tasks: modules
- fixture: toy

```python
trace = tl.trace(model, x)
root, fc1 = trace.modules["self"], trace.modules["fc1"]
print("address", fc1.address, root.address)
print("name", fc1.name, repr(root.name))
print("address_parent", fc1.address_parent, root.address_parent)
print("address_children", root.address_children)
print("address_depth", root.address_depth, fc1.address_depth)
print("all_addresses", fc1.all_addresses)
print("has_multiple_addresses", fc1.has_multiple_addresses)
# Output:
# address fc1 self
# name fc1 ''
# address_parent self None
# address_children ['fc1', 'fc2']
# address_depth 0 1
# all_addresses ['fc1']
# has_multiple_addresses False
```

### module-shared-addresses
One module object registered under two names has two addresses and two calls.

- covers: `Module.all_addresses`, `Module.has_multiple_addresses`, `Module.num_calls`, `ModuleCall.all_addresses`, `ModuleCall.has_multiple_addresses`
- tasks: modules, shared-module
- fixture: none

```python
lin = torch.nn.Linear(4, 4)
net = torch.nn.Sequential(lin, torch.nn.ReLU(), lin)
trace = tl.trace(net, torch.randn(2, 4))
shared = trace.modules["0"]
print("all_addresses", shared.all_addresses)
print("has_multiple_addresses", shared.has_multiple_addresses)
print("num_calls", shared.num_calls)
call = trace.module_calls["0:2"]
print("call", call.all_addresses, call.has_multiple_addresses)
# Output:
# all_addresses ['0', '2']
# has_multiple_addresses True
# num_calls 2
# call ['0', '2'] True
```

### module-call-tree-facts
Read where a module sits in the call tree: its calling parent, children and depth.

- covers: `Module.call_parent`, `Module.call_children`, `Module.call_depth`, `Module.call_labels`, `Module.num_calls`, `Module.num_descendant_calls`, `Module.max_descendant_depth`
- tasks: modules, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
root, fc1 = trace.modules["self"], trace.modules["fc1"]
print("call_parent", fc1.call_parent)
print("call_children", root.call_children)
print("call_depth", root.call_depth, fc1.call_depth)
print("call_labels", fc1.call_labels)
print("num_calls", fc1.num_calls)
print("num_descendant_calls", root.num_descendant_calls, fc1.num_descendant_calls)
print("max_descendant_depth", root.max_descendant_depth, fc1.max_descendant_depth)
# Output:
# call_parent self
# call_children ['fc1', 'fc2']
# call_depth 0 1
# call_labels ['fc1:1']
# num_calls 1
# num_descendant_calls 2 0
# max_descendant_depth 1 0
```

### module-call-tree-walk
Walk every call below the root, depth first.

- covers: `Module.walk_descendants`, `Module.call_tree`, `ModuleCall.label`
- tasks: modules, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
root = trace.modules["self"]
print("call_tree", root.call_tree.label)
print("descendants", [call.label for call in root.walk_descendants()])
# Output:
# call_tree self:1
# descendants ['fc1:1', 'fc2:1']
```

### module-show-call-tree
Print the call tree of a module as indented text.

- covers: `Module.show_call_tree`
- tasks: modules, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
trace.modules["self"].show_call_tree()
# Output:
# self:1 (forward_duration: ...)
# |-- fc1:1 (forward_duration: ...)
# `-- fc2:1 (forward_duration: ...)
```

## What kind of module it is

### module-class-facts
Read a module's class, its constructor settings and its training mode.

- covers: `Module.class_name`, `Module.class_qualname`, `Module.custom_attributes`, `Module.custom_methods`, `Module.training`, `Module.class_docstring`, `Module.handle`, `Module.trace`
- tasks: modules
- fixture: toy

```python
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("class_name", fc1.class_name)
print("class_qualname", fc1.class_qualname)
print("custom_attributes", fc1.custom_attributes)
print("custom_methods", fc1.custom_methods)
print("training", fc1.training)
print("has class_docstring", isinstance(fc1.class_docstring, str))
print("handle is model.fc1", fc1.handle is model.fc1)
print("trace", fc1.trace is trace)
# Output:
# class_name Linear
# class_qualname torch.nn.modules.linear.Linear
# custom_attributes {'in_features': 4, 'out_features': 8}
# custom_methods ['reset_parameters']
# training False
# has class_docstring True
# handle is model.fc1 True
# trace True
```

### module-source-class-forward
With source context saved, read where the module's class and forward live.

- covers: `Module.class_source_file`, `Module.class_source_line`, `Module.class_source_location`, `Module.forward_source_file`, `Module.forward_source_line`, `Module.forward_source_location`, `Module.forward_signature`, `Module.forward_docstring`
- tasks: modules, source
- fixture: toy

```python
import os

trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_source_context=True))
fc1 = trace.modules["fc1"]
print("class_source_file", os.path.basename(fc1.class_source_file))
print("class_source_line", isinstance(fc1.class_source_line, int))
print("class_source_location", fc1.class_source_location.startswith(fc1.class_source_file))
print("forward_source_file", os.path.basename(fc1.forward_source_file))
print("forward_source_line", isinstance(fc1.forward_source_line, int))
print("forward_source_location", fc1.forward_source_location.startswith(fc1.forward_source_file))
print("forward_signature has input", "input" in fc1.forward_signature)
print("forward_docstring is the class forward's", fc1.forward_docstring == type(model.fc1).forward.__doc__)
# Output:
# class_source_file linear.py
# class_source_line True
# class_source_location True
# forward_source_file linear.py
# forward_source_line True
# forward_source_location True
# forward_signature has input True
# forward_docstring is the class forward's True
```

### module-source-init
With source context saved, read where the module's `__init__` lives and its signature.

- covers: `Module.init_source_file`, `Module.init_source_line`, `Module.init_source_location`, `Module.init_signature`, `Module.init_docstring`
- tasks: modules, source
- fixture: toy

```python
import os

trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_source_context=True))
fc1 = trace.modules["fc1"]
print("init_source_file", os.path.basename(fc1.init_source_file))
print("init_source_line", isinstance(fc1.init_source_line, int))
print("init_source_location", fc1.init_source_location.startswith(fc1.init_source_file))
print("init_signature has in_features", "in_features" in fc1.init_signature)
print("init_docstring", fc1.init_docstring)
# Output:
# init_source_file linear.py
# init_source_line True
# init_source_location True
# init_signature has in_features True
# init_docstring None
```

### module-source-absent
Without `save_source_context` the source fields are `None`.

- covers: `Module.class_source_file`, `Module.forward_signature`, `Module.init_signature`
- tasks: modules, source
- fixture: toy

```python
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print(fc1.class_source_file, fc1.forward_signature, fc1.init_signature)
# Output:
# None None None
```

### module-hooks
Hooks registered on a module show up on its record.

- covers: `Module.forward_hooks`, `Module.forward_pre_hooks`, `Module.has_forward_hooks`, `Module.backward_hooks`, `Module.backward_pre_hooks`, `Module.full_backward_hooks`, `Module.full_backward_pre_hooks`, `Module.has_backward_hooks`
- tasks: modules, hooks
- fixture: toy

```python
model.fc1.register_forward_hook(lambda module, args, out: None)
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("forward_hooks", len(fc1.forward_hooks))
print("forward_pre_hooks", fc1.forward_pre_hooks)
print("has_forward_hooks", fc1.has_forward_hooks)
print("backward_hooks", fc1.backward_hooks)
print("backward_pre_hooks", fc1.backward_pre_hooks)
print("full_backward_hooks", fc1.full_backward_hooks)
print("full_backward_pre_hooks", fc1.full_backward_pre_hooks)
print("has_backward_hooks", fc1.has_backward_hooks)
# Output:
# forward_hooks 1
# forward_pre_hooks []
# has_forward_hooks True
# backward_hooks []
# backward_pre_hooks []
# full_backward_hooks []
# full_backward_pre_hooks []
# has_backward_hooks False
```

### module-attribute-writes
A value the forward stores on a module is recorded, and the module is restored after capture.

- covers: `Module.attribute_writes`
- tasks: modules, state
- fixture: none

```python
class Stash(torch.nn.Module):
    def forward(self, x):
        self.seen = x.shape[0]
        return x * 2


net = Stash()
trace = tl.trace(net, torch.ones(3, 2))
print(sorted(trace.modules["self"].attribute_writes))
print("restored", not hasattr(net, "seen"))
# Output:
# ['seen']
# restored True
```

## Parameters a module owns

### module-param-counts
Count a module's parameters, trainable and frozen.

- covers: `Module.num_params`, `Module.num_params_trainable`, `Module.num_params_frozen`, `Module.has_trainable_params`, `Module.has_frozen_params`, `Module.num_param_tensors_trainable`, `Module.num_param_tensors_frozen`, `Module.params`, `Module.param_memory`, `Module.internal_param_memory`
- tasks: modules, params
- fixture: toy

```python
model.fc1.bias.requires_grad_(False)
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("num_params", fc1.num_params)
print("num_params_trainable", fc1.num_params_trainable)
print("num_params_frozen", fc1.num_params_frozen)
print("has_trainable_params", fc1.has_trainable_params)
print("has_frozen_params", fc1.has_frozen_params)
print("num_param_tensors_trainable", fc1.num_param_tensors_trainable)
print("num_param_tensors_frozen", fc1.num_param_tensors_frozen)
print("params", [p.address for p in fc1.params])
print("param_memory", fc1.param_memory)
print("internal_param_memory", fc1.internal_param_memory)
# Output:
# num_params 40
# num_params_trainable 32
# num_params_frozen 8
# has_trainable_params True
# has_frozen_params True
# num_param_tensors_trainable 1
# num_param_tensors_frozen 1
# params ['fc1.weight', 'fc1.bias']
# param_memory 160 B
# internal_param_memory 160 B
```

### module-recursive-params
Count parameters of a module and everything below it.

- covers: `Module.recursive_params`, `Module.num_recursive_params`, `Module.num_recursive_params_trainable`, `Module.num_recursive_params_frozen`, `Module.num_recursive_param_tensors`, `Module.num_recursive_param_tensors_trainable`, `Module.num_recursive_param_tensors_frozen`
- tasks: modules, params
- fixture: toy

```python
model.fc2.requires_grad_(False)
trace = tl.trace(model, x)
root = trace.modules["self"]
print("recursive_params", [p.address for p in root.recursive_params])
print("num_recursive_params", root.num_recursive_params)
print("num_recursive_params_trainable", root.num_recursive_params_trainable)
print("num_recursive_params_frozen", root.num_recursive_params_frozen)
print("num_recursive_param_tensors", root.num_recursive_param_tensors)
print("num_recursive_param_tensors_trainable", root.num_recursive_param_tensors_trainable)
print("num_recursive_param_tensors_frozen", root.num_recursive_param_tensors_frozen)
# Output:
# recursive_params ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
# num_recursive_params 4
# num_recursive_params_trainable 2
# num_recursive_params_frozen 2
# num_recursive_param_tensors 4
# num_recursive_param_tensors_trainable 2
# num_recursive_param_tensors_frozen 2
```

## Layers and ops inside a module

### module-layers
List the layers inside a module and its boundary layers.

- covers: `Module.layer_labels`, `Module.layers`, `Module.input_layers`, `Module.output_layers`, `Module.input_ops`, `Module.output_ops`, `Module.buffer_layers`, `Module.ops`
- tasks: modules, layers
- fixture: toy

```python
trace = tl.trace(model, x)
root = trace.modules["self"]
print("layer_labels", root.layer_labels)
print("layers", [layer.layer_label for layer in root.layers])
print("input_layers", root.input_layers, "output_layers", root.output_layers)
print("input_ops", root.input_ops, "output_ops", root.output_ops)
print("buffer_layers", root.buffer_layers)
print("ops", type(root.ops).__name__, len(root.ops))
# Output:
# layer_labels ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
# layers ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
# input_layers ['input_1'] output_layers ['output_1']
# input_ops ['input_1'] output_ops ['output_1:1']
# buffer_layers []
# ops ModuleCallAccessor 1
```

### module-container-protocol
A module record behaves like a list of its layers.

- covers: `Module.__len__`, `Module.__iter__`, `Module.__getitem__`
- tasks: modules, layers
- fixture: toy

```python
trace = tl.trace(model, x)
root = trace.modules["self"]
print("len", len(root))
print("iter", [layer.layer_label for layer in root])
print("first", root[0].layer_label)
# Output:
# len 5
# iter ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
# first input_1
```

### module-edges
Count graph edges entering, leaving and inside a module.

- covers: `Module.num_edges`, `Module.num_input_edges`, `Module.num_internal_edges`, `Module.num_output_edges`
- tasks: modules, graph
- fixture: toy

```python
trace = tl.trace(model, x)
for address in ["self", "fc1"]:
    m = trace.modules[address]
    print(address, m.num_edges, m.num_input_edges, m.num_internal_edges, m.num_output_edges)
# Output:
# self 4 0 4 0
# fc1 2 1 0 1
```

## Outputs, gradients and cost

### module-outputs
Read a single-call module's output tensor and its shape.

- covers: `Module.out`, `Module.outs`, `Module.out_shape`, `Module.out_shapes`, `Module.output_structure`
- tasks: modules, activations
- fixture: toy

```python
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("out", tuple(fc1.out.shape))
print("outs", [tuple(t.shape) for t in fc1.outs])
print("out_shape", fc1.out_shape)
print("out_shapes", fc1.out_shapes)
print("output_structure", fc1.output_structure)
# Output:
# out (3, 8)
# outs [(3, 8)]
# out_shape (3, 8)
# out_shapes [(3, 8)]
# output_structure None
```

### module-grads
After a backward, read the gradient at a module's output.

- covers: `Module.grad`, `Module.grads`
- tasks: modules, gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
fc2 = trace.modules["fc2"]
print("grad", tuple(fc2.grad.shape), fc2.grad.unique().tolist())
print("grads", len(fc2.grads))
# Output:
# grad (3, 2) [1.0]
# grads 1
```

### module-memory
Read activation, gradient and autograd memory a module's tensors take.

- covers: `Module.total_output_activation_memory`, `Module.total_internal_activation_memory`, `Module.total_output_gradient_memory`, `Module.total_internal_gradient_memory`, `Module.total_autograd_memory`
- tasks: modules, memory
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
root = trace.modules["self"]
print("total_output_activation_memory", root.total_output_activation_memory)
print("total_internal_activation_memory", root.total_internal_activation_memory)
print("total_output_gradient_memory", root.total_output_gradient_memory)
print("total_internal_gradient_memory", root.total_internal_gradient_memory)
print("total_autograd_memory", root.total_autograd_memory)
# Output:
# total_output_activation_memory 24 B
# total_internal_activation_memory 264 B
# total_output_gradient_memory 24 B
# total_internal_gradient_memory 216 B
# total_autograd_memory 304 B
```

### module-flops
Read a module's estimated forward and backward FLOPs and MACs.

- covers: `Module.total_flops`, `Module.total_flops_forward`, `Module.total_flops_backward`, `Module.total_macs_forward`, `Module.total_macs_backward`, `Module.total_macs`
- tasks: modules, cost
- fixture: toy

```python
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("total_flops", fc1.total_flops)
print("total_flops_forward", fc1.total_flops_forward)
print("total_flops_backward", fc1.total_flops_backward)
print("total_macs_forward", fc1.total_macs_forward)
print("total_macs_backward", fc1.total_macs_backward)
print("total_macs", fc1.total_macs)
# Output:
# total_flops 648 FLOPs
# total_flops_forward 216 FLOPs
# total_flops_backward 432 FLOPs
# total_macs_forward 96.0 MACs
# total_macs_backward None
# total_macs None
```

### module-durations
Read how long a module's forward took; times vary, so check their type.

- covers: `Module.forward_duration`, `Module.total_forward_duration`, `Module.func_calls_duration`, `Module.total_func_calls_duration`
- tasks: modules, timing
- fixture: toy

```python
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("forward_duration", type(fc1.forward_duration).__name__)
print("total_forward_duration", type(fc1.total_forward_duration).__name__)
print("func_calls_duration", type(fc1.func_calls_duration).__name__)
print("total_func_calls_duration", type(fc1.total_func_calls_duration).__name__)
# Output:
# forward_duration Duration
# total_forward_duration Duration
# func_calls_duration Duration
# total_func_calls_duration Duration
```

### module-forward-args
TorchLens keeps a summary of a module's forward arguments, not the tensors.

- covers: `Module.forward_args`, `Module.forward_kwargs`, `Module.forward_args_summary`, `Module.forward_kwargs_summary`, `Module.forward_args_template`, `Module.forward_kwargs_template`
- tasks: modules, arguments
- fixture: toy

```python
trace = tl.trace(model, x)
fc1 = trace.modules["fc1"]
print("forward_args", fc1.forward_args)
print("forward_kwargs", fc1.forward_kwargs)
print("forward_args_summary", fc1.forward_args_summary)
print("forward_kwargs_summary", fc1.forward_kwargs_summary)
print("forward_args_template", type(fc1.forward_args_template).__name__)
print("forward_kwargs_template", fc1.forward_kwargs_template)
# Output:
# forward_args None
# forward_kwargs None
# forward_args_summary [Tensor(shape=(3, 4), dtype=float32)]
# forward_kwargs_summary {}
# forward_args_template CapturedArgTemplate
# forward_kwargs_template None
```

### module-receptive-field
Ask which input pixels a conv module's output sees, and what an early module's output reaches.

- covers: `Module.receptive_field`, `Module.projective_field`, `ModuleCall.receptive_field`, `ModuleCall.projective_field`
- tasks: modules, receptive-field
- fixture: cnn

```python
trace = tl.trace(model, x)
print("receptive_field", trace.modules["2"].receptive_field.size)
print("call receptive_field", trace.module_calls["2:1"].receptive_field.size)
proj = trace.modules["0"].projective_field
print("projective_field", type(proj).__name__, proj.status)
print("call projective_field", trace.module_calls["0:1"].projective_field.status)
# Output:
# receptive_field (5, 5)
# call receptive_field (5, 5)
# projective_field ReceptiveFieldView ReceptiveFieldStatus.EXACT
# call projective_field ReceptiveFieldStatus.EXACT
```

### module-facets
A module's facets view holds computed named values such as its output.

- covers: `Module.facets`, `ModuleCall.facets`
- tasks: modules, facets
- fixture: cnn

```python
trace = tl.trace(model, x)
view = trace.modules["0"].facets
print("out" in view, tuple(view["out"].value.shape))
print("call", "out" in trace.module_calls["0:1"].facets)
# Output:
# True (1, 4, 8, 8)
# call True
```

### module-collapse-score
Read the score the drawing's auto-collapse uses to fold a module.

- covers: `Module.collapse_score`
- tasks: modules, draw
- fixture: toy

```python
trace = tl.trace(model, x)
print({m.address: m.collapse_score for m in trace.modules})
# Output:
# {'self': 0.0, 'fc1': 0.0, 'fc2': 0.0}
```

### module-to-pandas
Export a module's call rows as a pandas table.

- covers: `Module.to_pandas`
- tasks: modules, pandas
- fixture: toy

```python
trace = tl.trace(model, x)
df = trace.modules["fc1"].to_pandas()
print(type(df).__name__, len(df))
# Output:
# DataFrame 1
```

### module-draw
Draw only one module's part of the graph; `file_format="gv"` writes DOT text.

- covers: `Module.draw`
- tasks: modules, draw
- fixture: toy

```python
trace = tl.trace(model, x)
drawing = trace.modules["fc1"].draw(path="fc1", file_format="gv")
print(type(drawing).__name__, drawing.file_format, "digraph" in drawing.dot)
# Output:
# Drawing gv True
```

## A module called several times

### module-multicall-out-wrong
A module called three times has no single output; reading `.out` on it raises.

- covers: `Module.out`, `Module.calls`
- tasks: modules, loops, multi-pass
- fixture: loop
- kind: wrong
- raises: InvalidArgumentError

```python
trace = tl.trace(model, x)
cell = trace.modules["cell"]
print(len(cell.calls))
print(cell.out.shape)
# Output:
# 3
```

### module-multicall-out-right
Pick one call by its label; iterating `.calls` yields 1-based call numbers, while `.calls[0]` is the first call.

- covers: `Module.calls`, `Module.call_labels`, `ModuleCall.out_shape`, `ModuleCall.label`
- tasks: modules, loops, multi-pass
- fixture: loop
- kind: right

```python
trace = tl.trace(model, x)
cell = trace.modules["cell"]
for label in cell.call_labels:
    print(label, cell.calls[label].out_shape)
print("keys", list(cell.calls), "first", cell.calls[0].label)
# Output:
# cell:1 (2, 4)
# cell:2 (2, 4)
# cell:3 (2, 4)
# keys [1, 2, 3] first cell:1
```

## One call of a module

### modulecall-identity
Read which module a call belongs to and where it sits among calls.

- covers: `ModuleCall.label`, `ModuleCall.address`, `ModuleCall.name`, `ModuleCall.call_index`, `ModuleCall.ordinal_index`, `ModuleCall.module`, `ModuleCall.parent_label`, `ModuleCall.call_parent`, `ModuleCall.trace`
- tasks: modules, module-calls
- fixture: loop

```python
trace = tl.trace(model, x)
call = trace.module_calls["cell:2"]
print("label", call.label)
print("address", call.address, "name", call.name)
print("call_index", call.call_index)
print("ordinal_index", call.ordinal_index)
print("module", call.module.address)
print("parent_label", call.parent_label, "call_parent", call.call_parent)
print("trace", call.trace is trace)
# Output:
# label cell:2
# address cell name cell
# call_index 2
# ordinal_index 2
# module cell
# parent_label self:1 call_parent self:1
# trace True
```

### modulecall-children
Read a call's child calls and walk everything below it.

- covers: `ModuleCall.call_children`, `ModuleCall.call_children_labels`, `ModuleCall.call_tree`, `ModuleCall.walk_descendants`, `ModuleCall.num_descendant_calls`, `ModuleCall.max_descendant_depth`
- tasks: modules, module-calls, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
root = trace.module_calls["self:1"]
print("call_children", root.call_children)
print("call_children_labels", root.call_children_labels)
print("call_tree", root.call_tree.label)
print("walk_descendants", [c.label for c in root.walk_descendants()])
print("num_descendant_calls", root.num_descendant_calls)
print("max_descendant_depth", root.max_descendant_depth)
# Output:
# call_children ['fc1:1', 'fc2:1']
# call_children_labels ['fc1:1', 'fc2:1']
# call_tree self:1
# walk_descendants ['self:1', 'fc1:1', 'fc2:1']
# num_descendant_calls 2
# max_descendant_depth 1
```

### modulecall-show-call-tree
Print the call tree under one call.

- covers: `ModuleCall.show_call_tree`
- tasks: modules, module-calls, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
trace.module_calls["self:1"].show_call_tree()
# Output:
# self:1 (forward_duration: ...)
# |-- fc1:1 (forward_duration: ...)
# `-- fc2:1 (forward_duration: ...)
```

### modulecall-ops
List the ops one call ran and its boundary ops.

- covers: `ModuleCall.ops`, `ModuleCall.num_layers`, `ModuleCall.__len__`, `ModuleCall.input_layers`, `ModuleCall.output_layers`, `ModuleCall.input_ops`, `ModuleCall.output_ops`, `ModuleCall.module_call_stack`
- tasks: modules, module-calls
- fixture: loop

```python
trace = tl.trace(model, x)
call = trace.module_calls["cell:2"]
print("ops", call.ops)
print("num_layers", call.num_layers, "len", len(call))
print("input_layers", call.input_layers, "output_layers", call.output_layers)
print("input_ops", call.input_ops, "output_ops", call.output_ops)
print("module_call_stack", call.module_call_stack)
# Output:
# ops ['linear_1_1:2']
# num_layers 1 len 1
# input_layers ['tanh_1_2'] output_layers ['linear_1_1']
# input_ops ['tanh_1_2'] output_ops ['linear_1_1:2']
# module_call_stack []
```

### modulecall-edges
Count graph edges entering, leaving and inside one call.

- covers: `ModuleCall.num_edges`, `ModuleCall.num_input_edges`, `ModuleCall.num_internal_edges`, `ModuleCall.num_output_edges`
- tasks: modules, module-calls, graph
- fixture: toy

```python
trace = tl.trace(model, x)
for label in ["self:1", "fc1:1"]:
    c = trace.module_calls[label]
    print(label, c.num_edges, c.num_input_edges, c.num_internal_edges, c.num_output_edges)
# Output:
# self:1 4 0 4 0
# fc1:1 2 1 0 1
```

### modulecall-outputs
Read one call's output, its shape, dtype and where it sits in the returned structure.

- covers: `ModuleCall.out`, `ModuleCall.outs`, `ModuleCall.out_shape`, `ModuleCall.out_shapes`, `ModuleCall.out_dtype`, `ModuleCall.out_dtypes`, `ModuleCall.output_paths`, `ModuleCall.output_structure`
- tasks: modules, module-calls, activations
- fixture: loop

```python
trace = tl.trace(model, x)
call = trace.module_calls["cell:2"]
print("out", tuple(call.out.shape))
print("outs", len(call.outs))
print("out_shape", call.out_shape, "out_shapes", call.out_shapes)
print("out_dtype", call.out_dtype, "out_dtypes", call.out_dtypes)
print("output_paths", call.output_paths)
print("output_structure", call.output_structure)
# Output:
# out (2, 4)
# outs 1
# out_shape (2, 4) out_shapes [(2, 4)]
# out_dtype torch.float32 out_dtypes [torch.float32]
# output_paths ((),)
# output_structure None
```

### modulecall-grads
After a backward, read the gradient at one call's output.

- covers: `ModuleCall.grad`, `ModuleCall.grads`
- tasks: modules, module-calls, gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
call = trace.module_calls["fc2:1"]
print("grad", tuple(call.grad.shape), call.grad.unique().tolist())
print("grads", len(call.grads))
# Output:
# grad (3, 2) [1.0]
# grads 1
```

### modulecall-args
Read how a call was invoked: argument names, counts and summaries.

- covers: `ModuleCall.forward_arg_names`, `ModuleCall.num_forward_args_total`, `ModuleCall.num_forward_pos_args`, `ModuleCall.num_forward_kwargs`, `ModuleCall.forward_args_summary`, `ModuleCall.forward_kwargs_summary`, `ModuleCall.forward_args`, `ModuleCall.forward_kwargs`, `ModuleCall.has_saved_forward_args`, `ModuleCall.forward_args_template`, `ModuleCall.forward_kwargs_template`
- tasks: modules, module-calls, arguments
- fixture: toy

```python
trace = tl.trace(model, x)
call = trace.module_calls["fc1:1"]
print("forward_arg_names", call.forward_arg_names)
print("counts", call.num_forward_args_total, call.num_forward_pos_args, call.num_forward_kwargs)
print("forward_args_summary", call.forward_args_summary)
print("forward_kwargs_summary", call.forward_kwargs_summary)
print("forward_args", call.forward_args, "forward_kwargs", call.forward_kwargs)
print("has_saved_forward_args", call.has_saved_forward_args)
print("forward_args_template", type(call.forward_args_template).__name__)
print("forward_kwargs_template", call.forward_kwargs_template)
# Output:
# forward_arg_names ['0']
# counts 1 1 0
# forward_args_summary [Tensor(shape=(3, 4), dtype=float32)]
# forward_kwargs_summary {}
# forward_args None forward_kwargs None
# has_saved_forward_args False
# forward_args_template CapturedArgTemplate
# forward_kwargs_template None
```

### modulecall-pre-hooks
A forward pre-hook that changes the input is recorded on the call.

- covers: `ModuleCall.has_pre_hook_input_change`, `ModuleCall.forward_pre_hook_effects`, `ModuleCall.inputs_before_pre_hooks`, `ModuleCall.inputs_after_pre_hooks`
- tasks: modules, module-calls, hooks
- fixture: toy

```python
model.fc1.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
trace = tl.trace(model, x)
call = trace.module_calls["fc1:1"]
print("has_pre_hook_input_change", call.has_pre_hook_input_change)
print("forward_pre_hook_effects", len(call.forward_pre_hook_effects))
print("inputs_before_pre_hooks", type(call.inputs_before_pre_hooks).__name__)
print("inputs_after_pre_hooks", type(call.inputs_after_pre_hooks).__name__)
# Output:
# has_pre_hook_input_change True
# forward_pre_hook_effects 1
# inputs_before_pre_hooks ModuleInputSnapshot
# inputs_after_pre_hooks ModuleInputSnapshot
```

### modulecall-params
Count the parameters one call used.

- covers: `ModuleCall.num_param_tensors`, `ModuleCall.num_param_tensors_trainable`, `ModuleCall.num_param_tensors_frozen`, `ModuleCall.has_trainable_params`, `ModuleCall.has_frozen_params`, `ModuleCall.param_memory`, `ModuleCall.internal_param_memory`
- tasks: modules, module-calls, params
- fixture: toy

```python
model.fc1.bias.requires_grad_(False)
trace = tl.trace(model, x)
call = trace.module_calls["fc1:1"]
print("num_param_tensors", call.num_param_tensors)
print("trainable", call.num_param_tensors_trainable, call.has_trainable_params)
print("frozen", call.num_param_tensors_frozen, call.has_frozen_params)
print("param_memory", call.param_memory)
print("internal_param_memory", call.internal_param_memory)
# Output:
# num_param_tensors 2
# trainable 1 True
# frozen 1 True
# param_memory 160 B
# internal_param_memory 160 B
```

### modulecall-memory
Read the activation, gradient and autograd memory of one call.

- covers: `ModuleCall.output_activation_memory`, `ModuleCall.internal_activation_memory`, `ModuleCall.output_gradient_memory`, `ModuleCall.internal_gradient_memory`, `ModuleCall.autograd_memory`
- tasks: modules, module-calls, memory
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
call = trace.module_calls["self:1"]
print("output_activation_memory", call.output_activation_memory)
print("internal_activation_memory", call.internal_activation_memory)
print("output_gradient_memory", call.output_gradient_memory)
print("internal_gradient_memory", call.internal_gradient_memory)
print("autograd_memory", call.autograd_memory)
# Output:
# output_activation_memory 24 B
# internal_activation_memory 264 B
# output_gradient_memory 24 B
# internal_gradient_memory 216 B
# autograd_memory 304 B
```

### modulecall-durations-context
Read a call's timings (they vary, so check the type) and the source line that made it.

- covers: `ModuleCall.forward_duration`, `ModuleCall.func_calls_duration`, `ModuleCall.backward_duration`, `ModuleCall.code_context`
- tasks: modules, module-calls, timing
- fixture: toy

```python
trace = tl.trace(model, x)
call = trace.module_calls["fc1:1"]
print("forward_duration", type(call.forward_duration).__name__)
print("func_calls_duration", type(call.func_calls_duration).__name__)
print("backward_duration", call.backward_duration)
print("code_context", len(call.code_context), type(call.code_context[0]).__name__)
# Output:
# forward_duration Duration
# func_calls_duration Duration
# backward_duration None
# code_context 2 FuncCallLocation
```

### modulecall-to-pandas
Export one call as a pandas row.

- covers: `ModuleCall.to_pandas`
- tasks: modules, module-calls, pandas
- fixture: toy

```python
trace = tl.trace(model, x)
df = trace.module_calls["fc1:1"].to_pandas()
print(type(df).__name__, len(df))
# Output:
# DataFrame 1
```

### module-buffers
List the buffers a module's forward read, such as an eval-mode BatchNorm's running statistics.

- covers: `Module.buffers`
- tasks: modules, buffers
- fixture: none

```python
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).eval()
trace = tl.trace(net, torch.randn(2, 4))
print([b.address for b in trace.modules["1"].buffers])
print("fc has none", len(trace.modules["0"].buffers))
# Output:
# ['1.running_mean', '1.running_var']
# fc has none 0
```
