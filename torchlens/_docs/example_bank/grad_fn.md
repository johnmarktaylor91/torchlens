# GradFn: the autograd nodes a backward pass ran

## Listing the backward graph

### grad-fn-list
After a backward, list the autograd nodes in the order the backward reached them.

- covers: `GradFn.label`, `GradFn.type`, `GradFn.type_index`, `GradFn.step_index`, `GradFn.order`, `GradFn.ordinal_index`
- tasks: backward, grad-fn
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
for g in list(trace.autograd.grad_fns)[:4]:
    print(g.label, g.type, g.type_index, g.step_index, g.order, g.ordinal_index)
# Output:
# sum_back_1_1 sum 1 1 1 0
# clone_back_1_2 clone 1 2 1 1
# addmm_back_1_3 addmm 1 3 1 2
# accumulategrad_back_1_4 accumulategrad 1 4 1 3
```

### grad-fn-class-facts
Read the autograd class behind a node.

- covers: `GradFn.class_name`, `GradFn.class_qualname`, `GradFn.is_custom`, `GradFn.handle`, `GradFn.trace`, `GradFn.grad_fn_object_id`
- tasks: backward, grad-fn
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
g = trace.autograd.grad_fns["relu_back_1_5"]
print("class_name", g.class_name)
print("class_qualname", g.class_qualname)
print("is_custom", g.is_custom)
print("handle", type(g.handle).__name__)
print("trace", g.trace is trace)
print("grad_fn_object_id", isinstance(g.grad_fn_object_id, int))
# Output:
# class_name ReluBackward0
# class_qualname builtins.ReluBackward0
# is_custom False
# handle ReluBackward0
# trace True
# grad_fn_object_id True
```

### grad-fn-neighbours
Walk the backward graph: a node's parents run before it, its children after it.

- covers: `GradFn.parents`, `GradFn.children`, `GradFn.siblings`, `GradFn.co_parents`, `GradFn.has_parents`, `GradFn.has_children`, `GradFn.has_siblings`, `GradFn.has_co_parents`, `GradFn.next_grad_fn_ids`
- tasks: backward, grad-fn, graph
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
g = trace.autograd.grad_fns["addmm_back_1_3"]
print("parents", g.parents, g.has_parents)
print("children", g.children, g.has_children)
print("siblings", g.siblings, g.has_siblings)
print("co_parents", g.co_parents, g.has_co_parents)
print("next_grad_fn_ids", len(g.next_grad_fn_ids))
# Output:
# parents ['clone_back_1_2'] True
# children ['accumulategrad_back_1_4', 'relu_back_1_5', 't_back_1_6'] True
# siblings [] False
# co_parents [] False
# next_grad_fn_ids 3
```

### grad-fn-forward-op
Go from an autograd node to the forward op it differentiates and that op's module.

- covers: `GradFn.has_op`, `GradFn.op_label`, `GradFn.op`, `GradFn.modules`, `GradFn.module_address`, `GradFn.module_membership_source`
- tasks: backward, grad-fn
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
for g in trace.autograd.grad_fns:
    if g.has_op:
        print(g.label, g.op_label, g.op.label)
leaf = trace.autograd.grad_fns["accumulategrad_back_4_11"]
print(leaf.modules, leaf.module_address, leaf.module_membership_source)
# Output:
# addmm_back_1_3 linear_2_3 linear_2_3:1
# relu_back_1_5 relu_1_2 relu_1_2:1
# addmm_back_2_7 linear_1_1 linear_1_1:1
# ['fc1'] fc1 paired
```

### grad-fn-calls
Each time the backward fired a node is one call; read the calls and timings.

- covers: `GradFn.calls`, `GradFn.call_labels`, `GradFn.num_calls`, `GradFn.has_saved_call`, `GradFn.backward_duration`, `GradFn.total_backward_duration`
- tasks: backward, grad-fn, timing
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
g = trace.autograd.grad_fns["relu_back_1_5"]
print("calls", len(g.calls), "call_labels", g.call_labels)
print("num_calls", g.num_calls)
print("has_saved_call", g.has_saved_call)
print("backward_duration", type(g.backward_duration).__name__)
print("total_backward_duration", type(g.total_backward_duration).__name__)
# Output:
# calls 1 call_labels ['relu_back_1_5:1']
# num_calls 1
# has_saved_call True
# backward_duration Duration
# total_backward_duration Duration
```

### grad-fn-builtin-source
Built-in autograd nodes have no Python source, so the source fields are `None`.

- covers: `GradFn.class_source_file`, `GradFn.class_source_line`, `GradFn.class_source_location`, `GradFn.class_docstring`, `GradFn.backward_source_file`, `GradFn.backward_signature`, `GradFn.forward_source_file`, `GradFn.init_source_file`
- tasks: backward, grad-fn, source
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
g = trace.autograd.grad_fns["relu_back_1_5"]
print("class", g.class_source_file, g.class_source_line, g.class_source_location)
print("class_docstring", g.class_docstring)
print("backward", g.backward_source_file, g.backward_signature)
print("forward_source_file", g.forward_source_file)
print("init_source_file", g.init_source_file)
# Output:
# class None None None
# class_docstring None
# backward None None
# forward_source_file None
# init_source_file None
```

## Custom autograd functions

### grad-fn-custom-node
A custom `torch.autograd.Function` applied to a traced output shows up as a node marked custom.

- covers: `GradFn.is_custom`, `GradFn.class_name`
- tasks: backward, grad-fn, custom-function
- fixture: none

```python
class Double(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        return t * 2

    @staticmethod
    def backward(ctx, g):
        return g * 2


trace = tl.trace(torch.nn.Linear(2, 2), torch.ones(1, 2), capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(Double.apply(trace[trace.output_layers[0]].out).sum())
print([g.label for g in trace.autograd.grad_fns if g.is_custom])
print(trace.autograd.grad_fns["double_back_1_2"].class_name)
# Output:
# ['double_back_1_2']
# DoubleBackward
```

## Higher-order gradients

### grad-fn-higher-order
A differentiable backward (`create_graph=True`) makes new autograd nodes; each records the pass that made it and its creator.

- covers: `GradFn.origin_backward_pass`, `GradFn.creator_object_id`, `GradFn.order`
- tasks: backward, grad-fn, higher-order
- fixture: none

```python
class Net(torch.nn.Module):
    def forward(self, x):
        return (torch.tanh(x) ** 4).sum()


torch.manual_seed(0)
x = torch.randn(3, requires_grad=True)
trace = tl.trace(Net(), x, capture=tl.options.CaptureOptions(save_grads="all"))
g1 = torch.autograd.grad(trace[trace.output_layers[0]].out, x, create_graph=True)[0]
torch.autograd.grad(g1.sum(), x, create_graph=True)
made = [g for g in trace.autograd.grad_fns if g.origin_backward_pass is not None]
print("first-order nodes", sum(g.origin_backward_pass is None for g in trace.autograd.grad_fns) > 0)
print("made in passes", sorted({g.origin_backward_pass for g in made}))
print("creator known", all(g.creator_object_id is not None for g in made), "orders", sorted({g.order for g in made if g.order is not None}))
# Output:
# first-order nodes True
# made in passes [1, 2]
# creator known True orders [2, 3]
```

### grad-fn-to-pandas
Export one autograd node as a pandas row.

- covers: `GradFn.to_pandas`
- tasks: backward, grad-fn, pandas
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
df = trace.autograd.grad_fns["relu_back_1_5"].to_pandas()
print(type(df).__name__, len(df))
# Output:
# DataFrame 1
```
