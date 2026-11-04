# Param and Buffer: the model's parameters and buffers in a trace

## Parameters

### params-identity
Find a parameter by address and read where it lives.

- covers: `Param.address`, `Param.name`, `Param.module_address`, `Param.module_name`, `Param.all_addresses`, `Param.all_module_addresses`, `Param.backend_address`, `Param.has_multiple_addresses`, `Param.index`, `Param.resolver_status`
- tasks: params
- fixture: toy

```python
trace = tl.trace(model, x)
p = trace.params["fc1.weight"]
print("address", p.address, "name", p.name)
print("module_address", p.module_address, "module_name", p.module_name)
print("all_addresses", p.all_addresses)
print("all_module_addresses", p.all_module_addresses)
print("backend_address", p.backend_address)
print("has_multiple_addresses", p.has_multiple_addresses)
print("index", p.index)
print("resolver_status", p.resolver_status)
# Output:
# address fc1.weight name weight
# module_address fc1 module_name fc1
# all_addresses ['fc1.weight']
# all_module_addresses ['fc1']
# backend_address fc1.weight
# has_multiple_addresses False
# index 0
# resolver_status resolved
```

### params-shape-facts
Read a parameter's shape, dtype, size and trainability.

- covers: `Param.shape`, `Param.dtype`, `Param.dtype_ref`, `Param.num_params`, `Param.__len__`, `Param.param_memory`, `Param.is_trainable`, `Param.is_quantized`, `Param.barcode`
- tasks: params
- fixture: toy

```python
trace = tl.trace(model, x)
p = trace.params["fc1.weight"]
print("shape", p.shape, "dtype", p.dtype)
print("dtype_ref", p.dtype_ref.name)
print("num_params", p.num_params, "len", len(p))
print("param_memory", p.param_memory)
print("is_trainable", p.is_trainable)
print("is_quantized", p.is_quantized)
print("barcode", isinstance(p.barcode, str))
# Output:
# shape (8, 4) dtype torch.float32
# dtype_ref torch.float32
# num_params 32 len 32
# param_memory 128 B
# is_trainable True
# is_quantized False
# barcode True
```

### params-usage
See which ops used a parameter and which module owns it.

- covers: `Param.used_by_layers`, `Param.used_by_ops`, `Param.num_calls`, `Param.co_parent_params`, `Param.module`, `Param.modules`, `Param.trace`
- tasks: params
- fixture: loop

```python
trace = tl.trace(model, x)
p = trace.params["cell.weight"]
print("used_by_layers", p.used_by_layers)
print("used_by_ops", p.used_by_ops)
print("num_calls", p.num_calls)
print("co_parent_params", p.co_parent_params)
print("module", p.module.address, [m.address for m in p.modules])
print("trace", p.trace is trace)
# Output:
# used_by_layers ['linear_1_1']
# used_by_ops ['linear_1_1:1', 'linear_1_1:2', 'linear_1_1:3']
# num_calls 3
# co_parent_params ['cell.bias']
# module cell ['cell']
# trace True
```

### params-value
Read the live parameter tensor behind a record.

- covers: `Param.handle`, `Param.value`, `Param.value_basis`
- tasks: params
- fixture: toy

```python
trace = tl.trace(model, x)
p = trace.params["fc2.bias"]
print("handle is live", p.handle is model.fc2.bias)
print("value", tuple(p.value.shape))
print("value_basis", p.value_basis.basis)
# Output:
# handle is live True
# value (2,)
# value_basis live_ref
```

### params-grad
After a backward, read a parameter's gradient and its facts.

- covers: `Param.grad`, `Param.has_grad`, `Param.grad_shape`, `Param.grad_dtype`, `Param.grad_memory`, `Param.grads`
- tasks: params, gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
p = trace.params["fc2.bias"]
print("grad", p.grad.tolist())
print("has_grad", p.has_grad)
print("grad_shape", p.grad_shape, "grad_dtype", p.grad_dtype)
print("grad_memory", p.grad_memory)
print("grads", len(trace.params["fc1.weight"].grads))
# Output:
# grad [3.0, 3.0]
# has_grad True
# grad_shape (2,) grad_dtype torch.float32
# grad_memory 8 B
# grads 1
```

### params-shared
One parameter tensor used by a module registered twice has two addresses.

- covers: `Param.all_addresses`, `Param.has_multiple_addresses`, `Param.num_calls`
- tasks: params, shared-module
- fixture: none

```python
lin = torch.nn.Linear(4, 4)
net = torch.nn.Sequential(lin, torch.nn.ReLU(), lin)
trace = tl.trace(net, torch.randn(2, 4))
p = trace.params["0.weight"]
print("all_addresses", p.all_addresses)
print("has_multiple_addresses", p.has_multiple_addresses)
print("num_calls", p.num_calls)
# Output:
# all_addresses ['0.weight', '2.weight']
# has_multiple_addresses True
# num_calls 2
```

### params-optimizer
Pass the optimizer at capture to mark which parameters it updates.

- covers: `Param.has_optimizer`, `CaptureOptions.optimizer`
- tasks: params, training
- fixture: toy

```python
opt = torch.optim.SGD(model.fc1.parameters(), lr=0.1)
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(optimizer=opt))
for p in trace.params:
    print(p.address, p.has_optimizer)
# Output:
# fc1.weight True
# fc1.bias True
# fc2.weight False
# fc2.bias False
```

### params-to-pandas
Export one parameter as a pandas row.

- covers: `Param.to_pandas`
- tasks: params, pandas
- fixture: toy

```python
trace = tl.trace(model, x)
df = trace.params["fc1.weight"].to_pandas()
print(type(df).__name__, len(df))
# Output:
# DataFrame 1
```

## Buffers

### params-buffer-identity
Find a BatchNorm buffer by address and read where it lives.

- covers: `Buffer.address`, `Buffer.name`, `Buffer.module_address`, `Buffer.module`, `Buffer.modules`, `Buffer.all_addresses`, `Buffer.has_multiple_addresses`, `Buffer.layer_label`, `Buffer.call_index`, `Buffer.trace`
- tasks: buffers
- fixture: none

```python
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
trace = tl.trace(net, torch.randn(5, 4))
print([b.address for b in trace.buffers])
b = trace.buffers["1.running_mean"]
print("name", b.name, "module_address", b.module_address)
print("module", b.module, "modules", b.modules)
print("all_addresses", b.all_addresses, b.has_multiple_addresses)
print("layer_label", b.layer_label, "call_index", b.call_index)
print("trace", b.trace is trace)
# Output:
# ['1.num_batches_tracked', '1.running_mean', '1.running_var']
# name running_mean module_address 1
# module 1:1 modules ('1:1',)
# all_addresses ['1.running_mean'] False
# layer_label buffer_5 call_index 2
# trace True
```

### params-buffer-writes
A training-mode BatchNorm overwrites its running mean; read the before and after values.

- covers: `Buffer.initial_value`, `Buffer.final_value`, `Buffer.is_overwritten`, `Buffer.num_overwrites`, `Buffer.overwrite_index`, `Buffer.writer_label`, `Buffer.write_ops`, `Buffer.versions`
- tasks: buffers, state
- fixture: none

```python
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
trace = tl.trace(net, torch.randn(5, 4))
b = trace.buffers["1.running_mean"]
print("initial_value", b.initial_value.tolist())
print("final_value", [round(v, 3) for v in b.final_value.tolist()])
print("is_overwritten", b.is_overwritten, "num_overwrites", b.num_overwrites)
print("overwrite_index", b.overwrite_index, "writer_label", b.writer_label)
print("write_ops", [op.label for op in b.write_ops])
print("versions", [v.label for v in b.versions])
# Output:
# initial_value [0.0, 0.0, 0.0, 0.0]
# final_value [-0.016, 0.052, -0.047, -0.001]
# is_overwritten True num_overwrites 1
# overwrite_index 2 writer_label batchnorm_1_3
# write_ops ['buffer_5:1']
# versions ['buffer_3:1', 'buffer_5:1']
```

### params-buffer-value-at
Read a buffer's value at one version, or after one overwrite (both 1-based).

- covers: `Buffer.value_at`, `Buffer.value_after`
- tasks: buffers, state
- fixture: none

```python
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
trace = tl.trace(net, torch.randn(5, 4))
b = trace.buffers["1.num_batches_tracked"]
print("value_at(1)", b.value_at(1).item())
print("value_at(2)", b.value_at(2).item())
print("value_after(1)", b.value_after(1).item())
# Output:
# value_at(1) 0
# value_at(2) 1
# value_after(1) 1
```

### params-buffer-facts
Read a buffer's shape, dtype, saved value and the live tensor behind it.

- covers: `Buffer.shape`, `Buffer.dtype`, `Buffer.has_saved_activation`, `Buffer.activation_memory`, `Buffer.handle`, `Buffer.has_grad`, `Buffer.grad_shape`, `Buffer.grad_dtype`, `Buffer.grad_memory`
- tasks: buffers
- fixture: none

```python
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
trace = tl.trace(net, torch.randn(5, 4))
b = trace.buffers["1.running_var"]
print("shape", b.shape, "dtype", b.dtype)
print("has_saved_activation", b.has_saved_activation)
print("activation_memory", b.activation_memory)
print("handle is live", b.handle is net[1].running_var)
print("has_grad", b.has_grad, "grad_shape", b.grad_shape, "grad_dtype", b.grad_dtype)
print("grad_memory", b.grad_memory)
# Output:
# shape (4,) dtype torch.float32
# has_saved_activation True
# activation_memory 16 B
# handle is live True
# has_grad False grad_shape None grad_dtype None
# grad_memory 0 B
```

### params-buffer-to-pandas
Export one buffer as a pandas row.

- covers: `Buffer.to_pandas`
- tasks: buffers, pandas
- fixture: none

```python
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
trace = tl.trace(net, torch.randn(5, 4))
df = trace.buffers["1.running_mean"].to_pandas()
print(type(df).__name__, len(df))
# Output:
# DataFrame 1
```
