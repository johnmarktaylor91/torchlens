# Traps: common mistakes and their fixes

## Multi-pass layers

### traps-layer-label-multipass-wrong
On a layer with several passes, per-pass fields such as `label` refuse: the layer has one op per pass.

- covers: `Layer.label`
- tasks: loops, multi-pass
- fixture: loop
- kind: wrong
- raises: InvalidArgumentError

```python
trace = tl.trace(model, x)
for layer in trace.layers:
    print(layer.label)
# Output:
# input_1:1
```

### traps-layer-label-multipass-right
Use the layer-level name `layer_label`, and reach one pass's fields through `layer.ops[k]`.

- covers: `Layer.layer_label`, `Layer.ops`, `Layer.num_passes`
- tasks: loops, multi-pass
- fixture: loop
- kind: right

```python
trace = tl.trace(model, x)
for layer in trace.layers:
    print(layer.layer_label, layer.num_passes, layer.ops[0].label)
# Output:
# input_1 1 input_1:1
# linear_1_1 3 linear_1_1:1
# tanh_1_2 3 tanh_1_2:1
# output_1 1 output_1:1
```

## Binding an edit for many forwards

### traps-bind-plain-call-wrong
`bind` returns an edited callable; calling the original model afterwards runs it unedited.

- covers: `tl.when`, `tl.zero_ablate`
- tasks: steer-generate
- fixture: toy
- kind: wrong

```python
bound = tl.when(tl.module("fc1"), tl.zero_ablate()).bind(model)
with torch.no_grad():
    out = model(x)
print("fc1 zeroed:", torch.allclose(out, model.fc2.bias.expand(3, 2)))
# Output:
# fc1 zeroed: False
```

### traps-bind-plain-call-right
Call the bound object (`bound(x)`, or `bound.generate(...)` for a language model).

- covers: `tl.when`, `tl.zero_ablate`
- tasks: steer-generate
- fixture: toy
- kind: right

```python
bound = tl.when(tl.module("fc1"), tl.zero_ablate()).bind(model)
with torch.no_grad():
    out = bound(x)
print("fc1 zeroed:", torch.allclose(out, model.fc2.bias.expand(3, 2)))
# Output:
# fc1 zeroed: True
```

### traps-generate-kv-cache-wrong
With the KV cache, generation steps after the first see only the new token, so a fixed position index fails.

- covers: `tl.when`
- tasks: steer-generate, record-generation
- fixture: lm
- requires: transformers
- kind: wrong
- raises: IndexError

```python
picked = []
def read(out, *, ctx):
    picked.append(out[0, 4].clone())
    return out
bound = tl.when(tl.module("transformer.h.0.mlp"), read).bind(lm)
bound.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
```

### traps-generate-kv-cache-right
Index the last position, `[:, -1]`: it is the newest token at every step.

- covers: `tl.when`
- tasks: steer-generate, record-generation
- fixture: lm
- requires: transformers
- kind: right

```python
lengths, picked = [], []
def read(out, *, ctx):
    lengths.append(out.shape[1])
    picked.append(out[:, -1].clone())
    return out
bound = tl.when(tl.module("transformer.h.0.mlp"), read).bind(lm)
bound.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
print(lengths, len(picked))
# Output:
# [5, 1, 1] 3
```

## Addressing sites

### traps-functional-relu-module-wrong
The toy model calls `torch.relu` as a function, so there is no `relu` module to address.

- covers: `tl.module`
- tasks: select, intervene
- fixture: toy
- kind: wrong

```python
edited = tl.trace(model, x, intervene=tl.when(tl.module("relu"), tl.zero_ablate()))
print(bool((edited["relu_1_2"].out == 0).all()))
# Output:
# False
```

### traps-functional-relu-module-right
Address a functional call by its function with `tl.func`.

- covers: `tl.func`
- tasks: select, intervene
- fixture: toy
- kind: right

```python
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
print(bool((edited["relu_1_2"].out == 0).all()))
# Output:
# True
```

### traps-label-before-capture-wrong
Op labels such as `relu_1_2` come from a capture; binding an edit addressed by label to a plain model refuses.

- covers: `tl.label`
- tasks: steer-generate
- fixture: toy
- kind: wrong
- raises: BindingPreflightError

```python
bound = tl.when(tl.label("relu_1_2"), tl.zero_ablate()).bind(model)
```

### traps-label-before-capture-right
Before a capture, address sites with selectors: `tl.func`, `tl.module` or `tl.in_module`.

- covers: `tl.func`
- tasks: steer-generate
- fixture: toy
- kind: right

```python
bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
with torch.no_grad():
    print(torch.allclose(bound(x), model.fc2.bias.expand(3, 2)))
# Output:
# True
```

### traps-tuple-output-wrong
A module that returns a tuple (attention returns output and weights) has one site per tensor; its name alone is ambiguous.

- covers: `Trace.__getitem__`
- tasks: select
- fixture: none
- kind: wrong
- raises: AmbiguousOpLookupError

```python
torch.manual_seed(0)
attn = torch.nn.MultiheadAttention(4, 2, batch_first=True)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = attn
    def forward(self, h):
        return self.attn(h, h, h)[0]
trace = tl.trace(Net().eval(), torch.randn(2, 3, 4))
print(trace["attn"].out.shape)
```

### traps-tuple-output-right
List the module's outputs with `trace.modules[name].outs` (here the attention weights, then the output) and pick the tensor you mean.

- covers: `Trace.modules`, `Module.outs`
- tasks: select
- fixture: none
- kind: right

```python
torch.manual_seed(0)
attn = torch.nn.MultiheadAttention(4, 2, batch_first=True)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = attn
    def forward(self, h):
        return self.attn(h, h, h)[0]
trace = tl.trace(Net().eval(), torch.randn(2, 3, 4))
print([tuple(out.shape) for out in trace.modules["attn"].outs])
# Output:
# [(2, 3, 3), (2, 3, 4)]
```

## Reading the right tensor

### traps-final-norm-wrong
The last block's output is not the model's last hidden state.

- covers: `Trace.modules`
- tasks: lm, final-norm
- fixture: lm
- requires: transformers
- kind: wrong

```python
trace = tl.trace(lm, ids)
with torch.no_grad():
    last_hidden = lm(ids, output_hidden_states=True).hidden_states[-1]
block_out = trace.modules["transformer.h.1"].outs[0]
print(torch.allclose(block_out, last_hidden, atol=1e-6))
# Output:
# False
```

### traps-final-norm-right
The last hidden state is the final norm's output (`transformer.ln_f` in GPT-2, `model.norm` in Llama).

- covers: `Trace.modules`
- tasks: lm, final-norm
- fixture: lm
- requires: transformers
- kind: right

```python
trace = tl.trace(lm, ids)
with torch.no_grad():
    last_hidden = lm(ids, output_hidden_states=True).hidden_states[-1]
print(torch.allclose(trace.modules["transformer.ln_f"].out, last_hidden, atol=1e-6))
# Output:
# True
```

### traps-padding-last-token-wrong
In a right-padded batch, `[:, -1]` reads a padding position for the shorter rows.

- covers: `tl.trace`
- tasks: lm
- fixture: lm
- requires: transformers
- kind: wrong

```python
batch = torch.tensor([[5, 17, 42, 8, 3], [9, 4, 11, 0, 0]])
mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
hidden = tl.trace(lm, batch, input_kwargs={"attention_mask": mask})["transformer.ln_f"].out
alone = tl.trace(lm, batch[1:, :3])["transformer.ln_f"].out[0, -1]
print(torch.allclose(hidden[1, -1], alone, atol=1e-5))
# Output:
# False
```

### traps-padding-last-token-right
Pass the mask with `input_kwargs` and read each row's last real token.

- covers: `tl.trace`
- tasks: lm
- fixture: lm
- requires: transformers
- kind: right

```python
batch = torch.tensor([[5, 17, 42, 8, 3], [9, 4, 11, 0, 0]])
mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
hidden = tl.trace(lm, batch, input_kwargs={"attention_mask": mask})["transformer.ln_f"].out
alone = tl.trace(lm, batch[1:, :3])["transformer.ln_f"].out[0, -1]
last = mask.sum(dim=1) - 1
print(torch.allclose(hidden[1, last[1]], alone, atol=1e-5))
# Output:
# True
```

## Edits and live tensors

### traps-edit-keeps-live-tensor-wrong
An edit sees the live output; a later in-place op (here `ReLU(inplace=True)`) overwrites a tensor you kept.

- covers: `tl.when`
- tasks: intervene
- fixture: none
- kind: wrong

```python
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=True), torch.nn.Linear(4, 2))
kept = []
def keep(out, *, ctx):
    kept.append(out)
    return out
with torch.no_grad():
    tl.when(tl.module("0"), keep).bind(net.eval())(torch.randn(8, 4))
print("kept negatives:", bool((kept[0] < 0).any()))
# Output:
# kept negatives: False
```

### traps-edit-keeps-live-tensor-right
Clone what you keep inside an edit.

- covers: `tl.when`
- tasks: intervene
- fixture: none
- kind: right

```python
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=True), torch.nn.Linear(4, 2))
kept = []
def keep(out, *, ctx):
    kept.append(out.clone())
    return out
with torch.no_grad():
    tl.when(tl.module("0"), keep).bind(net.eval())(torch.randn(8, 4))
print("kept negatives:", bool((kept[0] < 0).any()))
# Output:
# kept negatives: True
```

### traps-module-edit-read-op-wrong
An edit on a module's output (`tl.module`) records the edited value as a new op after the module's last op, so that op's label still holds the unedited value.

- covers: `Trace.do`
- tasks: ablate
- fixture: toy
- kind: wrong

```python
trace = tl.trace(model, x)
edited = trace.fork().do(tl.module("fc1"), tl.zero_ablate(), model=model, inputs=x)
print(bool((edited["linear_1_1"].out == 0).all()))
# Output:
# False
```

### traps-module-edit-read-op-right
Read an edited module's output through `trace.modules[address].out`; downstream op labels can shift by one.

- covers: `Trace.do`, `Trace.modules`, `Module.out`
- tasks: ablate
- fixture: toy
- kind: right

```python
trace = tl.trace(model, x)
edited = trace.fork().do(tl.module("fc1"), tl.zero_ablate(), model=model, inputs=x)
print(bool((edited.modules["fc1"].out == 0).all()))
print(edited.layer_labels)
# Output:
# True
# ['input_1', 'linear_1_1', 'interventionreplacement_1_2', 'relu_1_3', 'linear_2_4', 'output_1']
```

## Gradients through saved values

### traps-vjp-copy-wrong
Saved values are copies by default, detached from the graph, so autograd cannot differentiate between them.

- covers: `tl.trace`
- tasks: gradients
- fixture: toy
- kind: wrong
- raises: RuntimeError

```python
trace = tl.trace(model, x)
hidden, out = trace["linear_1_1"].out, trace["linear_2_3"].out
grads = torch.autograd.grad(out[0], hidden, grad_outputs=torch.eye(2), is_grads_batched=True)
```

### traps-vjp-copy-right
Capture with `save_mode="view"` so saved values stay in the graph, then take one batched VJP.

- covers: `tl.trace`
- tasks: gradients
- fixture: toy
- kind: right

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_mode="view"))
hidden, out = trace["linear_1_1"].out, trace["linear_2_3"].out
(grads,) = torch.autograd.grad(out[0], hidden, grad_outputs=torch.eye(2), is_grads_batched=True)
print(tuple(grads.shape))
# Output:
# (2, 3, 8)
```

## Errors

### traps-error-fields
TorchLens errors are typed: branch on `error.fields["code"]` and follow `error.fields["remedy"]`.

- covers: `Trace.__getitem__`
- tasks: errors
- fixture: toy

```python
trace = tl.trace(model, x)
try:
    trace["relu_9_9"]
except tl.errors.TorchLensError as error:
    print(type(error).__name__, error.fields["code"], "remedy" in error.fields)
# Output:
# InvalidArgumentError op_lookup_not_found True
```
