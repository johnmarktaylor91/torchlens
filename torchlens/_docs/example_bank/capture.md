# Capture: tl.trace, tl.record, tl.extract and their options

## Capturing a forward pass

### capture-trace-read-op
Capture one forward pass and read an op's saved output by its label.

- covers: `tl.trace`, `Trace.ops`, `Op.out`, `Op.label`
- tasks: capture
- fixture: toy

```python
trace = tl.trace(model, x)
relu = trace.ops["relu_1_2"]
print(relu.label, tuple(relu.out.shape))
# Output:
# relu_1_2:1 (3, 8)
```

### capture-trace-list-op-labels
List every op a capture recorded, in execution order.

- covers: `tl.trace`, `Trace.op_labels`
- tasks: capture
- fixture: toy

```python
trace = tl.trace(model, x)
print(trace.op_labels)
# Output:
# ['input_1:1', 'linear_1_1:1', 'relu_1_2:1', 'linear_2_3:1', 'output_1:1']
```

### capture-save-selector
Keep only the relu activations; the other ops are recorded without their values.

- covers: `tl.trace`, `tl.func`, `Op.has_saved_activation`
- tasks: select-save, capture
- fixture: toy

```python
trace = tl.trace(model, x, save=tl.func("relu"))
for label in trace.op_labels:
    print(label, trace.ops[label].has_saved_activation)
# Output:
# input_1:1 False
# linear_1_1:1 False
# relu_1_2:1 True
# linear_2_3:1 False
# output_1:1 False
```

### capture-extract-module-outputs
Get named modules' outputs as plain tensors without keeping a trace.

- covers: `tl.extract`
- tasks: extract-layers
- fixture: toy

```python
features = tl.extract(model, x, ["fc1", "fc2"])
print({name: tuple(t.shape) for name, t in features.items()})
# Output:
# {'fc1': (3, 8), 'fc2': (3, 2)}
```

### capture-record-cheap
Record a few sites over a forward with the light recorder, then read one.

- covers: `tl.record`, `tl.func`
- tasks: record
- fixture: toy

```python
recording = tl.record(model, x, save=tl.func("relu"))
print(type(recording).__name__)
# Output:
# Recording
```

### capture-loop-passes
A module called several times gives one layer with several passes.

- covers: `tl.trace`, `Trace.layers`, `Layer.layer_label`, `Layer.num_passes`
- tasks: loops, multi-pass
- fixture: loop

```python
trace = tl.trace(model, x)
for layer in trace.layers:
    print(layer.layer_label, layer.num_passes)
# Output:
# input_1 1
# linear_1_1 3
# tanh_1_2 3
# output_1 1
```

### capture-lm-final-norm
In a language model the last hidden state is the final norm's output.

- covers: `tl.trace`, `tl.module`, `Trace.modules`
- tasks: lm, final-norm
- fixture: lm
- requires: transformers

```python
trace = tl.trace(lm, ids, save=tl.module("transformer.ln_f"))
print(tuple(trace.modules["transformer.ln_f"].out.shape))
# Output:
# (1, 5, 32)
```
