# Workflows: multi-step reads and edits

## Loops and multi-pass layers

### workflows-loop-read-passes
A layer run three times keeps one op per pass: read each pass by its pass label or through `layer.ops`.

- covers: `Trace.layers`, `Layer.num_passes`, `Layer.ops`, `Trace.ops`
- tasks: loops, multi-pass
- fixture: loop

```python
trace = tl.trace(model, x)
layer = trace.layers["linear_1_1"]
print(layer.num_passes, [op.label for op in layer.ops])
for k in (1, 2, 3):
    print(k, tuple(trace.ops[f"linear_1_1:{k}"].out.shape))
# Output:
# 3 ['linear_1_1:1', 'linear_1_1:2', 'linear_1_1:3']
# 1 (2, 4)
# 2 (2, 4)
# 3 (2, 4)
```

### workflows-loop-pass-values-differ
Each pass of a recurrent layer sees a different input, so its saved outputs differ.

- covers: `Trace.ops`, `Op.out`
- tasks: loops, multi-pass
- fixture: loop

```python
trace = tl.trace(model, x)
first, second = trace.ops["tanh_1_2:1"].out, trace.ops["tanh_1_2:2"].out
print(torch.equal(first, second))
print(torch.equal(trace.ops["tanh_1_2:3"].out, trace["output_1"].out))
# Output:
# False
# True
```

### workflows-loop-edit-one-pass
Edit only the second pass of a multi-pass layer: address the op with its pass label.

- covers: `Trace.fork`, `Trace.do`, `tl.zero_ablate`
- tasks: loops, multi-pass, ablate
- fixture: loop

```python
clean = tl.trace(model, x)
edited = clean.fork().do("linear_1_1:2", tl.zero_ablate(), model=model, inputs=x)
for k in (1, 2, 3):
    label = f"linear_1_1:{k}"
    same = torch.equal(edited.ops[label].out, clean.ops[label].out)
    print(label, "unchanged" if same else "changed", bool((edited.ops[label].out == 0).all()))
# Output:
# linear_1_1:1 unchanged False
# linear_1_1:2 changed True
# linear_1_1:3 changed False
```

### workflows-loop-edit-all-passes
A function or module selector matches every pass, so the edit runs on all three.

- covers: `tl.when`, `tl.func`, `tl.scale`
- tasks: loops, multi-pass, intervene
- fixture: loop

```python
clean = tl.trace(model, x)
edited = tl.trace(model, x, intervene=tl.when(tl.func("linear"), tl.scale(0.0)))
for k in (1, 2, 3):
    print(k, bool((edited.ops[f"linear_1_1:{k}"].out == 0).all()))
print(edited.ops["linear_1_1:2"].is_edited)
# Output:
# 1 True
# 2 True
# 3 True
# True
```

## Gradients

### workflows-gradients-per-pass
Capture with gradients, run backward from the output, and read each pass's gradient.

- covers: `Trace.backward`, `Op.grad`
- tasks: gradients, loops
- fixture: loop

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
for k in (1, 2, 3):
    grad = trace.ops[f"tanh_1_2:{k}"].grad
    print(k, tuple(grad.shape))
# Output:
# 1 (2, 4)
# 2 (2, 4)
# 3 (2, 4)
```

### workflows-gradients-match-autograd
The saved gradient of a layer equals what autograd computes for the same tensor.

- covers: `Trace.backward`, `Layer.grad`
- tasks: gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
hidden = torch.relu(model.fc1(x)).detach().requires_grad_()
model.fc2(hidden).sum().backward()
print(torch.allclose(trace.layers["relu_1_2"].grad, hidden.grad))
# Output:
# True
```

## Saving and loading

### workflows-save-load-roundtrip
Save a trace, load it in a fresh object, and check the values and labels survive.

- covers: `Trace.save`, `tl.load`
- tasks: save-load
- fixture: toy

```python
trace = tl.trace(model, x)
trace.save("run.tlspec")
loaded = tl.load("run.tlspec")
print(loaded.op_labels == trace.op_labels)
print(all(torch.equal(loaded[label].out, trace[label].out) for label in trace.layer_labels))
# Output:
# True
# True
```

### workflows-save-load-edited
An edited fork saves with its edit, and the loaded copy still says so.

- covers: `Trace.save`, `tl.load`, `Op.is_edited`
- tasks: save-load, ablate
- fixture: toy

```python
edited = tl.trace(model, x).fork().do("relu_1_2", tl.zero_ablate(), model=model, inputs=x)
edited.save("edited.tlspec")
loaded = tl.load("edited.tlspec")
print(loaded["relu_1_2"].is_edited, bool((loaded["relu_1_2"].out == 0).all()))
# Output:
# True True
```

## Bundles of several traces

### workflows-bundle-several-inputs
Capture the same model on several inputs, bundle them, and read one site across all of them; the members' batches are stacked in order.

- covers: `tl.Bundle`, `Bundle.ops`
- tasks: compare-runs
- fixture: toy

```python
traces = {f"x{i}": tl.trace(model, x * i) for i in range(3)}
bundle = tl.Bundle(traces, baseline="x1")
stacked = bundle.ops["relu_1_2:1"].out
print(tuple(stacked.shape))
print(torch.equal(stacked[:3], traces["x0"]["relu_1_2:1"].out))
# Output:
# (9, 8)
# True
```

### workflows-bundle-rank-edits
Bundle a clean run with two edited forks and rank which sites moved most.

- covers: `tl.Bundle`, `Bundle.rank_changes`
- tasks: compare-runs, ablate
- fixture: toy

```python
clean = tl.trace(model, x)
half = clean.fork().do("relu_1_2", tl.scale(0.5), model=model, inputs=x)
zero = clean.fork().do("relu_1_2", tl.zero_ablate(), model=model, inputs=x)
bundle = tl.Bundle({"clean": clean, "half": half, "zero": zero}, baseline="clean")
ranked = bundle.rank_changes(k=2, metric="relative_l2")
print([label for label, _ in ranked])
# Output:
# ['relu_1_2', 'linear_2_3']
```

## Drawing

### workflows-draw-trace-file
Draw a captured trace to an SVG in the working directory.

- covers: `Trace.draw`
- tasks: draw
- fixture: toy

```python
import os

trace = tl.trace(model, x)
trace.draw("toy.svg")
print(os.path.exists("toy.svg"))
# Output:
# True
```

### workflows-draw-rolled-loop
Draw a recurrent model rolled up, one node per layer instead of one per pass.

- covers: `Trace.draw`
- tasks: draw, loops
- fixture: loop

```python
import os

trace = tl.trace(model, x)
trace.draw("unrolled.svg")
trace.draw("rolled.svg", rolled=True)
print(os.path.exists("unrolled.svg"), os.path.exists("rolled.svg"))
# Output:
# True True
```

### workflows-draw-options
Drawing options: direction, theme and legend, and the backward graph after a backward pass.

- covers: `Trace.draw`
- tasks: draw, gradients
- fixture: toy

```python
import os

trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
trace.draw("left_right.svg", direction="leftright", theme="dark", show_legend=True)
trace.draw("backward.svg", graph="backward")
print(os.path.exists("left_right.svg"), os.path.exists("backward.svg"))
# Output:
# True True
```

### workflows-draw-one-call-pdf
`tl.draw` captures and draws in one call; the file extension picks the format.

- covers: `tl.draw`
- tasks: draw
- fixture: cnn

```python
import os

drawing = tl.draw(model, x, path="cnn.pdf")
print(drawing.file_format, os.path.exists("cnn.pdf"))
# Output:
# TorchLens graph written to cnn.pdf
# pdf True
```
