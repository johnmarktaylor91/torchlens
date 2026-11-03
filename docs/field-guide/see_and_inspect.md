# See the graph and inspect any operation

**Result in one line:** a picture of every operation the forward ran, and the output,
module and arguments of any one of them.

<!-- JMT review: plain factual prose, written for review. -->

**Requirements:** `pip install torchlens torchvision`; drawing also needs the Graphviz `dot`
binary (`apt install graphviz`).

## Minimal example on the running model

This page uses [ResNet-18 on one image](running_examples.md#resnet-18-on-one-image).

```python
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.rand(1, 3, 224, 224)

trace = tl.trace(model, x)
```

Draw it, then read one module's output and one operation's metadata:

```python
drawing = trace.draw(collapse="auto", file_format="svg")
block = trace["layer3"]
print(block.out.shape)
relu = next(op for op in trace.ops if op.func_name == "relu")
print(relu.label, relu.func_name, tuple(relu.shape))
```

## Output

`trace.summary()` prints the recorded layers with their shapes. `trace.audit()` returns the
health report for the capture, including the replay check.

```python
print(trace.summary())
print(trace.audit())
```

## A check

The recorded output equals the model's own output for the same input, and the module output
read from the trace equals the one a forward hook on the same module sees.

```python
seen = {}
handle = model.layer3.register_forward_hook(lambda module, args, out: seen.update(out=out))
model(x)
handle.remove()
assert torch.equal(trace["layer3"].out, seen["out"])
assert torch.equal(trace.output, model(x))
```

## Vary it

Change `"layer3"` to any module name in `model.named_modules()`, or select by function
instead of by module with `tl.func("conv2d")` or `tl.in_module("layer2")`. Operations that no
module returns, such as the residual adds, are in `trace.ops` with labels of their own. Draw a
smaller picture with `trace.draw(module="layer2")`.

```python
adds = [op.label for op in trace.ops if "add" in op.func_name]
print(len(adds), "residual adds, none of them a module output:", adds[:2])
```

## Performance note

A full trace saves every activation, so memory grows with the model. Pass `save=` with a
selector to keep only what you read (`tl.trace(model, x, save=tl.module("layer3"))`); the
structure of every operation is still recorded. The cost of capture against a plain forward
is measured on the [capture benchmark page](../benchmarks/capture.md).

## Limits

- Fused kernels are one operation: TorchLens does not expose tensors that exist only inside a
  kernel.
- `torch.compile`, TorchScript and `torch.export` artifacts are not traced; trace the eager
  module.
- Drawing needs Graphviz; without it, `trace.summary()` and `trace.ops` still work.

## Next task

[Extract features for a stimulus set](extract_features.md), or [intervene on a site](intervene.md).
