# Intervention Visibility Classes

TorchLens can intervene on operations that its decoration layer sees during a
normal Python `forward` pass. In practice, that means tensor operations routed
through wrapped PyTorch callables, plus module output boundaries that TorchLens
records during capture.

## Leaf Op

A leaf op is a single decorated PyTorch operation such as `torch.relu`,
`torch.add`, `torch.matmul`, or a tensor method that TorchLens wraps. These are
the most direct intervention targets:

```python
log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
log.find_sites(tl.func("relu"))
edited = log.fork("zero_relu").attach_hooks(tl.func("relu"), tl.zero_ablate()).replay()
```

Replay works best when the downstream graph is stable and the saved call
metadata is sufficient to recompute the affected cone.

## Compound Module

A compound module is a user or library `nn.Module` that contains several leaf
ops. TorchLens records module entry and exit metadata, so you can discover
inside a module with `tl.in_module("encoder.layers.0")` and then target a finer
leaf op, or target a module output boundary with `tl.module(...)` when that is
the right semantic site.

The conservative workflow is discover first:

```python
sites = log.find_sites(tl.in_module("block"), max_fanout=32)
target = tl.label(sites.where(lambda site: site.func_name == "relu").labels()[0])
```

## Opaque Fused Kernel

Opaque fused kernels execute internal math that is not exposed as separate
Python-level PyTorch calls. TorchLens can target the visible fused call, but not
hidden intermediates inside it.

Important examples:

- `torch.nn.functional.scaled_dot_product_attention` is fused from TorchLens'
  perspective. You can intervene on the visible SDPA call output, but not the
  internal Q/K/V scores, attention probabilities, or value aggregation steps.
- FlashAttention kernels have the same limitation: the kernel internals are not
  TorchLens leaf ops.
- Compiler stacks such as `torch.compile`, TorchScript, and exported graphs can
  bypass the Python calls TorchLens decorates. Capture the original uncompiled
  `nn.Module` when you need intervention visibility.

If you need intervention access to Q/K/V intermediates, use a manual attention
implementation that calls visible PyTorch operations for projections, scores,
softmax, and value aggregation.
