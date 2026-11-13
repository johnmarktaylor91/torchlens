# Coming from flax tabulate

This page is for people who print model tables with Flax's `nnx.tabulate` (or Linen's
`Module.tabulate`) and now have a PyTorch model. `tl.summary(model, x)` is the counterpart:
one call runs the model on a real input and returns one table with a row per module, its
output shape and its parameters, and a totals footer. The Flax example was written against
the Flax release TorchLens's JAX backend tests pin (`flax==0.12.8` with `jax[cpu]==0.10.2`)
and skips where Flax is not installed. Checked 2026-11-03.

## The same task in both

The task: the table of a two-layer network, and its total parameter count.

### In Flax

```python
# migration-test: tool=flax expected=True
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(8, 16, rngs=rngs)
        self.linear2 = nnx.Linear(16, 4, rngs=rngs)

    def __call__(self, x):
        return self.linear2(nnx.relu(self.linear1(x)))


table = nnx.tabulate(MLP(nnx.Rngs(0)), jnp.zeros((2, 8)))
RESULT = "Total Parameters: 212" in table
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=(212, 848, True)
import torch
from torch import nn

import torchlens as tl


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


report = tl.summary(MLP(), torch.zeros(2, 8), columns=["+input"])
RESULT = (report.total_params, report.totals.params_bytes, "linear1 (Linear)" in report)
```

`print(report)` shows:

```text
MLP | input (2, 8) float32 | eval mode, no_grad, state restored
view: hybrid, all 3 ops
name (type)       input    output   params        (%)  fwd flops
----------------------------------------------------------------
linear1 (Linear)  (2, 8)   (2, 16)     144  ##. 67.9%        544
relu_1_2          (2, 16)  (2, 16)       -          -         32
linear2 (Linear)  (2, 16)  (2, 4)       68  #.. 32.1%        264
----------------------------------------------------------------
Params   212 declared | trainable 212 (100%) | size 848 B
Compute  840 FLOPs fwd (fma=2) | 384 MACs | 3/3 ops known (formula-exact)
```

The footer goes on with Memory, Graph and Capture lines (left out here). The functional
`torch.relu` gets its own row because it belongs to no module; Flax's table has no row for
`nnx.relu` for the same reason, and TorchLens shows it so that every executed operation is
counted once.

## Translation table

| flax `tabulate` | TorchLens `tl.summary` | Notes |
| --- | --- | --- |
| `nnx.tabulate(model, x)` | `tl.summary(model, x)` | Both run the model on the input. TorchLens returns the report (a `str` subclass) instead of a string, and prints nothing until you print it. |
| `path` column | `name` column | The attribute name, indented by depth. |
| `type` column | the `(type)` part of `name (type)` | The module class. |
| `inputs` column | `columns=["+input"]` | Shapes of the distinct tensors entering the row. |
| `outputs` column | `output` column | Shapes only; the input dtype is in the header line. |
| One column per variable type (`Param`, ...) with shapes, count and bytes | `params` column (count per module path) and `(%)`; bytes in the Params footer (`size 848 B`, `report.totals.params_bytes`) | Per-tensor shapes are not a column: read `model.named_parameters()`. Tied weights are counted once and named. |
| `BatchStat` column | Not a column | BatchNorm running statistics are registered buffers in PyTorch, not parameters, so `params` does not count them; `trace.buffers` lists them ([buffers](../buffers.md)). |
| `RngState` column | No counterpart | PyTorch modules hold no random-number state; dropout draws from the global generator. |
| `compute_flops=True` (`flops` column) | `fwd flops` column, in the default view; `view="compute"` adds `macs` and the evidence class | Counted from the executed operations, not compiled cost analysis. |
| `compute_vjp_flops=True` (`vjp_flops` column) | Not a summary column | After a captured backward, `tl.report.cost_tree(trace)` reports executed backward FLOPs per module call (`self_backward_flops`; [backward](../backward.md)). |
| `depth=2` | `depth=2` | `depth="all"` for the full tree. |
| `row_filter=` | `where=` (regex, selector or callable) | Totals stay whole-model, and the header says what share the shown rows cover. |
| `method="encode"` | No counterpart | TorchLens runs the module's `forward`; wrap another method in a small module whose `forward` calls it. |
| `table_kwargs=`, `column_kwargs=`, `console_kwargs=` (rich) | `report.print()`, `report.render("unicode")`, `report.to_markdown()`, `report.to_html()` | No rich dependency; `str(report)` is byte-stable ASCII. |
| `Total` row and the `Total Parameters: N (size)` line | Params footer line | `report.total_params` and `report.totals` hold the numbers as plain values. |

## What TorchLens adds

- Executed FLOPs and true MACs per row, with an evidence class per row, and a Compute footer
  that names any operation it could not count instead of skipping it.
- Rows for functional operations that belong to no module, and parameters that never ran
  split out by name.
- The report as data: `report.to_pandas()`, `report.to_dict()`, and the totals as plain ints.
- A capture under the table: `tl.trace(model, x)` keeps every operation's output for
  inspection, and `trace.summary()` prints the same kind of table for it.

## What flax tabulate does that TorchLens does not

- It summarizes Flax and JAX models. TorchLens summarizes PyTorch models; its JAX backend
  captures functions and Equinox and Flax NNX modules, without
  `nnx.tabulate`'s table ([JAX backend](../backends/jax.md)), and replacing `nnx.tabulate` is
  not a goal ([non-goals](../non_goals.md)).
- It shows every variable collection, including random-number state, as its own column.
- It estimates backward FLOPs in the same table.

## Quantized models

The table above is for ordinary floating-point models. For a quantized model TorchLens
reports what the capture recorded, and quantized activation metadata is best-effort today
(the `quantized_tensor` row of the capability table in the repository's top-level `LIMITATIONS.md`). There is no
bits-per-weight line in the footer: that line needs quantized capture TorchLens does not yet provide.

## Using both

Keep `nnx.tabulate` for the Flax model and use `tl.summary` for its PyTorch port. Comparing
`Total Parameters` with `report.total_params` is a quick check that a port has the same
parameters; the per-row counts show where a difference sits.

## Semantic differences

- Flax's `Total` row adds up every variable column the table shows, so batch statistics
  count toward it. TorchLens's `params` and `size` count parameters only; a BatchNorm layer's
  running statistics are buffers and sit outside both. For the network above both tools
  report 212 parameters and 848 bytes.
- TorchLens runs the model in eval mode with gradients off by default
  (`execution_mode="eval"`) and restores its state afterwards; pass
  `execution_mode="same"` to keep the model's own flags.
- Repeated modules fold into one row (`h.0..11 (GPT2Block x12)`) unless
  `fold_repeats=False`.
