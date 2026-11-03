# Which tool for this job?

What TorchLens and seven other tools do, job by job, including the jobs another tool does and
TorchLens does not. Checked 2026-11-03 against the versions named in the table header; tool
status and sources are on [the landscape page](landscape.md).

<!-- JMT review: plain factual prose, written for review. -->

## When you need TorchLens, and when five lines of hooks are enough

Five lines of `register_forward_hook` are enough when the value you want is the output of one
module, you know which module, and the model is yours to read. Hooks ship with torch, add no
dependency, and cost almost nothing per call.

You need a capture when the value is not a module output (a functional call, a residual add, an
in-place op inside `forward`), when you do not know in advance which site matters, when you
want to check that what you recorded is what the forward computed, or when you will edit a
site and compare runs. The fence below shows the first case: the hook sees the module outputs,
and the ReLU that `forward` calls as a function is not one of them.

```python
import torch
from torch import nn

import torchlens as tl


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + x


torch.manual_seed(0)
model = Block().eval()
x = torch.randn(2, 4)

hooked = []
handles = [m.register_forward_hook(lambda m, a, out: hooked.append(out)) for m in model.modules()]
model(x)
for handle in handles:
    handle.remove()

trace = tl.trace(model, x)
funcs = [op.func_name for op in trace.ops]
assert len(hooked) == 2  # the Linear and the Block itself
assert "relu" in funcs  # the functional ReLU is a recorded operation
```

## The table

Cells read: **yes**, **partial** (with what is missing), **no**, or **by hand** (possible with
code you write, not provided). Rows where another tool covers a job TorchLens does not are
marked in the last column.

| Job | TorchLens | Plain hooks (torch) | TransformerLens 4.3 | nnsight 0.8 | pyvene 0.1.8 | Captum 0.9 | thingsvision 1.4 | torchvision feature extraction | Another tool covers more |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Read operations no module returns (functional calls, residual adds) | yes | no: module boundaries only | partial: the hook points its model classes define | yes: `.source` | no: named components | no: layer hooks | no: module outputs | partial: fx graph nodes, symbolically traceable models only | |
| Any PyTorch architecture, no per-family adapter | yes | yes | partial: transformers, through `TransformerBridge` | yes | partial: component names per model type | yes | partial: a model registry plus custom models | partial: symbolically traceable models | |
| Data-dependent control flow in `forward` | yes: the executed path | yes | yes | yes | yes | yes | yes | no: symbolic trace | |
| Check the recording against the real forward by replay | yes | no | no | no | no | no | no | no | |
| Draw the executed graph | yes | no | no | no | no | no | no | partial: prints the fx graph | |
| Edit an activation and rerun (ablate, patch, steer) | yes | by hand | yes | yes | yes | partial: ablation inside attribution methods | no | no | |
| Edit applied live at forward-hook cost, including `generate()` | yes: `tl.when(...).bind(model)` | yes | yes | yes | yes | no | no | no | |
| Batch feature extraction over a stimulus set, with provenance | yes: `tl.extract_dataset` | by hand | partial: `run_with_cache` in your loop | by hand | no | no | yes | partial: your loop | |
| Save a recording, load it, diff two runs | yes | by hand | partial: the cache as tensors | partial: saved values | partial: saved interventions | no | partial: saves features | no | |
| Gradients at internal sites | yes | yes: tensor hooks | yes | yes | partial | yes: layer gradients | no | partial: autograd on outputs | |
| Trainable interventions (DAS, LoReFT) | no: reads pyreft artifacts, ships no trainer | by hand | no | by hand | yes | no | no | no | pyvene, pyreft |
| Sparse autoencoder training | no: [use a trainer](../field-guide/sae_trainers.md) | no | no: SAELens builds on it | no | no | no | no | no | SAELens |
| Attribution method breadth (TCAV, influence functions, attacks, text DeepLift and LIME) | partial: a gradient and perturbation set in `tl.attribution` | no | no | no | no | yes | no | no | Captum, inseq |
| Serving-scale and remote capture (vLLM, remote GPUs, tensor parallel) | no: local eager forward | no | no | yes: vLLM back end, NDIF | no | no | no | no | nnsight |
| Model-specific LLM conveniences (canonical hook names across families, pretrained loader, folded LayerNorm) | partial: resolves TransformerLens names on Hugging Face models | no | yes | partial: `LanguageModel` | no | no | no | no | TransformerLens |
| One module's output, nothing to install | works, with a dependency and capture cost | yes | no | no | no | no | no | yes, with torchvision | plain hooks |
| Non-PyTorch models | preview: JAX, TensorFlow, MLX, Paddle, tinygrad | no | no | no | no | no | partial: TensorFlow and Keras models | no | |

## What the table does not say

- Reading module-less operations, diffusers pipelines, quantized models and vision-language
  models is shared ground with nnsight 0.8; the first row is not a claim TorchLens alone has.
- "No" means the tool does not provide it on the check date, not that it cannot be built on top.
- A tool's own documentation is the authority for its column; corrections are welcome as issues.

Where a row points to another tool, [where to go instead](where_to_go_instead.md) says how to
get there and how to bring the result back to TorchLens.
