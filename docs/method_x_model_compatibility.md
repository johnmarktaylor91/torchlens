# Method x Model Compatibility

Generated from `tl.compat.report` on representative eager PyTorch models. These rows are a
smoke reference for ordinary dense eager execution, not a complete certification matrix.

Generation snippet:

```python
import torch
from torch import nn
import torchlens as tl

models = {
    "linear_mlp": nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).eval(),
    "conv_pool": nn.Sequential(
        nn.Conv2d(1, 2, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    ).eval(),
}
inputs = {
    "linear_mlp": torch.ones(1, 4),
    "conv_pool": torch.ones(1, 1, 5, 5),
}

for name, model in models.items():
    print(name)
    print(tl.compat.report(model, inputs[name]).to_markdown())
```

## Representative Results

| Model | Rows | Non-pass rows |
| --- | ---: | --- |
| `linear_mlp` | 17 | none |
| `conv_pool` | 17 | none |

Both representative models report `pass` for Hugging Face wrapper detection, Accelerate
offload/device-map detection, bitsandbytes detection, tied parameters, multi-GPU RNG context,
`nn.DataParallel`, DDP, FSDP, DeepSpeed, `torch.compile`, FX `GraphModule`, Lightning
training-step context, functorch/vmap markers, quantized tensors/modules, device-context factory
handling, and single-thread design.

Interpretation: plain eager dense PyTorch models are the compatibility baseline. For wrappers,
compiled execution, sharding/offload, quantization, or concurrent capture, run
`tl.compat.report(model, x)` on the exact model/input pair and include the report when filing an
issue.
