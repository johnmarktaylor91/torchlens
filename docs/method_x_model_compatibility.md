# Method x Model Compatibility

<!-- GENERATED FILE; do not hand-edit. tests/test_schema_lockstep.py
regenerates and diffs this doc. Refresh from the repo root with:
python -c "import sys; sys.path.insert(0, 'tests'); import test_schema_lockstep as m; m.write_method_x_model_compatibility_doc()"
-->

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
| `linear_mlp` | 25 | `Runtime capability snapshot` |
| `conv_pool` | 25 | `Runtime capability snapshot` |

Non-pass rows above name the checks that did not report `pass`; every other check
reported `pass`. The full check list:

- HF Transformers wrapper
- Accelerate device_map='auto'
- Accelerate CPU/disk offload
- bitsandbytes 8-bit/4-bit
- PEFT/LoRA adapters
- Tied/shared parameters
- Multi-GPU RNG
- nn.DataParallel
- DistributedDataParallel
- FSDP
- DTensor / sharded tensors
- Device mesh
- Tensor parallel (TP)
- Pipeline parallel (PP)
- DeepSpeed
- torch.compile
- FX GraphModule
- Runtime capability snapshot
- Protocol-invisible belt coverage
- Lightning training_step mid-loop
- vmap/functorch
- Quantized tensors/modules
- fp8 (float8_*) tensors
- DeviceContext factory injection
- Single-thread design

Interpretation: plain eager dense PyTorch models are the compatibility baseline. For wrappers,
compiled execution, sharding/offload, quantization, or concurrent capture, run
`tl.compat.report(model, x)` on the exact model/input pair and include the report when filing an
issue.
