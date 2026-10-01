# ruff: noqa -- harvest-time provenance script, committed as run (lint header added at import)
import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
out = sys.argv[1]
import torch
import torchvision

import torchlens as tl

print("torchlens", getattr(tl, "__version__", "?"))
from torchlens import _io

print("IO_FORMAT_VERSION", getattr(_io, "IO_FORMAT_VERSION", None))
m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
x = torch.randn(1, 3, 32, 32)
fn = getattr(tl, "log_forward_pass", None) or getattr(tl, "get_model_activations", None)
print("capture fn:", fn.__name__ if fn else None)
log = fn(m, x)
tl.save(log, out)
d = json.load(open(os.path.join(out, "manifest.json")))
print("MANIFEST_KEYS", json.dumps(sorted(d.keys())))
for k in ("io_format_version", "tlspec_version", "torchlens_version", "save_level"):
    if k in d:
        print("FIELD", k, "=", d[k])
