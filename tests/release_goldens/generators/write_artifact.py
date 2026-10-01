# ruff: noqa -- harvest-time provenance script, committed as run (lint header added at import)
"""Write a genuine artifact using whatever torchlens is on PYTHONPATH.
Usage: write_artifact.py <outdir> <level>
Real model: torchvision resnet18 IMAGENET1K_V1 (cached), 32x32 input (small artifact).
"""

import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
outdir, level = sys.argv[1], sys.argv[2]
import torch
import torchvision

import torchlens as tl

print("torchlens", getattr(tl, "__version__", "?"))
try:
    from torchlens import _io

    print(
        "TLSPEC_VERSION",
        getattr(_io, "TLSPEC_VERSION", None),
        "MIN_TLSPEC_VERSION",
        getattr(_io, "MIN_TLSPEC_VERSION", None),
        "MIN_TORCHLENS_VERSION_TEXT",
        getattr(_io, "MIN_TORCHLENS_VERSION_TEXT", None),
    )
except Exception as e:
    print("io-probe-fail", type(e).__name__, e)
m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
m.eval()
x = torch.randn(1, 3, 32, 32)
t = tl.trace(m, x)
print("trace ok; ops:", len(t))
tl.save(t, outdir, level=level)
mp = os.path.join(outdir, "manifest.json")
d = json.load(open(mp))
print("MANIFEST_KEYS", json.dumps(sorted(d.keys())))
for k in (
    "tlspec_version",
    "io_format_version",
    "torchlens_version",
    "save_level",
    "torch_version",
):
    if k in d:
        print("FIELD", k, "=", d[k])
