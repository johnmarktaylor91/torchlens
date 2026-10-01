import json
from pathlib import Path

from jax import random

import torchlens as tl


def identity_key(params, key):
    del params
    return key


trace = tl.trace(identity_key, (None, random.PRNGKey(123)), backend="jax")
path = Path("/tmp/jax_old_style_prng_key.tlspec")
trace.save(path, level="portable")
manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
for entry in manifest["tensors"]:
    print(entry.get("label"), entry.get("logical_dtype"))
