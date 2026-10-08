# ruff: noqa
"""F1: copy.deepcopy of a model after tl.trace/tl.record runs the ORIGINAL's submodules."""

import copy, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl

model, x, _ = make()
tl.trace(model, x)  # any capture (or tl.record) prepares the model
clone = copy.deepcopy(model)
with torch.no_grad():
    for p in clone.parameters():
        p.mul_(2.0)  # the clone now has different weights
fresh = MLP().eval()
fresh.load_state_dict(clone.state_dict())
with torch.no_grad():
    print("clone(x) vs a fresh model with the clone's weights:", md(clone(x), fresh(x)))
    print("clone(x) vs the ORIGINAL model(x):              ", md(clone(x), model(x)))
print("clone.fc1 has an instance 'forward':", "forward" in vars(clone.fc1))
try:
    tl.trace(clone, x)
    print("tl.trace(clone) ok")
except Exception as e:
    print("tl.trace(clone) raised", type(e).__name__, str(e).splitlines()[0][:80])
tl.release_model(model)
clone2 = copy.deepcopy(model)
with torch.no_grad():
    for p in clone2.parameters():
        p.mul_(2.0)
fresh.load_state_dict(clone2.state_dict())
with torch.no_grad():
    print("after tl.release_model: clone2(x) vs fresh:", md(clone2(x), fresh(x)))
