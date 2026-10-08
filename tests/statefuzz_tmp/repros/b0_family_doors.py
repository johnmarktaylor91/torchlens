# ruff: noqa
"""B0 family: the rerun hook doubling reaches attach_hooks + run and do(engine='rerun') too."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl

model, x, x2 = make()
fires = [0]


def cnt(out, *, hook):
    fires[0] += 1
    return out * 0.5


t = tl.trace(model, x)
t.attach_hooks(tl.module("fc2"), cnt, confirm_mutation=True)
counts = []
for _ in range(3):
    fires[0] = 0
    t.run(model, x)
    counts.append(fires[0])
print(
    "attach_hooks + run x3: hook fires per rerun =",
    counts,
    "staged hooks =",
    len(t._ensure_intervention_spec().hook_specs),
)
d = torch.randn(16, generator=torch.Generator().manual_seed(7))
f = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
opts = tl.options.InterventionOptions(engine="rerun")
for n in (1, 2, 3):
    f.do(
        tl.module("fc2"),
        tl.steer(d, magnitude=1.0, feature_axis=-1),
        model=model,
        x=x,
        intervention=opts,
    )
    o = f.output_ops[0].out
    best = min(range(1, 9), key=lambda k: md(o, hooked(model, "fc2", lambda z, k=k: z + k * d, x)))
    print(f"do(engine='rerun') call {n}: result equals steer x{best} (expected x{n})")
