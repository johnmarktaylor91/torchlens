# ruff: noqa
"""F3: post-hoc tl.module()/tl.in_module() on a module whose output is returned also matches the
synthetic output node, so fork.do() applies the edit twice (x0.25 instead of x0.5)."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl

model, x, x2 = make()
want = hooked(model, "fc3", lambda o: o * 0.5, x)
twice = hooked(model, "fc3", lambda o: o * 0.25, x)
ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
print(
    "sites matched by tl.module('fc3'):",
    [s.layer_label for s in ready.find_sites(tl.module("fc3"))],
)
f = ready.fork()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f.do(tl.module("fc3"), tl.scale(0.5))
show_warnings(w)
print(
    "fork.do vs truth (x0.5):",
    md(f.output_ops[0].out, want),
    "| vs x0.25:",
    md(f.output_ops[0].out, twice),
)
t = tl.trace(model, x, intervene=tl.when(tl.module("fc3"), tl.scale(0.5)))
with torch.no_grad():
    b = tl.when(tl.module("fc3"), tl.scale(0.5)).bind(model)(x)
print(
    "tl.trace(intervene=) vs truth:", md(t.output_ops[0].out, want), "| bind vs truth:", md(b, want)
)
