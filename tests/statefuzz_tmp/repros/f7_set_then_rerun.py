# ruff: noqa
"""F7: Trace.set(label, value) followed by the legacy rerun refuses with a live-label error whose
remedy does not apply to set(); a staged attach_hooks rerun warns ControlFlowDivergenceWarning."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl

model, x, x2 = make()
t = tl.trace(model, x)
label = t.find_sites(tl.module("fc2"))[0].layer_label
t.set(label, torch.full_like(t[label].out, 0.3), confirm_mutation=True)
try:
    t.run(model, x2)
    print("set + run(model, x2): ok")
except Exception as e:
    print("set + run(model, x2) raised", type(e).__name__, ":", str(e).splitlines()[0][:230])
t = tl.trace(model, x)
t.attach_hooks(tl.module("fc2"), tl.scale(0.5), confirm_mutation=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t.run(model, x)
want = hooked(model, "fc2", lambda o: o * 0.5, x)
print("attach_hooks + run(model, x): value vs truth =", md(t.output_ops[0].out, want))
show_warnings(w)
