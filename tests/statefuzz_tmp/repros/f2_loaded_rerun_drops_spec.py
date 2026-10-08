# ruff: noqa
"""F2: a saved+loaded intervened trace reruns UN-intervened through trace.run(model, x), silently."""

import sys, os, tempfile

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl

model, x, x2 = make()
want = hooked(model, "fc2", lambda o: o * 0.5, x2)
with torch.no_grad():
    clean = model(x2)
for door in ("intervene", "attach_hooks", "fork_do"):
    if door == "intervene":
        t = tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.scale(0.5)))
    elif door == "attach_hooks":
        t = tl.trace(model, x)
        t.attach_hooks(tl.module("fc2"), tl.scale(0.5), confirm_mutation=True)
    else:
        t = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
        t.do(tl.module("fc2"), tl.scale(0.5))
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.tlspec")
        t.save(p)
        t2 = tl.load(p)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t2.run(model, x2)
        t.run(model, x2)
        print(
            f"{door:12s} live hooks={len(t._ensure_intervention_spec().hook_specs)} loaded hooks="
            f"{len(t2._ensure_intervention_spec().hook_specs)} spec_revision(loaded)={t2._spec_revision} | "
            f"live rerun vs truth={md(t.output_ops[0].out, want):.3g}  loaded rerun vs truth="
            f"{md(t2.output_ops[0].out, want):.3g}  loaded rerun vs CLEAN={md(t2.output_ops[0].out, clean):.3g}"
        )
        show_warnings(w)
