# ruff: noqa
"""F4: tl.module('fc:2') (pass-qualified) matches zero sites on the live capture doors and on reruns."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl


class TwicePass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc, self.out = nn.Linear(8, 8), nn.Linear(8, 3)

    def forward(self, x):
        return self.out(torch.relu(self.fc(torch.relu(self.fc(x)))))


torch.manual_seed(0)
model = TwicePass().eval()
x, x2 = torch.randn(2, 8), torch.randn(2, 8)


def truth(xx):
    calls = [0]

    def hk(m, a, o):
        calls[0] += 1
        return o * 0 if calls[0] == 2 else o

    h = model.fc.register_forward_hook(hk)
    with torch.no_grad():
        y = model(xx)
    h.remove()
    return y


spec = tl.when(tl.module("fc:2"), tl.zero_ablate())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t = tl.trace(model, x, intervene=spec)
    r = tl.record(model, x, default_op=True, intervene=spec).to_trace()
    with torch.no_grad():
        b = spec.bind(model)(x)
    f = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    f.do(tl.module("fc:2"), tl.zero_ablate())
    fd = f.output_ops[0].out.clone()
    f.run(model, x2)
with torch.no_grad():
    clean = model(x)
print(
    "trace(intervene=) vs truth:",
    md(t.output_ops[0].out, truth(x)),
    "| vs clean:",
    md(t.output_ops[0].out, clean),
)
print("record(intervene=) vs truth:", md(r.output_ops[0].out, truth(x)))
print("bind vs truth:", md(b, truth(x)))
print("fork.do vs truth:", md(fd, truth(x)))
print("fork.run(model, x2) vs truth:", md(f.output_ops[0].out, truth(x2)))
show_warnings(w)
