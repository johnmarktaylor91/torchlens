# ruff: noqa
"""F6: tl.trace reseeds the global RNG for the forward and restores it after; spec.bind does not.
Same seed, same spec: the trace door and the bind door give different outputs on a dropout model."""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import torchlens as tl


class DropNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.drop, self.fc2 = nn.Linear(8, 16), nn.Dropout(0.5), nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


torch.manual_seed(0)
model = DropNet().train()
x = torch.randn(4, 8)
spec = tl.when(tl.module("fc2"), tl.scale(1.0))
torch.manual_seed(42)
eager = model(x)
after_eager = torch.rand(2)
torch.manual_seed(42)
t = tl.trace(model, x, intervene=spec)
after_trace = torch.rand(2)
torch.manual_seed(42)
b = spec.bind(model)(x)
torch.manual_seed(42)
nothing = torch.rand(2)
print("trace(intervene=identity) vs eager, same seed:", md(t.output_ops[0].out, eager))
print("bind(identity) vs eager, same seed:           ", md(b, eager))
print(
    "RNG after tl.trace equals RNG as if nothing ran:",
    torch.equal(after_trace, nothing),
    "| equals RNG after one eager forward:",
    torch.equal(after_trace, after_eager),
)
print("trace.random_seed:", t.random_seed)
