# Task RNN circuits: windowed interventions, the recurrence Jacobian and fixed points

Systems neuroscience studies trained recurrent networks the way it studies circuits: train a
small RNN on a memory task, find the fixed points its dynamics settle into, and test causally
that those fixed points carry the memory. This recipe does all of that on a CPU in seconds.
TorchLens captures every timestep of the recurrence as one pass of one layer, reads the hidden
state at any pass, and edits chosen passes: a window of timesteps, or a recurrent weight at
every timestep it is used. The recurrence Jacobian and the fixed-point search are plain
`torch.autograd` code inside the recipe.

**Requirements:** `pip install torchlens`. No download, no GPU; the whole page runs in about
10 seconds on one CPU core.

## 1. Train a task RNN

The task is the 2-bit flip-flop: two input channels receive occasional +1 or -1 pulses, and
each output must hold the sign of its channel's most recent pulse. A trained network needs
four stable memory states, one per bit pattern. The model is a vanilla tanh RNN written as an
explicit loop over timesteps, so each step is a separate call TorchLens can address. (The
fused `nn.RNN` and `nn.GRU` kernels run the whole sequence as one operation, which hides the
per-step hidden states.)

```python
# recipe-test: tool=torchlens expected=True
import torch
from torch import nn

import torchlens as tl

N_BITS, HIDDEN, STEPS = 2, 32, 40


class FlipFlopRNN(nn.Module):
    """A vanilla tanh RNN: h[t] = tanh(W_in x[t] + W_rec h[t-1] + b)."""

    def __init__(self) -> None:
        super().__init__()
        self.w_in = nn.Linear(N_BITS, HIDDEN)
        self.w_rec = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.readout = nn.Linear(HIDDEN, N_BITS)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.w_in(x_t) + self.w_rec(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.new_zeros(x.shape[0], HIDDEN)
        outputs = []
        for t in range(x.shape[1]):
            h = self.step(x[:, t], h)
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1)


def flip_flop_batch(batch: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random +1/-1 pulses on each input bit; the target holds each bit's last pulse."""

    pulses = torch.rand(batch, STEPS, N_BITS, generator=generator) < 0.1
    pulses[:, 0] = True  # every trial starts by setting every bit
    signs = torch.randint(0, 2, (batch, STEPS, N_BITS), generator=generator) * 2.0 - 1.0
    x = pulses * signs
    target = torch.empty_like(x)
    target[:, 0] = x[:, 0]
    for t in range(1, STEPS):
        target[:, t] = torch.where(pulses[:, t], x[:, t], target[:, t - 1])
    return x, target


torch.manual_seed(0)
generator = torch.Generator().manual_seed(0)
model = FlipFlopRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(400):
    x, target = flip_flop_batch(64, generator)
    loss = nn.functional.mse_loss(model(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

x, target = flip_flop_batch(256, generator)
with torch.no_grad():
    accuracy = (model(x).sign() == target).float().mean().item()
print(f"held-out sign accuracy: {accuracy:.3f}")
RESULT = accuracy > 0.98
```

The held-out accuracy prints as 1.000.

## 2. Capture every timestep

The probe batch has one trial per bit pattern: both bits are set at the first step, then the
input is silent and the network must hold the pattern for the remaining 39 steps.

```python
# recipe-test: tool=torchlens expected=(40, True)
patterns = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
probe = torch.zeros(4, STEPS, N_BITS)
probe[:, 0] = patterns  # set both bits once, then hold

trace = tl.trace(model, probe)
STATE = trace.find_sites(tl.func("tanh"), max_fanout=STEPS).first().layer_label
states = torch.stack([trace[f"{STATE}:{t}"].out for t in range(1, STEPS + 1)], dim=1)

with torch.no_grad():
    h, own = probe.new_zeros(4, HIDDEN), []
    for t in range(STEPS):
        h = model.step(probe[:, t], h)
        own.append(h)
RESULT = (trace.layers[STATE].num_passes, torch.equal(states, torch.stack(own, dim=1)))
```

The `tanh` that produces the hidden state is one layer with 40 passes, one per timestep, and
`STATE:t` (for example `tanh_1_6:15`) addresses the hidden state after step `t`. The stacked
states equal the ones the model computes step by step, so `states` (4 trials, 40 steps, 32
units) is the trajectory with no hooks written.

## 3. Windowed interventions across passes

A cross-pass selector takes an explicit window of passes. During the silent hold the hidden
state stops moving: every unit's variance across passes 20 to 40 is below `1e-6`, the
signature of a fixed point.

```python
# recipe-test: tool=torchlens expected=1.0
def selected_fraction(selection) -> float:
    return sum(int(e.mask.sum()) for e in selection) / sum(e.mask.numel() for e in selection)


held = trace[tl.pass_variance(STATE, below=1e-6, passes=range(20, STEPS + 1))]
RESULT = selected_fraction(held)
```

Now edit a window of timesteps. Zeroing the hidden state at passes 15 to 20 and replaying the
rest of the sequence from the edited state tests whether the memory lives in the state: every
trial ends in the same pattern, so both stored bits are lost.

```python
# recipe-test: tool=torchlens expected=(True, 1)
def final_patterns(run: tl.Trace) -> list[tuple[int, ...]]:
    return [tuple(row) for row in run["output_1"].out[:, -1].sign().int().tolist()]


window = {tl.label(f"{STATE}:{t}"): tl.zero_ablate() for t in range(15, 21)}
zeroed = trace.fork().do(window)
clean_ok = final_patterns(trace) == [tuple(row) for row in patterns.int().tolist()]
RESULT = (clean_ok, len(set(final_patterns(zeroed))))
```

A parameter used at every timestep can be edited at every timestep at once. `tl.param` stages
the edit at each pass that consumes `w_rec.weight` (all 40), and the replay recomputes each
pass. With the recurrent weights zeroed the state no longer depends on its past, so all four
trials again end in one pattern. The live parameter is never written.

```python
# recipe-test: tool=torchlens expected=(1, True)
before = model.w_rec.weight.detach().clone()
lesioned = trace.fork().do(tl.param("w_rec.weight"), tl.zero_ablate())
RESULT = (len(set(final_patterns(lesioned))), torch.equal(model.w_rec.weight, before))
```

## 4. The recurrence Jacobian and fixed points, with plain torch.autograd

With the input silent, one step of the network is the map `F(h) = tanh(W_rec h + b)`. A fixed
point satisfies `F(h) = h`. The search below minimises the speed `q(h) = |F(h) - h|^2 / 2`
from many starting states at once (Sussillo and Barak, 2013): the captured trajectory, plus
perturbed copies of it so that some starts land away from the attractors. The Jacobian of `F`
at each fixed point decides its stability: every eigenvalue inside the unit circle means a
perturbation decays, so the point is stable.

```python
# recipe-test: tool=torchlens expected=4
silent = torch.zeros(1, N_BITS)


def update(h: torch.Tensor) -> torch.Tensor:
    return model.step(silent.expand(h.shape[0], -1), h)


torch.manual_seed(1)
starts = states.reshape(-1, HIDDEN).detach()
h = torch.cat([starts, starts + 0.5 * torch.randn_like(starts)]).requires_grad_(True)
optimizer = torch.optim.Adam([h], lr=1e-2)
for _ in range(1500):
    speed = 0.5 * (update(h) - h).pow(2).sum(dim=1)
    optimizer.zero_grad()
    speed.sum().backward()
    optimizer.step()

with torch.no_grad():
    speed = 0.5 * (update(h) - h).pow(2).sum(dim=1)
fixed_points: list[torch.Tensor] = []
for point in h.detach()[speed < 1e-10]:
    if all((point - kept).norm() > 1e-3 for kept in fixed_points):
        fixed_points.append(point)


def spectral_radius(point: torch.Tensor) -> float:
    jacobian = torch.autograd.functional.jacobian(lambda v: update(v[None])[0], point)
    return torch.linalg.eigvals(jacobian).abs().max().item()


stable = [point for point in fixed_points if spectral_radius(point) < 1.0]
print(f"{len(fixed_points)} fixed points, {len(stable)} stable")
RESULT = len(stable)
```

The search finds four stable fixed points (it also finds unstable ones, the saddles that
separate them; how many depends on the starts).

## 5. Read out the fixed points

Each stable fixed point, passed through the readout, gives one of the four bit patterns, and
the captured state at the end of each probe trial sits on one of them.

```python
# recipe-test: tool=torchlens expected=([(-1, -1), (-1, 1), (1, -1), (1, 1)], True)
with torch.no_grad():
    readouts = [tuple(model.readout(point).sign().int().tolist()) for point in stable]
settled = all(
    min((state - point).norm().item() for point in stable) < 1e-3 for state in states[:, -1]
)
print(sorted(readouts), settled)
RESULT = (sorted(readouts), settled)
```

The causal test puts the hidden state on one fixed point, the `(-1, -1)` memory, for passes 15
to 20 of every trial. The selection is the `passes=` window of section 3; its mask lands on
every pass in the window, so `do()` edits all six. After the window every trial reports
`(-1, -1)` until the end of the sequence: the network holds whatever fixed point it is placed
on, which is what it means for the fixed points to store the memory.

```python
# recipe-test: tool=torchlens expected=True
target = stable[readouts.index((-1, -1))]
moved = trace.fork()
window_sites = moved[tl.pass_variance(STATE, below=1e-6, passes=range(15, 21))]
moved.do(window_sites, tl.replace_with(target.expand(4, HIDDEN).clone()))
RESULT = set(final_patterns(moved)) == {(-1, -1)}
```

## Where to go next

- The same addressing works on any model whose recurrence is a Python loop over a module or a
  function: `layer:pass` labels, cross-pass selectors with `passes=` windows, and `tl.param`
  edits at every consuming pass. A fused `nn.LSTM`, `nn.GRU` or `nn.RNN` is one operation, so
  use a cell (`nn.GRUCell`) in a loop, or an explicit step as above, when you need the steps.
- Replace the flip-flop with delayed match-to-sample, context-dependent integration or any task
  from the task-RNN literature; the capture, window edits and fixed-point code do not change.
- [The NeuroAI feature-extraction journey](journey.md) covers recurrent time courses on a
  vision model (CORnet-S), and [torchlens.neuro](../reference/neuro.md) the comparison tools.

The Jacobian and fixed-point helpers here are plain `torch.autograd` code in the recipe, so they
work on any TorchLens version. They may become TorchLens functions after the 3.0 release.
