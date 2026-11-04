# TorchLens example bank

Small tested snippets for agents writing TorchLens code: one task each, grouped by the object
they use, with the printed output. Every example runs in the TorchLens test suite against this
version, so the calls here exist and behave as shown.

- Find an example by public name or by task in `INDEX.md` beside this file.
- Each example names a fixture: run the fixture's setup below first, then the snippet. A
  snippet with fixture `none` needs only the imports in that fixture.
- `# Output:` comment lines at the end of a snippet are what it prints; `...` stands for a
  value that varies between runs or machines.
- A pair marked `kind: wrong` / `kind: right` shows a known trap and its fix.
- Public functions and classes not listed in `INDEX.md` carry a runnable example in their
  docstring: `help(tl.stats.Mean)`.

## Fixtures

### fixture: toy

A two-layer network with a functional relu. Its layers are `linear_1_1`, `relu_1_2` and
`linear_2_3` (each op label adds the pass, `relu_1_2:1`); its modules are `fc1` and `fc2`.

```python
import torch
from torch import nn

import torchlens as tl


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


torch.manual_seed(0)
model = Toy().eval()
x = torch.randn(3, 4)
```

### fixture: cnn

A small convolutional network; its modules are `0`, `1` and `2`. Docstring examples use the
same model.

```python
import torch
from torch import nn

import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 2, 3, padding=1))
model = model.eval()
x = torch.randn(1, 3, 8, 8)
```

### fixture: loop

A recurrent model: one module called three times per forward, so its layers have three passes.

```python
import torch
from torch import nn

import torchlens as tl


class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


torch.manual_seed(0)
model = Loop().eval()
x = torch.randn(2, 4)
```

### fixture: branch

A model with a data-dependent `if`, so its trace records a conditional.

```python
import torch
from torch import nn

import torchlens as tl


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        if h.sum() > 0:
            return torch.relu(h)
        return -h


torch.manual_seed(0)
model = Gate().eval()
x = torch.randn(2, 4)
```

### fixture: lm

A two-block GPT-2 with random weights (no download; needs `transformers`). Its blocks are
`transformer.h.0` and `transformer.h.1`, its final norm `transformer.ln_f`.

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

torch.manual_seed(0)
config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
lm = GPT2LMHeadModel(config).eval()
ids = torch.tensor([[5, 17, 42, 8, 3]])
```

### fixture: none

Only the imports; the snippet builds what it uses.

```python
import torch

import torchlens as tl
```
