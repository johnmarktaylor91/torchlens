# Coming from torchinfo

This page is for people who print model summaries with `torchinfo.summary` or its predecessor
`torchsummary`. `tl.summary(model, x)` maps to `torchinfo.summary(model, input_data=x)`: one call,
one table, params and FLOPs per row, a totals footer. The runnable example uses only
`summary(...).total_params`, so it does not depend on a particular torchinfo release; the
measurements below were taken on gpt2 against torchinfo's behavior at that time. Checked 2026-11-03.

## The same task in both

The task: the total parameter count of a small network from one call.

### In torchinfo

Needs torchinfo; the fence is skipped where it is not installed.

```python
# migration-test: tool=torchinfo expected=212
import torch.nn as nn
from torchinfo import summary

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
result = summary(model, input_size=(2, 8), verbose=0)
RESULT = result.total_params
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=212
import torch
import torch.nn as nn

import torchlens as tl

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
report = tl.summary(model, torch.zeros(2, 8))
RESULT = report.total_params
```

## Translation table

| torchinfo | TorchLens |
|---|---|
| `summary(model, input_data=x)` | `tl.summary(model, x)` |
| `summary(model, input_size=(1, 3, 224, 224))` | `tl.summary(model, input_shape=(1, 3, 224, 224))` |
| `depth=3` | `depth=3` |
| `col_names=("output_size", "num_params", "mult_adds")` | `columns=[...]` / `view="compute"` |
| `col_names=(..., "input_size", "kernel_size", "groups")` | `columns=["+input", "+kernel_size", "+groups"]` |
| `col_names=(..., "trainable")` | `columns=["+params_trainable"]` (a count; the `train` column says yes/no/partial) |
| Lightning's `Mode` column | `columns=["+mode"]` (`module.training` as captured) |
| params size (MB) footer line | `size` in the Params footer |
| `mode="eval"` | `execution_mode="eval"` (the default; state restored) |
| result object (`.total_params`, ...) | the returned report (`.total_params`, `.to_pandas()`, ...) |
| printed side effect | RETURN-only; the result self-displays |

### Coming from torchsummary

`torchsummary` (and its fork `pytorch-model-summary`) is frozen; its own
README points to torchinfo. The swap is one import:

```text
# before
from torchsummary import summary
summary(model, (3, 224, 224))

# after
import torchlens as tl
tl.summary(model, input_shape=(1, 3, 224, 224))  # the batch dimension is explicit
```

torchsummary's `input_size` leaves out the batch dimension and its
`device="cuda"` default fails on a CPU-only machine; TorchLens takes the
full shape and runs wherever the model is.

Inputs that crash torchsummary (its own issue tracker) and run here:
any LSTM, GRU, or other tuple-returning layer (the row prints every
member shape: an LSTM shows `output`, `h_n`, and `c_n`), a stock
`nn.TransformerEncoderLayer`, integer or embedding inputs, dict inputs
(`input_kwargs=`), and Hugging Face GPT-2.

Totals: torchsummary counts a reused module's parameters again at every
call and skips any parameter not named `weight` or `bias`. TorchLens
counts each Parameter object once (ties and reuse are named in the Params
footer) and counts every parameter whatever its name. Its "Params size
(MB)" line is the `size` part of the Params footer.

## What TorchLens adds

1. **Functional ops exist.** torchinfo (like torchsummary before it) sees
   `nn.Module`s only: torchvision's `torch.flatten`, a residual `+`, a
   functional attention call -- invisible. The TorchLens default view
   shows every executed op exactly once (vgg16 renders all 40 ops
   including the orphan `torch.flatten`).
2. **A default that never cliffs.** torchinfo's `depth=` either spells
   out twelve identical GPT-2 blocks or hides whole subtrees. The auto
   ladder folds repeats (`h.0..11 (GPT2Block x12)` -- gpt2's COMPLETE
   tree in 19 rows) and, when a flat model can't fit any depth, elides
   the lowest-share run into an accounting row instead of truncating.
3. **Aggregate rows that conserve totals.** torchinfo's container rows
   don't sum to the footer. Every TorchLens view is an exact partition:
   every parameter identity and executed op event is owned by exactly one
   row, at every depth, fold, filter, and elision -- CI-enforced.
4. **Honest unknowns.** An op with no FLOP rule renders `?`, is named in
   the footer with a remedy, and turns totals into labeled lower bounds
   -- never silently zero.
5. **A safe one-call door.** `tl.summary` runs eval + `no_grad` and
   restores every training flag, BatchNorm buffer, and RNG stream
   bit-identically (state-dict-hash pinned). Measurements come from a
   real capture of YOUR forward, not shape arithmetic -- so multi-pass
   modules, weight-tying, and data-dependent branches are facts, not
   guesses.

### Where the numbers differ from torchinfo's

Measured on gpt2:

- **Params: torchinfo prints 163,037,184.** The `wte`/`lm_head` weight is
  ONE tied Parameter; torch's own count is **124,439,808**. TorchLens
  headlines the identity-true count, prints the per-path total beside it,
  and names the tie -- so you can still reconcile against torchinfo.
- **"mult-adds": torchinfo prints ~163.43G** for a batch-1, 16-token
  forward whose true figure is on the order of 2G MACs. TorchLens counts
  MACs from the per-op two-term compute record: a ReLU has zero MACs, a
  biased Linear's bias adds are FLOPs but not MACs, and nothing is ever
  relabeled `flops // 2`.
- **Alias double counting.** The output pseudo-row must own nothing: the
  producing op already ran. On gpt2 that error is +31% of true forward
  FLOPs (invisible on CNNs at +0.05%, which is why it survived
  everywhere). TorchLens alias rows are non-accounting by construction.

Constants above re-derive from the committed script:
`python tools/derive_summary_pins.py gpt2`.

### Side by side (rendered output, from the test goldens)

ASCII (the canonical `str(report)` -- what logs and CI see):

```
_GoldenToy | input (1, 3, 8, 8) float32
view: hybrid, all 4 ops
name (type)    output        params    (%)  fwd flops
-----------------------------------------------------
conv (Conv2d)  (1, 4, 8, 8)     108   7.8%      13.8k
relu (ReLU)    (1, 4, 8, 8)       -      -        256
flatten_1      (1, 256)           -      -          0
fc (Linear)    (1, 5)          1.3k  92.2%       2.6k
-----------------------------------------------------
Params   1,388 declared | trainable 1,388 (100%) | size 5.42 KiB
Compute  16.6k FLOPs fwd (fma=2) | 8.2k MACs | 4/4 ops known
Memory   <masked>
Graph    4 ops, 4 rows shown | 6 tracked tensor rows
Capture  torch | complete | verification not checked | no NaN/Inf (exact)
Legend   1 MAC = 2 FLOPs
more     result.details() | view='compute' for per-row MACs | docs/reference/summary.md
```

(The Memory line is masked because its live measurement varies by host; every other
byte above is pinned by the dual-charset goldens in
`tests/test_summary_rebuild_charset.py`.)

Unicode (`report.print()` in a verified interactive terminal -- same
bytes through a declared glyph table):

```
_GoldenToy | input (1, 3, 8, 8) float32
view: hybrid, all 4 ops
name (type)    output        params    (%)  fwd flops
─────────────────────────────────────────────────────
conv (Conv2d)  (1, 4, 8, 8)     108   7.8%      13.8k
relu (ReLU)    (1, 4, 8, 8)       -      -        256
flatten_1      (1, 256)           -      -          0
fc (Linear)    (1, 5)          1.3k  92.2%       2.6k
─────────────────────────────────────────────────────
Params   1,388 declared | trainable 1,388 (100%) | size 5.42 KiB
Compute  16.6k FLOPs fwd (fma=2) | 8.2k MACs | 4/4 ops known
Memory   <masked>
Graph    4 ops, 4 rows shown | 6 tracked tensor rows
Capture  torch | complete | verification not checked | no NaN/Inf (exact)
Legend   1 MAC = 2 FLOPs
more     result.details() | view='compute' for per-row MACs | docs/reference/summary.md
```

## What torchinfo does that TorchLens does not

- torchinfo is a single-purpose, lightweight table printer. A `tl.summary` call runs a real
  forward through the TorchLens capture machinery, so it costs more than shape arithmetic.
- torchinfo's printed-table muscle memory (`verbose=1` printing as a side effect, `col_names`
  tuples) maps onto different spellings: TorchLens returns a report object instead of printing,
  and columns use the `columns=[...]` grammar above. These names are experimental
  and may change in any release.
- For profilers and per-kernel timing that neither summary tool provides, see
  [profilers_alongside.md](profilers_alongside.md) and [landscape.md](landscape.md).

## Using both

### torchinfo feeds TorchLens

There is no bridge, and none is needed: both read a plain `nn.Module`. Compare the two footers
by total parameter count; the tie disclosure and the `fma=2` convention line explain the two
systematic divergences (below).

### TorchLens feeds torchinfo

There is no bridge. The returned report exposes `.total_params`, `.to_pandas()` and the printed
text, which can sit beside a torchinfo table in a notebook or a log.

## Semantic differences

- The one-call door is safe by default: eval mode, `no_grad`, and every training flag, BatchNorm
  buffer and RNG stream restored afterwards. `torchinfo` documents its own `mode=` argument for
  this; the TorchLens spelling is `execution_mode=`.
- Weight tying changes the headline: a tied weight is counted once, as torch counts it.

```python
# migration-test: tool=torchlens expected=True
import torch
import torch.nn as nn

import torchlens as tl


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        return self.head(self.embed(ids))


model = Tied()
report = tl.summary(model, torch.tensor([[1, 2, 3]]))
RESULT = report.total_params == sum(p.numel() for p in model.parameters())
```

### What to expect when numbers differ

If your torchinfo baseline disagrees with TorchLens, check the footer
first: the tie disclosure (params) and the convention line (`fma=2`,
MACs) explain the two systematic divergences above. Everything else is a
bug report we want: the identity-partition suite and the committed
derivation script make every constant reproducible.
