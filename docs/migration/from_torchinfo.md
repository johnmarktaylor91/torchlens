# Coming from torchinfo

`tl.summary(model, x)` is a drop-in mental model for
`torchinfo.summary(model, input_data=x)`: one call, one table, params and
FLOPs per row, a totals footer. This page maps the spellings, then names
what changes -- including the numbers torchinfo gets wrong that TorchLens
now gets right.

## Spelling map

| torchinfo | TorchLens |
|---|---|
| `summary(model, input_data=x)` | `tl.summary(model, x)` |
| `summary(model, input_size=(1, 3, 224, 224))` | `tl.summary(model, input_size=(1, 3, 224, 224))` |
| `depth=3` | `depth=3` |
| `col_names=("output_size", "num_params", "mult_adds")` | `columns=[...]` / `view="compute"` |
| `mode="eval"` | `execution_mode="eval"` (the default; state restored) |
| result object (`.total_params`, ...) | the returned report (`.total_params`, `.to_pandas()`, ...) |
| printed side effect | RETURN-only; the result self-displays |

## The five asks torchinfo does not serve

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

## Numbers torchinfo gets wrong that this surface gets right

Measured on gpt2 (the most-compared model in the world):

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

## Side by side (real rendered output, from the test goldens)

ASCII (the canonical `str(report)` -- what logs and CI see):

```
_GoldenToy | input (1, 3, 8, 8) float32
view: hybrid, all 4 ops
name (type)    output        params        (%)  fwd flops
---------------------------------------------------------
conv (Conv2d)  (1, 4, 8, 8)     108   ... 7.8%     13.82K
relu (ReLU)    (1, 4, 8, 8)       -          -        256
flatten_1_3    (1, 256)           -          -          0
fc (Linear)    (1, 5)         1.28K  ### 92.2%      2.56K
---------------------------------------------------------
Params   1,388 declared | trainable 1,388 (100%)
Compute  16.64K FLOPs fwd (fma=2) | 8.19K MACs | 4/4 ops known (formula-exact)
Memory   forward peak 3.4 MB (torch) | activations at capture 3.9 KB, retained now 3.9 KB
Graph    4 ops, 4 rows shown | 6 tracked tensor rows
Capture  torch | complete | health CHECKED-AND-CLEAN
More:    result.details() | view='compute' for per-row MACs | docs/reference/summary.md
```

(The Memory line is a live measurement and varies by host; every other
byte above is pinned by the dual-charset goldens in
`tests/test_summary_rebuild_charset.py`.)

Unicode (`report.print()` in a verified interactive terminal -- same
bytes through a declared glyph table):

```
_GoldenToy | input (1, 3, 8, 8) float32
view: hybrid, all 4 ops
name (type)    output        params        (%)  fwd flops
─────────────────────────────────────────────────────────
conv (Conv2d)  (1, 4, 8, 8)     108   ░░░ 7.8%     13.82K
relu (ReLU)    (1, 4, 8, 8)       -          -        256
flatten_1_3    (1, 256)           -          -          0
fc (Linear)    (1, 5)         1.28K  ███ 92.2%      2.56K
─────────────────────────────────────────────────────────
Params   1,388 declared | trainable 1,388 (100%)
Compute  16.64K FLOPs fwd (fma=2) | 8.19K MACs | 4/4 ops known (formula-exact)
Memory   forward peak 3.4 MB (torch) | activations at capture 3.9 KB, retained now 3.9 KB
Graph    4 ops, 4 rows shown | 6 tracked tensor rows
Capture  torch | complete | health CHECKED-AND-CLEAN
More:    result.details() | view='compute' for per-row MACs | docs/reference/summary.md
```

## What to expect when numbers differ

If your torchinfo baseline disagrees with TorchLens, check the footer
first: the tie disclosure (params) and the convention line (`fma=2`,
MACs) explain the two systematic divergences above. Everything else is a
bug report we want: the identity-partition suite and the committed
derivation script make every constant reproducible.
