# Model summaries

`tl.summary(model, x)` is the one-call door: it runs ONE metadata-only
capture (eval + `no_grad` by default, every training flag and RNG stream
restored bit-identically), resolves the automatic view ladder, and RETURNS
a summary result. It never auto-prints; the result displays itself in a
REPL or notebook. `trace.summary()` renders the same report from an
existing capture without re-running anything.

All spellings on this page are DOCUMENTED-UNSTABLE pending the naming
session; the accounting laws (identity partition, raw-numbers pin, ASCII
contract) are stable.

## The sixty-second tour

```python
import torch, torchlens as tl
from torchvision import models

report = tl.summary(models.resnet50(weights=None), torch.randn(1, 3, 224, 224))
report.print()          # auto-detected charset at YOUR terminal
print(report)           # the canonical byte-stable ASCII form (logs, files)
report.total_params     # 25557032  -- a plain int, never a formatter object
report.to_pandas()      # typed DataFrame of the rendered rows
```

The default view is an automatic ladder under a 48-body-row budget
(48 is derived, not chosen: the smallest budget that renders the entire
VGG family at complete op level):

1. **Coalesced hybrid**, full depth -- every executed op appears exactly
   once; a module owning one op coalesces into a single `fc1 (Linear)`
   row; orphan functional ops (the classic torchsummary blind spot,
   e.g. torchvision's `torch.flatten`) get their own rows.
2. **Module tree**, full depth, strictly folded (`h.0..11 (GPT2Block x12)`).
3. **Module tree**, descending depth.
4. **Elision**: protected, totals-conserving accounting rows replace the
   lowest-share runs -- never truncation, so no model at any budget
   collapses to opaque container rows.

The disclosure line always states what resolved
(`view: module tree, depth 2 of 4, 4 folds`). On gpt2 the default renders
the COMPLETE folded tree in 19 body rows -- nothing hidden.

## The numbers are the true ones

- **Params** use torch's own Parameter-object identity:
  `report.total_params == sum(p.numel() for p in model.parameters())` by
  construction. Tied weights (gpt2's `wte`/`lm_head`) are counted once,
  DISCLOSED by name, and the per-module-path total prints beside the
  headline when the two differ -- so numbers are comparable against tools
  that double-count the tie.
- **FLOPs** come from the per-op two-term compute record. Alias
  (input/output) rows own nothing. Unknown ops stay unknown, are named
  with a remedy, and make totals explicit lower bounds.
- **MACs** are true multiply-accumulates (a ReLU has zero; a biased
  Linear's bias adds are FLOPs, not MACs) -- never `flops // 2`.
- Declared-but-never-executed parameters are split out and named
  (`3,326,696 never ran: AuxLogits.*` on eval-mode inception_v3).
- Every parameter identity and executed op event is owned by exactly one
  accounting row at every depth, fold, filter, and elision (the identity
  partition -- CI-enforced).

## The grammar

```text
tl.summary(model, input_args=None, input_kwargs=None,
    # Everything below is keyword-only.
    input_size=None,          # synthetic input shape (XOR real input_args)
    execution_mode="eval",    # eval | train | same    (one-call only)
    grad_mode="off",          # off | same             (one-call only)
    level="auto",             # auto | module | op
    view="overview",          # overview | compute (column bundles)
    depth="auto",             # auto | all | int
    columns=None,             # bundle name, exact list, or +name/-name deltas
    filter=None,              # regex or callable; totals stay whole-model
    buffers="summary",        # summary | hide
    fold_repeats="auto", max_rows=48,
    flop_convention="fma2",   # fma2 | fma1 (refuses typed if underivable)
    units="human",            # human | raw
    style="auto",             # auto | ascii | unicode (display helpers only)
)
```

Granularity, presentation, selection, numbers, and output are orthogonal
axes. Legacy spellings (`level="graph"`, `preset=`, `fields=`,
`show_ops=`, `count_fma_as_two=`) keep their historical byte-stable
rendering through one compatibility table; mixing the two grammars raises
`summary_option_conflict`. Nothing is ever accepted-and-ignored.

With NO input at all, `tl.summary(model)` infers a verified input via
`infer_input_shape`, reuses its verification trace (no second capture),
and disclosures the synthesis; decoded-output views refuse synthetic
inputs typed.

## The string contract

`str(report)` is canonical byte-stable ASCII with no ANSI or OSC-8 bytes
-- the equality, logging, and issue-paste contract. Unicode appears only
at explicit display boundaries:

- `report.print()` detects the actual sink (explicit argument >
  `TORCHLENS_SUMMARY_STYLE` > CI env > verified interactive terminal >
  ASCII; every uncertain check falls toward ASCII).
- `report.render("unicode")` is unconditional.
- The two renders differ ONLY through a declared one-to-one glyph table;
  `ascii == degrade(unicode)` holds byte-for-byte in CI.

In Jupyter, the report (and a bare `trace` cell) renders a
dependency-free escaped HTML table: sticky header, right-aligned
numerics, dark-mode safe, zero JavaScript.

## The result object

The report is a `str` subclass carrying detached typed data -- it retains
neither the model nor the Trace and survives cleanup and GC:

- Scalars (plain ints or None): `total_params`, `executed_params`,
  `unexecuted_params`, `trainable_params`, `frozen_params`,
  `total_flops_forward`, `total_macs_forward`, `unknown_flop_ops`,
  `capture_status`.
- Projections: `to_pandas(scope="display"|"all")`, `to_dict()` (bounded,
  JSON-safe), `to_markdown()`, `to_html()`, `render(style=)`, `print()`.
- `report.details()` serves the capture facts;
  `trace.provenance()` returns the full historical provenance preamble
  byte-for-byte (it no longer opens the table).

Capture honesty travels with the table: structure-only, unverified,
partial, rescued, and episode captures banner ABOVE the table via the one
shared honesty chokepoint, and the footer's five subjects (Params /
Compute / Memory / Graph / Capture) carry coverage, both payload scopes
(`at capture` vs `retained now`), and the health verdict.

## Performance

`tl.summary` runs one real capture (that is why its numbers are
measured facts, not estimates). The measured cost decomposition against a
plain forward is published and regenerated by
`tools/benchmark_summary.py` at `docs/benchmarks/summary_performance.md`.
