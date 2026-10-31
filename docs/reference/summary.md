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
    input_shape=None,          # synthetic input shape (XOR real input_args)
    execution_mode="eval",    # eval | train | same    (one-call only)
    grad_mode="off",          # off | same             (one-call only)
    level="auto",             # auto | module | op
    view="overview",          # overview | compute | stats (column bundles; stats = per-site payload statistics)
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
axes. Contradictory axes (`level="op"` with a `depth=` cut) raise
`summary_option_conflict`. Nothing is ever accepted-and-ignored.

The legacy spellings are removed (no aliases). Each refuses typed and names
its successor:

| Removed spelling | Use instead |
|---|---|
| `preset=` | `view=` (`"overview"` / `"compute"` / `"stats"`) for columns, `level=` for row grain |
| `fields=` | `columns=` |
| `show_ops=`, `include_ops=` | `level="op"` |
| `mode=` | `level="op"`, or `fold_repeats=False` to unfold repeated runs |
| `print_to=` | `report.print(file=...)`, or `print_to(str(report))` |
| `count_fma_as_two=` | `flop_convention="fma2"` (was `True`) / `"fma1"` (was `False`) |
| `show_input_preprocessing_details=` | `trace.provenance()` and `trace.input_preprocessor` |
| `level="overview"` | `view="overview"` (the default) |
| `level="compute"`, `level="cost"` | `view="compute"` |
| `level="graph"` | `trace.to_agent_json()` (op rows, edges, hierarchy) or `trace.draw()` |
| `level="memory"` | `trace.profile(sort_by="activation_memory")`; totals stay in the footer |
| `level="control_flow"` | `trace.conditional_records` and the `conditional_*` columns of `trace.to_pandas()` |
| `level="waterfall"` | `trace.profile(level="op")`, or `trace.to_pandas()` in execution order |
| `level="output"` | `trace.output_table()` |

## The stats view

`trace.summary(view="stats")` is the per-site payload statistics table: one row per
executed op (`level="module"` keeps the module-output payloads), read in graph order, with
typed row states (`ok` / `unsaved` / `disk_backed` / `unsupported` / `unscanned`) so an
absent number is a state, never a zero. `max_scan_elements=` is the scan budget (accepted only
with `view="stats"`; later retained rows become `unscanned` and the header says how many).
Sampled families print with a `~` mark and never a clearance; `= parent` marks come only from
the corroborated distribution verdict and are never folded. `str(report)` is the table,
`report.rows` the typed rows, `report.to_pandas()` the frame; `columns=` takes the stats
column names (`name`, `state`, `n`, `mean`, `sd`, `min`, `max`, `zero_count`, `nan_count`,
`inf_count`, `same_as_parent`, `reason`) or `+name` / `-name` deltas.

## Columns

`columns=` takes a bundle name (`"overview"`, `"compute"`, `"stats"`), an exact
ordered list, or `+name` / `-name` deltas on the view's bundle. A `+name`
lands at its registry position (`+input` before `output`, `+flops_pct`
beside `fwd flops`).

| Column | Header | Cell |
|---|---|---|
| `name` | `name (type)` | attribute name and class, indented by depth; `[never ran]`, `[pruned]`, and `*` (tied) tags |
| `input` | `input` | shapes of the distinct tensors entering the row |
| `output` | `output` | every member shape of the row's output in return order (an LSTM row shows `output`, `h_n`, `c_n`) |
| `kernel_size` | `kernel` | kernel size of the row's own conv or pooling op |
| `groups` | `groups` | conv groups of the row's own conv op |
| `params` | `params` | parameters per module path |
| `params_pct` | `(%)` | share of the declared parameter total |
| `params_trainable` | `trainable` | trainable parameters per module path (a count) |
| `flops` | `fwd flops` | forward FLOPs under `flop_convention` |
| `flops_pct` | `(%)` | share of the whole-model known forward FLOPs |
| `macs` | `macs` | MACs the row owns |
| `train` | `train` | `yes` / `no` / `partial` over `requires_grad` (auto-shown when trainability is mixed) |
| `mode` | `mode` | `train` / `eval` from `module.training` as captured; `mixed` when the row's modules disagree |
| `passes` | `passes` | call count |
| `evidence` | `evidence` | weakest FLOP evidence class of the row |

`mode` reports the flags the capture ran under: the one-call door
defaults to `execution_mode="eval"`, so pass `execution_mode="same"` (or
summarize a `tl.trace` capture) to see the model's own flags.

A FLOP cell whose count includes `scaled_dot_product_attention` ends in
`^`, and the footer explains it: attention FLOPs count the full
query x key square, and attention masks and `is_causal` are not
discounted.

```python
import torch, torchlens as tl
from torch import nn

net = nn.Sequential(nn.Conv2d(3, 8, 3, groups=1), nn.ReLU(), nn.Conv2d(8, 8, 3, groups=8))
report = tl.summary(
    net,
    torch.randn(1, 3, 16, 16),
    columns=["+input", "+kernel_size", "+groups", "+params_trainable", "+flops_pct", "+mode"],
)
print(report)
```

With NO input at all, `tl.summary(model)` infers a verified input via
`guess_input_shape`, reuses its verification trace (no second capture),
and discloses the synthesis.

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
- `report.totals` (`SummaryTotals`) also carries `params_bytes` (the
  unique parameters' storage), `params_effective` (set on a pruned model:
  the raw total minus mask zeros), and `optimizer_state_params` /
  `optimizer_state_bytes` (set when the capture was given an optimizer).
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

The Params footer also carries the parameter storage (`size 5.6 KB`); on
a model pruned with `torch.nn.utils.prune` it names the pruned tensors
with raw and effective nonzero counts (`pruned: 3,016 effective nonzero
of 3,124 raw (108 masked in conv.weight)`); and when the capture was
given an optimizer (`tl.trace(model, x,
capture=tl.options.CaptureOptions(optimizer=opt))`) it adds
`optimizer state N params, X KB`, read from the optimizer's live state
(zero before the first `step()`).

## Performance

`tl.summary` runs one real capture (that is why its numbers are
measured facts, not estimates). The measured cost decomposition against a
plain forward is published and regenerated by
`tools/benchmark_summary.py` at `docs/benchmarks/summary_performance.md`.
