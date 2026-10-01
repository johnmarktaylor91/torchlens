# Semantic lenses (themes roster v1)

**Every spelling on this page is DOCUMENTED-UNSTABLE** pending the naming
session; the public `draw(theme=..., skin=...)` kwargs are a named
[UI-SPRINT] fork. Until ratification the door is:

```python
from torchlens.visualization import lenses

lenses.draw_with_lens(trace, "speed", skin="paper", vis_fileformat="svg")
resolution = lenses.resolve_lens(trace, "speed")   # inspect without drawing
```

## The two axes

A **LENS** answers a question (which draw settings, channels, and label
rows); a **SKIN** styles the ink (palette, ramps, typography). They compose
freely. Presets are DEFAULTS, never overrides: an explicit user kwarg
always wins, and the user's `node_spec_fn` keeps the last word. The full
precedence chain:

```
skin defaults < lens members < explicit user kwargs < intervention/status
marks < active channel specs < preset spec slot < user node_spec_fn
```

## The roster (nine rows)

Inclusion rule (the standing arbiter): a preset earns a ROW iff it has its
own HEADLINE evidence requirement AND its own mandatory disclosure text a
composition would not inherit.

| Row | Tier | Question | Refusal semantics |
|---|---|---|---|
| `overview` (default candidate) | CORE | what is this model? | never refuses |
| `blueprint` | CORE | show me everything | never refuses; pins today's bare-draw contract |
| `debug` | CORE | what is broken? | refuses above the detail ceiling (`lens_above_detail_budget`) |
| `speed` | CORE | where does time go? | refuses without timing evidence; zero resolved coverage refuses |
| `dims` | CORE | what shape is the data? | refuses above the detail ceiling |
| `transformer` | CORE | how is attention wired? | refuses on zero attention structures (`lens_attention_subject_absent`) |
| `memory` | EXTENDED | what is big in bytes? | refuses without activation-memory evidence |
| `sequence` | EXTENDED | how do the passes unfold? | refuses on a single-pass trace (`lens_single_pass_no_sequence`) |
| `compute` | EXTENDED | where are the FLOPs? | refuses without FLOP evidence |

A CORE row failing its full legibility stratum at freeze blocks the
release; an EXTENDED row drops individually to wave 2 with its named
promotion path (themes memo section 10), never silently.

Named compositions (gallery recipes, full strata, pre-authorized
promotion): `runtime_storage` (speed + `size_by="bytes"`), `vision` (dims +
leftright + `color_by="flops"`), `debug_edge_shapes`.

## Honesty machinery

- **View-aware source families (N16).** Perf rows declare a FAMILY
  (`time`/`bytes`/`flops`); resolution binds `func_duration` on unrolled
  views and `total_func_duration` on rolled views, the rolled case carries
  the mandatory aggregation line, and zero coverage refuses typed
  (`lens_source_zero_coverage`) -- never a populated legend over an
  unencoded graph.
- **The budget resolver (N10).** Rows that compact search the shipped
  dials (float collapse schedule below the optimizer ceiling; call-depth
  truncation above it) for the coarsest setting inside the two-sided band
  (floor 100 / target 160 / ceiling 220), plus a coverage floor (0.80) when
  a colour channel is active. `collapse="none"` wins when it fits; the
  resolved dial is disclosed BY NAME, always.
- **Six-state nonfinite channel (N6).** The debug lens derives
  finite / nan / +inf / -inf / mixed / not_checked per op from saved
  payloads and paints motifs per the per-shape table (striped fills on
  boxes, wedged on ellipses, border-only degrade elsewhere -- box3d
  explicitly covered). Degrade is two-mode: zero coverage = one legend
  line, no motifs; partial coverage = per-node `not_checked` marks.
  NOT-CHECKED-as-FINITE is a zero-tolerance honesty class.
- **Display filter (N9).** `DisplayFilter(exclude=...)` /
  `DisplayFilter(include=...)` with the closed v1 tokens `reshapes`,
  `constants`, `non_module_ops` (documented as torchview MIGRATION PARITY,
  never the headline), a same-trace Selection, or a raw predicate (the
  power spelling; strict boundary raise preserved and teaching). Every
  filtered artifact carries the rendered caption ("displayed X of Y
  eligible ops; Z filtered (...)"), dashed bridged edges with midpoint
  "via N hidden" labels, and the bridged-reachability legend line.
- **Rank transform.** Perf rows encode by RANK (ordinal, not ratio) -- the
  only measured candidate that cannot degenerate; linear min-max painted
  77-99% of real-model nodes into the bottom decile. Degenerate domains
  (min == max) render UNENCODED with a note, never mid-ramp.
- **Skins are live (N4).** Semantic role colours resolve through the
  active skin's palette at the one theme seam; the Okabe-Ito set ships as
  data with its CVD evidence attached (the default flip is FORK-2, JMT's).
  Each skin carries a 3-anchor ramp and the N17 neutral "aggregate, not
  encoded" fill for collapsed boxes under an active channel.

## The validation harness

`torchlens.visualization.lenses.audit`: the evidence corpus (toy builders +
guarded real members + per-artifact manifests), the Stage-0 deterministic
audit (`run_stage0` -- geometry via `dot -Tjson`, size caps, colour spread,
disclosure presence, the label-spelling audit, CVD palette gates),
RenderIR-derived answer keys (`generate_answer_key`; hand keys prohibited),
and the naive-evaluator battery harness (`build_packet`, `score_responses`,
`freeze_threshold` -- the anchor-midpoint procedure). The nine honesty
classes are zero-tolerance forever. Running evaluators is D03's job; the
normative gallery/battery matrix is `docs/reference/lens_gallery_spec.md`.
