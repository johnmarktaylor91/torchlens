# Transformer pictures (`torchlens.transformer`)

`torchlens.transformer` renders the transformer-visualization picture families --
token-labeled attention heatmaps and grids, the attention atlas, colored-token
strips, logit-lens prediction pictures, per-token loss/entropy strips, the
term-complete score decomposition -- plus the one picture the dormant
incumbents (CircuitsVis, BertViz, Ecco, inspectus) never shipped: attention
annotated with **measured** ablation effects.

Every spelling is DOCUMENTED-UNSTABLE pending the naming session. Import as
`import torchlens.transformer` (the root-name budget is frozen; a root spelling
awaits the registration sweep). Display records are session-only values and
are never persisted into `.tlspec` artifacts.

## Install

Static is the product: every numeric picture saves genuine paper-ready PNG,
SVG, and PDF through matplotlib, resolved at **call time**:

```
pip install "torchlens[viz]"
```

Without matplotlib, any save-format request refuses typed
(`tv_matplotlib_missing`) printing that exact command -- and the
zero-dependency SVG/HTML token-strip emitters (`token_strip_html`,
`token_strip_svg`, `prediction_table_html`) still work on a bare install.
PDF is the documented paper format (vector, self-contained,
text-extractable). SVG text defaults to `svg.fonttype='path'` (renders
identically everywhere); pass `svg_fonttype="none"` for editable text in
Illustrator/Inkscape.

## Typed records

One closed family of immutable records feeds matplotlib, the HTML emitters,
and the CircuitsVis bridge without re-derivation:

| record | picture |
|---|---|
| `AttentionView` | attention heatmaps, grids, atlas, bridge payloads |
| `TokenScores` (+ `TokenScoreRow`) | colored-token strips, NMF factor rows |
| `PredictionTrajectory` | layer-prediction ribbon, answer trajectory + rank panel |
| `PredictionTable` | top-k tokens per position |
| `TokenMetrics` | per-token loss / per-context entropy strips |
| `ScoreDecomposition` | the BertViz neuron-view data (term-complete) |
| `CausalReceipt` / `Annotation` | measured-effect attachment (closed 4-kind grammar) |
| `SavedFigure` | every save's result: paths + every rendered disclosure line |

Attention semantics are fixed and printed: canonical
`[query_head, destination_query, source_key]` layout, "rows attend from
(query); columns attend to (key)" on the artifact, separate query/key token
axes even for self-attention (T5 cross-attention is rectangular), a fixed
shared `[0, 1]` probability color domain (per-panel rescale is expert-only
and renders visibly labeled "panels not comparable"), cropping that never
renormalizes (`AttentionView.crop_to` measures the omitted keys and mass on
the uncropped pattern), and the permanent footer: *attention weights are
routing measurements, not causal importance*.

Mask provenance is a three-source hierarchy, never inferred from zeros:
recorded SDPA call arguments, the captured eager additive-mask operand
(compared against `finfo(dtype).min` -- real HF fills are not `-inf`), then
explicit user metadata. A view with none of these renders unmarked with the
exact wording "zero and masked positions are not distinguished".

## Quick tour

```python
import torchlens as tl
import torchlens.transformer as tviz

log = tl.trace(model, input_ids)  # eager attention captures the real pattern
views = tviz.attention_views(log, tokens=tokens)

tl.viz.attention(views[0], "layer0.pdf")            # per-head grid, paged
tl.viz.attention(views[0].head_view(9), "h9.svg")   # one square panel
tl.viz.attention_atlas(views, "atlas.pdf")           # overview + detail pages

# Streaming lens pictures (probabilities always full-vocabulary denominator)
from torchlens.semantic.logit_lens import logit_lens_predictions
preds = logit_lens_predictions(log, tokens=[answer_id])
traj = tviz.prediction_trajectory(preds, tokenizer=tok, target_token_id=answer_id)
tviz.render_prediction_ribbon(traj, "ribbon.pdf")
tviz.render_answer_trajectory(traj, "answer.pdf")

# Layers-by-positions heatmap over the full logit-lens result
from torchlens.semantic.logit_lens import logit_lens
tl.viz.logit_lens(logit_lens(log), "lens.pdf", tokens=tokens, tokenizer=tok)

# Loss + entropy strips from ONE logits source (alignment proof travels)
loss, entropy = tviz.token_metrics(logits, input_ids[0], token_strings)
tl.viz.metric_strip(loss, "loss.pdf")

# Colored tokens from the attribution kit's payload
scores = tviz.TokenScores.from_attribution(result.payload())
tl.viz.token_strip(scores, "tokens.pdf")
print(tviz.token_strip_html(scores))  # zero-dependency notebook path
```

## Causal receipts (the differentiator)

The mech-interp kit **executes** counterfactuals
(`torchlens.mechinterp.patch_heads_grid`); tviz owns attachment validation,
the visual grammar, and the artifact:

```python
import torchlens.mechinterp as mi

grid = mi.patch_heads_grid(model, clean, corrupted, metric, budget=200)
receipt = tviz.receipt_from_patch_grid(
    grid,
    layer="transformer.h.0.attn",
    negative_control=neg,     # measured self-patch delta (must be ~0)
    positive_control=pos,     # measured large perturbation (must move)
    joint_effect=joint,       # one extra run: all heads patched together, as a delta
)
tl.viz.ablation_grid(receipt, "receipts.pdf")
ann = tviz.Annotation.from_receipt(receipt)
tl.viz.attention(views[0], "annotated.pdf", annotation=ann)
```

A grid's `values` are the metric of each patched run. Its receipts also
record the unpatched baselines, `grid.receipts["clean_metric"]` and
`grid.receipts["corrupted_metric"]`. A receipt cell is an effect: the patched
metric minus `corrupted_metric`, because every cell patches clean values into
a corrupted rerun, so a cell that changes nothing reads `+0.00` (the same
quantity the self-patch negative control measures). The figure's footer
prints the baseline value, and the joint effect and both controls are deltas
from the same run. `receipt_from_patch_grid` refuses a grid without
`corrupted_metric` (`tv_receipt_invalid`) rather than plotting raw metric
values. The footer's height is reserved below the matrix, so the disclosure
lines never cover a cell or a label.

The validity rules are enforced, not advisory: zero fires refuses (a hook
that did not fire is an exception, never a number), a moved negative control
refuses, an unmoved positive control refuses (the only check that catches
the silent no-fire class), and the grid **figure** refuses without the
measured joint effect -- single-head effects need not be additive (measured on
gpt2 layer 0: one head `+13.27` alone, all twelve together `-0.68`), so
every grid prints the joint measurement beside the cell sum. Unmeasured
cells are blank, never zero, and the figure prints "measured N of M".

Under grouped-query attention a key or value site holds one tensor per KV
group, read by several query heads, so a grid over that site
(`mi.patch_grid(..., sites="v", axes=("layer", "head"))`) measures groups, not
query heads. Pass the grouping and the receipt becomes a kv-group receipt:

```python
gqa = tviz.GqaInfo(num_query_heads=14, n_kv_heads=2)
receipt = tviz.receipt_from_patch_grid(
    v_grid, layer="model.layers.0.self_attn", negative_control=neg,
    positive_control=pos, joint_effect=joint, kv_groups=gqa,
)
receipt.sharing_lines()  # ("kv group 1 of 2, shared by query heads 0-6", ...)
tl.viz.ablation_grid(receipt, "kv_receipts.pdf")
```

The grid labels each column `kv g` over its query-head range and prints that
no cell is a per-query-head effect. `Annotation.from_receipt` refuses a
kv-group receipt (`tv_annotation_invalid`): attention panels are per query
head, and one group's effect is not any single query head's.

To measure one query head at a value site, patch its own copy of the group
with `mi.patch_query_heads_grid`. The grid counts query heads, so its
receipt is an ordinary query-head receipt; the grid's `effect_site` becomes
the receipt's `effect_source`, printed on the figure:

```python
q_grid = mi.patch_query_heads_grid(model, clean, corrupted, metric, site="v")
receipt = tviz.receipt_from_patch_grid(
    q_grid, layer="model.layers.0.self_attn", negative_control=neg,
    positive_control=pos, joint_effect=joint,
)
receipt.effect_source  # "expanded value copy read by one query head (op-level interceptor)"
ann = tviz.Annotation.from_receipt(receipt)
```

Annotation kinds are a closed set of four -- `descriptive_head_score`,
`additive_logit_contribution`, `screening_estimate`, `intervention_effect`
-- each with its own glyph and legend section; only a valid `CausalReceipt`
can mint `intervention_effect`, and effects never color attention cells
(the pattern channel stays the pattern; effects ride headers and borders).

## Score decomposition (neuron view data)

```python
record = tviz.score_decomposition(log, "transformer.h.0.attn", head=3,
                                  destination=4, source=1)
tviz.render_neuron_card(record, "card.pdf")
```

The record carries post-transform q/k vectors, per-dimension products, the
scale, and every additive term under a hard sum-to-score invariant: if the
terms do not reproduce the captured scores facet within tolerance, the
record refuses (`tv_decomposition_unclosed`) rather than mis-decomposing.
gpt2/bert-class families close today; RoPE families refuse per-family until
term extraction covers them. The full neuron *browser* is deliberately
skipped -- per-architecture reverse engineering is the treadmill that killed
it; any future browser reads this record.

## Bridges

CircuitsVis is the one supported bridge (`circuitsvis_attention`): the
payload is built from the same typed records the static renderers read,
size-estimated before handoff, refused typed above the tested ceiling
(`tv_bridge_payload_too_large`), and every handoff carries the
dormancy/CDN disclosure -- the no-network guarantee attaches only to
torchlens's own emitters. `bertviz_tuple(views)` emits the HF
`output_attentions` tuple layout; it is the BERT numeric parity **oracle**
and a documented recipe, not a supported bridge.

## Recipe: Ecco-style NMF factor view (docs recipe, hardened)

NMF is a docs recipe by decision (D20): the multi-row strip renderer is the
product; the factorization is yours, with the disclosures below mandatory.

```python
import numpy as np
from sklearn.decomposition import NMF  # your environment; record its version

# 1. A NAMED activation matrix: [n_tokens, n_features], nonnegative transform.
acts = log["transformer.h.4.mlp.act_1"].out[0].float().numpy()
matrix = np.maximum(acts, 0.0)          # ReLU-style nonnegativity; NAME yours

# 2. Factorize with an explicit rank and seed; record the error.
rank, seed = 8, 0
nmf = NMF(n_components=rank, init="nndsvda", random_state=seed, max_iter=500)
weights = nmf.fit_transform(matrix)     # [n_tokens, rank]
recon_err = float(nmf.reconstruction_err_)

# 3. Stability over >= 3 seeds: rerun and compare factor correlations; a
#    factor that does not reappear across seeds is noise, not structure.

# 4. Render: one strip row per factor, disclosures in the footer.
import torchlens.transformer as tviz
rows = tuple(
    tviz.TokenScoreRow(label=f"factor {k}", scores=tuple(weights[:, k].tolist()))
    for k in range(rank)
)
record = tviz.TokenScores(
    tokens=token_strings, rows=rows, domain="magnitude_sequential",
    footer_lines=(
        "matrix: h.4.mlp.act_1, ReLU nonnegativity",
        f"sklearn.decomposition.NMF {sklearn.__version__}, rank={rank}, seed={seed}",
        f"reconstruction error: {recon_err:.4g}; stability checked over 3 seeds",
        "factors are descriptive; order is unidentifiable; no factor is causal",
    ),
    provenance="NMF recipe (docs/reference/tviz.md)",
)
tl.viz.token_strip(record, "factors.pdf")
```

The last footer line is not optional: factors are descriptive, order is
unidentifiable, and no factor is causal. Promotion to a native appliance is
a docs-to-API move on measured demand; the record/renderer seam already
exists.

## Wave 2 (plumbed now, rendered later)

Episode/generation views: `EpisodeCoordinates` (step / role / completion /
member) lands on the records from day one, so per-step sheets and the
staircase render against the episode substrate without recomputation.
A value-site grid measures KV groups (a group is shared storage); kv-group
receipts (above) render the disclosure ("kv group g of G, shared by query
heads a-b"). Per-query-head value-site receipts come from the op-level
interceptor (`mi.patch_query_heads_grid`, above), which patches one query
head's expanded copy.

## Refusal codes

All tviz refusals ride `torchlens.errors.RenderError` with a stable
`fields["code"]`; the contract rows live in
`docs/reference/error_refusal_contract.md`.
