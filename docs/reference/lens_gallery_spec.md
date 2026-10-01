# Lens gallery + naive-battery specification (NORMATIVE)

This document is the normative gallery and evaluation matrix for the lens
roster (themes memo sections 7-9). It is the contract D03 executes: which
artifacts exist, what each must disclose, and what gates them. The memo
section is the authority; this spec operationalizes it.

Artifacts (DOT + SVG + 150-DPI PNG + per-artifact run manifest) live under
sprint scratch, NEVER the repo.

## 1. The corpus

Real weights, real processors and inputs, eval mode, fixed seeds, pinned
revisions. The manifest names the EXACT checkpoint.

- Core real (D03, pinned revisions): torchvision `resnet18` + `resnet50`
  (pinned weight enums, one real ImageNet image); HF
  `openai-community/gpt2` + `distilbert/distilgpt2` (>= 8-token prompt);
  `google/vit-base-patch16-224` with its real processor; `t5-small`
  (cross-attention is not self-attention); `openai/clip-vit-base-patch32`
  (two towers); timm `convit_base`; `densenet121` (917 ops); `gpt2-large`
  (above the optimizer ceiling); pinned Milesial `unet_carvana`;
  torchvision `raft_small` (GRU iterations); menagerie `phased_lstm`; one
  `nn.LSTM` classifier beside a Python-loop `LSTMCell`; `yolov4`
  (menagerie); one real double-backprop render.
- **The single most valuable member: the distilgpt2 10-token generation
  episode (2,802 ops)** -- the only member exercising the above-ceiling
  path, the rolled/unrolled source pairing, and the aggregation-line
  requirement together.
- The mixed-precision leg is a real model under `torch.autocast` (measured
  46% dtype split on resnet18), NOT a bf16-loaded model; keep a plain fp32
  control.
- Coverage stratum: every channel-bearing row on resnet18, resnet50,
  vit_b_16, distilgpt2, coverage recorded against the 0.80 floor.
- Toys (additional, never substitutes) and stress members: shipped in
  `torchlens.visualization.lenses.audit.corpus` (residual diamond, tied
  LSTMCell loop, nonfinite chain, encoder/decoder with skips, filtered
  chain, 2-node minimal, attention toy, StressDiamond10k; random-init
  resnet18/densenet121 for structural strata).

## 2. The gallery matrix

One gallery entry per cell; every entry carries its run manifest.

| Stratum | Members | Gate |
|---|---|---|
| 9 lens rows x core-real corpus | roster x members above | Stage-0 + battery accuracy thresholds |
| 3 compositions (`runtime_storage`, `vision`, `debug_edge_shapes`) | full strata | same as rows; promotion pre-authorized if no higher harmful-inference rate than constituents |
| 5 skins x {overview, speed} | skin matrix | Stage-0 + all-four CVD simulations |
| Legendless arm | each channel-bearing row, one member | 5 judges; legend-forcing rule (>=60% legendless ships legend-optional; 60%>x>=90% aided ships legend-FORCED; <90% aided redesigns) |
| Sentinels | the nine honesty classes, five judges each | ZERO tolerance -- one confirmed hit sends the lens back |
| Controls (every round) | known-bad (white-anchored min-max time render; random-rainbow fill), known-good (hand-tuned candidate), blueprint (no-encoding floor) | a battery that cannot separate them is a broken battery |

## 3. Stage-0 (deterministic; blocks before any evaluator)

Shipped as `lenses.audit.run_stage0`: `dot -Tjson` geometry (node overlap +
label penetration > 0.25 pt; head/tail labels counted as candidates, the R0
baseline is the blocking reference), PNG <= 40 MPx / longest side <= 12,000
px / aspect within [1:6, 6:1], >= 8 distinct encoded fills where
cardinality permits, <= 40% of encoded nodes per 5-luminance bucket, <= 10%
above luminance 245, coverage line whenever < 100%, disclosure presence
(dial by name, filter caption, aggregation line, bridged legend line), the
label-spelling audit (bridged disclosures are midpoint `label=`; no preset
introduces a new xlabel/headlabel family), and the CVD palette gates.
Panel-added checks that need render-pair orchestration (compaction
non-identity, unit invariance, channel exclusivity) ship as harness
helpers/tests and run per stratum.

## 4. The battery (Stage 1) and beauty pass (Stage 2)

Evaluators are fresh Fable instances with no repo, docs, TorchLens name,
lens name, or channel vocabulary; filenames masked; one evaluator, one
image. Packets (`lenses.audit.build_packet`) fix the question order (free
response -> free inventory -> forced choice -> RenderIR task probes ->
honesty probes -> transcription -> legend reveal in the withheld arm) and
the presentation rules (downscaled whole first, then original + 3x-4x crops
of densest labels / legend / extrema / marked path; every difference
question on a crop). Answer keys are GENERATED
(`lenses.audit.generate_answer_key`); hand keys are prohibited.

Counts: 3 judges per normal image, 5 per legendless/sentinel image; ~700
single-image judgements per full round, ~20 for the dev loop.

Accuracy thresholds are PROVISIONAL priors frozen from anchors in R0 by the
anchor-midpoint procedure (`lenses.audit.freeze_threshold`): 90%
post-legend semantic identification; 65% pre-legend headline (no core cue
below 60%); 90% extrema/status/boundary and IO/direction agreement; 85%
path F1; 95% honesty precision, < 2% harmful-inference rate; 99% label
character accuracy, zero clipped labels. Stage 2: five blind pairwise
reviewers prefer-or-tie vs today's default on ResNet-18, GPT-2, ViT, real
U-Net, and the recurrent model; preference can never waive a semantic gate;
the audit outranks taste.

## 5. Rounds and stop criteria

R0 baseline + calibration on TODAY's code (geometry gate baselined; anchors
frozen). R1 wave-1 plumbing. R2 ONE tuning session (size_range, ramp
anchors, coverage-floor value, rank-vs-log for bytes, transformer
direction, dims threshold) decided from side-by-side rendered candidates.
R3 full battery. R4 fix-and-confirm. DONE when every applicable gate is
green on the full matrix and two consecutive rounds add no new gate
failure; three flat rounds = BLOCKED, never a fourth. Failure handling:
classify (disclosure / encoding / layout / density / battery ambiguity),
change ONE mechanism, regenerate the affected stratum, re-audit, re-judge.

## 6. Credits page candidates

torchview (compact overview, visibility ergonomics), penzai/treescope
(diverging-around-zero, NaN-as-pattern), Okabe-Ito and ColorBrewer
(palettes), matplotlib (style-sheets-as-defaults precedence), netron
(categorical type colours), bertviz/circuitsvis (the transformer
audience's vocabulary), graphviz (the substrate).
