# T2: Jacobian lens on a byte decoder with alternating block types

Implement a corpus-averaged Jacobian lens for the model in `model.py`.
`build_model(weight_seed)` returns a seeded decoder with 256 byte symbols,
width 64 and six blocks: causal depthwise convolution at blocks 0, 2 and 4,
causal four-head attention at blocks 1, 3 and 5. Each block has a learned gated
residual scale. The decoder ends in RMS normalization with nonuniform learned
gains and an untied vocabulary readout. The shipped implementation defines
all details and draws every weight with `numpy.random.default_rng(weight_seed)`.
Run in evaluation mode, CPU float32; do not modify weights.

Encode every text in `corpus.jsonl` as UTF-8 bytes with ids 0 through 255,
without special tokens, and truncate to `max_seq_len` (64) bytes. For a text
with T bytes, the valid positions are V = {8, ..., T-2}, using
`skip_first_positions` from `config.json`. No next-token target exists for
the last byte. Use the same 16 texts, file order and equal per-text weighting
as the small-decoder Jacobian-lens task.

Let h_l[p] be the residual after block l at position p. The target is h_5,
before the final RMS normalization. For every l in `source_layers` (0 to 4),
compute exactly with automatic differentiation:

    J_l(text) = (1 / |V|) sum_{p in V} sum_{p' in V} d h_5[p'] / d h_l[p]
    J_l = mean over texts of J_l(text)

Rows are final-block output dimensions; columns are source-block input
dimensions. Causality makes terms p' < p zero. Do not divide by |V| twice.
Do not use finite differences. Summing target positions before differentiating
is a permitted exact way to evaluate this estimator.

For each text in `readout_prompts.jsonl`, encode as bytes without truncation.
For each source layer, take the last byte's residual h_l, form J_l h_l, apply
the decoder's final RMS normalization and then its untied readout. Record the
ten largest-logit byte ids in descending score order. Also record the model's
own ten largest-logit next-byte ids from h_5 with final normalization/readout.

Inputs: `config.json`, `model.py`, `corpus.jsonl`, `readout_prompts.jsonl`.
Required outputs:

- `lens.npz`: float32 `J_0` through `J_4`, each `[64,64]`.
- `readout.json`: `{"layers": [0,1,2,3,4], "top10": T, "model_top10": M}`;
  T[prompt][layer] and M[prompt] are lists of ten integer byte ids.

The grader computes exact position-by-position Jacobians with
`torch.func.jacrev`, selecting valid source and target positions explicitly,
then caches these matrices under `--cache`. Shapes and finite matrices must
match. Each layer's relative Frobenius error must be below 1e-3, and the mean
top-ten set overlap with the oracle must be at least 0.9 on both visible and
held-out readout prompts. The model's own top-ten lists must also reach 0.9
overlap. The held-out run uses the same fitting corpus and unseen readout
prompts; the second solution invocation must still write all outputs.

Pitfalls: distinguish bytes from language-model tokens, retain the source
residual in the differentiation graph, sum over all valid target positions,
and keep the output/input matrix orientation. RMS normalization includes
nonuniform gains, so omitting it can change vocabulary rankings.
