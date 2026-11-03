# T1: Jacobian lens on a small decoder (corpus-averaged layer-to-final Jacobian and vocabulary readout)

## Prompt to the agent

Implement the Jacobian lens for the decoder-only language model named in
`config.json` (`EleutherAI/pythia-70m`, 6 transformer blocks, residual width
512, loaded with the `transformers` library in float32 on CPU, evaluation mode).

Definitions. Number the transformer blocks 0 to 5. The *residual stream after
block l*, written `h_l[p]`, is the 512-dimensional vector that block l outputs
at token position p (the vector that enters block l+1; after block 5 it is the
vector that enters the final layer norm). The *final block* is block 5.

Fitting. For each text in `corpus.jsonl`, in file order: tokenise it without
adding special tokens (no beginning-of-sequence token; the published code
prepends one by default, and this task deliberately does not) and truncate it to
`max_seq_len` tokens, giving T tokens. A text with no valid position (fewer
than `skip_first_positions + 2` tokens) is skipped; none in this corpus is. Let the valid positions be `V = {skip_first_positions, ..., T-2}`
(the first 16 positions are excluded because of attention-sink statistics and
the last position because it has no next-token target). For each source layer
l in `source_layers`, the per-text Jacobian estimate is the 512 x 512 matrix

    J_l(text) = (1/|V|) * sum over p in V of sum over p' in V of d h_5[p'] / d h_l[p]

where `d h_5[p'] / d h_l[p]` is the Jacobian matrix of the final block's
output at position p' with respect to the residual after block l at position
p, with rows indexed by output dimensions and columns by input dimensions
(terms with p' < p are zero by causality). The lens matrix `J_l` is the plain
mean of `J_l(text)` over the texts (equal weight per text). Compute it exactly
with automatic differentiation, not by finite differences; note that summing
the cotangent over all target positions at once is allowed.

Readout. For each prompt in `readout_prompts.jsonl` (tokenised the same way,
no truncation), take the residual `h_l` at the last token position, transport
it with `J_l h_l`, apply the model's final layer norm and then its unembedding
matrix, and record the ten highest-scoring vocabulary token ids in descending
score order. Also record the ten highest-scoring token ids of the model's own
next-token distribution at that position.

Inputs (in `--inputs`): `config.json` (model, `max_seq_len`,
`skip_first_positions`, `source_layers`, `target_layer`, `readout_top_k`),
`corpus.jsonl` (one `{"text": ...}` per line), `readout_prompts.jsonl` (same format).

Required outputs (in `--outputs`):

- `lens.npz`: arrays `J_0`, `J_1`, `J_2`, `J_3`, `J_4`, each float32 of shape
  `[512, 512]`, rows = output (final-block) dimensions, columns = input
  (source-layer) dimensions.
- `readout.json`: `{"layers": [0, 1, 2, 3, 4], "top10": T, "model_top10": M}`
  where `T[i][k]` is the list of ten token ids for prompt i and source layer
  `layers[k]`, and `M[i]` is the list of ten token ids from the model's own
  logits for prompt i.

## Oracle and pass bars

The oracle is a port of the published reference estimator (companion code of
the July 2026 Jacobian lens paper) run on the same model, corpus and settings,
with an internal consistency check against per-target-position vector-Jacobian
products on one text. The held-out run uses the same corpus and different
readout prompts.
Held-out readout prompts, never shown to the solution, are used for the
readout bars.

| Bar | Pass |
|---|---|
| shapes | the five matrices and the readout lists are present with the right shapes |
| jacobian | for every source layer, `||J_agent - J_oracle||_F / ||J_oracle||_F < 1e-3` (maximum over layers reported) |
| readout (visible prompts) | mean over prompts and layers of `|top10_agent & top10_oracle| / 10` >= 0.9 (set overlap; the recorded order is not graded) |
| model top-10 (visible prompts) | the model's own top-10 set equals the oracle's for every prompt |
| readout (held-out prompts) | same statistic on the held-out prompts >= 0.9 |

Reference: Gurnee, Sofroniew et al., "Verbalizable Representations Form a
Global Workspace in Language Models" (July 2026); the estimator ported is the
one documented in the companion repository's fitting module at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.
