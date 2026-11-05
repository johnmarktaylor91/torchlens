# T3: causal restoration at attention, MLP and residual sites

Compute a causal restoration heatmap for the pretrained `gpt2` decoder
(12 blocks, CPU float32, evaluation mode, no cache). Each factual prompt in
`prompts.json` supplies its text, a half-open character interval identifying
the subject, and the correct next-token string, including its leading space.
Tokenize without special tokens. The answer string must encode to one token.

A subject token is any token whose character interval overlaps the supplied
subject interval. Start a single `numpy.random.default_rng(noise_seed)` for
the input set, then process prompts in file order. For each prompt draw
`standard_normal((number_of_subject_tokens, 768))`, cast to float32, and multiply
by the float32 scalar `3 * std(W_E)`; the population standard deviation is
computed over all entries of the token embedding matrix in float32. Add this
noise to the subject token embeddings, before adding position embeddings.
Consume no other random draws. The multiplier and seed are in `config.json`.

First measure the clean and corrupted probabilities of the correct next token.
For each block l, each token position p, and each of the following three sites,
run the corrupted prompt again, replacing only that site's vector at p by its
value from the clean run:

- `attn`: the attention sub-layer output after its output projection and
  residual dropout, before addition to the incoming residual.
- `mlp`: the feed-forward sub-layer output after its output projection and
  dropout, before addition to the intermediate residual.
- `resid`: the residual stream immediately after the block's second addition,
  before the next block or the final layer normalization.

The heatmap entry is the resulting probability of the correct next token,
using the last token's logits and a softmax over the whole vocabulary.
Restore exactly one site and one position per experiment. All other values
are recomputed from the corrupted run. Positions include the subject and the
last token; block indices and token positions start at zero.

Write `heatmap.npz` using `numpy.savez`, with float32 arrays `attn`, `mlp`,
`resid` of shape `[n_prompts, 12, n_tokens_max]`, padding the right end of each
prompt's token dimension with NaN. Also save float32 `clean_prob` and
`corrupt_prob`, each shape `[n_prompts]`. Inputs are `config.json` and
`prompts.json`. A second run uses unseen prompts and a different noise seed.

The grader independently recomputes attention and residual equations, checks
its baseline against the model, and caches its oracle under the supplied
cache directory. For both input sets every shape and NaN padding mask must
match, Pearson correlation over finite entries must be at least 0.99 for each
of the three sites separately, and each baseline probability must have
maximum absolute error at most 1e-4. Missing or nonfinite values fail.

Pitfalls: clone saved module outputs; attention can return a tuple whereas a
block may return a tensor. A forward hook must preserve the output container.
Replacing an entire block is not equivalent to replacing its attention or MLP
output. Use the identical noise samples for every restoration of a prompt.
