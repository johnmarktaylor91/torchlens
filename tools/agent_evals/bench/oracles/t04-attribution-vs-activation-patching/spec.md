# T4: first-order attribution versus exact attention-head replacement

Measure how attention heads affect an indirect-object decision in pretrained
`gpt2` (12 blocks, 12 heads per block). Run on CPU in float32, evaluation mode,
without caching. `config.json` names the model; `prompts.json` provides one
clean/corrupt text pair and positive/negative next-token strings. Tokenize
without special tokens. The pair has equal token length, and each name string
including its leading space encodes to one token.

Define the decision metric M as the last position's positive-token logit minus
its negative-token logit. Keep these two token identities fixed for all
experiments, including the corrupt baseline. Record M_clean and M_corrupt.

For each layer l and attention head h, let z[l,h,p] be the 64-dimensional
attention result at token position p, before concatenating heads and applying
the attention output projection. It is the attention-weighted sum of value
vectors, before that projection, its bias, or residual addition.

Compute two matrices, each indexed `[layer, head]`:

- Exact replacement: run the clean prompt with z[l,h,p] replaced at every
  position by the corresponding corrupt-prompt vector. Replace one head at a
  time. Set `exact[l,h] = M_replaced - M_clean`.
- First-order attribution: differentiate M on the unmodified clean run with
  respect to z. Set `attribution[l,h] = sum_p sum_d
  (d M_clean / d z_clean[l,h,p,d]) *
  (z_corrupt[l,h,p,d] - z_clean[l,h,p,d])`.

Write `patching.npz` with float32 `exact` and `attribution`, both `[12,12]`,
and scalar float32 `clean_metric`, `corrupt_metric`. The attribution is a
linear estimate; do not replace it by exact replacement scores, normalize it,
change its sign, or restrict the sum to the last token.

The grader independently recomputes attention equations and runs individual
head replacements. All shapes and finite values are required. Each matrix
must have Spearman correlation at least 0.98 with its corresponding oracle;
each baseline metric must have maximum absolute error at most 1e-4. The same
bars apply to an unseen second pair in a separate held-out run.

Pitfalls: the input of the attention output-projection module concatenates
all heads. Replacing that entire input changes twelve heads, not one. A
forward pre-hook can isolate head slices. Retain the clean computation graph
for gradients, and detach corrupt values. Gradients at earlier positions can
contribute even though the metric is read at the last position.
