# T6: indirect-object circuit scores from outgoing residual-edge attribution

Use pretrained `gpt2`, CPU float32, evaluation mode and no cache. The input
`config.json` supplies a name list, seed and 256-example count. Generate examples
with `prompts.py:make_prompts`, which draws a sentence template, place and
object for each example. Examples alternate ABBA and BABA: A is the
indirect object, B the repeated subject, and the final A is the next-token
answer. The corrupt prompt replaces the repeated subject occurrence by a
third, distinct name, destroying the duplicate-name relation while preserving
the earlier names. Tokenize without special tokens. All names encode as one
token including their leading space; all generated texts have equal length.

For each example, M is the last position's clean indirect-object-name logit
minus the clean subject-name logit. These two name identities stay fixed when
processing the corrupt text. Use the clean computation for all derivatives.

For sender head h at layer l, let c[l,h,p] be its contribution in residual
coordinates: its 64-dimensional attention result multiplied by its own rows
of the attention output projection. Exclude the shared projection bias. Let
Delta c = c_corrupt - c_clean, measured at every token position p.

The following defines the required edge-attribution score precisely. At every
receiving layer j, imagine giving each head k its own copy r[j,k,p] of that
layer's incoming raw residual. Apply the usual layer normalization, query,
key and value transformations and causal attention for just that head to its
copy. All copies have the ordinary clean residual value, so this computes the
same model. Let G[j,k,p] = d M / d r[j,k,p], holding the copies for the other
heads independent. This derivative includes subsequent model computation.
Let G_final[p] = d M / d h_final[p], where h_final is the residual before the
final layer normalization, with no intermediate branches beyond that point.

For each prompt compute the direct edge and all later attention-head edges:

    e_direct(l,h) = sum_{p,d} Delta c[l,h,p,d] * G_final[p,d]
    e(l,h,j,k) = sum_{p,d} Delta c[l,h,p,d] * G[j,k,p,d], for j > l
    importance(l,h) = abs(e_direct(l,h)) + sum_{j>l,k} abs(e(l,h,j,k))
    scores[l,h] = mean over all 256 prompts of importance(l,h)

This is a sum of magnitudes of outgoing first-order residual-edge effects.
Receiver derivatives are evaluated through one receiver head at a time;
using the total gradient of a shared residual would mix heads. The incoming
edge is the direct residual connection from sender to receiver, so other
intervening heads and MLP branches are held independent for that edge. The
receiver's ordinary downstream metric derivative is retained. Absolute values
are taken after summing positions and features and before averaging prompts.
Do not sum signed edges and then take one absolute value.

Write `head_scores.npz` with finite nonnegative float32 `scores`, `[12,12]`,
indexed `[layer,head]`, with higher values meaning greater importance.

The grader computes its own scores using independent per-head residual inputs,
and asserts its own scores reach ROC AUC >= 0.85 against the published IOI head
set, including name movers, negative name movers, S-inhibition, induction,
duplicate-token, previous-token and backup-name-mover heads. The output must
also have membership AUC >= 0.85 and Spearman correlation >= 0.98 with the
oracle scores. The membership labels are used only for grading, never to
compute the score. There is no second input set for this task.

Pitfalls: a forward pre-hook on the attention output projection records
concatenated head results. For receiver gradients, a local vector-Jacobian
product for one head's attention computation, weighted by that head's global
metric gradient, is equivalent to making independent copies of the raw
residual. The gradient is with respect to the raw residual before layer
normalization, not the normalized vectors.
