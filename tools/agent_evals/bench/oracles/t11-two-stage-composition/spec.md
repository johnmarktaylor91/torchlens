# T11: compose two local representation experiments

Use pretrained `gpt2` and its fast tokenizer in CPU float32, evaluation mode,
with eager attention. Number the twelve blocks from zero. The residual after
block 6 means the vector at each position that enters block 7, before final
normalization. Tokenize each entire sentence separately, with no special
tokens, padding, or truncation. Do not tokenize an entity in isolation.

For each ordered item in `items.json`:
1. Encode `source`, run stage one, and retain the residual after block 6 at
   the token span covering `source_span`, a `[start,end)` character interval.
2. Encode `target`, a sentence of the form "The capital of X is". Run stage
   two, replacing the residual after block 6 at the token positions covering
   `target_span` with stage one's representation in source-token order.
   Continue the target computation through all later blocks and its readout.
3. Also run the target unmodified. Record the full vocabulary scores at the
   final target position, before softmax, in both cases.

Character offsets are zero-based and end-exclusive. A token belongs to the
span exactly when its nonempty tokenizer character interval has positive
overlap with the named character interval. Use the smallest contiguous
`[start_token,end_token)` range covering these tokens. Leading spaces inside a
token are allowed; exclude all tokens with no overlap. There are six visible
items and four held-out items. Entities appear in different source contexts
and at different positions from their target template positions. The same
entity text appears internally in source and target, ensuring matching span
lengths, including multiple-token spans.

For any length mismatch, truncate trailing source tokens when the source is
longer; when shorter, repeat the final source-token vector until the target
span is filled. Report `truncate`, `repeat_last`, or `none` for each item.
Replacement affects all residual features at all positions in that span,
only at block 6, once in stage two. Source and target are separate runs, with
independent token positions and no reused attention cache.

Write:
- `composition.npz`: float32 arrays `patched_logits` and `baseline_logits`,
  both `[n_items,vocab_size]` in item order.
- `alignment.json`: an ordered list of objects `{"source_range": [s,e],
  "target_range": [s,e], "adaptation": "none"}` (or the applicable adaptation).

Use `solve.py --inputs DIR --outputs DIR`. `config.json` fixes the model,
zero-based block and attention mode; no sampling or other random choices
are needed. The reserved seed is 0.

The oracle explicitly runs the two stages sequentially. It independently
computes token spans from the tokenizer's offset mapping and captures and
replaces the input to block 7. All array shapes and finite float32 types must
match. Alignment ranges and adaptation labels must match exactly; maximum
absolute patched-logit error must be strictly below `1e-4`, and baseline-logit
error must be strictly below `1e-4`, on visible and held-out items alike.

Pitfalls: character offsets and token offsets differ, and tokenizing only the
entity can change its segmentation. Block 5 and block 6 produce different
representations. Replacing only the first token loses the remaining span.
Forward hooks must preserve tensor-versus-tuple block outputs; autograd is
not required for this experiment.
