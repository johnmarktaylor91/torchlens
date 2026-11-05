# T5: fit a trait direction, steer generation, and monitor it

Use the pretrained `EleutherAI/pythia-70m` causal language model and its
associated tokenizer. Load in CPU float32, evaluation mode, with eager
attention. Number blocks from zero. The residual after block 3 is the
512-vector at each token position before block 4. Tokenize each text
individually without adding special tokens, padding or truncation.
Set `torch.set_num_threads(1)` for both generation paths. These short,
single-sequence runs use the same fixed CPU-thread policy for their timings.

`prompts.json` contains 24 positive-sentiment sentences (set A), 24
negative-sentiment sentences (set B), and eight generation prompts, all ordered.
For every sentence in each fitting set, take the last token's residual after
block 3. Define direction d = mean(A) minus mean(B), with equal weight per
sentence. Save this raw difference, then form u = d / ||d||_2 for steering.

For each generation prompt, generate exactly 12 new tokens greedily (argmax,
smallest token id in a tie), both with the native network and with steering.
In the steered run add 4.0*u to the residual after block 3 at EVERY position
at EVERY generation step. Evaluate the entire accumulated sequence each step;
do not reuse cached states from earlier steps, stop at end-of-text tokens,
or sample. Before each addition record the scalar projection of the LAST
position's residual onto u. The sequence then continues through later blocks
with the modified residual. Keep fitting and generation text orders unchanged.

Write:
- `direction.npz`: float32 `direction` `[512]`, the raw mean difference d.
- `generation.json`: `{"items": [{"native_token_ids": [...],
  "steered_token_ids": [...], "projections": [...]}, ...]}` in prompt order.
  Token lists contain ONLY the 12 newly generated integer token ids; projections
  are 12 scalars from the steered run, measured before addition.
- `steered_first_logits.npz`: float32 `logits` `[8,vocab_size]`, the full
  steered next-token scores at the FIRST generation step, before softmax.
- `timing.json`: positive finite `native_seconds` and `steered_seconds`, measured
  with `time.perf_counter` around each whole prompt-set generation loop in the
  same process. Exclude model loading, fitting and serialization from both.

Use `solve.py --inputs DIR --outputs DIR`. `config.json` fixes the model,
zero-based layer index, coefficient, new-token count and attention mode. The
seed is reserved for deterministic extensions; this task requires no random
sampling. A held-out run supplies eight different generation prompts and the
same fitting sentences and bars.

The oracle independently fits and intervenes with a forward hook. Bars:
all shapes and finite float32 outputs; direction cosine at least .999;
max absolute first-step logit error divided by max absolute oracle logit at
most .01; steered new-token agreement at least 90%; native new-token agreement
100%; maximum projection absolute error strictly below `1e-3`; steered/native
time ratio at most 1.5. The ratio is measured in the same process and is
advisory across machines. Both visible and held-out runs must pass.

Pitfalls: subtract B from A; normalize for addition and monitoring, but save
the raw direction. Adding only at the final position changes the computation.
A projection measured after addition is offset by the coefficient. Forward
hooks must preserve tensor-versus-tuple block return conventions.
