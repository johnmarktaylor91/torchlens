# T9: replace a recurrent network's hidden state

Implement a causal state-replacement experiment on the seeded language model
in `model.py`. It embeds 64 symbols into 32 features, uses a two-layer GRU
with 48 features per layer, and has an untied linear readout. Run CPU float32,
evaluation mode, without changing weights. All inputs and weights come from
`numpy.random.default_rng` with the seeds in `config.json`.

Call `build_model(weight_seed)` and `make_pairs(stimulus_seed, n_pairs, length)`.
Pairs are ordered; the first array is clean and the second is corrupt. Both
have shape `[n_pairs,24]`. All initial hidden states are zero. Steps are
zero-based: state at k means the state AFTER consuming token k.

For every pair and k in `[4,8,12,16]`, run the corrupt sequence through k,
replace BOTH layers' states with the clean sequence's states after token k,
and continue on corrupt tokens k+1 through 23. Replace the state once only.
Record the readout logits after each subsequent token. Also run both sequences
without any replacement. Preserve pair order and step order.

Write `patched.npz` with float32 arrays:
- `logits`: `[n_pairs,4,24,64]`; NaN before and at k, finite at every later step.
- `clean_logits` and `corrupt_logits`: `[n_pairs,24,64]`, finite throughout.

The grader computes a separate stepwise implementation of the GRU reset,
update and candidate equations from the weights. Shapes, float32 types and
NaN locations must match exactly; max absolute error across ALL defined
logits (including both unmodified runs) must be strictly below `1e-5`.
A held-out run changes both weight and stimulus seeds, with the same bars.

Pitfalls: patching only the top layer changes the experiment; reusing a state
at every later step prevents the intervened recurrence from evolving. A state
before token k is different from a state after token k. For PyTorch's GRU,
the reset gate multiplies the hidden affine candidate including its bias.
Use the standard `solve.py --inputs DIR --outputs DIR` interface.
